"""The two-tower contrastive model: image and report encoders, and the loss.

A knee MRI study is three images (one per anatomical plane) and its report is one
piece of text. Both are mapped to unit vectors in a shared space, where the paired
study and report should be each other's nearest neighbour.

The image tower is the part that is not standard CLIP. It encodes each plane
separately with a shared DINOv2 backbone, then pools the three plane vectors with
learned attention rather than averaging them, because the planes are not equally
informative for a given finding -- a meniscal tear reads from sagittal images, a
collateral ligament from coronal. Learned plane embeddings let the attention depend
on which plane a vector came from, not only on its content.

Both backbones are mostly frozen; see :func:`freeze_then_unfreeze_last`. With a few
thousand studies, adapting the last blocks plus the projections is what fits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet statistics, which the pretrained DINOv2 weights were trained against.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _encoder_blocks(model: nn.Module) -> list[nn.Module]:
    """Find a Hugging Face backbone's transformer block list.

    Vision and text architectures disagree on the attribute path, and none of them
    expose a common accessor, so the known layouts are probed in order.
    """
    for path in (
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("transformer", "layer"),
        ("transformer", "layers"),
    ):
        value: Any = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None:
            return list(value)
    raise ValueError(f"Cannot locate transformer blocks in {type(model).__name__}")


def freeze_then_unfreeze_last(model: nn.Module, count: int) -> int:
    """Freeze a backbone, then re-enable its last ``count`` blocks. Returns block count.

    The final layer norm is unfrozen alongside them: it sits after the blocks being
    adapted, and holding it fixed fights the shift they introduce. ``count == 0``
    leaves the backbone entirely frozen, in which case its tower will also keep it
    in eval mode so dropout and norm statistics stay fixed.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    blocks = _encoder_blocks(model)
    if count:
        for block in blocks[max(0, len(blocks) - count) :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name in ("layernorm", "layer_norm"):
            if (normalization := getattr(model, name, None)) is not None:
                for parameter in normalization.parameters():
                    parameter.requires_grad = True
    return len(blocks)


class ProjectionHead(nn.Module):
    """Map backbone features to a unit vector in the shared embedding space.

    The input norm stabilizes the very different feature scales the two backbones
    produce, and the output is L2 normalized so a dot product is a cosine
    similarity -- which is what the loss and every retrieval metric assume.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(inputs), dim=-1)


class _FrozenAwareModule(nn.Module):
    """Base for towers that must keep a fully frozen backbone in eval mode.

    A frozen backbone still has dropout and normalization layers whose train-mode
    behaviour would inject noise into features nothing can learn to compensate for.
    """

    backbone: nn.Module
    backbone_frozen: bool

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self


class ImageTower(_FrozenAwareModule):
    """DINOv2 plane encoder with learned attention pooling across the MRI planes."""

    def __init__(self, backbone: nn.Module, embedding_dim: int, n_planes: int = 3):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        # Each plane is described by its CLS token concatenated with its mean patch
        # token, hence the doubled width everywhere below.
        feature_dim = hidden * 2
        # Small random init: a learned marker of which plane a vector came from,
        # added before attention so the pooling weights can be plane-dependent.
        self.plane_embeddings = nn.Parameter(torch.randn(n_planes, feature_dim) * 0.02)
        self.plane_attention = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.projection = ProjectionHead(feature_dim, embedding_dim)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.backbone_frozen = not any(parameter.requires_grad for parameter in backbone.parameters())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, planes, 3 slice channels, H, W]`` into ``[B, embedding_dim]``.

        The three sampled slices occupy the RGB channels the pretrained backbone
        expects, which is why ``slices_per_plane`` is fixed at three.
        """
        if images.ndim != 5 or images.shape[1:3] != (self.plane_embeddings.shape[0], 3):
            raise ValueError(
                "Expected [B, planes, 3 slice channels, H, W] with "
                f"{self.plane_embeddings.shape[0]} planes, got {tuple(images.shape)}"
            )
        batch, planes = images.shape[:2]
        # Planes are folded into the batch so one backbone pass encodes all of them.
        flat = images.reshape(batch * planes, *images.shape[2:]).float()
        flat = (flat - self.mean) / self.std
        tokens = self.backbone(pixel_values=flat).last_hidden_state
        features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1)
        features = features.reshape(batch, planes, -1) + self.plane_embeddings[:planes]
        weights = self.plane_attention(features).squeeze(-1).softmax(dim=1)
        study_features = torch.einsum("bp,bpd->bd", weights, features)
        return self.projection(study_features)


class TextTower(_FrozenAwareModule):
    """Transformer report encoder, pooled to a single vector then projected.

    ``cls`` pooling suits encoders trained with a sentence-level objective;
    ``mean`` pooling suits those that are not, and is what E5 expects. Mean pooling
    respects the attention mask so padding cannot dilute a short report.
    """

    def __init__(self, backbone: nn.Module, embedding_dim: int, pooling: str = "cls"):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("text pooling must be 'cls' or 'mean'")
        self.backbone = backbone
        self.pooling = pooling
        self.projection = ProjectionHead(int(backbone.config.hidden_size), embedding_dim)
        self.backbone_frozen = not any(parameter.requires_grad for parameter in backbone.parameters())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            features = tokens[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
            features = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(features)


class ImageReportContrastiveModel(nn.Module):
    """Both towers plus the learned temperature that scales their similarities.

    Temperature is stored as a log inverse scale so it stays positive under
    unconstrained gradient updates; training clamps its upper end each step.
    """

    def __init__(self, image_tower: ImageTower, text_tower: TextTower, temperature: float = 0.07):
        super().__init__()
        self.image_tower = image_tower
        self.text_tower = text_tower
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_tower(images)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_tower(input_ids, attention_mask)

    def forward(
        self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scaled ``[B, B]`` similarity logits and both embedding batches."""
        image_embeddings = self.encode_images(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        logits = self.logit_scale.exp() * image_embeddings @ text_embeddings.T
        return logits, image_embeddings, text_embeddings


def symmetric_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """CLIP-style paired InfoNCE, averaged over both retrieval directions.

    Row ``i`` of ``logits`` scores study ``i`` against every report in the batch, so
    the correct answer is always the diagonal: cross entropy against the index range
    pushes each pair together and every other pair in the batch apart. Both
    directions are averaged so neither retrieval task is favoured.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"Expected square paired logits, got {tuple(logits.shape)}")
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def build_model(
    image_model_name: str,
    text_model_name: str,
    embedding_dim: int,
    image_unfreeze_last: int,
    text_unfreeze_last: int,
    *,
    image_model_revision: str | None = None,
    text_model_revision: str | None = None,
    text_pooling: str = "cls",
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    pretrained_backbones: bool = True,
    local_files_only: bool = False,
) -> tuple[ImageReportContrastiveModel, Any, dict[str, Any]]:
    """Assemble the model and tokenizer, and report what was actually resolved.

    ``pretrained_backbones=False`` builds architecture-only backbones from config,
    for the case where a checkpoint already carries every weight: it avoids
    downloading weights that are about to be overwritten. The returned info dict
    records the resolved commit hashes, which is the provenance a compact
    checkpoint needs to be reloadable.

    Transformers is imported lazily so the rest of the package -- config, cohort
    manifests, metrics -- stays usable in environments without it installed.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if pretrained_backbones:
        image_backbone = AutoModel.from_pretrained(
            image_model_name, revision=image_model_revision, local_files_only=local_files_only
        )
        text_backbone = AutoModel.from_pretrained(
            text_model_name, revision=text_model_revision, local_files_only=local_files_only
        )
    else:
        image_backbone = AutoModel.from_config(
            AutoConfig.from_pretrained(
                image_model_name, revision=image_model_revision, local_files_only=local_files_only
            )
        )
        text_backbone = AutoModel.from_config(
            AutoConfig.from_pretrained(
                text_model_name, revision=text_model_revision, local_files_only=local_files_only
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(
        text_model_name,
        use_fast=True,
        revision=text_model_revision,
        local_files_only=local_files_only,
    )
    if use_lora:
        from contrastive_learning.lora import apply_lora_to_backbone

        image_blocks = apply_lora_to_backbone(
            image_backbone, count=image_unfreeze_last, rank=lora_rank,
            alpha=lora_alpha, dropout=lora_dropout,
        )
        text_blocks = apply_lora_to_backbone(
            text_backbone, count=text_unfreeze_last, rank=lora_rank,
            alpha=lora_alpha, dropout=lora_dropout,
        )
    else:
        image_blocks = freeze_then_unfreeze_last(image_backbone, image_unfreeze_last)
        text_blocks = freeze_then_unfreeze_last(text_backbone, text_unfreeze_last)
    model = ImageReportContrastiveModel(
        ImageTower(image_backbone, embedding_dim),
        TextTower(text_backbone, embedding_dim, pooling=text_pooling),
    )
    return model, tokenizer, {
        "image_blocks": image_blocks,
        "text_blocks": text_blocks,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "lora_dropout": lora_dropout if use_lora else None,
        "image_revision": getattr(image_backbone.config, "_commit_hash", image_model_revision),
        "text_revision": getattr(text_backbone.config, "_commit_hash", text_model_revision),
    }
