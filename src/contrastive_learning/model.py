"""The two-tower contrastive model: image and report encoders, and the loss.

A knee MRI study is three images (one per anatomical plane) and its report is one
piece of text. Both are mapped to unit vectors in a shared space, where the paired
study and report should be each other's nearest neighbour.

The image tower is the part that is not standard CLIP. It encodes each plane
separately with a shared DINOv2 backbone, then pools the three plane vectors into one
study vector. Two poolings are available. ``attention`` learns the weights, on the
premise that planes are not equally informative for a given finding -- a meniscal tear
reads from sagittal images, a collateral ligament from coronal -- with learned plane
embeddings so the weights can depend on which plane a vector came from and not only on
its content. ``uniform`` simply averages them.

That premise is not borne out on this cohort. Measured on the trained 4-block
checkpoints, the learned weights are near-constant at roughly ``[0.26, 0.28, 0.46]``:
normalised entropy 0.947, per-plane standard deviation only 0.06-0.09 against a
0.20 spread in the means, and no study anywhere near a one-plane collapse. The
attention implements approximately a fixed reweighting, not content-dependent routing.
Reading the same checkpoints out with a uniform plane mean instead of those weights
scores higher on the label probe in 8 of 8 arms measured (+0.0059 to +0.0228 macro
AUROC), which is why ``uniform`` exists as an option. See
``digest/plane-pooling-protocol.md``.

Both backbones are mostly frozen; see :func:`freeze_then_unfreeze_last`. With a few
thousand studies, adapting the last blocks plus the projections is what fits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Plane poolings the image tower accepts; see the module docstring for why both exist.
PLANE_POOLINGS = frozenset({"attention", "uniform"})

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


def restrict_drop_path_to_adapted(model: nn.Module, count: int) -> int:
    """Zero stochastic depth outside the last ``count`` blocks. Returns blocks left on.

    Transformers builds one drop-path module per block from a single config rate, with
    no decay by depth, so switching it on switches it on everywhere -- including the
    frozen blocks. Noise there is unlearnable: nothing downstream can compensate for
    it, which is the same reason :class:`_FrozenAwareModule` holds a frozen backbone in
    eval mode. Only the blocks actually being adapted should carry it.
    """
    blocks = _encoder_blocks(model)
    adapted = set(id(block) for block in blocks[max(0, len(blocks) - count) :]) if count else set()
    enabled = 0
    for block in blocks:
        keep = id(block) in adapted
        for module in block.modules():
            if not hasattr(module, "drop_prob"):
                continue
            if keep:
                enabled += module.drop_prob > 0.0
            else:
                module.drop_prob = 0.0
    return enabled


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
    """DINOv2 plane encoder, pooling the MRI planes by learned attention or a mean.

    ``plane_pooling="uniform"`` builds no attention parameters at all, so it changes
    both the pooling and the trainable parameter count. The two effects are not
    separable in this tower and a comparison between the modes must say so.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        n_planes: int = 3,
        *,
        plane_pooling: str = "attention",
    ):
        super().__init__()
        if plane_pooling not in PLANE_POOLINGS:
            raise ValueError(
                f"plane pooling must be one of {sorted(PLANE_POOLINGS)}, got {plane_pooling!r}"
            )
        self.plane_pooling = plane_pooling
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        # Each plane is described by its CLS token concatenated with its mean patch
        # token, hence the doubled width everywhere below.
        feature_dim = hidden * 2
        self.n_planes = n_planes
        if plane_pooling == "attention":
            # Small random init: a learned marker of which plane a vector came from,
            # added before attention so the pooling weights can be plane-dependent.
            self.plane_embeddings = nn.Parameter(torch.randn(n_planes, feature_dim) * 0.02)
            self.plane_attention = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
        else:
            # No attention parameters exist in this mode, so a checkpoint saved by the
            # other mode fails a strict load on its plane_* keys. A compact checkpoint
            # loads with strict=False, which ignores them: an attention checkpoint read
            # into this mode silently drops its attention, and the reverse leaves the
            # attention freshly initialized. Build with the pooling the checkpoint's
            # saved config names, as evaluate_checkpoint does.
            self.plane_embeddings = None
            self.plane_attention = None
        self.projection = ProjectionHead(feature_dim, embedding_dim)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.backbone_frozen = not any(parameter.requires_grad for parameter in backbone.parameters())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, planes, 3 slice channels, H, W]`` into ``[B, embedding_dim]``.

        The three sampled slices occupy the RGB channels the pretrained backbone
        expects, which is why ``slices_per_plane`` is fixed at three.
        """
        if images.ndim != 5 or images.shape[1:3] != (self.n_planes, 3):
            raise ValueError(
                "Expected [B, planes, 3 slice channels, H, W] with "
                f"{self.n_planes} planes, got {tuple(images.shape)}"
            )
        batch, planes = images.shape[:2]
        # Planes are folded into the batch so one backbone pass encodes all of them.
        flat = images.reshape(batch * planes, *images.shape[2:]).float()
        flat = (flat - self.mean) / self.std
        tokens = self.backbone(pixel_values=flat).last_hidden_state
        # DINOv2-with-registers and DINOv3 place register tokens between CLS
        # and patches. They are learned summary tokens, not spatial patches.
        first_patch = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        features = torch.cat((tokens[:, 0], tokens[:, first_patch:].mean(dim=1)), dim=-1)
        features = features.reshape(batch, planes, -1)
        if self.plane_pooling == "attention":
            features = features + self.plane_embeddings[:planes]
            weights = self.plane_attention(features).squeeze(-1).softmax(dim=1)
            study_features = torch.einsum("bp,bpd->bd", weights, features)
        else:
            study_features = features.mean(dim=1)
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


def label_similarity_targets(
    values: torch.Tensor, mask: torch.Tensor, weight: float
) -> torch.Tensor:
    """Blend LM-label agreement into the one-hot contrastive target.

    ``values`` is +1/-1 per finding and ``mask`` is 1 where the finding was
    actually observed, both ``[batch, findings]``. A pair is scored only on the
    findings both studies observed, so a shared silence never counts as agreement:

        sim[i, j] = mean over co-observed findings of (agree ? +1 : -1)

    rescaled to ``[0, 1]``, and set to 0 for the rare pair sharing no observed
    finding, which is "no evidence of affinity" rather than the 0.5 that rescaling
    a zero would have produced.

    Two guards keep the true pair the dominant target. The diagonal is forced to
    1.0 before row-normalizing, so a study is always its own best label match even
    when every one of its findings is UNK; and the result is mixed as
    ``(1 - weight) * I + weight * rownorm(sim)``. Both matter because roughly half
    of all rows in a batch of 16 have some other study agreeing perfectly on every
    co-observed finding -- normalized agreement alone would hand those rows a
    target that outranks the correct report.

    Row sums are never zero (the diagonal is 1.0), so the normalization needs no
    epsilon and rows of unlabelled studies degrade to exactly the one-hot target.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"label_similarity_weight must be in [0, 1], got {weight}")
    values, mask = values.float(), mask.float()
    # Enforce the mask here rather than trusting it to hold already. If any
    # unobserved finding carried a nonzero sign, ``agreement`` could exceed the
    # co-observed count and the rescaled similarity would leave [0, 1] -- which
    # surfaces as small negative probabilities in the target, not as an error.
    values = values * mask
    co_observed = mask @ mask.T
    agreement = values @ values.T
    similarity = (agreement / co_observed.clamp(min=1.0) + 1.0) / 2.0
    similarity = torch.where(co_observed > 0, similarity, torch.zeros_like(similarity))
    identity = torch.eye(len(similarity), device=similarity.device, dtype=similarity.dtype)
    # Overwrite rather than scale: a study whose findings are all UNK has a zero
    # diagonal here, and that row must still normalize to the one-hot target.
    similarity = similarity * (1.0 - identity) + identity
    soft = similarity / similarity.sum(dim=1, keepdim=True)
    return (1.0 - weight) * identity + weight * soft


def soft_target_contrastive_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Symmetric cross entropy against a soft target distribution.

    The same target serves both directions: the label similarity matrix is
    symmetric, so normalizing the rows of its transpose gives back the rows of the
    original. With ``targets`` one-hot this is exactly
    :func:`symmetric_contrastive_loss`, which is what makes a weight of 0 a true
    reproduction of the unmodified run rather than an approximation of it.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"Expected square paired logits, got {tuple(logits.shape)}")
    if targets.shape != logits.shape:
        raise ValueError(f"Target shape {tuple(targets.shape)} does not match {tuple(logits.shape)}")
    return (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    ) / 2

def _enable_gradient_checkpointing_for_dinov2(backbone: nn.Module) -> None:
    """Patch Dinov2Encoder forward during training to checkpoint each transformer block.

    Transformers' Dinov2Model defines gradient_checkpointing_enable on PreTrainedModel,
    setting self.gradient_checkpointing = True, but Dinov2Encoder.forward does not implement
    checkpointing over its layers. Patching encoder.forward to wrap each layer in
    torch.utils.checkpoint drops activation memory by >60%, enabling dinov2-base at BS=64.
    """
    import torch.utils.checkpoint

    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        encoder = backbone.encoder
        if not getattr(encoder, "_custom_checkpointing_patched", False):
            orig_forward = encoder.forward

            def _checkpointed_forward(hidden_states: torch.Tensor, **kwargs: Any) -> Any:
                if not backbone.training:
                    return orig_forward(hidden_states, **kwargs)
                for layer_module in encoder.layer:
                    def _create_forward(module: nn.Module):
                        def _custom_forward(*inputs: Any) -> torch.Tensor:
                            return module(*inputs)
                        return _custom_forward
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _create_forward(layer_module), hidden_states, use_reentrant=False
                    )
                from transformers.modeling_outputs import BaseModelOutput
                return BaseModelOutput(last_hidden_state=hidden_states)

            encoder.forward = _checkpointed_forward
            encoder._custom_checkpointing_patched = True


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
    image_plane_pooling: str = "attention",
    image_drop_path_rate: float = 0.0,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    gradient_checkpointing: bool = False,
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

    def image_config():
        config = AutoConfig.from_pretrained(
            image_model_name, revision=image_model_revision, local_files_only=local_files_only
        )
        config.drop_path_rate = image_drop_path_rate
        return config

    if pretrained_backbones:
        # A zero rate takes the plain path, so every run recorded before this knob
        # existed rebuilds byte-identically rather than through a config override.
        image_backbone = AutoModel.from_pretrained(
            image_model_name,
            revision=image_model_revision,
            local_files_only=local_files_only,
            **({} if image_drop_path_rate == 0.0 else {"config": image_config()}),
        )
        text_backbone = AutoModel.from_pretrained(
            text_model_name, revision=text_model_revision, local_files_only=local_files_only
        )
    else:
        image_backbone = AutoModel.from_config(image_config())
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

        for p in image_backbone.parameters():
            p.requires_grad = False
        for p in text_backbone.parameters():
            p.requires_grad = False
        image_blocks = apply_lora_to_backbone(
            image_backbone,
            count=image_unfreeze_last,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        drop_path_blocks = restrict_drop_path_to_adapted(image_backbone, image_unfreeze_last)
        text_blocks = apply_lora_to_backbone(
            text_backbone,
            count=text_unfreeze_last,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
    else:
        image_blocks = freeze_then_unfreeze_last(image_backbone, image_unfreeze_last)
        drop_path_blocks = restrict_drop_path_to_adapted(image_backbone, image_unfreeze_last)
        text_blocks = freeze_then_unfreeze_last(text_backbone, text_unfreeze_last)

    if gradient_checkpointing:
        if hasattr(image_backbone, "gradient_checkpointing_enable"):
            image_backbone.gradient_checkpointing_enable()
        if hasattr(text_backbone, "gradient_checkpointing_enable"):
            text_backbone.gradient_checkpointing_enable()
        if hasattr(image_backbone, "enable_input_require_grads"):
            image_backbone.enable_input_require_grads()
        if hasattr(text_backbone, "enable_input_require_grads"):
            text_backbone.enable_input_require_grads()
        _enable_gradient_checkpointing_for_dinov2(image_backbone)

    model = ImageReportContrastiveModel(
        ImageTower(image_backbone, embedding_dim, plane_pooling=image_plane_pooling),
        TextTower(text_backbone, embedding_dim, pooling=text_pooling),
    )
    return model, tokenizer, {
        "image_blocks": image_blocks,
        "image_drop_path_rate": image_drop_path_rate,
        "image_drop_path_blocks": drop_path_blocks,
        "text_blocks": text_blocks,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "lora_dropout": lora_dropout if use_lora else None,
        "gradient_checkpointing": gradient_checkpointing,
        "image_revision": getattr(image_backbone.config, "_commit_hash", image_model_revision),
        "text_revision": getattr(text_backbone.config, "_commit_hash", text_model_revision),
    }
