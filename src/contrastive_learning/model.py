from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _encoder_blocks(model: nn.Module) -> list[nn.Module]:
    candidates = (
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("transformer", "layer"),
        ("transformer", "layers"),
    )
    for path in candidates:
        value: Any = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None:
            return list(value)
    raise ValueError(f"Cannot locate transformer blocks in {type(model).__name__}")


def freeze_then_unfreeze_last(model: nn.Module, count: int) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    blocks = _encoder_blocks(model)
    for block in blocks[max(0, len(blocks) - count) :] if count else []:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for name in ("layernorm", "layer_norm"):
        if (normalization := getattr(model, name, None)) is not None and count:
            for parameter in normalization.parameters():
                parameter.requires_grad = True
    return len(blocks)


class ProjectionHead(nn.Module):
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


class ImageTower(nn.Module):
    """DINOv2 plane encoder with learned pooling across the three MRI planes."""

    def __init__(self, backbone: nn.Module, embedding_dim: int, n_planes: int = 3):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        self.plane_embeddings = nn.Parameter(torch.randn(n_planes, hidden * 2) * 0.02)
        self.plane_attention = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.projection = ProjectionHead(hidden * 2, embedding_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone_frozen = not any(parameter.requires_grad for parameter in backbone.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, planes = images.shape[:2]
        flat = images.reshape(batch * planes, *images.shape[2:]).float()
        flat = (flat - self.mean) / self.std
        tokens = self.backbone(pixel_values=flat).last_hidden_state
        features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1)
        features = features.reshape(batch, planes, -1) + self.plane_embeddings[:planes]
        weights = self.plane_attention(features).squeeze(-1).softmax(dim=1)
        study_features = torch.einsum("bp,bpd->bd", weights, features)
        return self.projection(study_features)


class TextTower(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int, pooling: str = "cls"):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("text pooling must be 'cls' or 'mean'")
        self.backbone = backbone
        self.pooling = pooling
        self.projection = ProjectionHead(int(backbone.config.hidden_size), embedding_dim)
        self.backbone_frozen = not any(parameter.requires_grad for parameter in backbone.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            features = tokens[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
            features = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(features)


class ImageReportContrastiveModel(nn.Module):
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
        image_embeddings = self.encode_images(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        logits = self.logit_scale.exp() * image_embeddings @ text_embeddings.T
        return logits, image_embeddings, text_embeddings


def symmetric_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """CLIP-style paired InfoNCE in both retrieval directions."""
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
    pretrained_backbones: bool = True,
    local_files_only: bool = False,
) -> tuple[ImageReportContrastiveModel, Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if pretrained_backbones:
        image_backbone = AutoModel.from_pretrained(
            image_model_name, revision=image_model_revision, local_files_only=local_files_only
        )
        text_backbone = AutoModel.from_pretrained(
            text_model_name, revision=text_model_revision, local_files_only=local_files_only
        )
    else:
        image_config = AutoConfig.from_pretrained(
            image_model_name, revision=image_model_revision, local_files_only=local_files_only
        )
        text_config = AutoConfig.from_pretrained(
            text_model_name, revision=text_model_revision, local_files_only=local_files_only
        )
        image_backbone = AutoModel.from_config(image_config)
        text_backbone = AutoModel.from_config(text_config)
    tokenizer = AutoTokenizer.from_pretrained(
        text_model_name,
        use_fast=True,
        revision=text_model_revision,
        local_files_only=local_files_only,
    )
    image_blocks = freeze_then_unfreeze_last(image_backbone, image_unfreeze_last)
    text_blocks = freeze_then_unfreeze_last(text_backbone, text_unfreeze_last)
    model = ImageReportContrastiveModel(
        ImageTower(image_backbone, embedding_dim),
        TextTower(text_backbone, embedding_dim, pooling=text_pooling),
    )
    return model, tokenizer, {
        "image_blocks": image_blocks,
        "text_blocks": text_blocks,
        "image_revision": getattr(image_backbone.config, "_commit_hash", image_model_revision),
        "text_revision": getattr(text_backbone.config, "_commit_hash", text_model_revision),
    }
