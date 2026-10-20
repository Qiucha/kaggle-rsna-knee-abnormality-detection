"""Low-Rank Adaptation (LoRA) for transformer backbones.

Implements parameter-efficient fine-tuning for linear projections:
    W_eff = W_base + (alpha / r) * (B @ A)
where W_base is frozen, A is initialized with Kaiming uniform / Gaussian,
and B is initialized to zero, ensuring an identity transform at initialization.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap an existing nn.Linear with trainable low-rank adapter matrices."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base + lora


def _find_encoder_blocks(model: nn.Module) -> list[nn.Module]:
    """Find transformer blocks in vision or text Hugging Face backbones."""
    for path in (
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("transformer", "layer"),
        ("transformer", "layers"),
    ):
        val: Any = model
        for name in path:
            val = getattr(val, name, None)
            if val is None:
                break
        if val is not None:
            return list(val)
    raise ValueError(f"Cannot locate transformer blocks in {type(model).__name__}")


def apply_lora_to_linear_modules(
    module: nn.Module,
    rank: int,
    alpha: float = 32.0,
    targets: Sequence[str] | None = None,
    dropout: float = 0.0,
) -> int:
    """Recursively replace nn.Linear submodules matching targets with LoRALinear."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if targets is None or any(t in name for t in targets):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                replaced += 1
        elif not isinstance(child, LoRALinear):
            replaced += apply_lora_to_linear_modules(child, rank, alpha, targets, dropout)
    return replaced


def apply_lora_to_backbone(
    backbone: nn.Module,
    count: int,
    rank: int,
    alpha: float = 32.0,
    targets: Sequence[str] | None = None,
    dropout: float = 0.0,
) -> int:
    """Freeze the backbone and apply LoRA to its last `count` transformer blocks."""
    for p in backbone.parameters():
        p.requires_grad = False

    if count <= 0 or rank <= 0:
        return 0

    blocks = _find_encoder_blocks(backbone)
    target_blocks = blocks[max(0, len(blocks) - count) :]
    total_replaced = 0
    for block in target_blocks:
        total_replaced += apply_lora_to_linear_modules(
            block, rank=rank, alpha=alpha, targets=targets, dropout=dropout
        )
    return total_replaced


def get_lora_parameter_summary(model: nn.Module) -> dict[str, int]:
    """Report total, trainable, and LoRA parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and ("lora_A" in name or "lora_B" in name)
    )
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lora_parameters": lora_params,
        "trainable_fraction": trainable_params / max(1, total_params),
    }
