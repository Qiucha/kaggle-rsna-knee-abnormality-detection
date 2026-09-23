from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from contrastive_learning.config import TrainConfig
from contrastive_learning.lora import (
    LoRALinear,
)
def test_lora_linear_forward_and_identity_init():
    """Verify that LoRALinear produces base layer output at initialization (delta is zero)."""
    base = nn.Linear(32, 64)
    lora = LoRALinear(base, rank=8, alpha=16.0)
    x = torch.randn(4, 32)

    with torch.no_grad():
        out_base = base(x)
        out_lora = lora(x)

    torch.testing.assert_close(out_lora, out_base)


def test_lora_linear_backward():
    """Verify gradients flow to lora_A and lora_B, while base weight stays frozen."""
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(2, 16)
    out = lora(x).sum()
    out.backward()

    assert base.weight.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_train_config_lora_defaults_and_validation():
    config = TrainConfig()
    assert config.use_lora is False
    assert config.lora_rank == 16
    assert config.lora_alpha == 32.0

    # Serialization roundtrip
    data = config.as_dict()
    restored = TrainConfig.from_dict(data)
    assert restored.use_lora is False
    assert restored.lora_rank == 16

    # Validation errors
    bad_config = TrainConfig(use_lora=True, lora_rank=0)
    with pytest.raises(ValueError, match="lora_rank must be positive"):
        bad_config.validate_training()
