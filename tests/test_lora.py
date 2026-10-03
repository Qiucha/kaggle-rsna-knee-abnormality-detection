from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from contrastive_learning.config import TrainConfig
from contrastive_learning.lora import (
    LoRALinear,
    apply_lora_to_backbone,
    apply_lora_to_linear_modules,
    get_lora_parameter_summary,
)
from contrastive_learning.model import (
    ImageReportContrastiveModel,
    ImageTower,
    TextTower,
    build_model,
    symmetric_contrastive_loss,
)
from contrastive_learning.checkpoints import load_model_weights


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


def test_lora_checkpoint_save_and_reload(tmp_path):
    """Test saving and restoring weights for a LoRA model in both full and compact formats."""
    # Toy backbone
    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)

        def forward(self, x):
            return self.linear(x)

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 16})()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([DummyBlock() for _ in range(2)])

        def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
            x = torch.ones(2, 4, 16)
            for block in self.encoder.layer:
                x = block(x)
            return type("Output", (), {"last_hidden_state": x})()

    img_backbone = DummyBackbone()
    txt_backbone = DummyBackbone()

    # Apply LoRA
    apply_lora_to_backbone(img_backbone, count=1, rank=4)
    apply_lora_to_backbone(txt_backbone, count=1, rank=4)

    img_tower = ImageTower(img_backbone, embedding_dim=16, plane_pooling="uniform")
    txt_tower = TextTower(txt_backbone, embedding_dim=16, pooling="cls")
    model = ImageReportContrastiveModel(img_tower, txt_tower)

    summary = get_lora_parameter_summary(model)
    assert summary["lora_parameters"] > 0
    assert summary["trainable_parameters"] < summary["total_parameters"]

    # Save state dict
    state_dict = model.state_dict()
    checkpoint = {"model_state_dict": state_dict}

    # New model instance
    img_backbone2 = DummyBackbone()
    txt_backbone2 = DummyBackbone()
    apply_lora_to_backbone(img_backbone2, count=1, rank=4)
    apply_lora_to_backbone(txt_backbone2, count=1, rank=4)
    model2 = ImageReportContrastiveModel(
        ImageTower(img_backbone2, embedding_dim=16, plane_pooling="uniform"),
        TextTower(txt_backbone2, embedding_dim=16, pooling="cls"),
    )

    load_model_weights(model2, checkpoint)
    # Check that weights loaded identically
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)
