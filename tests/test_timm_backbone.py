import pytest
import torch

timm = pytest.importorskip("timm")

from contrastive_learning.finetune import WindowModel
from contrastive_learning.model import _IMAGENET_MEAN, _IMAGENET_STD, freeze_then_unfreeze_last
from contrastive_learning.timm_backbone import TimmBackbone


def tiny_coatnet():
    model = timm.create_model("coatnet_nano_rw_224", pretrained=False, num_classes=0,
                              embed_dim=(32, 32, 64, 64), depths=(1, 1, 2, 2), stem_width=16, img_size=64)
    return TimmBackbone(model.eval())


def test_tokens_are_max_then_flattened_feature_map():
    backbone = tiny_coatnet()
    images = torch.rand(2, 3, 64, 64)
    tokens = backbone(pixel_values=images).last_hidden_state
    assert tokens.shape == (2, 1 + 2 * 2, backbone.config.hidden_size)
    torch.testing.assert_close(tokens[:, 0], tokens[:, 1:].amax(1))


def test_imagenet_normalised_input_reaches_timm_with_its_own_statistics():
    backbone = tiny_coatnet()
    x = torch.rand(1, 3, 64, 64)
    imagenet = (x - torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    cfg = backbone.model.pretrained_cfg
    native = (x - torch.tensor(cfg["mean"]).view(1, 3, 1, 1)) / torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    torch.testing.assert_close(imagenet * backbone.scale + backbone.shift, native)


def test_unfreeze_counts_blocks_across_stages_and_keeps_timm_names():
    backbone = tiny_coatnet()
    assert freeze_then_unfreeze_last(backbone, 3) == 6
    model = WindowModel(backbone, [8], dropout=0.0, pooling="mean")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("backbone.model.stages.3.blocks.1.") for n in trainable)
    assert any(n.startswith("backbone.model.stages.2.blocks.1.") for n in trainable)
    assert not any(n.startswith("backbone.model.stages.2.blocks.0.") for n in trainable)
    assert any(n.startswith("backbone.model.norm.") for n in trainable)
    assert all(n.startswith(("backbone.model.", "head.")) for n, _ in model.named_parameters())


def test_window_model_backpropagates_through_the_unfrozen_blocks_only():
    backbone = tiny_coatnet()
    freeze_then_unfreeze_last(backbone, 1)
    model = WindowModel(backbone, [8], dropout=0.0, pooling="mean")
    images, valid = torch.rand(1, 3, 2, 3, 64, 64), torch.ones(1, 3, 2, dtype=torch.bool)
    valid[0, 1, 1] = False
    model(images, valid).sum().backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith("backbone.model.stages.3.blocks.1.") for n in grads)
    assert not any(n.startswith("backbone.model.stages.0.") for n in grads)


@pytest.mark.parametrize("unfreeze", [1, 6])
def test_batchnorm_statistics_do_not_drift_in_training_frozen_or_not(unfreeze):
    backbone = tiny_coatnet()
    freeze_then_unfreeze_last(backbone, unfreeze)
    model = WindowModel(backbone, [8], dropout=0.0, pooling="mean").train()
    norms = [m for m in backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert norms
    assert any(p.requires_grad for m in norms for p in m.parameters()) == (unfreeze == 6)
    before = [m.running_mean.clone() for m in norms]
    model(torch.rand(1, 3, 2, 3, 64, 64), torch.ones(1, 3, 2, dtype=torch.bool))
    for m, mean in zip(norms, before):
        assert not m.training
        torch.testing.assert_close(m.running_mean, mean)


def test_grad_checkpointing_matches_plain_gradients_behind_a_frozen_stage():
    torch.manual_seed(0)
    plain = tiny_coatnet()
    checkpointed = tiny_coatnet()
    checkpointed.load_state_dict(plain.state_dict())
    checkpointed.model.set_grad_checkpointing(True)
    images = torch.rand(2, 3, 64, 64)
    grads = []
    for backbone in (plain, checkpointed):
        freeze_then_unfreeze_last(backbone, 5)  # stage 0 frozen, so stage 1's input needs no grad
        backbone.train()
        backbone(pixel_values=images).last_hidden_state.sum().backward()
        grads.append({n: p.grad for n, p in backbone.named_parameters() if p.requires_grad})
    assert any(n.startswith("model.stages.1.") and g is not None for n, g in grads[1].items())
    for name, grad in grads[0].items():
        torch.testing.assert_close(grads[1][name], grad, rtol=1e-4, atol=1e-5)
