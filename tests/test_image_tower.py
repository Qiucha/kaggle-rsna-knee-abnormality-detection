from types import SimpleNamespace

import pytest
import torch

from contrastive_learning.model import ImageTower, restrict_drop_path_to_adapted
from contrastive_learning.preprocessing import augment_study_images


class CaptureBackbone(torch.nn.Module):
    config = SimpleNamespace(hidden_size=3)

    def forward(self, pixel_values):
        self.pixels = pixel_values.detach().clone()
        channels = pixel_values.mean((-2, -1))
        return SimpleNamespace(last_hidden_state=channels[:, None, :].expand(-1, 3, -1))


def test_register_tokens_are_excluded_from_patch_mean():
    class RegisterBackbone(torch.nn.Module):
        config = SimpleNamespace(hidden_size=3, num_register_tokens=4)

        def forward(self, pixel_values):
            batch = pixel_values.shape[0]
            cls = pixel_values.new_full((batch, 1, 3), 2)
            registers = pixel_values.new_full((batch, 4, 3), 100)
            patches = pixel_values.new_full((batch, 2, 3), 3)
            return SimpleNamespace(last_hidden_state=torch.cat((cls, registers, patches), dim=1))

    tower = ImageTower(RegisterBackbone(), 4, plane_pooling='uniform')
    pooled = []
    hook = tower.projection.register_forward_pre_hook(lambda _, args: pooled.append(args[0]))
    tower(torch.zeros(1, 3, 3, 2, 2))
    hook.remove()
    torch.testing.assert_close(pooled[0], torch.tensor([[2., 2., 2., 3., 3., 3.]]))
    from contrastive_learning.submission import image_features
    torch.testing.assert_close(image_features(tower.backbone, torch.zeros(1, 3, 3, 2, 2)), pooled[0])


@pytest.mark.parametrize('noncontiguous', [False, True])
def test_all_slice_channels_and_study_plane_order_survive(noncontiguous):
    backbone = CaptureBackbone()
    tower = ImageTower(backbone, 4)
    images = torch.arange(2*3*3*2*4).reshape(2, 3, 3, 2, 4).float() / 144
    if noncontiguous:
        images = images.transpose(-1, -2)
    # Uniform plane attention permits an independent expected pooling calculation.
    with torch.no_grad():
        tower.plane_embeddings.zero_()
        for parameter in tower.plane_attention.parameters():
            parameter.zero_()
    pooled = []
    hook = tower.projection.register_forward_pre_hook(lambda _, args: pooled.append(args[0].detach()))
    tower(images)
    hook.remove()
    recovered = backbone.pixels * tower.std + tower.mean
    for batch in range(2):
        for plane in range(3):
            for channel in range(3):
                torch.testing.assert_close(recovered[batch*3+plane, channel], images[batch, plane, channel])
    normalized = (images - tower.mean) / tower.std
    expected = normalized.mean((-2, -1)).mean(1)
    torch.testing.assert_close(pooled[0], torch.cat((expected, expected), -1))


def test_all_slice_channels_contribute_without_mixing_studies():
    from transformers import Dinov2Config, Dinov2Model
    torch.manual_seed(0)
    backbone = Dinov2Model(Dinov2Config(
        hidden_size=24, num_hidden_layers=1, num_attention_heads=3,
        image_size=28, patch_size=14,
    )).eval()
    tower = ImageTower(backbone, 8).eval()
    images = torch.rand(2, 3, 3, 28, 28, requires_grad=True)
    output = tower(augment_study_images(images, rotation_degrees=0, zoom=0, shift=0, intensity=0))
    # Every selected slice contributes; one study cannot affect the other.
    output[0, 0].backward()
    assert torch.all(images.grad[0].abs().sum((-2, -1)) > 0)
    assert torch.count_nonzero(images.grad[1]) == 0


@pytest.mark.parametrize('shape', [(2,3,28,28), (2,2,3,28,28), (2,3,1,28,28)])
def test_image_tower_rejects_missing_plane_or_channel_axes(shape):
    with pytest.raises(ValueError, match='slice channels'):
        ImageTower(CaptureBackbone(), 4)(torch.zeros(shape))


def test_uniform_pooling_averages_planes_and_builds_no_attention():
    # The zeroed-attention case above shows what a uniform mean must reproduce, so
    # the two poolings have to agree there: it pins "uniform" to the plane mean
    # rather than merely to something plausible.
    images = torch.arange(2 * 3 * 3 * 2 * 4).reshape(2, 3, 3, 2, 4).float() / 144
    uniform = ImageTower(CaptureBackbone(), 4, plane_pooling='uniform')
    assert uniform.plane_attention is None and uniform.plane_embeddings is None
    assert not any('plane' in name for name, _ in uniform.named_parameters())

    attention = ImageTower(CaptureBackbone(), 4)
    with torch.no_grad():
        attention.plane_embeddings.zero_()
        for parameter in attention.plane_attention.parameters():
            parameter.zero_()
    # Share the projection so only the pooling differs between the two towers.
    uniform.projection.load_state_dict(attention.projection.state_dict())
    torch.testing.assert_close(uniform(images), attention(images))


def test_uniform_pooling_still_rejects_the_wrong_plane_count():
    # The plane check used to read its expected count off plane_embeddings, which
    # uniform pooling does not have.
    with pytest.raises(ValueError, match='slice channels'):
        ImageTower(CaptureBackbone(), 4, plane_pooling='uniform')(torch.zeros(2, 2, 3, 28, 28))


def test_unknown_plane_pooling_is_rejected():
    with pytest.raises(ValueError, match='plane pooling'):
        ImageTower(CaptureBackbone(), 4, plane_pooling='mean')


class DropPathBlock(torch.nn.Module):
    """A transformer block carrying one drop-path module, as Dinov2 blocks do."""

    def __init__(self, rate):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.drop_path = SimpleNamespace(drop_prob=rate)

    def modules(self):
        yield self
        yield self.drop_path


def _backbone_with_drop_path(rate, block_count=12):
    blocks = torch.nn.ModuleList([DropPathBlock(rate) for _ in range(block_count)])
    return SimpleNamespace(encoder=SimpleNamespace(layer=blocks))


def test_drop_path_is_zeroed_outside_the_adapted_blocks():
    # Transformers builds one drop-path module per block from a single rate, so
    # switching it on switches it on in the frozen blocks too. Noise there is
    # unlearnable, and this is the guard against measuring that instead.
    backbone = _backbone_with_drop_path(0.05)
    enabled = restrict_drop_path_to_adapted(backbone, 4)
    blocks = backbone.encoder.layer
    assert enabled == 4
    assert [block.drop_path.drop_prob for block in blocks[:8]] == [0.0] * 8
    assert [block.drop_path.drop_prob for block in blocks[8:]] == [0.05] * 4


def test_drop_path_with_no_adapted_blocks_is_fully_disabled():
    backbone = _backbone_with_drop_path(0.05)
    assert restrict_drop_path_to_adapted(backbone, 0) == 0
    assert all(block.drop_path.drop_prob == 0.0 for block in backbone.encoder.layer)


def test_the_default_rate_leaves_every_block_untouched():
    # The default path must rebuild every run recorded before this knob existed.
    backbone = _backbone_with_drop_path(0.0)
    assert restrict_drop_path_to_adapted(backbone, 4) == 0
    assert all(block.drop_path.drop_prob == 0.0 for block in backbone.encoder.layer)
