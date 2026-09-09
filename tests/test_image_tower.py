from types import SimpleNamespace

import pytest
import torch

from contrastive_learning.model import ImageTower
from contrastive_learning.preprocessing import augment_study_images


class CaptureBackbone(torch.nn.Module):
    config = SimpleNamespace(hidden_size=3)

    def forward(self, pixel_values):
        self.pixels = pixel_values.detach().clone()
        channels = pixel_values.mean((-2, -1))
        return SimpleNamespace(last_hidden_state=channels[:, None, :].expand(-1, 3, -1))


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
