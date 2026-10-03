"""Wrap a timm convolutional/hybrid backbone (e.g. CoAtNet) in the HF ViT interface.

The fine-tuning models call ``backbone(pixel_values=...).last_hidden_state`` and pool
``token 0`` with the mean of the remaining tokens, and ``freeze_then_unfreeze_last``
finds blocks at ``encoder.layer`` and the final norm at ``layernorm``. This adapter
gives a timm model that surface without touching those paths:

- The patch tokens are the final feature map, flattened. There is no CLS token, so
  token 0 is the feature map's spatial max: the pooled feature is then
  ``[max, mean]`` over positions, where the mean is timm's own pre-logits pool.
- ``encoder.layer`` lists every stage block in depth order, so ``image_unfreeze_last``
  counts blocks across stage boundaries. It is a plain list, not a submodule, so
  parameter names stay timm's.
- Callers normalise with ImageNet statistics. Inputs are mapped back to [0, 1] and
  renormalised with the checkpoint's own mean and std.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .model import _IMAGENET_MEAN, _IMAGENET_STD


class TimmBackbone(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(hidden_size=model.num_features, num_register_tokens=0)
        self.encoder = SimpleNamespace(layer=[block for stage in model.stages for block in stage.blocks])
        # x = x_in * imagenet_std + imagenet_mean, then (x - mean) / std, as one affine map.
        mean, std = torch.tensor(model.pretrained_cfg["mean"]), torch.tensor(model.pretrained_cfg["std"])
        self.register_buffer("scale", (torch.tensor(_IMAGENET_STD) / std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((torch.tensor(_IMAGENET_MEAN) - mean) / std).view(1, 3, 1, 1),
                             persistent=False)

    @property
    def layernorm(self) -> nn.Module:
        return self.model.norm

    def train(self, mode: bool = True) -> "TimmBackbone":
        # Every BatchNorm layer stays in eval mode, trainable or not; trainable ones still
        # learn their affine weights. Otherwise training would update running statistics
        # that best.pt (trainable parameters only) never saves, and a micro-batch of one or
        # two studies' correlated windows would make those statistics noisy anyway.
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        features = self.model.forward_features(pixel_values * self.scale + self.shift)
        patches = features.flatten(2).transpose(1, 2)
        tokens = torch.cat((patches.amax(1, keepdim=True), patches), dim=1)
        return SimpleNamespace(last_hidden_state=tokens)


def load_timm_backbone(name: str, revision: str, grad_checkpointing: bool = False) -> TimmBackbone:
    """``timm/<arch>.<tag>`` at ``revision`` from the local Hugging Face cache, without its classifier.

    The architecture comes from timm's registry and only the weights file from the
    cache, so this works offline and loads the same way from a Kaggle dataset path.
    ``grad_checkpointing`` recomputes each block in the backward pass (timm's
    non-reentrant mode, so blocks behind a frozen stage still get gradients). It trades
    time for memory when many high-resolution blocks are unfrozen.
    """
    import timm
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(name, "model.safetensors", revision=revision, local_files_only=True)
    model = timm.create_model(name.split("/", 1)[1], pretrained=True, num_classes=0,
                              pretrained_cfg_overlay={"file": weights})
    model.set_grad_checkpointing(grad_checkpointing)
    return TimmBackbone(model)
