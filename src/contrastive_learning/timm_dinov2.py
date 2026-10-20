"""Load timm-format DINOv2 and DINO ViTs (e.g. OmniRad, RadioDino) as HF ``Dinov2Model``.

Radiology-pretrained backbones such as ``Snarcy/OmniRad-small`` are published for
timm as ``vit_small_patch14_dinov2``, the same architecture as
``facebook/dinov2-small``. Renaming their weights onto HF's names keeps every
downstream path unchanged: ``freeze_then_unfreeze_last``, LoRA, ``export_member``
and the Kaggle loader, which rebuilds from the saved config without timm.
"""

from __future__ import annotations

import re

import torch

_BLOCK = re.compile(r"blocks\.(\d+)\.(.+)")
_BLOCK_RENAMES = {
    "norm1": "norm1",
    "norm2": "norm2",
    "attn.proj": "attention.output.dense",
    "mlp.fc1": "mlp.fc1",
    "mlp.fc2": "mlp.fc2",
}
_TOP_RENAMES = {
    "cls_token": "embeddings.cls_token",
    "pos_embed": "embeddings.position_embeddings",
    "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
    "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
    "norm.weight": "layernorm.weight",
    "norm.bias": "layernorm.bias",
}


def convert_state_dict(timm_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """timm ``vit_*_dinov2`` weights under ``Dinov2Model`` names.

    The fused ``qkv`` projection splits into query, key and value in that order.
    ``mask_token`` has no timm counterpart and is only read by masked pretraining,
    so it is zero. Unknown keys raise rather than being dropped silently.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in timm_state.items():
        if key in _TOP_RENAMES:
            out[_TOP_RENAMES[key]] = value
            continue
        match = _BLOCK.fullmatch(key)
        if not match:
            raise KeyError(f"Unmapped timm key: {key}")
        prefix, rest = f"encoder.layer.{match[1]}.", match[2]
        if rest.startswith("attn.qkv."):
            kind = rest.rsplit(".", 1)[1]
            for name, part in zip(("query", "key", "value"), value.chunk(3, dim=0)):
                out[f"{prefix}attention.attention.{name}.{kind}"] = part.clone()
        elif rest in ("ls1.gamma", "ls2.gamma"):
            out[f"{prefix}layer_scale{rest[2]}.lambda1"] = value
        else:
            module, kind = rest.rsplit(".", 1)
            if module not in _BLOCK_RENAMES:
                raise KeyError(f"Unmapped timm key: {key}")
            out[f"{prefix}{_BLOCK_RENAMES[module]}.{kind}"] = value
    width = out["embeddings.cls_token"].shape[-1]
    # Original-DINO ViTs (e.g. RadioDino) have no layer scale; ones make the
    # DINOv2 block compute exactly the plain ViT block.
    for name in [k for k in out if k.endswith("norm1.weight") and k.startswith("encoder.layer.")]:
        prefix = name[: -len("norm1.weight")]
        for i in (1, 2):
            out.setdefault(f"{prefix}layer_scale{i}.lambda1", torch.ones(width))
    out["embeddings.mask_token"] = torch.zeros(1, width)
    return out


def load_timm_dinov2(repo: str, revision: str, *, base_model: str, base_revision: str,
                     local_files_only: bool = True, drop_path_rate: float = 0.0):
    """``repo``'s weights in a ``Dinov2Model`` whose config comes from ``base_model``.

    The base supplies width, depth and heads. ``image_size`` is set from the stored
    position embedding (OmniRad stores a 224 px grid, Meta's DINOv2 a 518 px one),
    so the parameter shapes match and HF interpolates only for other input sizes.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, Dinov2Model

    path = hf_hub_download(repo, "model.safetensors", revision=revision,
                           local_files_only=local_files_only)
    state = convert_state_dict(load_file(path))
    config = AutoConfig.from_pretrained(base_model, revision=base_revision,
                                        local_files_only=local_files_only)
    # Patch size and grid come from the weights, so a patch-16 model can borrow a
    # patch-14 base's width and depth.
    config.patch_size = state["embeddings.patch_embeddings.projection.weight"].shape[-1]
    grid = int(round((state["embeddings.position_embeddings"].shape[1] - 1) ** 0.5))
    config.image_size = grid * config.patch_size
    config.drop_path_rate = drop_path_rate
    # The config source is the architecture, but these are the radiology weights.
    config._commit_hash = revision
    model = Dinov2Model(config)
    model.load_state_dict(state, strict=True)
    return model
