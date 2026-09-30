import pytest
import torch

from contrastive_learning.timm_dinov2 import convert_state_dict


def timm_state(width=8, depth=2, grid=2):
    state = {
        "cls_token": torch.randn(1, 1, width),
        "pos_embed": torch.randn(1, 1 + grid * grid, width),
        "patch_embed.proj.weight": torch.randn(width, 3, 14, 14),
        "patch_embed.proj.bias": torch.randn(width),
        "norm.weight": torch.randn(width),
        "norm.bias": torch.randn(width),
    }
    for i in range(depth):
        for name, shape in {"norm1": (width,), "norm2": (width,), "attn.qkv": (3 * width, width),
                            "attn.proj": (width, width), "mlp.fc1": (4 * width, width),
                            "mlp.fc2": (width, 4 * width)}.items():
            state[f"blocks.{i}.{name}.weight"] = torch.randn(shape)
            state[f"blocks.{i}.{name}.bias"] = torch.randn(shape[0])
        state[f"blocks.{i}.ls1.gamma"] = torch.randn(width)
        state[f"blocks.{i}.ls2.gamma"] = torch.randn(width)
    return state


def test_converted_weights_load_strictly_and_split_qkv_in_order():
    from transformers import Dinov2Config, Dinov2Model

    state = timm_state()
    converted = convert_state_dict(state)
    model = Dinov2Model(Dinov2Config(hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                                     intermediate_size=32, image_size=28, patch_size=14))
    model.load_state_dict(converted, strict=True)
    query, key, value = state["blocks.1.attn.qkv.weight"].chunk(3)
    attention = model.encoder.layer[1].attention.attention
    assert torch.equal(attention.query.weight, query)
    assert torch.equal(attention.key.weight, key)
    assert torch.equal(attention.value.weight, value)
    assert torch.equal(model.encoder.layer[0].layer_scale2.lambda1, state["blocks.0.ls2.gamma"])
    assert not converted["embeddings.mask_token"].any()


def test_unknown_timm_keys_raise_instead_of_being_dropped():
    state = timm_state()
    state["blocks.0.attn.q_norm.weight"] = torch.ones(8)
    with pytest.raises(KeyError):
        convert_state_dict(state)
    with pytest.raises(KeyError):
        convert_state_dict({**timm_state(), "reg_token": torch.zeros(1, 4, 8)})
