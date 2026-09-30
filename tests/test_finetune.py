import numpy as np
import pandas as pd
import pytest
import torch

from contrastive_learning.finetune import FinetuneModel, soft_targets, validate_config
from contrastive_learning.submission import LABELS, _adapt, image_features, predict_head


def tiny_backbone():
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
        image_size=28, patch_size=14,
    )
    return Dinov2Model(config)


def test_exported_member_predicts_what_the_trained_model_predicts():
    torch.manual_seed(0)
    model = FinetuneModel(tiny_backbone(), [16], dropout=0.2).eval()
    _adapt(model.backbone, {"use_lora": False, "image_unfreeze_last": 2})
    images = torch.rand(2, 3, 3, 28, 28)
    with torch.no_grad():
        trained = model(images).sigmoid().numpy()
    deployed = predict_head(image_features(model.backbone, images).numpy(), model.export_head())
    np.testing.assert_allclose(deployed, trained, atol=1e-5)


def test_only_the_last_blocks_and_head_train():
    from contrastive_learning.model import freeze_then_unfreeze_last

    backbone = tiny_backbone()
    freeze_then_unfreeze_last(backbone, 1)
    model = FinetuneModel(backbone, [16], dropout=0.0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("backbone.encoder.layer.3.") for n in trainable)
    assert not any(n.startswith("backbone.encoder.layer.2.") for n in trainable)
    assert any(n.startswith("head.") for n in trainable)


def test_soft_targets_keep_unk_masked(tmp_path):
    uids = ["a", "b"]
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for label in LABELS:
        frame[label] = [0.94, 0.28]
    frame.to_csv(tmp_path / "lm.csv", index=False)
    masks = np.zeros((2, len(LABELS)), np.float32)
    masks[0] = 1
    values = soft_targets(tmp_path / "lm.csv", uids, masks)
    assert np.allclose(values[0], 0.94)
    assert np.allclose(values[1], 0.0)


@pytest.mark.parametrize("change", [{"targets": "probability"}, {"image_unfreeze_last": 0},
                                    {"slices_per_plane": 5}, {"epochs": 0}])
def test_invalid_rung_configs_are_rejected(change):
    config = {"targets": "verdict", "image_unfreeze_last": 3, "slices_per_plane": 3,
              "epochs": 20, "batch_size": 16, "patience": 5, **change}
    with pytest.raises(ValueError):
        validate_config(config)


def test_jittered_slices_stay_in_their_strata_and_replay_under_a_seed():
    from contrastive_learning.preprocessing import _slice_indices, jittered_slice_indices

    draws = [jittered_slice_indices(40, (0.2, 0.8), 3, np.random.default_rng(s)) for s in range(200)]
    for indices in draws:
        assert len(indices) == 3 and list(indices) == sorted(indices)
        assert 7 <= indices[0] < 15 <= indices[1] < 23 <= indices[2] <= 31
    assert len({tuple(d) for d in draws}) > 50
    assert (jittered_slice_indices(40, (0.2, 0.8), 3, np.random.default_rng(1))
            == jittered_slice_indices(40, (0.2, 0.8), 3, np.random.default_rng(1))).all()
    # Too few slices to stratify: the fixed sampler's answer.
    assert (jittered_slice_indices(4, (0.2, 0.8), 3, np.random.default_rng(0))
            == _slice_indices(4, (0.2, 0.8), 3)).all()


def test_only_the_training_dataset_samples_slices_at_random(tmp_path):
    from contrastive_learning.data import KneeReportDataset
    from contrastive_learning.finetune import RandomSliceDataset, _dataset

    config = {"image_size": 28, "crop_mm": 130.0, "slices_per_plane": 3, "slice_band": [0.2, 0.8],
              "image_cache_dir": str(tmp_path),
              "train_slice_sampling": "random"}
    assert isinstance(_dataset([], config, training=True), RandomSliceDataset)
    assert isinstance(_dataset([], config, training=False), KneeReportDataset)
    config["train_slice_sampling"] = "fixed"
    assert isinstance(_dataset([], config, training=True), KneeReportDataset)


def test_unknown_slice_sampling_is_rejected():
    with pytest.raises(ValueError):
        validate_config({"targets": "verdict", "image_unfreeze_last": 3, "slices_per_plane": 3,
                         "epochs": 20, "batch_size": 16, "patience": 5,
                         "train_slice_sampling": "shuffled"})


def test_random_slices_read_from_the_local_mirror(tmp_path):
    from pathlib import Path

    from contrastive_learning.finetune import RandomSliceDataset

    (tmp_path / "local" / "study" / "series").mkdir(parents=True)
    dataset = RandomSliceDataset([], {}, {"from": "/mnt/d/train_series", "to": str(tmp_path / "local")})
    assert dataset._series(Path("/mnt/d/train_series/study/series")) == tmp_path / "local/study/series"
    with pytest.raises(FileNotFoundError):
        dataset._series(Path("/mnt/d/train_series/study/other"))
