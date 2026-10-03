"""Deployment invariants: schema, leakage boundary, and offline weight fidelity."""

import numpy as np
import pandas as pd
import pytest
import torch

from contrastive_learning.config import TrainConfig
from contrastive_learning.submission import (
    LABELS,
    UID,
    benchmark_head,
    combine_members,
    export_member,
    fit_head,
    image_features,
    load_encoder,
    load_member,
    predict_head,
    validate_submission,
)
from contrastive_learning.submission_training import training_population


def test_submission_requires_exact_order_and_probabilities():
    sample = pd.DataFrame(np.full((2, 12), 0.5), columns=LABELS)
    sample.insert(0, UID, ["a", "b"])
    validate_submission(sample.copy(), sample)
    for bad in (
        sample.iloc[::-1],
        sample.assign(ACL=np.nan),
        sample.assign(ACL=1.1),
        sample.drop(columns=["Fracture"]),
        sample.assign(StudyInstanceUID=["a", "a"]),
    ):
        with pytest.raises(ValueError):
            validate_submission(bad, sample)


def test_rank_mean_averages_per_label_ranks_across_members():
    from scipy.stats import rankdata

    rng = np.random.default_rng(0)
    members = [rng.random((7, 12)), rng.random((7, 12))]
    members[1][:2, 0] = 0.25  # a tie takes the average rank
    expected = np.mean([np.column_stack([rankdata(m[:, j]) / len(m) for j in range(12)])
                        for m in members], axis=0)
    np.testing.assert_allclose(combine_members(members, "rank_mean"), expected)


def test_ensemble_mode_defaults_to_probability_mean_and_rejects_unknown():
    members = [np.full((3, 12), 0.2), np.full((3, 12), 0.6)]
    np.testing.assert_allclose(combine_members(members, "mean_probability"), 0.4)
    with pytest.raises(ValueError):
        combine_members(members, "geometric")


def test_portable_head_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(17)
    x = rng.normal(size=(32, 6))
    y = (x[:, 0] > 0).astype(int)[:, None]
    head = fit_head(x, y)
    reference = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(
        x, y[:, 0]
    )
    np.testing.assert_allclose(
        predict_head(x, head)[:, 0], reference.predict_proba(x)[:, 1], atol=1e-12
    )


def test_oof_preprocessing_sees_only_training_fold():
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 4))
    x[0] = 1e4
    y = np.tile([0.0, 1.0], 12)[:, None]
    oof, assignments = benchmark_head(x, y, folds=3)
    for i, (train, val) in enumerate(
        StratifiedKFold(3, shuffle=True, random_state=2026).split(x, y[:, 0])
    ):
        expected = predict_head(x[val], fit_head(x[train], y[train]))
        np.testing.assert_allclose(oof[val], expected)
        assert (assignments[val] == i).all()
    with pytest.raises(ValueError, match="examples"):
        benchmark_head(x, np.zeros_like(y), folds=3)


def test_locked_test_population_is_rejected_before_reading():
    with pytest.raises(ValueError, match="locked test"):
        training_population(None, None, ["test"])


@pytest.mark.parametrize("lora", [False, True])
def test_image_export_reload_is_offline_and_exact(tmp_path, monkeypatch, lora):
    from transformers import AutoConfig, AutoModel, Dinov2Config, Dinov2Model

    from contrastive_learning.submission import _adapt

    config = TrainConfig(use_lora=lora, image_unfreeze_last=1)
    adaptation = {
        key: getattr(config, key)
        for key in (
            "use_lora",
            "image_unfreeze_last",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
        )
    }
    backbone = Dinov2Model(
        Dinov2Config(
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            image_size=28,
            patch_size=14,
        )
    ).eval()
    _adapt(backbone, adaptation)
    images = torch.rand(2, 3, 3, 28, 28)
    expected = image_features(backbone, images)
    path = tmp_path / "member.pt"
    export_member(path, backbone, config, adaptation, {})

    def no_download(*args, **kwargs):
        raise AssertionError("Inference attempted a download")

    monkeypatch.setattr(AutoConfig, "from_pretrained", no_download)
    monkeypatch.setattr(AutoModel, "from_pretrained", no_download)
    restored, _ = load_member(path, "cpu")
    torch.testing.assert_close(
        image_features(restored, images), expected, atol=0, rtol=0
    )


def test_incomplete_compact_image_delta_fails(tmp_path, monkeypatch):
    from transformers import AutoConfig, AutoModel, Dinov2Config, Dinov2Model

    architecture = Dinov2Config(
        hidden_size=12, num_hidden_layers=1, num_attention_heads=3
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: architecture)
    monkeypatch.setattr(
        AutoModel, "from_pretrained", lambda *a, **kw: Dinov2Model(architecture)
    )
    config = TrainConfig(image_unfreeze_last=1, image_model_revision="pinned")
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": config.as_dict(), "trainable_state_dict": {}}, path)
    with pytest.raises(ValueError, match="Incomplete image delta"):
        load_encoder(path)
