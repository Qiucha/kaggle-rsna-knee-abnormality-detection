"""Missing-label supervision, split isolation, and portable dense heads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from contrastive_learning.labels import LabelTable
from contrastive_learning.lm_supervision import paired_gold_interval, partition_records
from contrastive_learning.submission import predict_head
from contrastive_learning.supervised_head import fit_dense_head, masked_bce


def test_unknown_targets_have_zero_gradient_and_no_effect_on_loss():
    logits = torch.tensor([[0.3, -0.4], [0.8, 0.1]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[1.0, float("nan")], [999.0, 0.0]])
    loss = masked_bce(logits, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask.bool()], targets[mask.bool()]
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[~mask.bool()], torch.zeros(2))


def test_fixed_observation_rates_make_batch_loss_match_full_loss():
    logits = torch.randn(4, 2)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    expected = masked_bce(logits, targets, mask)
    actual = (
        sum(
            masked_bce(
                logits[i : i + 2], targets[i : i + 2], mask[i : i + 2], mask.mean(0)
            )
            for i in (0, 2)
        )
        / 2
    )
    torch.testing.assert_close(actual, expected)


def record(uid, split, group=None):
    return SimpleNamespace(study_uid=uid, split=split, patient_group_id=group or uid)


def test_gold_lm_rows_and_locked_test_never_enter_head_training():
    rows = [
        record("a", "train"),
        record("b", "validation"),
        record("g", "excluded"),
        record("t", "test"),
        record("u", "train"),
        record("relative", "train", "g"),
    ]
    table = LabelTable(
        ("ACL",),
        {r.study_uid: torch.ones(1) for r in rows},
        {r.study_uid: torch.tensor([0.0 if r.study_uid == "u" else 1.0]) for r in rows},
    )
    groups, exclusions = partition_records(rows, table, ["g", "t"])
    assert [r.study_uid for r in groups["train"]] == ["a"]
    assert [r.study_uid for r in groups["validation"]] == ["b"]
    assert [r.study_uid for r in groups["gold"]] == ["g"]
    assert exclusions["gold_locked_test"] == 1
    assert exclusions["gold_group"] == 1
    assert exclusions["all_unknown"] == 1
    with pytest.raises(ValueError, match="encoder exposure"):
        partition_records([*rows, record("leaked", "train")], table, ["g", "leaked"])


def test_all_gold_cases_can_be_scored_without_entering_head_fitting():
    rows = [
        record("a", "train"),
        record("b", "validation"),
        record("g", "excluded"),
        record("t", "test"),
        record("other_test", "test"),
        record("relative", "train", "t"),
    ]
    table = LabelTable(
        ("ACL",),
        {r.study_uid: torch.ones(1) for r in rows},
        {r.study_uid: torch.ones(1) for r in rows},
    )
    groups, exclusions = partition_records(
        rows, table, ["t", "g"], gold_splits=("excluded", "test")
    )
    assert [r.study_uid for r in groups["train"]] == ["a"]
    assert [r.study_uid for r in groups["validation"]] == ["b"]
    assert [r.study_uid for r in groups["gold"]] == ["t", "g"]
    assert exclusions["gold_test_evaluated"] == 1
    assert exclusions["gold_locked_test"] == 0
    assert exclusions["test"] == 1
    assert exclusions["gold_group"] == 1


def test_portable_dense_ensemble_matches_pytorch_and_roundtrips(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(10, 4)).astype(np.float32)
    torch.manual_seed(7)
    layers = [torch.nn.Linear(4, 6), torch.nn.Linear(6, 2)]
    head = {
        "kind": "dense_v1",
        "mean": torch.ones(4),
        "scale": torch.full((4,), 2.0),
        "layers": [
            {"weight": layer.weight.detach(), "bias": layer.bias.detach()}
            for layer in layers
        ],
    }
    with torch.no_grad():
        expected = (
            layers[1](layers[0]((torch.from_numpy(x) - 1) / 2).relu()).sigmoid().numpy()
        )
    ensemble = {"kind": "mean_heads_v1", "heads": [head, head]}
    path = tmp_path / "head.pt"
    torch.save(ensemble, path)
    actual = predict_head(x, torch.load(path, weights_only=True))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    with pytest.raises(ValueError, match="Unsupported head"):
        predict_head(x, {"kind": "bad"})


def test_training_standardizer_uses_only_train_and_selects_validation_minimum():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(48, 4)).astype(np.float32)
    val = rng.normal(size=(24, 4)).astype(np.float32) + 1
    y = np.tile([[0.0, 1.0], [1.0, 0.0]], (24, 1)).astype(np.float32)
    vy = np.tile([[0.0, 1.0], [1.0, 0.0]], (12, 1)).astype(np.float32)
    options = {
        "hidden_dims": [6],
        "dropout": 0.2,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "batch_size": 16,
        "max_epochs": 8,
        "patience": 3,
        "min_delta": 1e-5,
        "seed": 17,
    }
    head, result = fit_dense_head(
        train, y, np.ones_like(y), val, vy, np.ones_like(vy), **options
    )
    np.testing.assert_allclose(head["mean"].numpy(), train.mean(0), atol=1e-7)
    best = min(result["history"], key=lambda r: r["validation_masked_bce"])
    assert result["best_epoch"] == best["epoch"]
    assert result["validation_masked_bce"] == best["validation_masked_bce"]
    actual = predict_head(val, head)
    # Independent BCE from exported probabilities verifies saved weights are the selected epoch.
    bce = -(vy * np.log(actual) + (1 - vy) * np.log1p(-actual)).mean()
    assert bce == pytest.approx(best["validation_masked_bce"], abs=1e-6)
    same, _ = fit_dense_head(
        train, y, np.ones_like(y), val, vy, np.ones_like(vy), **options
    )
    np.testing.assert_array_equal(predict_head(val, same), actual)


def test_paired_bootstrap_uses_shared_study_resamples():
    targets = np.tile([[0.0, 1.0], [1.0, 0.0]], (20, 1))
    values = np.random.default_rng(2).random(targets.shape)
    result = paired_gold_interval(targets, values, values, resamples=30, seed=9)
    assert result["difference"] == 0
    assert result["ci95"] == [0, 0]


def test_feature_cache_reuses_tuple_preprocessing_after_json_roundtrip(
    tmp_path, monkeypatch
):
    import contrastive_learning.lm_supervision as workflow

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint identity")
    labels = tmp_path / "labels.csv"
    labels.write_text("labels identity")
    config = {
        "checkpoint": str(checkpoint),
        "cohort_manifest": str(tmp_path / "manifest"),
        "manifest_sha256": "pinned",
        "lm_labels_csv": str(labels),
        "gold_labels_csv": str(labels),
        "feature": "uniform_mean_cls_patch",
    }
    groups = {
        split: [record(split, split)] for split in ("train", "validation", "gold")
    }
    backbone = torch.nn.Linear(4, 4)
    backbone.config = SimpleNamespace(hidden_size=2)
    run_config = SimpleNamespace(
        cohort_manifest=config["cohort_manifest"],
        image_kwargs={"slice_band": (0.2, 0.8)},
    )
    monkeypatch.setattr(workflow, "load_population", lambda *a: (groups, {}, {}, {}))
    monkeypatch.setattr(workflow, "load_encoder", lambda *a: (backbone, run_config, {}))
    monkeypatch.setattr(
        workflow, "read_cohort_manifest", lambda *a: ([], {"manifest_sha256": "pinned"})
    )
    monkeypatch.setattr(
        workflow, "extract_features", lambda *a: np.ones((1, 4), dtype=np.float32)
    )
    path = tmp_path / "features.npz"
    first = workflow.prepare_features(
        config, path, device="cpu", batch_size=1, image_cache_dir=None
    )

    def must_not_extract(*args):
        raise AssertionError("A valid feature cache must not re-encode images")

    monkeypatch.setattr(workflow, "extract_features", must_not_extract)
    second = workflow.prepare_features(
        config, path, device="cpu", batch_size=1, image_cache_dir=None
    )
    assert first[5] == second[5]
    for split in groups:
        np.testing.assert_array_equal(first[0][split], second[0][split])
