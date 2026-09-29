"""Locked-test LM proxy: population isolation and masked AUROC."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from contrastive_learning.labels import LabelTable
from contrastive_learning.lm_proxy import (
    lm_test_population,
    masked_macro_auroc,
    paired_intervals,
)
from contrastive_learning.submission import LABELS


def record(uid, split, group=None):
    return SimpleNamespace(study_uid=uid, split=split, patient_group_id=group or uid)


def test_population_is_non_gold_locked_test_with_an_lm_verdict():
    rows = [
        record("train", "train"),
        record("val", "validation"),
        record("gold_excluded", "excluded"),
        record("gold_test", "test"),
        record("gold_relative", "test", "gold_test"),
        record("all_unknown", "test"),
        record("kept", "test"),
    ]
    value = torch.tensor([1.0, -1.0])
    table = LabelTable(
        ("ACL", "MCL"),
        {r.study_uid: value for r in rows},
        {r.study_uid: torch.tensor([0.0, 0.0] if r.study_uid == "all_unknown" else [1.0, 0.0]) for r in rows},
    )
    kept, targets, masks = lm_test_population(rows, table, ["gold_excluded", "gold_test"])
    assert [r.study_uid for r in kept] == ["kept"]
    # YES -> 1, unobserved cells are zero-filled and masked out.
    np.testing.assert_array_equal(targets, [[1.0, 0.0]])
    np.testing.assert_array_equal(masks, [[1.0, 0.0]])


def test_empty_population_is_an_error():
    table = LabelTable(("ACL",), {"a": torch.ones(1)}, {"a": torch.ones(1)})
    with pytest.raises(ValueError, match="no LM-labelled"):
        lm_test_population([record("a", "train")], table, [])


def test_masked_auroc_ignores_unknown_cells_and_single_class_labels():
    rng = np.random.default_rng(0)
    n = 40
    targets = rng.integers(0, 2, (n, len(LABELS))).astype(float)
    masks = (rng.random((n, len(LABELS))) > 0.3).astype(float)
    targets[:, 1] = 1  # one class only -> not scored
    predictions = rng.random((n, len(LABELS)))
    # Values under the mask must not matter.
    noisy = predictions.copy()
    noisy[masks == 0] = 99.0
    result = masked_macro_auroc(noisy, targets, masks)
    assert result["per_label"][LABELS[1]] is None
    expected = [
        roc_auc_score(targets[masks[:, j] > 0, j], predictions[masks[:, j] > 0, j])
        for j in range(len(LABELS))
        if j != 1
    ]
    np.testing.assert_allclose(
        [v for v in result["per_label"].values() if v is not None], expected
    )
    assert result["labels_scored"] == len(LABELS) - 1
    assert result["macro_auroc"] == pytest.approx(np.mean(expected))


def test_paired_intervals_are_zero_for_identical_predictions():
    rng = np.random.default_rng(1)
    targets = rng.integers(0, 2, (50, len(LABELS))).astype(float)
    masks = np.ones_like(targets)
    values = rng.random(targets.shape)
    result = paired_intervals({"a": values, "b": values}, targets, masks, resamples=20)
    assert result["a - b"] == {"difference": 0.0, "ci95": [0.0, 0.0]}
