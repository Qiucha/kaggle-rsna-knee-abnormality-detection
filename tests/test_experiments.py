import json
from pathlib import Path

import pytest
import torch

from contrastive_learning import experiments
from contrastive_learning.checkpoints import SELECTION_POLICY


@pytest.fixture
def matrix(tmp_path):
    value = {
        "requires_cuda": True,
        "manifest_sha256": "frozen",
        "selection_policy": SELECTION_POLICY,
        "output_root": str(tmp_path / "runs"),
        "base_args": ["--cohort-manifest", "cohort.jsonl", "--batch-size", "16",
                      "--image-cache-dir", "cache"],
        "experiments": [
            {"name": "first", "purpose": "baseline", "args": []},
            {"name": "second", "purpose": "comparison", "args": ["--batch-size", "32"]},
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(value))
    return path, value


def test_matrix_dry_run_and_execution_use_identical_commands(matrix, tmp_path, monkeypatch):
    path, _ = matrix
    monkeypatch.setattr(experiments, "read_cohort_manifest", lambda _: (
        [], {"manifest_sha256": "frozen"}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []
    monkeypatch.setattr(experiments.subprocess, "run", lambda command, **kw: calls.append(command))
    planned = experiments.run_matrix(path, tmp_path, experiments=["second"], dry_run=True)
    assert calls == []
    assert [run["name"] for run in planned] == ["second"]
    command = planned[0]["command"]
    assert command.count("--batch-size") == 1
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[-1] == str(tmp_path / "runs" / "second")
    with pytest.raises(ValueError, match="requires CUDA"):
        experiments.run_matrix(path, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert experiments.run_matrix(path, tmp_path, experiments=["second"]) == planned
    assert calls == [command]


@pytest.mark.parametrize("failure", ["checksum", "selection", "unknown"])
def test_matrix_rejects_invalid_plan_before_execution(matrix, tmp_path, monkeypatch, failure):
    path, value = matrix
    if failure == "selection":
        value["selection_policy"] = "wrong"
        path.write_text(json.dumps(value))
    monkeypatch.setattr(experiments, "read_cohort_manifest", lambda _: (
        [], {"manifest_sha256": "wrong" if failure == "checksum" else "frozen"}))
    with pytest.raises(ValueError, match={"checksum": "checksum", "selection": "Selection", "unknown": "Unknown"}[failure]):
        experiments.run_matrix(path, tmp_path, dry_run=True,
                               experiments=["missing"] if failure == "unknown" else None)


def _save_best(root, name, epoch=1, recall=0.5):
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    metrics = {"epoch": epoch, "mean_r@1": recall, "validation_loss": 1.0}
    (run / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    torch.save({"epoch": epoch, "metrics": metrics}, run / "best.pt")


def test_finalization_verifies_ties_and_locks_selection_before_test(matrix, tmp_path, monkeypatch):
    path, value = matrix
    root = Path(value["output_root"])
    _save_best(root, "first", epoch=4)
    with pytest.raises(ValueError, match="Incomplete"):
        experiments.finalize_matrix(path, tmp_path)
    assert not (root / "validation-selection.json").exists()
    _save_best(root, "second", epoch=1)
    calls = []

    def evaluate(command, **kwargs):
        selection = json.loads((root / "validation-selection.json").read_text())
        assert selection["selected_experiment"] == "first"
        assert selection["locked_test_was_used_for_selection"] is False
        assert command[command.index("--split") + 1] == "test"
        calls.append(command)

    monkeypatch.setattr(experiments.subprocess, "run", evaluate)
    result = experiments.finalize_matrix(path, tmp_path)
    assert calls == []
    assert result == experiments.finalize_matrix(path, tmp_path, evaluate_locked_test=True)
    assert len(calls) == 1
    original = (root / "validation-selection.json").read_bytes()
    _save_best(root, "second", recall=0.9)
    with pytest.raises(ValueError, match="Immutable"):
        experiments.finalize_matrix(path, tmp_path, evaluate_locked_test=True)
    assert len(calls) == 1
    assert (root / "validation-selection.json").read_bytes() == original


def test_rescoring_reuses_reports_and_aggregates_changes(tmp_path, monkeypatch):
    _save_best(tmp_path, "first")
    _save_best(tmp_path, "second")
    result = {"metric_deltas": {"mean_r@1": 0.25}, "queries_with_tied_candidates": 2}
    (tmp_path / "first" / "evaluation-validation-best.json").write_text(json.dumps(result))
    calls = []

    def evaluate(command, **kwargs):
        calls.append(command)
        checkpoint = Path(command[2])
        (checkpoint.parent / "evaluation-validation-best.json").write_text(json.dumps(result))

    monkeypatch.setattr(experiments.subprocess, "run", evaluate)
    summary_path = tmp_path / "summary.json"
    summary = experiments.rescore_checkpoints(tmp_path, tmp_path / "cache", summary_path, tmp_path)
    assert len(calls) == 1
    assert summary["checkpoint_count"] == 2
    assert summary["checkpoints_with_metric_changes"] == 2
    assert summary["queries_with_tied_candidates"] == 4
    assert json.loads(summary_path.read_text()) == summary
    experiments.rescore_checkpoints(tmp_path, tmp_path / "cache", summary_path, tmp_path, overwrite=True)
    assert len(calls) == 3
    assert all("--overwrite" in call for call in calls[1:])
