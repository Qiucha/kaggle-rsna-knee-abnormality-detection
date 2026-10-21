"""When a rerun counts as reproducing its recorded run."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_run_history", Path(__file__).resolve().parents[1] / "scripts/analysis/compare_run_history.py")
compare_run_history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_run_history)
compare = compare_run_history.compare


def run(directory: Path, losses, *, report: bool = False, **extra) -> Path:
    directory.mkdir()
    rows = [{"epoch": e, "train_masked_bce": loss, "validation_masked_bce": loss + 0.1,
             "validation_macro_auroc": 0.8, "seconds": 1.0, **extra} for e, loss in enumerate(losses, 1)]
    (directory / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if report:
        scores = {"macro_auroc": 0.89}
        (directory / "report.json").write_text(json.dumps(
            {"best_epoch": 1, "validation_masked_bce": 0.5, "lm_test": scores, "gold": scores}))
    return directory


def test_small_drift_passes_without_the_optional_objective(tmp_path):
    record = run(tmp_path / "record", [0.50, 0.40], report=True)
    rerun = run(tmp_path / "rerun", [0.501, 0.399], report=True)
    assert compare(record, rerun) == []


def test_training_loss_beyond_tolerance_fails(tmp_path):
    record = run(tmp_path / "record", [0.50, 0.40])
    rerun = run(tmp_path / "rerun", [0.50, 0.42])
    assert compare(record, rerun) == ["epoch 2 train_masked_bce +0.0200"]


def test_a_nan_loss_fails(tmp_path):
    record = run(tmp_path / "record", [0.50])
    rerun = run(tmp_path / "rerun", [float("nan")])
    assert compare(record, rerun) == ["epoch 1 train_masked_bce +nan"]


def test_no_shared_epoch_fails(tmp_path):
    record = run(tmp_path / "record", [0.50])
    (tmp_path / "rerun").mkdir()
    (tmp_path / "rerun" / "history.jsonl").write_text("")
    assert compare(record, tmp_path / "rerun") == ["no epoch in common"]


def test_finished_runs_that_stopped_at_different_epochs_fail(tmp_path):
    record = run(tmp_path / "record", [0.50, 0.40, 0.35], report=True)
    rerun = run(tmp_path / "rerun", [0.50, 0.40], report=True)
    assert compare(record, rerun) == ["finished runs differ in epochs: record only [3], rerun only []"]


def test_an_unfinished_rerun_is_judged_on_the_epochs_it_has(tmp_path):
    record = run(tmp_path / "record", [0.50, 0.40, 0.35], report=True)
    rerun = run(tmp_path / "rerun", [0.50])
    assert compare(record, rerun) == []


def test_inherited_epochs_can_be_skipped(tmp_path):
    record = run(tmp_path / "record", [0.50, 0.40])
    rerun = run(tmp_path / "rerun", [0.50, 0.40])
    assert compare(record, rerun, first_epoch=3) == ["no epoch in common"]


def test_no_training_loss_in_common_fails(tmp_path):
    for name, key in (("record", "train_objective"), ("rerun", "train_masked_bce")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "history.jsonl").write_text(json.dumps({"epoch": 1, key: 0.5}) + "\n")
    assert compare(tmp_path / "record", tmp_path / "rerun") == ["epoch 1 has no training loss logged by both runs"]


def test_a_partial_last_line_is_skipped(tmp_path):
    record = run(tmp_path / "record", [0.50, 0.40])
    rerun = run(tmp_path / "rerun", [0.50])
    with (rerun / "history.jsonl").open("a") as handle:
        handle.write('{"epoch": 2, "train_ma')
    assert compare(record, rerun) == []
