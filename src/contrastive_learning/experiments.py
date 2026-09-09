"""Run a frozen experiment grid, then select one checkpoint on validation alone.

An experiment matrix is a JSON file that fixes, before any run starts: the cohort
(by checksum), the shared training arguments, the per-experiment overrides, the
output root, and the selection policy. This module's job is to refuse to execute
anything that no longer matches that frozen description, and to make the selection
step a one-way door -- so a grid cannot be quietly re-selected after its results
are known.

Three guarantees are worth stating because they are the reason the code is shaped
this way:

* **Runs are subprocesses.** Each experiment trains in a fresh interpreter, so GPU
  memory, CUDA context, and library global state cannot leak between comparisons.
* **Selection reads validation only.** ``validation-selection.json`` is written
  first, records ``locked_test_was_used_for_selection: false``, and refuses to be
  overwritten with different content. Test evaluation can only run afterwards.
* **A dry run is total.** ``--dry-run`` performs every check and returns the exact
  commands, so a grid can be validated on a CPU-only host before booking a GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from contrastive_learning.artifacts import read_json, sha256_file, write_json
from contrastive_learning.checkpoints import SELECTION_POLICY, selection_key, verified_best_epoch
from contrastive_learning.manifest import read_cohort_manifest


SELECTION_FILENAME = "validation-selection.json"


def _replace_args(arguments: list[str], overrides: list[str]) -> list[str]:
    """Apply ``--flag value`` overrides onto base arguments, in place or appended.

    Overriding rather than appending matters because the training CLI takes the
    last occurrence of a flag; rewriting keeps the emitted command readable and
    makes an experiment's difference from the baseline visible at a glance.
    """
    result = list(arguments)
    for flag, value in zip(overrides[::2], overrides[1::2], strict=True):
        if flag in result:
            result[result.index(flag) + 1] = value
        else:
            result.extend((flag, value))
    return result


def _load_matrix(matrix_path: Path, *, verify_cohort: bool) -> dict[str, Any]:
    """Read a matrix and verify the frozen facts it claims.

    A selection policy mismatch means the code's ranking rule moved out from under
    a comparison already specified, so it is rejected wherever a matrix is used.
    The cohort checksum is only verified before training, which is the step that
    reads the data; finalizing and reporting must stay possible on a host that
    does not have the DICOM mount attached.
    """
    matrix = read_json(matrix_path)
    if matrix["selection_policy"] != SELECTION_POLICY:
        raise ValueError("Selection policy differs from the frozen experiment matrix")
    if verify_cohort:
        base_args = matrix["base_args"]
        manifest_path = Path(base_args[base_args.index("--cohort-manifest") + 1])
        _, metadata = read_cohort_manifest(manifest_path)
        if metadata["manifest_sha256"] != matrix["manifest_sha256"]:
            raise ValueError("Cohort checksum differs from the frozen experiment matrix")
    return matrix


def _script(repo_root: Path, name: str) -> str:
    return str(Path(repo_root) / "scripts" / name)


def run_matrix(
    matrix_path: Path,
    repo_root: Path,
    *,
    experiments: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Validate a frozen matrix and train the selected experiments in fresh processes.

    ``experiments`` restricts the grid by name -- the normal way to run one
    experiment first and inspect its cost before committing to the rest. Returns
    the commands in matrix order whether or not they were executed.
    """
    matrix = _load_matrix(matrix_path, verify_cohort=True)
    selected = [
        experiment
        for experiment in matrix["experiments"]
        if not experiments or experiment["name"] in experiments
    ]
    if experiments and len(selected) != len(set(experiments)):
        known = {experiment["name"] for experiment in matrix["experiments"]}
        missing = sorted(set(experiments) - known)
        raise ValueError(f"Unknown experiment(s): {', '.join(missing)}")
    if matrix.get("requires_cuda") and not torch.cuda.is_available() and not dry_run:
        raise ValueError("This matrix requires CUDA; use --dry-run on a CPU-only host")

    output_root = Path(matrix["output_root"])
    commands = []
    for experiment in selected:
        command = [
            sys.executable,
            _script(repo_root, "train_contrastive.py"),
            *_replace_args(matrix["base_args"], experiment["args"]),
            "--output-dir",
            str(output_root / experiment["name"]),
        ]
        commands.append({"name": experiment["name"], "command": command})
        if not dry_run:
            subprocess.run(command, cwd=repo_root, check=True)
    return commands


def finalize_matrix(
    matrix_path: Path,
    repo_root: Path,
    *,
    allow_incomplete: bool = False,
    evaluate_locked_test: bool = False,
) -> dict[str, Any]:
    """Verify every run's best epoch, write the immutable selection, then optionally test.

    Each experiment's ``best.pt`` is re-derived from its own ``metrics.jsonl``, so a
    checkpoint saved under a superseded rule is caught here rather than reported.
    Experiment ties are broken by matrix order: the winner comparison drops the
    epoch term from the selection key, and ``max`` keeps the earliest of equals.

    Passing ``evaluate_locked_test`` spends the one-shot test split on the already
    written selection. Leave it off during development.
    """
    matrix = _load_matrix(matrix_path, verify_cohort=False)
    output_root = Path(matrix["output_root"])
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for experiment in matrix["experiments"]:
        run_dir = output_root / experiment["name"]
        metrics_path = run_dir / "metrics.jsonl"
        checkpoint_path = run_dir / "best.pt"
        if not metrics_path.is_file() or not checkpoint_path.is_file():
            missing.append(experiment["name"])
            continue
        results.append(
            {
                "name": experiment["name"],
                "purpose": experiment["purpose"],
                "best_checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "best_validation": verified_best_epoch(metrics_path, checkpoint_path),
            }
        )
    if missing and not allow_incomplete:
        raise ValueError("Incomplete experiment(s): " + ", ".join(missing))
    if not results:
        raise ValueError("No completed experiments found")

    selected = max(results, key=lambda result: selection_key(result["best_validation"])[:2])
    selection = {
        "schema_version": 1,
        "selection_metric": SELECTION_POLICY + "; matrix order breaks experiment ties",
        "locked_test_was_used_for_selection": False,
        "selected_experiment": selected["name"],
        "selected_checkpoint": selected["best_checkpoint"],
        "experiments": results,
    }
    selection_path = output_root / SELECTION_FILENAME
    # Compared as parsed JSON rather than as bytes, so the immutability check tests
    # the selection itself and cannot be broken by a change to write_json's layout.
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError("Immutable validation selection already exists with different content")
    write_json(selection_path, selection)

    if evaluate_locked_test:
        base_args = matrix["base_args"]
        subprocess.run(
            [
                sys.executable,
                _script(repo_root, "evaluate_contrastive.py"),
                selected["best_checkpoint"],
                "--split",
                "test",
                "--output",
                str(output_root / "locked-test-evaluation.json"),
                "--rankings",
                str(output_root / "locked-test-rankings.jsonl"),
                "--image-cache-dir",
                base_args[base_args.index("--image-cache-dir") + 1],
            ],
            cwd=repo_root,
            check=True,
        )
    return selection


def rescore_checkpoints(
    checkpoint_root: Path,
    image_cache_dir: Path,
    summary_path: Path,
    repo_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Re-evaluate every saved ``best.pt`` under the current metric definitions.

    This exists so a metric change can be applied to historical runs without
    retraining: the summary counts how many checkpoints actually moved, which
    distinguishes a definition that mattered from one that did not. Without
    ``overwrite`` an existing per-checkpoint report is reused rather than
    recomputed, making the sweep resumable.
    """
    checkpoints = sorted(Path(checkpoint_root).rglob("best.pt"))
    completed = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        report_path = checkpoint.parent / "evaluation-validation-best.json"
        print(f"[{index}/{len(checkpoints)}] {checkpoint}", flush=True)
        if not report_path.exists() or overwrite:
            command = [
                sys.executable,
                _script(repo_root, "evaluate_contrastive.py"),
                str(checkpoint),
                "--image-cache-dir",
                str(image_cache_dir),
            ]
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, cwd=repo_root, check=True)
        completed.append(read_json(report_path))

    summary = {
        "schema_version": 1,
        "metric_tie_policy": "uniform_expected_credit",
        "checkpoint_count": len(completed),
        "checkpoints_with_metric_changes": sum(
            any(abs(delta) > 1e-8 for delta in result["metric_deltas"].values())
            for result in completed
        ),
        "queries_with_tied_candidates": sum(
            result["queries_with_tied_candidates"] for result in completed
        ),
        "results": completed,
    }
    write_json(summary_path, summary)
    return summary
