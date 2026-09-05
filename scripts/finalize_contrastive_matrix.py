#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.selection import SELECTION_POLICY, selection_key, verified_best_epoch
from contrastive_learning.manifest import _sha256_file


def _best_epoch(path: Path) -> dict:
    return verified_best_epoch(path, path.with_name("best.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select on validation metrics, then optionally evaluate the locked test split once."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/contrastive/full-multilingual-matrix.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--evaluate-locked-test", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    output_root = Path(matrix["output_root"])
    results = []
    missing = []
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
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "best_validation": _best_epoch(metrics_path),
            }
        )
    if missing and not args.allow_incomplete:
        raise SystemExit("Incomplete experiment(s): " + ", ".join(missing))
    if not results:
        raise SystemExit("No completed experiments found")
    selected = max(
        results,
        key=lambda result: selection_key(result["best_validation"])[:2],
    )
    selection = {
        "schema_version": 1,
        "selection_metric": SELECTION_POLICY + "; matrix order breaks experiment ties",
        "locked_test_was_used_for_selection": False,
        "selected_experiment": selected["name"],
        "selected_checkpoint": selected["best_checkpoint"],
        "experiments": results,
    }
    selection_path = output_root / "validation-selection.json"
    serialized = json.dumps(selection, indent=2, sort_keys=True) + "\n"
    if selection_path.exists() and selection_path.read_text(encoding="utf-8") != serialized:
        raise SystemExit("Immutable validation selection already exists with different content")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")

    if args.evaluate_locked_test:
        cache_index = matrix["base_args"].index("--image-cache-dir") + 1
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_contrastive.py"),
            selected["best_checkpoint"],
            "--split",
            "test",
            "--output",
            str(output_root / "locked-test-evaluation.json"),
            "--rankings",
            str(output_root / "locked-test-rankings.jsonl"),
            "--image-cache-dir",
            matrix["base_args"][cache_index],
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
