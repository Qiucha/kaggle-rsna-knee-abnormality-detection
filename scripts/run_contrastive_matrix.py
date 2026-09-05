#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.manifest import read_cohort_manifest
from contrastive_learning.selection import SELECTION_POLICY


def _replace_args(arguments: list[str], overrides: list[str]) -> list[str]:
    result = list(arguments)
    index = 0
    while index < len(overrides):
        flag = overrides[index]
        value = overrides[index + 1]
        if flag in result:
            result[result.index(flag) + 1] = value
        else:
            result.extend((flag, value))
        index += 2
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the versioned contrastive experiment matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/contrastive/full-multilingual-matrix.json"),
    )
    parser.add_argument("--experiment", action="append", help="Run only the named experiment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    selected = [
        experiment
        for experiment in matrix["experiments"]
        if not args.experiment or experiment["name"] in args.experiment
    ]
    if args.experiment and len(selected) != len(set(args.experiment)):
        known = {experiment["name"] for experiment in matrix["experiments"]}
        missing = sorted(set(args.experiment) - known)
        raise SystemExit(f"Unknown experiment(s): {', '.join(missing)}")
    if matrix.get("requires_cuda") and not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("This matrix requires CUDA; use --dry-run on a CPU-only host")

    manifest_path = Path(matrix["base_args"][matrix["base_args"].index("--cohort-manifest") + 1])
    _, metadata = read_cohort_manifest(manifest_path)
    if metadata["manifest_sha256"] != matrix["manifest_sha256"]:
        raise SystemExit("Cohort checksum differs from the frozen experiment matrix")
    if matrix["selection_policy"] != SELECTION_POLICY:
        raise SystemExit("Selection policy differs from the frozen experiment matrix")

    output_root = Path(matrix["output_root"])
    for experiment in selected:
        experiment_args = _replace_args(matrix["base_args"], experiment["args"])
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_contrastive.py"),
            *experiment_args,
            "--output-dir",
            str(output_root / experiment["name"]),
        ]
        if args.dry_run:
            print(json.dumps({"name": experiment["name"], "command": command}))
        else:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
