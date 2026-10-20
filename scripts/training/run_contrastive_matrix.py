#!/usr/bin/env python3
"""Run a frozen contrastive experiment matrix, one fresh ``train_contrastive.py`` process each.

Reads the matrix JSON (default ``configs/contrastive/full-multilingual-matrix.json``)
and verifies its cohort checksum. Each experiment writes its run directory under the
matrix's ``output_root``. ``--dry-run`` prints one JSON command per experiment and
trains nothing; ``--experiment NAME`` (repeatable) restricts the grid.

    .venv-cuda/bin/python scripts/training/run_contrastive_matrix.py --matrix <matrix.json> --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.experiments import run_matrix


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

    commands = run_matrix(
        args.matrix, REPO_ROOT, experiments=args.experiment, dry_run=args.dry_run
    )
    if args.dry_run:
        for command in commands:
            print(json.dumps(command))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from error
