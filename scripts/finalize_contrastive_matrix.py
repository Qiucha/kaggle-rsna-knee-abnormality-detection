#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.experiments import finalize_matrix


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

    selection = finalize_matrix(
        args.matrix, REPO_ROOT,
        allow_incomplete=args.allow_incomplete,
        evaluate_locked_test=args.evaluate_locked_test,
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from error
