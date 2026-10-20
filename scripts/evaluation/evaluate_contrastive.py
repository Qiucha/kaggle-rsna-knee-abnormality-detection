#!/usr/bin/env python3
"""Score one saved contrastive checkpoint on a split with tie-aware retrieval metrics.

Reads: the checkpoint, its cohort manifest and the image cache. The ``test`` split is the locked one;
spend it only through ``finalize_contrastive_matrix.py``.
Writes: the metrics report (``--output``) and per-query rankings (``--rankings``) when given; prints the metrics.
Run: ``.venv/bin/python scripts/evaluation/evaluate_contrastive.py CHECKPOINT [--split validation]``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved contrastive checkpoint with tie-aware retrieval metrics."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--image-cache-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        output_path=args.output,
        rankings_path=args.rankings,
        image_cache_dir=args.image_cache_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
