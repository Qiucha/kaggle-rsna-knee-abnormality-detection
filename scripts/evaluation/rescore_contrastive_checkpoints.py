#!/usr/bin/env python3
"""Re-score every saved ``best.pt`` under the current metric definitions and summarize which moved.

Reads: every ``best.pt`` under ``--checkpoint-root`` and the image cache.
Writes: ``evaluation-validation-best.json`` beside each checkpoint (via ``evaluate_contrastive.py``;
reused unless ``--overwrite``) and the summary JSON (``--summary``, default
``digest/contrastive-checkpoint-rescore.json``).
Run: ``.venv/bin/python scripts/evaluation/rescore_contrastive_checkpoints.py [--overwrite]``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.experiments import rescore_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score every saved best contrastive checkpoint.")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/contrastive")
    )
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=Path("checkpoints/contrastive/session-20260902/image-cache-v2"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("digest/contrastive-checkpoint-rescore.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = rescore_checkpoints(
        args.checkpoint_root, args.image_cache_dir, args.summary, REPO_ROOT,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from error
