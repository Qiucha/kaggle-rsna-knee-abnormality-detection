#!/usr/bin/env python3
"""Train one contrastive run: knee MRI studies against their reports.

The single training entry point. It only forwards to ``contrastive_learning.cli``,
which defines every flag (``--help`` lists them) and resolves unset data paths from
the paths configuration. Reads the cohort manifest and its DICOM series; writes
checkpoints, ``metrics.jsonl`` and reports under ``--output-dir``.

    .venv-cuda/bin/python scripts/training/train_contrastive.py --output-dir checkpoints/contrastive/<run> [flags]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from contrastive_learning.cli import main


if __name__ == "__main__":
    main()
