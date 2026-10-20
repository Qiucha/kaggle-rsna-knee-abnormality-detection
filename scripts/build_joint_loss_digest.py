"""Snapshot the completed joint-loss run and its saved paired comparisons.

Run from the repository root after scoring with compare_joint_vs_finetune.py.
The site consumes this snapshot without loading checkpoints or training libraries.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_tools.joint_loss import build_snapshot  # noqa: E402

if __name__ == "__main__":
    print(build_snapshot())
