"""Snapshot the completed joint-loss run and its saved paired comparisons.

Run after scoring with compare_joint_vs_finetune.py (the snapshot logic is experiment_tools.joint_loss.build_snapshot).
The site consumes this snapshot without loading checkpoints or training libraries.

    .venv/bin/python scripts/digests/build_joint_loss_digest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiment_tools.joint_loss import build_snapshot  # noqa: E402

if __name__ == "__main__":
    print(build_snapshot())
