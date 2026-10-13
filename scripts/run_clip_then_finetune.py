"""Run the approved OmniRad CLIP pre-training and matched supervised follow-up."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning.artifacts import read_json
from contrastive_learning.clip_pipeline import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_pipeline(read_json(args.config), args.output, device=args.device)
