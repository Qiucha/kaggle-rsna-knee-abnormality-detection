"""Fine-tune the image backbone end to end on LM labels: one ladder rung per config."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning.artifacts import read_json
from contrastive_learning.finetune import run_finetune


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new experiment directory")
    run_finetune(read_json(args.config), args.output, device=args.device)


if __name__ == "__main__":
    main()
