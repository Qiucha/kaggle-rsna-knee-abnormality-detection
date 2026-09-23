"""Benchmark trained contrastive encoders and export a self-contained Kaggle bundle."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning.submission_training import train_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/kaggle/lora-holdout-v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-cache-dir", type=Path)
    args = parser.parse_args()
    train_bundle(
        args.config,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.image_cache_dir,
    )


if __name__ == "__main__":
    main()
