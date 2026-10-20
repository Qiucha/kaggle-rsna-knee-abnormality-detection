"""Run the approved OmniRad CLIP pre-training and matched supervised follow-up.

Reads ``--config``, a fine-tuning config whose ``contrastive_pretrain_config`` names the
CLIP stage. Pre-trains (or reuses a completed checkpoint), then fine-tunes from it into
``--output``, which must not exist. Pipeline progress is written to the config's
``pipeline_status_dir``. ``scripts/training/finetune_supervised.py`` reaches the same
pipeline for configs that name a pre-training stage.

    .venv-cuda/bin/python scripts/training/run_clip_then_finetune.py \\
        --config configs/finetune/<rung>.json --output checkpoints/finetune/<rung>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning.artifacts import read_json
from contrastive_learning.clip_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paths-config", type=Path,
                        help="Host paths file (default: $CONTRASTIVE_PATHS_CONFIG, then paths.json at the repository root)")
    args = parser.parse_args()
    run_pipeline(read_json(args.config), args.output, device=args.device, paths_config=args.paths_config)


if __name__ == "__main__":
    main()
