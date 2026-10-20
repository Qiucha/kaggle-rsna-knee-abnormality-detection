"""Fine-tune the image backbone end to end on LM labels: one ladder rung per config.

Reads ``--config`` (``configs/finetune/<rung>.json``) and writes the run directory
``--output``, which must not exist. A config with ``contrastive_pretrain_config`` is a
two-stage experiment: CLIP pre-training first, then this rung from its checkpoint.
``--paths-config`` names the host paths file (default ``$CONTRASTIVE_PATHS_CONFIG``,
then ``paths.json`` at the repository root).

    .venv-cuda/bin/python scripts/training/finetune_supervised.py \\
        --config configs/finetune/<rung>.json --output checkpoints/finetune/<rung>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning import clip_pipeline
from contrastive_learning.artifacts import read_json
from contrastive_learning.finetune import run_finetune, with_host_mirror


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
    if args.output.exists():
        parser.error("Output already exists; use a new experiment directory")
    config = read_json(args.config)
    if config.get("contrastive_pretrain_config"):
        # A two-stage experiment: CLIP pre-training, then this rung from its checkpoint.
        clip_pipeline.run_pipeline(config, args.output, device=args.device, paths_config=args.paths_config)
        return
    config = with_host_mirror(config, args.paths_config)
    run_finetune(config, args.output, device=args.device)


if __name__ == "__main__":
    main()
