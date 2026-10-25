"""Train masked-label MLP/linear heads on a frozen contrastive image encoder.

Reads a head config (``configs/kaggle/lm-*.json``) and extracts frozen-tower features
into ``--features`` (an ``.npz`` cache, reused when its recorded identity matches). Then
fits the head and writes the experiment directory ``--output``, which must not exist.
Unless the config is an image baseline, it ends by scoring LM-verdict AUROC on the
locked test split. ``--features-only`` stops after the feature cache.

    .venv-cuda/bin/python scripts/training/train_lm_head.py \\
        --config configs/kaggle/lm-mlp-v1.json --output checkpoints/kaggle/<run>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning.artifacts import read_json
from contrastive_learning.lm_supervision import prepare_features


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/kaggle/lm-mlp-v1.json")
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("checkpoints/kaggle/lm-mlp-features-v1.npz"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=Path("checkpoints/contrastive/full-multilingual-v1/image-cache"),
    )
    args = parser.parse_args()
    if not args.features_only and args.output is None:
        parser.error("--output is required unless --features-only")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; use a new experiment directory")
    config = read_json(args.config)
    prepared = prepare_features(
        config,
        args.features,
        device=args.device,
        batch_size=args.batch_size,
        image_cache_dir=args.image_cache_dir,
    )
    if not args.features_only:
        from contrastive_learning.lm_supervision import run_experiment

        from contrastive_learning.lm_proxy import score_run

        run_experiment(config, args.output, prepared, device=args.device)
        if config.get("image_baseline"):
            # score_run re-extracts LM-test features from the adapted checkpoint; gold scores are in report.json.
            print("image_baseline: LM-test scoring skipped", flush=True)
            return
        # Standard local report: LM-verdict AUROC on the locked test split.
        report, *_ = score_run(
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            image_cache_dir=args.image_cache_dir,
        )
        print(f"LM-test macro AUROC={report['macro_auroc']:.6f}", flush=True)


if __name__ == "__main__":
    main()
