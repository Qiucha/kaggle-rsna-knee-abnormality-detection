from __future__ import annotations

import argparse
import json
from pathlib import Path

from contrastive_learning.config import TEXT_MODEL_CHOICES, TrainConfig
from contrastive_learning.training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DINOv2/medical-language-model image-report contrastive encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d"))
    parser.add_argument("--reports-csv", type=Path, default=Path("data/sample/train.csv"))
    parser.add_argument(
        "--cohort-manifest", type=Path, help="Use fixed train/validation/test assignments from JSONL"
    )
    parser.add_argument("--report-column", default="Report")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/contrastive"))
    parser.add_argument(
        "--text-model",
        default="radbert-va",
        help="A short name (%s) or Hugging Face model ID" % ", ".join(TEXT_MODEL_CHOICES),
    )
    parser.add_argument("--image-model", default="facebook/dinov2-small")
    parser.add_argument("--text-model-revision")
    parser.add_argument("--image-model-revision")
    parser.add_argument("--text-pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--text-prefix", default="")
    parser.add_argument(
        "--report-view",
        choices=("full", "impression", "findings_impression"),
        default="full",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--image-unfreeze-last", type=int, default=3)
    parser.add_argument("--text-unfreeze-last", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument(
        "--fold-count",
        type=int,
        help="Use deterministic K-fold validation instead of --val-fraction",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        help="Zero-based validation fold used with --fold-count",
    )
    parser.add_argument("--lr-image", type=float, default=1e-5)
    parser.add_argument("--lr-text", type=float, default=2e-5)
    parser.add_argument("--lr-projection", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-studies", type=int)
    parser.add_argument("--skip-decode-validation", action="store_true")
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Deprecated alias for a 64-study bounded memory cache",
    )
    parser.add_argument(
        "--image-memory-cache-size",
        type=int,
        default=0,
        help="Maximum preprocessed study tensors held in each loader process",
    )
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        help="Persist deterministic preprocessed study tensors for reuse across runs",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume from last.pt in --output-dir")
    parser.add_argument("--allow-text-policy-mismatch", action="store_true")
    parser.add_argument(
        "--trainable-checkpoints",
        action="store_true",
        help="Store trainable deltas instead of duplicate frozen backbone weights; requires pinned revisions",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text_model = TEXT_MODEL_CHOICES.get(args.text_model, args.text_model)
    config = TrainConfig(
        data_root=args.data_root,
        reports_csv=args.reports_csv,
        cohort_manifest=args.cohort_manifest,
        report_column=args.report_column,
        output_dir=args.output_dir,
        text_model=text_model,
        text_model_revision=args.text_model_revision,
        text_pooling=args.text_pooling,
        text_prefix=args.text_prefix,
        report_view=args.report_view,
        image_model=args.image_model,
        image_model_revision=args.image_model_revision,
        embedding_dim=args.embedding_dim,
        max_text_length=args.max_text_length,
        image_unfreeze_last=args.image_unfreeze_last,
        text_unfreeze_last=args.text_unfreeze_last,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        fold_count=args.fold_count,
        fold_index=args.fold_index,
        lr_image=args.lr_image,
        lr_text=args.lr_text,
        lr_projection=args.lr_projection,
        seed=args.seed,
        max_studies=args.max_studies,
        validate_pixels=not args.skip_decode_validation,
        cache_images=args.cache_images,
        image_cache_dir=args.image_cache_dir,
        image_memory_cache_size=(
            args.image_memory_cache_size or (64 if args.cache_images else 0)
        ),
        allow_text_policy_mismatch=args.allow_text_policy_mismatch,
        trainable_checkpoints=args.trainable_checkpoints,
        local_files_only=args.local_files_only,
        amp=not args.no_amp,
        prepare_only=args.prepare_only,
        resume=args.resume,
    )
    required_paths = (
        (config.cohort_manifest, config.cohort_manifest.with_suffix(".meta.json"))
        if config.cohort_manifest is not None
        else (config.reports_csv, config.series_csv, config.series_root)
    )
    missing = [path for path in required_paths if path is not None and not path.exists()]
    if missing:
        raise SystemExit("Missing required data path(s): " + ", ".join(map(str, missing)))
    print(json.dumps(run_training(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
