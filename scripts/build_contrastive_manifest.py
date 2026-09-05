#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.manifest import ManifestConfig, build_cohort_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a versioned, patient-grouped contrastive-learning cohort manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reports-csv", type=Path, default=Path("data/sample/train.csv"))
    parser.add_argument("--series-csv", type=Path, default=Path("/mnt/d/train_series.csv"))
    parser.add_argument("--series-root", type=Path, default=Path("/mnt/d/train_series"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/contrastive-cohort.jsonl")
    )
    parser.add_argument(
        "--text-policy",
        choices=("multilingual_raw", "translated_english"),
        default="multilingual_raw",
    )
    parser.add_argument("--translations-csv", type=Path)
    parser.add_argument("--report-column", default="Report")
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validate-pixels", action="store_true")
    parser.add_argument("--pixel-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_cohort_manifest(
        ManifestConfig(
            reports_csv=args.reports_csv,
            series_csv=args.series_csv,
            series_root=args.series_root,
            output_path=args.output,
            text_policy=args.text_policy,
            translations_csv=args.translations_csv,
            report_column=args.report_column,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            validate_pixels=args.validate_pixels,
            pixel_workers=args.pixel_workers,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
