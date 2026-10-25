"""Shared helpers for the read-only cohort and preprocessing audits under scripts/.

These scripts kept re-deriving the same three things -- how to sample studies from the
cohort, how many millimetres a slice step is, and how to print a distribution -- so they
live here once. Nothing in this module trains, writes to the cohort, or touches the test
split.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from contrastive_learning.manifest import read_manifest_rows

DEFAULT_MANIFEST = "data/processed/contrastive-cohort.jsonl"
DEFAULT_SEED = 20260914


def load_studies(manifest: str | Path = DEFAULT_MANIFEST, split: str | None = "train",
                 sample: int = 0, seed: int = DEFAULT_SEED) -> list[dict]:
    """Cohort rows that have slots, optionally one split and a reproducible sample."""
    rows = [row for row in read_manifest_rows(manifest) if row["slots"]]
    if split:
        rows = [row for row in rows if row.get("split") == split]
    if sample:
        rows = random.Random(seed).sample(rows, min(sample, len(rows)))
    return rows


def series_dirs(rows: list[dict]) -> list[tuple[str, Path]]:
    """Every (plane, directory) the pipeline actually reads, across the given rows."""
    return [(plane, Path(directory)) for row in rows for plane, directory in row["slots"].items()]


def slice_step_mm(ds) -> float | None:
    """Millimetres between slice centres: spacing between slices if stated, else thickness."""
    for tag in ("SpacingBetweenSlices", "SliceThickness"):
        value = getattr(ds, tag, None)
        if value:
            try:
                step = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(step) and step > 0:
                return step
    return None


def describe(values, label: str, unit: str = "mm", width: int = 34) -> str:
    """One line of distribution summary, or a note that nothing was measurable."""
    values = [v for v in values if v is not None]
    if not values:
        return f"  {label}: no data"
    a = np.asarray(values, dtype=float)
    return (f"  {label:<{width}} median {np.median(a):7.1f}{unit}  "
            f"p10 {np.percentile(a, 10):7.1f}  p90 {np.percentile(a, 90):7.1f}  (n={len(a)})")
