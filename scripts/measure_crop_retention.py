#!/usr/bin/env python
"""How much tissue a centred physical crop keeps, for candidate crop widths.

`read_slot` crops a fixed `crop_mm` box about the image centre. That constant was
never derived from the data. This measures, per candidate width, the share of
foreground (non-air) intensity mass such a crop retains -- so the constant can be
chosen from evidence rather than assumed.

Foreground is everything above a per-slice Otsu threshold; air and the background
ramp fall below it. Reads pixels, so it is slower than the header-only audits.
No training, no test split.

Usage: measure_crop_retention.py [--sample N] [--candidates 120,130,...]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import DEFAULT_MANIFEST, DEFAULT_SEED, load_studies  # noqa: E402
from contrastive_learning.preprocessing import (  # noqa: E402
    _decode_pixels, ordered_slice_paths, series_fov_mm,
)

DEFAULT_CANDIDATES = (110, 120, 130, 140, 150, 160, 180)


def otsu(values: np.ndarray) -> float:
    """Threshold splitting the histogram into air and tissue."""
    counts, edges = np.histogram(values, bins=64)
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(counts)
    total = weight[-1]
    if total == 0 or counts.sum() == 0:
        return float(values.mean())
    mean = np.cumsum(counts * centres)
    valid = (weight > 0) & (total - weight > 0)
    if not valid.any():
        return float(values.mean())
    between = np.zeros_like(centres, dtype=float)
    between[valid] = ((mean[-1] * weight[valid] / total - mean[valid]) ** 2 /
                      (weight[valid] * (total - weight[valid]) / total ** 2 + 1e-12))
    return float(centres[int(np.argmax(between))])


def retention(image: np.ndarray, spacing_mm: float, candidates) -> dict[int, float]:
    """Share of foreground mass kept by a centred crop of each candidate width."""
    foreground = np.clip(image - otsu(image), 0, None)
    total = foreground.sum()
    if total <= 0:
        return {}
    height, width = foreground.shape
    cy, cx = height // 2, width // 2
    kept = {}
    for mm in candidates:
        half = int(round(mm / spacing_mm)) // 2
        box = foreground[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        kept[mm] = float(box.sum() / total)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--candidates", default=",".join(str(c) for c in DEFAULT_CANDIDATES))
    parser.add_argument("--out", default="digest/crop-retention.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    candidates = [int(c) for c in args.candidates.split(",")]

    rows = load_studies(args.manifest, args.split, args.sample, args.seed)

    per_candidate = {c: [] for c in candidates}
    n = 0
    for row in rows:
        for directory in row["slots"].values():
            directory = Path(directory)
            fov = series_fov_mm(directory)
            if fov is None:
                continue
            try:
                # Anatomical order, not filename order: filenames are random SOP UIDs,
                # so files[n // 2] is an arbitrary slice rather than a central one.
                paths_in_order = ordered_slice_paths(directory)
                image = _decode_pixels(paths_in_order[len(paths_in_order) // 2])
            except Exception:
                continue
            if image is None:
                continue
            spacing = fov[0] / image.shape[0]
            kept = retention(image.astype(np.float32), spacing, candidates)
            if not kept:
                continue
            n += 1
            for c, share in kept.items():
                per_candidate[c].append(share)

    result = {"series": n, "split": args.split, "candidates": {}}
    for c in candidates:
        a = np.asarray(per_candidate[c])
        result["candidates"][str(c)] = {
            "mean_retained": float(a.mean()), "median_retained": float(np.median(a)),
            "p10_retained": float(np.percentile(a, 10)),
            "share_of_series_keeping_95pct": float((a >= 0.95).mean()),
            "share_of_series_keeping_99pct": float((a >= 0.99).mean())}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print(f"{n} series\n")
    print(f"{'crop mm':>8} {'mean kept':>10} {'median':>8} {'p10':>8} {'>=95% kept':>11} {'>=99% kept':>11}")
    for c in candidates:
        e = result["candidates"][str(c)]
        mark = "   <- current" if c == 130 else ""
        print(f"{c:>8} {e['mean_retained']:10.3f} {e['median_retained']:8.3f} {e['p10_retained']:8.3f} "
              f"{e['share_of_series_keeping_95pct']:11.2f} {e['share_of_series_keeping_99pct']:11.2f}{mark}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
