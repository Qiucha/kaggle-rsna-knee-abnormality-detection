#!/usr/bin/env python
"""Compare depth coverage: the current sampler versus contiguous tiled groups.

Reports, per series, how much of the volume each scheme observes and how far apart
the channels of one input are. Header-only -- no pixels, no training, no test split.

Usage: check_slice_grouping.py [--groups K] [--sample N]
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiment_tools.cohort_audit import DEFAULT_MANIFEST, DEFAULT_SEED, load_studies, slice_step_mm  # noqa: E402
from contrastive_learning.preprocessing import _slice_indices, slice_group_indices  # noqa: E402
from experiment_tools.dicom_series import middle_slice_header  # noqa: E402

BAND = (0.20, 0.80)
GROUP_SIZE = 3  # fixed by DINOv2's 3-channel patch embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--sample", type=int, default=120)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rows = load_studies(args.manifest, args.split, args.sample, args.seed)

    now_seen, new_seen, now_spread, new_spread, inputs = [], [], [], [], []
    for row in rows:
        for directory in row["slots"].values():
            header = middle_slice_header(directory)
            if header is None:
                continue
            ds, n = header
            step = slice_step_mm(ds)
            if not step:
                continue
            current = _slice_indices(n, BAND, GROUP_SIZE)
            grouped = slice_group_indices(n, BAND, GROUP_SIZE, args.groups)
            now_seen.append(len(set(int(i) for i in current)) / n)
            new_seen.append(len(set(int(i) for i in grouped.ravel())) / n)
            # Spread of one input's channels: what the 3 channels straddle, in mm.
            now_spread.append((current.max() - current.min()) * step)
            new_spread.append(float(np.mean(grouped.max(1) - grouped.min(1))) * step)
            inputs.append(len(grouped))

    def line(label, values, unit=""):
        a = np.asarray(values, dtype=float)
        return (f"  {label:<34} median {np.median(a):6.1f}{unit}  "
                f"p10 {np.percentile(a, 10):6.1f}  p90 {np.percentile(a, 90):6.1f}")

    print(f"{len(now_seen)} series | current: 1 input of {GROUP_SIZE} spread slices | "
          f"proposed: {args.groups} inputs of {GROUP_SIZE} contiguous slices\n")
    print("VOLUME OBSERVED (share of slices)")
    print(line("current", [v * 100 for v in now_seen], "%"))
    print(line(f"grouped (K={args.groups})", [v * 100 for v in new_seen], "%"))
    print("\nCHANNEL SPREAD WITHIN ONE INPUT (should be small: one coherent stack)")
    print(line("current", now_spread, "mm"))
    print(line(f"grouped (K={args.groups})", new_spread, "mm"))
    print(f"\nCOST: {np.mean(inputs):.0f}x the image forward passes per study "
          f"({3} planes -> {3 * args.groups} encoded views).")


if __name__ == "__main__":
    main()
