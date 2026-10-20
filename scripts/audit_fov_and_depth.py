#!/usr/bin/env python
"""Audit two preprocessing assumptions against the DICOM itself.

1. FOV. `read_slot` crops a fixed `crop_mm` physical box. This reports the true
   per-series field of view, how often that constant crops rather than no-ops, and
   how much physical extent it discards.
2. Depth. `_slice_indices` picks `slices_per_plane` slices spread across a fixed
   band, so most of the volume is never seen. This reports the physical gap between
   adjacent sampled slices -- the width of anatomy the model cannot observe.

Read-only: DICOM headers only, no pixels, no training, no test split.

Usage: audit_fov_and_depth.py [--sample N] [--split SPLIT] [--out PATH]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_tools.cohort_audit import (  # noqa: E402
    DEFAULT_MANIFEST, DEFAULT_SEED, describe, load_studies, slice_step_mm,
)
from contrastive_learning.preprocessing import series_fov_mm, _slice_indices  # noqa: E402

CROP_MM = 130.0        # the constant currently hardcoded in read_slot
SLICES_PER_PLANE = 3   # config.slices_per_plane, hard-constrained to 3 by DINOv2's 3 channels
BAND = (0.20, 0.80)


def _require_pydicom():
    try:
        import pydicom
    except ModuleNotFoundError:
        sys.exit("pydicom is required; use .venv, not .venv-cuda")
    return pydicom


def audit_series(pydicom, series_dir: Path) -> dict | None:
    try:
        files = sorted(os.listdir(series_dir))
    except OSError:
        return None
    if not files:
        return None
    fov = series_fov_mm(series_dir)
    if fov is None:
        return None
    try:
        ds = pydicom.dcmread(series_dir / files[len(files) // 2],
                             stop_before_pixels=True, force=True)
    except Exception:
        return None
    step = slice_step_mm(ds)
    indices = _slice_indices(len(files), BAND, SLICES_PER_PLANE)
    gaps = np.diff(np.unique(indices))
    return {
        "n_slices": len(files),
        "fov_rows_mm": fov[0],
        "fov_cols_mm": fov[1],
        # What the fixed crop keeps, and what it throws away.
        "crop_is_a_no_op": bool(CROP_MM >= min(fov)),
        "discarded_rows_mm": max(0.0, fov[0] - CROP_MM),
        "slice_step_mm": step,
        "sampled_indices": [int(i) for i in indices],
        "unseen_slices": len(files) - len(set(int(i) for i in indices)),
        "gap_mm": float(np.mean(gaps) * step) if step and len(gaps) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--sample", type=int, default=120, help="studies to sample; 0 = all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="digest/fov-and-depth-audit.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    pydicom = _require_pydicom()
    rows = load_studies(args.manifest, args.split, args.sample, args.seed)

    records = []
    for row in rows:
        for plane, directory in row["slots"].items():
            entry = audit_series(pydicom, Path(directory))
            if entry:
                records.append({"plane": plane, **entry})

    fov_rows = [r["fov_rows_mm"] for r in records]
    result = {
        "manifest": args.manifest, "split": args.split,
        "studies_sampled": len(rows), "series_audited": len(records),
        "crop_mm_constant": CROP_MM, "slices_per_plane": SLICES_PER_PLANE, "band": BAND,
        "fov_rows_mm": {"median": float(np.median(fov_rows)),
                        "p10": float(np.percentile(fov_rows, 10)),
                        "p90": float(np.percentile(fov_rows, 90)),
                        "min": float(min(fov_rows)), "max": float(max(fov_rows))},
        "share_where_crop_is_a_no_op": float(np.mean([r["crop_is_a_no_op"] for r in records])),
        "discarded_rows_mm": {
            "median": float(np.median([r["discarded_rows_mm"] for r in records])),
            "p90": float(np.percentile([r["discarded_rows_mm"] for r in records], 90))},
        "unseen_slice_share": float(np.mean([r["unseen_slices"] / r["n_slices"] for r in records])),
        "gap_mm": {"median": float(np.median([r["gap_mm"] for r in records if r["gap_mm"]])),
                   "p90": float(np.percentile([r["gap_mm"] for r in records if r["gap_mm"]], 90))},
        "per_series": records,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print(f"{len(records)} series from {len(rows)} {args.split} studies\n")
    print("FIELD OF VIEW (true, from DICOM)")
    print(describe(fov_rows, "rows"))
    print(describe([r["fov_cols_mm"] for r in records], "cols"))
    print(f"  the fixed {CROP_MM:g}mm crop is a no-op in "
          f"{100 * result['share_where_crop_is_a_no_op']:.1f}% of series")
    print(describe([r["discarded_rows_mm"] for r in records], "physical extent discarded by it"))
    print("\nDEPTH SAMPLING")
    print(describe([r["n_slices"] for r in records], "slices per series", unit=""))
    print(f"  unseen slices: {100 * result['unseen_slice_share']:.1f}% of every series")
    print(describe([r["gap_mm"] for r in records], "physical gap between sampled slices"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
