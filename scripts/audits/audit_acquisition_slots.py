#!/usr/bin/env python
"""Recover weighting and fat suppression per series, then measure slot fill rates.

A slot is (plane x weighting x fat-suppressed). Slots must be defined from measured
fill rates, not from the product of every flag value: a slot present in 20% of studies
is mostly a masked-out zero, and that budget buys more elsewhere.

Reads DICOM headers only -- no pixels, no training, no test-split evaluation. Planes come
from train_series.csv, which is the one column of it that is not known to be broken;
weighting and fat suppression are recovered from the header because the CSV's two flags
are byte-identical (digest/acquisition-metadata-trust.md).

Usage: audit_acquisition_slots.py [--sample N] [--out PATH]
"""
import argparse, json, os, random, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning import paths  # noqa: E402
from contrastive_learning.acquisition import recover  # noqa: E402
from contrastive_learning.manifest import read_manifest_rows  # noqa: E402

TAGS = ("SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
        "RepetitionTime", "EchoTime")


def probe(args):
    """Read one series' header fields and recover its acquisition."""
    import pydicom
    study_uid, series_dir = args
    try:
        files = sorted(entry.name for entry in os.scandir(series_dir) if entry.name.endswith(".dcm"))
        if not files:
            return None
        ds = pydicom.dcmread(os.path.join(series_dir, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
    except Exception:
        return None
    values = {}
    for tag in TAGS:
        value = getattr(ds, tag, None)
        if value is not None and not isinstance(value, (str, int, float)):
            value = "\\".join(str(item) for item in value)
        values[tag] = value
    acquisition = recover(
        description=values["SeriesDescription"], sequence_name=values["SequenceName"],
        scan_options=values["ScanOptions"], scanning_sequence=values["ScanningSequence"],
        repetition_time=values["RepetitionTime"], echo_time=values["EchoTime"],
    )
    return {"study_uid": study_uid, "series_uid": os.path.basename(series_dir),
            "n_slices": len(files), "weighting": acquisition.weighting,
            "fat_suppressed": acquisition.fat_suppressed,
            "fluid_sensitive": acquisition.fluid_sensitive, "source": acquisition.source,
            "description": str(values["SeriesDescription"] or "")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/contrastive-cohort.jsonl")
    parser.add_argument("--series-csv", type=Path, help="default: train_series.csv on this host's data_root")
    parser.add_argument("--sample", type=int, default=0, help="studies; 0 = all")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", default="digest/acquisition-slots.json")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    rows = [r for r in read_manifest_rows(args.manifest) if r["slots"]]
    if args.sample:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))

    jobs, selected = [], set()
    for row in rows:
        study_dir = Path(next(iter(row["slots"].values()))).parent
        selected.update(os.path.basename(p) for p in row["slots"].values())
        for entry in os.scandir(study_dir):
            if entry.is_dir():
                jobs.append((row["study_uid"], entry.path))

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        series = [r for r in pool.map(probe, jobs) if r]

    planes = pd.read_csv(args.series_csv or paths.series_csv(paths.host_data_root()))
    plane_of = dict(zip(planes["SeriesInstanceUID"], planes["Anatomical_Plane"]))
    for record in series:
        record["plane"] = plane_of.get(record["series_uid"], "?")
        record["selected_by_pipeline"] = record["series_uid"] in selected
        record["slot"] = (f"{record['plane']} / {record['weighting']} / "
                          f"{'fatsat' if record['fat_suppressed'] else 'no-fatsat'}")

    frame = pd.DataFrame(series)
    n_studies = frame["study_uid"].nunique()
    fill = (pd.crosstab(frame["study_uid"], frame["slot"]) > 0).mean().sort_values(ascending=False)

    result = {
        "studies": int(n_studies), "series": int(len(frame)),
        "weighting_counts": frame["weighting"].value_counts().to_dict(),
        "decided_by": frame["source"].value_counts().to_dict(),
        "fat_suppressed_share": float(frame["fat_suppressed"].mean()),
        "fluid_sensitive_share": float(frame["fluid_sensitive"].mean()),
        "slot_fill": {slot: float(share) for slot, share in fill.items()},
        "selected_by_pipeline": {
            "weighting": frame[frame.selected_by_pipeline]["weighting"].value_counts().to_dict(),
            "fat_suppressed_share": float(frame[frame.selected_by_pipeline]["fat_suppressed"].mean()),
            "fluid_sensitive_share": float(frame[frame.selected_by_pipeline]["fluid_sensitive"].mean()),
        },
        "discarded_by_pipeline": {
            "weighting": frame[~frame.selected_by_pipeline]["weighting"].value_counts().to_dict(),
            "fat_suppressed_share": float(frame[~frame.selected_by_pipeline]["fat_suppressed"].mean()),
            "fluid_sensitive_share": float(frame[~frame.selected_by_pipeline]["fluid_sensitive"].mean()),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print(f"{result['series']:,} series across {n_studies:,} studies\n")
    print("weighting:", ", ".join(f"{k} {v:,}" for k, v in result["weighting_counts"].items()))
    print("decided by:", ", ".join(f"{k} {v:,}" for k, v in result["decided_by"].items()))
    print(f"fat-suppressed {result['fat_suppressed_share']:.1%}  "
          f"fluid-sensitive {result['fluid_sensitive_share']:.1%}")
    print("\nSLOT FILL (share of studies holding at least one such series)")
    for slot, share in fill.items():
        if share < 0.02:
            continue
        print(f"  {share:>6.1%}  {'#' * int(round(share * 40)):<40}  {slot}")
    print(f"\nslots present in over 80% of studies: {int((fill > 0.8).sum())} of {len(fill)}")
    sel, dis = result["selected_by_pipeline"], result["discarded_by_pipeline"]
    print(f"\nWHAT THE CURRENT PIPELINE PICKS vs DISCARDS")
    print(f"  selected : fatsat {sel['fat_suppressed_share']:.1%}  "
          f"fluid-sensitive {sel['fluid_sensitive_share']:.1%}  {sel['weighting']}")
    print(f"  discarded: fatsat {dis['fat_suppressed_share']:.1%}  "
          f"fluid-sensitive {dis['fluid_sensitive_share']:.1%}  {dis['weighting']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
