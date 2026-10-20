"""Header-only near-duplicate screen over all 4,407 studies, including the test split.

The existing pixel evidence covers 3,967 cached train+validation tensors (a 16x16
pooled screen) plus one exact-pixel verification of the known pair. Nothing covers
test, and nothing covers the raw volumes: `/mnt/d/train_series` is 532 GB, and test
pixels are gated regardless. So this screens the *acquisition* instead -- geometry and
sequence tags that two exports of one scan must share.

Two stages, because the cheap stage is coarse:

  Stage 1 reads one header per selected series and keeps only tags that are constant
  across a series (matrix, spacing, TR/TE, coil, scanner, protocol name) plus the slice
  count. Three reads per study, ~4 minutes for the cohort. Many studies share a scanner
  and protocol, so a stage-1 collision is a candidate, not a finding.

  Stage 2 reads every slice header inside a stage-1 group and compares the *sorted
  multiset of SliceLocation values* per plane. That is invariant to file naming, which
  matters here: filenames are randomised SOP UIDs, so "the middle file" is a different
  physical slice in each copy of a duplicated study.

That last point is not hypothetical. A first version of this screen hashed the middle
file's geometry and MISSED the known quarantined pair, whose two copies differ in
InstanceNumber, SliceLocation, ImagePositionPatient and window values purely because
name-sorting picked a different slice from each. The pair is kept as a positive
control: if the screen cannot find a duplicate we already know is there, a null result
means nothing.

What this never reads: pixels, labels, report text. Test studies contribute an
acquisition fingerprint and a split name only.

Output: digest/acquisition-duplicate-screen.json
"""

from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import pydicom

from contrastive_learning.manifest import read_manifest_rows

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl"
OUT = ROOT / "digest/acquisition-duplicate-screen.json"

KNOWN_PAIR = [
    "study-b65cfd4f5265d33b",
    "study-7bbbc0e80da5e8ac",
]

# Constant across every slice of a series, and preserved by re-export. No UID, patient
# tag or date: this dataset re-randomises all of those per study, which is precisely
# why they cannot be used to link two copies of one scan.
SERIES_TAGS = [
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "FlipAngle",
    "EchoTrainLength",
    "PixelBandwidth",
    "ImagingFrequency",
    "MagneticFieldStrength",
    "AcquisitionMatrix",
    "PercentSampling",
    "PercentPhaseFieldOfView",
    "NumberOfPhaseEncodingSteps",
    "SeriesDescription",
    "ScanningSequence",
    "SequenceVariant",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "ReceiveCoilName",
    "PatientPosition",
]
GEOMETRY_TAGS = ["SliceLocation", "ImagePositionPatient"]

# Stage 2 reads every slice in a group. A pathological stage-1 group (hundreds of
# studies on one common protocol) would cost hours, so it is skipped and reported
# rather than silently truncated.
STAGE2_MAX_GROUP = 40


def series_signature(directory: str) -> tuple[str, int] | None:
    """Series-constant tags plus slice count, read from a single arbitrary slice."""
    files = sorted(Path(directory).glob("*.dcm"))
    if not files:
        return None
    try:
        ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True, specific_tags=SERIES_TAGS)
    except Exception:
        return None
    payload = {tag: str(getattr(ds, tag, "")) for tag in SERIES_TAGS}
    payload["slice_count"] = str(len(files))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), len(files)


def slice_geometry(directory: str) -> str | None:
    """Sorted multiset of slice positions: invariant to the randomised file names."""
    locations = []
    for path in sorted(Path(directory).glob("*.dcm")):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=GEOMETRY_TAGS)
        except Exception:
            return None
        location = getattr(ds, "SliceLocation", None)
        if location is None:
            location = str(getattr(ds, "ImagePositionPatient", ""))
        locations.append(f"{float(location):.4f}" if isinstance(location, (int, float)) else str(location))
    if not locations:
        return None
    return hashlib.sha256("|".join(sorted(locations)).encode()).hexdigest()


def main() -> None:
    rows = read_manifest_rows(COHORT)
    split_of = {r["study_uid"]: r["split"] for r in rows}
    slots_of = {r["study_uid"]: dict(sorted(r.get("slots", {}).items())) for r in rows}

    started = time.time()
    stage1: dict[str, str] = {}
    incomplete: list[str] = []
    for uid, slots in slots_of.items():
        parts = []
        for plane, directory in slots.items():
            signature = series_signature(directory)
            if signature is None:
                parts = []
                break
            parts.append(f"{plane}:{signature[0]}:{signature[1]}")
        if parts:
            stage1[uid] = hashlib.sha256("|".join(parts).encode()).hexdigest()
        else:
            incomplete.append(uid)
    stage1_seconds = time.time() - started

    index: dict[str, list[str]] = collections.defaultdict(list)
    for uid, fingerprint in stage1.items():
        index[fingerprint].append(uid)
    candidates = {fp: sorted(uids) for fp, uids in index.items() if len(uids) > 1}

    control_stage1 = (
        stage1.get(KNOWN_PAIR[0]) is not None and stage1.get(KNOWN_PAIR[0]) == stage1.get(KNOWN_PAIR[1])
    )

    started = time.time()
    confirmed, skipped, geometry_cache = [], [], {}
    for fingerprint, uids in sorted(candidates.items(), key=lambda kv: -len(kv[1])):
        if len(uids) > STAGE2_MAX_GROUP:
            skipped.append({"fingerprint": fingerprint[:16], "members": len(uids)})
            continue
        refined: dict[str, list[str]] = collections.defaultdict(list)
        for uid in uids:
            per_plane = []
            for plane, directory in slots_of[uid].items():
                if (uid, plane) not in geometry_cache:
                    geometry_cache[(uid, plane)] = slice_geometry(directory)
                value = geometry_cache[(uid, plane)]
                if value is None:
                    per_plane = []
                    break
                per_plane.append(f"{plane}:{value}")
            if per_plane:
                refined["|".join(per_plane)].append(uid)
        for _key, members in refined.items():
            if len(members) > 1:
                splits = sorted({split_of[u] for u in members})
                confirmed.append(
                    {
                        "members": len(members),
                        "splits": splits,
                        "crosses_splits": len(splits) > 1,
                        "involves_test": "test" in splits,
                        "study_uids": sorted(members),
                    }
                )
    stage2_seconds = time.time() - started

    control_stage2 = any(sorted(KNOWN_PAIR) == g["study_uids"] for g in confirmed)
    audit = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cohort": str(COHORT.relative_to(ROOT)),
        "method": (
            "Stage 1: sha256 of series-constant acquisition tags plus slice count, per "
            "selected plane, one header read per series. Stage 2: within each stage-1 "
            "collision group, the sorted multiset of SliceLocation values per plane. No "
            "pixels, labels or report text; no UIDs, patient tags or dates."
        ),
        "stage1_tags": SERIES_TAGS,
        "stage2_tags": GEOMETRY_TAGS,
        "studies_fingerprinted": len(stage1),
        "studies_incomplete": len(incomplete),
        "incomplete_examples": incomplete[:10],
        "stage1_seconds": round(stage1_seconds, 1),
        "stage2_seconds": round(stage2_seconds, 1),
        "positive_control": {
            "pair": KNOWN_PAIR,
            "stage1_groups_the_pair": control_stage1,
            "stage2_confirms_the_pair": control_stage2,
            "verdict": (
                "screen recovers the known duplicate; a null result elsewhere is "
                "meaningful within the stated limitation"
                if control_stage2
                else "screen does NOT recover the known duplicate; treat any null result "
                "as uninformative"
            ),
        },
        "stage1": {
            "candidate_groups": len(candidates),
            "studies_in_candidate_groups": sum(len(u) for u in candidates.values()),
            "largest_candidate_group": max((len(u) for u in candidates.values()), default=0),
            "group_size_distribution": dict(
                sorted(collections.Counter(len(u) for u in candidates.values()).items())
            ),
        },
        "stage2": {
            "confirmed_groups": len(confirmed),
            "studies_in_confirmed_groups": sum(g["members"] for g in confirmed),
            "groups_crossing_splits": sum(1 for g in confirmed if g["crosses_splits"]),
            "groups_involving_test": sum(1 for g in confirmed if g["involves_test"]),
            "groups_skipped_as_too_large": skipped,
            "max_group_examined": STAGE2_MAX_GROUP,
            "detail": sorted(confirmed, key=lambda g: -g["members"]),
        },
        "limitation": (
            "A confirmed group means two studies' selected series agree on every "
            "series-constant acquisition tag, slice count, and the full set of slice "
            "positions. That is strong evidence of the same acquisition, not proof of "
            "identical pixels. Confirming a group by pixels is cheap for train and "
            "validation and is a locked-test decision for any group containing a test "
            "study. The screen also covers only the three selected series per study, "
            "not the full raw volume."
        ),
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in {"stage1_tags", "stage2_tags"}}, indent=1)[:6000])


if __name__ == "__main__":
    main()
