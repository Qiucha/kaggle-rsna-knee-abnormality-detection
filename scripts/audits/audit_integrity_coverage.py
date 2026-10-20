"""Read-only §7 integrity audit: patient independence, join coverage, duplicate scope.

Answers the open half of the knee CLIP feedback's §7 from local files plus DICOM
*headers only*. Nothing here decodes a pixel, opens the locked test's content, or
writes to the cohort. Test-split studies contribute counts, never labels or text.

Why headers and not raw volumes: `/mnt/d/train_series` is 532 GB over 4,407 studies
on a 9p-mounted Windows drive. A header read is ~27 ms/study (~2 min for the cohort);
a full raw-volume content hash is a different order of magnitude and is recorded as a
scoped gap rather than attempted.

Output: digest/integrity-coverage-audit.json
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning.manifest import read_manifest_rows

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl"
SERIES_CSV = ROOT / "data/train_series.csv"
OUT = ROOT / "digest/integrity-coverage-audit.json"

# One middle DICOM per study carries the patient/session tags; the rest of the series
# repeats them. Reading more would not add identity information.
IDENTITY_TAGS = [
    "PatientID",
    "PatientSex",
    "PatientName",
    "PatientBirthDate",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "FrameOfReferenceUID",
    "Manufacturer",
    "ManufacturerModelName",
    "DeviceSerialNumber",
    "InstitutionName",
]


def load_rows() -> list[dict]:
    return read_manifest_rows(COHORT)


def collision_groups(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Map each key shared by more than one member to its members."""
    index: dict[str, list[str]] = collections.defaultdict(list)
    for key, member in pairs:
        index[key].append(member)
    return {key: sorted(members) for key, members in index.items() if len(members) > 1}


def split_profile(groups: dict[str, list[str]], split_of: dict[str, str]) -> dict:
    """Summarise collision groups by how they sit against the split boundary."""
    crossing, within, sizes = [], [], []
    for key, members in groups.items():
        splits = sorted({split_of.get(uid, "unknown") for uid in members})
        sizes.append(len(members))
        entry = {"key": key, "members": len(members), "splits": splits}
        (crossing if len(splits) > 1 else within).append(entry)
    return {
        "groups": len(groups),
        "studies_involved": sum(sizes),
        "largest_group": max(sizes, default=0),
        "groups_crossing_splits": len(crossing),
        "studies_in_crossing_groups": sum(e["members"] for e in crossing),
        "groups_within_one_split": len(within),
        "crossing_detail": sorted(crossing, key=lambda e: -e["members"])[:25],
    }


def patient_grouping(rows: list[dict]) -> dict:
    """Whether the manifest's own grouping can make a split patient-disjoint."""
    counts = collections.Counter(r["patient_group_id"] for r in rows)
    split_of = {r["study_uid"]: r["split"] for r in rows}
    by_group: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        by_group[row["patient_group_id"]].add(row["split"])
    return {
        "rows": len(rows),
        "distinct_group_ids": len(counts),
        "group_size_distribution": dict(sorted(collections.Counter(counts.values()).items())),
        "groups_spanning_splits": sum(1 for splits in by_group.values() if len(splits) > 1),
        "every_group_is_a_single_study": set(counts.values()) == {1},
        "split_sizes": dict(sorted(collections.Counter(split_of.values()).items())),
    }


def series_join(rows: list[dict]) -> dict:
    """Many-to-one joins: one series reused by more than one study."""
    split_of = {r["study_uid"]: r["split"] for r in rows}

    # (a) the source table, independent of what the manifest selected
    csv_pairs: list[tuple[str, str]] = []
    header = None
    import csv as _csv

    with SERIES_CSV.open(encoding="utf-8", newline="") as handle:
        reader = _csv.DictReader(handle)
        header = reader.fieldnames
        for record in reader:
            csv_pairs.append((record["SeriesInstanceUID"], record["StudyInstanceUID"]))
    csv_shared = {
        series: sorted(set(studies))
        for series, studies in collision_groups(csv_pairs).items()
        if len(set(studies)) > 1
    }

    # (b) what the manifest actually points training at
    slot_pairs = [
        (str(path), row["study_uid"]) for row in rows for path in row.get("slots", {}).values()
    ]
    slot_shared = collision_groups(slot_pairs)

    return {
        "series_csv": {
            "columns": header,
            "rows": len(csv_pairs),
            "distinct_series": len({s for s, _ in csv_pairs}),
            "series_under_multiple_studies": len(csv_shared),
            "examples": dict(list(csv_shared.items())[:10]),
        },
        "manifest_slots": {
            "slot_references": len(slot_pairs),
            "distinct_series_directories": len({p for p, _ in slot_pairs}),
            "directories_shared_by_multiple_studies": len(slot_shared),
            "detail": split_profile(slot_shared, split_of) if slot_shared else None,
            "examples": {k: v for k, v in list(slot_shared.items())[:10]},
        },
    }


def report_collisions(rows: list[dict]) -> dict:
    """Exact-duplicate report text, translated and raw, against the split boundary."""
    split_of = {r["study_uid"]: r["split"] for r in rows}
    result = {}
    for label, field in [
        ("translated_report", "report_sha256"),
        ("source_report", "source_report_sha256"),
    ]:
        pairs = [(r[field], r["study_uid"]) for r in rows if r.get(field)]
        result[label] = split_profile(collision_groups(pairs), split_of)
        result[label]["rows_with_field"] = len(pairs)
    return result


def fingerprint_collisions(rows: list[dict]) -> dict:
    """Collision scan over `series_fingerprints`, with its semantics stated.

    `series_fingerprint` (preprocessing.py) hashes the preprocessing version plus each
    file's *name, size and mtime* — stat metadata, not content. It therefore cannot see
    a re-exported or re-identified duplicate, and a collision means an identical file
    tree rather than identical pixels. Reported for completeness with that limitation.
    """
    split_of = {r["study_uid"]: r["split"] for r in rows}
    per_plane = {}
    for plane in ("Axial", "Coronal", "Sagittal"):
        pairs = [
            (r["series_fingerprints"][plane], r["study_uid"])
            for r in rows
            if r.get("series_fingerprints", {}).get(plane)
        ]
        per_plane[plane] = split_profile(collision_groups(pairs), split_of)
        per_plane[plane]["rows_with_plane"] = len(pairs)

    whole = [
        (json.dumps(r["series_fingerprints"], sort_keys=True), r["study_uid"])
        for r in rows
        if r.get("series_fingerprints")
    ]
    return {
        "semantics": (
            "sha256 of PREPROCESSING_VERSION plus each .dcm file name, st_size and "
            "st_mtime_ns (preprocessing.py:196). Stat metadata, not pixel content: "
            "cannot detect a re-exported duplicate, and a collision proves an identical "
            "file tree rather than identical images."
        ),
        "per_plane": per_plane,
        "all_three_planes": split_profile(collision_groups(whole), split_of),
    }


def identity_probe(rows: list[dict]) -> dict:
    """Header-only patient/session identity across studies.

    The standing claim is that the competition supplies no reliable cross-study patient
    identity. This checks it instead of inheriting it. A collision here is potential
    cross-split leakage; unique-per-study values confirm the claim with evidence.
    """
    split_of = {r["study_uid"]: r["split"] for r in rows}
    records: dict[str, dict] = {}
    unreadable: list[str] = []
    started = time.time()

    for row in rows:
        uid = row["study_uid"]
        slots = row.get("slots", {})
        chosen = None
        for _plane, directory in sorted(slots.items()):
            files = sorted(Path(directory).glob("*.dcm"))
            if files:
                chosen = files[len(files) // 2]
                break
        if chosen is None:
            unreadable.append(uid)
            continue
        try:
            ds = pydicom.dcmread(
                chosen, stop_before_pixels=True, force=True, specific_tags=IDENTITY_TAGS
            )
        except Exception:
            unreadable.append(uid)
            continue
        records[uid] = {
            tag: ("" if getattr(ds, tag, None) is None else str(getattr(ds, tag, "")))
            for tag in IDENTITY_TAGS
        }

    elapsed = time.time() - started
    per_tag = {}
    for tag in IDENTITY_TAGS:
        values = [(v[tag], uid) for uid, v in records.items() if v[tag] != ""]
        groups = collision_groups(values)
        per_tag[tag] = {
            "studies_with_a_value": len(values),
            "distinct_values": len({v for v, _ in values}),
            "collision_profile": split_profile(groups, split_of),
        }

    # A PatientID shared by two studies is the leakage case worth naming explicitly.
    pid_pairs = [(v["PatientID"], uid) for uid, v in records.items() if v["PatientID"]]
    pid_groups = collision_groups(pid_pairs)
    leaking = {
        pid: sorted({split_of[uid] for uid in members})
        for pid, members in pid_groups.items()
        if len({split_of[uid] for uid in members}) > 1
    }

    return {
        "studies_probed": len(records),
        "studies_unreadable": len(unreadable),
        "unreadable_examples": unreadable[:10],
        "seconds": round(elapsed, 1),
        "per_tag": per_tag,
        "patient_id": {
            "multi_study_patients": len(pid_groups),
            "studies_in_multi_study_patients": sum(len(m) for m in pid_groups.values()),
            "patients_spanning_splits": len(leaking),
            "studies_in_split_spanning_patients": sum(
                len(pid_groups[p]) for p in leaking
            ),
            "split_pairs_observed": dict(
                sorted(collections.Counter("+".join(s) for s in leaking.values()).items())
            ),
            "largest_patient_group": max((len(m) for m in pid_groups.values()), default=0),
        },
        "record_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()
        ).hexdigest(),
    }


def main() -> None:
    rows = load_rows()
    audit = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cohort": str(COHORT.relative_to(ROOT)),
        "cohort_sha256": hashlib.sha256(COHORT.read_bytes()).hexdigest(),
        "scope": {
            "reads": "local CSV/manifest plus one middle DICOM header per study",
            "decodes_pixels": False,
            "opens_locked_test_content": False,
            "raw_volume_content_hashing": (
                "not attempted: 532 GB over 4,407 studies on a 9p-mounted Windows drive"
            ),
        },
        "patient_grouping": patient_grouping(rows),
        "series_join": series_join(rows),
        "report_collisions": report_collisions(rows),
        "series_fingerprints": fingerprint_collisions(rows),
        "identity_probe": identity_probe(rows),
    }
    OUT.write_text(json.dumps(audit, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "identity_probe"}, indent=1)[:3000])
    print("identity_probe:", json.dumps(audit["identity_probe"], indent=1)[:3000])


if __name__ == "__main__":
    main()
