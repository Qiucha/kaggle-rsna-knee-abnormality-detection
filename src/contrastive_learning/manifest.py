"""Freeze one cohort: which studies exist, what text they train on, and their split.

Everything downstream reads a *cohort manifest* -- a JSON-lines file with one row
per study plus a sidecar ``.meta.json`` -- rather than re-deriving the cohort from
CSVs and DICOM directories. That indirection is the point: split assignment,
translation choice, and pixel eligibility are decided once, recorded with
checksums, and afterwards can only be read, never recomputed. A run's numbers are
therefore attributable to a specific cohort by hash.

Two decisions this module owns are easy to get wrong elsewhere:

**Splitting is by patient, not by study.** A patient with two studies would
otherwise put near-duplicate anatomy on both sides of the train/validation
boundary and inflate retrieval. Rows are grouped by a salted pseudonym of the
DICOM ``PatientID``, groups are shuffled, and whole groups are assigned -- so
targets are approximate by construction, since a group cannot be split.

**Text policy is explicit.** ``multilingual_raw`` trains on reports as written and
requires a multilingual text encoder. ``translated_english`` trains on English,
using the source text when it is already English and a supplied translation
otherwise; a study with neither becomes ineligible rather than silently training
on text in the wrong language. Both the chosen text and the original are recorded
with hashes, so a translation can always be audited after the fact.
"""

from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from contrastive_learning import paths
from contrastive_learning.artifacts import (
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from contrastive_learning.data import (
    StudyRecord,
    pick_study_slots,
    read_csv_rows,
    read_series_manifest,
    resolve_slot,
    slot_identity,
)
from contrastive_learning.preprocessing import (
    PLANES,
    PREPROCESSING_VERSION,
    image_kwargs,
    middle_dicom,
    read_slot,
    resolve_laterality,
    series_fingerprint,
)


MANIFEST_VERSION = 1
TEXT_POLICIES = ("multilingual_raw", "translated_english")
SPLITS = ("train", "validation", "test", "excluded")

# Salt for the patient pseudonym. It is versioned and constant: grouping must be
# reproducible across manifest builds, while raw PatientIDs stay out of artifacts.
_PSEUDONYM_SALT = "novo-contrastive-patient-v1"


@dataclass(frozen=True)
class ManifestConfig:
    """Everything that determines a cohort's contents, splits, and eligibility."""

    reports_csv: Path
    series_csv: Path
    series_root: Path
    output_path: Path
    text_policy: str = "multilingual_raw"
    translations_csv: Path | None = None
    report_column: str = "Report"
    translated_report_column: str = "Translated Report"
    language_column: str = "Original Language"
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    seed: int = 2026
    # Decoding every slice of every study is expensive; enable it to guarantee that
    # no run can fail mid-epoch on an unreadable study, and to record pixel QC.
    validate_pixels: bool = False
    pixel_workers: int = 4
    image_size: int = 224
    crop_mm: float = 130.0
    slices_per_plane: int = 3
    slice_band_low: float = 0.20
    slice_band_high: float = 0.80
    overwrite: bool = False

    @property
    def image_kwargs(self) -> dict[str, Any]:
        """Preprocessing settings used for pixel validation, matching TrainConfig."""
        return image_kwargs(self)


@dataclass(frozen=True)
class ManifestSummary:
    """Counts a caller can print to judge a freshly built cohort at a glance."""

    total_studies: int
    eligible_studies: int
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    preprocessing_counts: dict[str, int]
    output_path: str
    metadata_path: str


# --------------------------------------------------------------------------- #
# Patient identity
# --------------------------------------------------------------------------- #


def study_pseudonym(uid: str) -> str:
    """Public, unsalted pseudonym for a StudyInstanceUID, used in committed results.

    ``"study-" + sha256(uid)[:16]``: anyone holding the competition data can map it
    back; nobody else learns the UID.
    """
    return f"study-{sha256_text(uid)[:16]}"


def _pseudonym(value: str) -> str:
    return f"patient-{sha256_text(f'{_PSEUDONYM_SALT}\0{value}')[:20]}"


def _patient_id(slots: dict[str, Path] | None, fallback: str) -> str:
    """Read ``PatientID`` from one slice, falling back to a per-study identity.

    The fallback is deliberately unique per study: an unreadable tag must never
    collapse unrelated studies into one grouping key, which would move them into
    the same split together.
    """
    try:
        import pydicom

        path = middle_dicom(slots[PLANES[0]]) if slots else None
        if path is not None:
            dataset = pydicom.dcmread(
                path, stop_before_pixels=True, force=True, specific_tags=["PatientID"]
            )
            value = str(getattr(dataset, "PatientID", "")).strip()
            if value:
                return value
    except Exception:
        pass
    return f"study-fallback:{fallback}"


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _resolve_text(
    config: ManifestConfig, report: str, source_language: str, translated: str
) -> tuple[str, str, str]:
    """Choose the training text under the configured policy.

    Returns the text, its language, and its provenance. An empty text marks the
    study ineligible, which is how "no usable English text" is expressed without
    inventing one.
    """
    if config.text_policy == "multilingual_raw":
        return report, source_language, "raw_multilingual"
    if source_language == "en":
        return report, "en", "source_english"
    if source_language != "und" and translated:
        return translated, "en", "translated_english"
    return "", "und", "missing_translation"


def _assign_splits(
    rows: list[dict[str, Any]], train_fraction: float, validation_fraction: float, seed: int
) -> None:
    """Assign whole patient groups to train/validation/test, in place.

    Groups are walked in seeded random order and filled up to each target, so the
    realised proportions land near the targets but never split a patient. Rows that
    are not training eligible are marked ``excluded`` and never enter a split.
    """
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must be positive and sum to less than one")
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["training_eligible"]:
            by_patient[row["patient_group_id"]].append(row)
        else:
            row["split"] = "excluded"
    groups = list(by_patient.values())
    random.Random(seed).shuffle(groups)
    total = sum(len(group) for group in groups)
    train_target = round(total * train_fraction)
    validation_target = train_target + round(total * validation_fraction)
    assigned = 0
    for group in groups:
        if assigned < train_target:
            split = "train"
        elif assigned < validation_target:
            split = "validation"
        else:
            split = "test"
        for row in group:
            row["split"] = split
        assigned += len(group)


def _validate_row_pixels(row: dict[str, Any], config: ManifestConfig) -> tuple[str, dict[str, Any]]:
    """Decode every plane of one study, returning a status and pixel QC summary."""
    if not row["slots"]:
        return "missing_plane", {}
    tensors = [
        read_slot(resolve_slot(row["slots"][plane], config.series_root), **config.image_kwargs)
        for plane in PLANES
    ]
    if any(tensor is None for tensor in tensors):
        return "decode_failed", {}
    return "passed", {
        "shape": [len(PLANES), config.slices_per_plane, config.image_size, config.image_size],
        "finite": all(bool(tensor.isfinite().all()) for tensor in tensors),
        "min": min(float(tensor.min()) for tensor in tensors),
        "max": max(float(tensor.max()) for tensor in tensors),
    }


def _manifest_row(
    config: ManifestConfig,
    source: dict[str, str],
    study_uid: str,
    slots: dict[str, Path] | None,
    translation: dict[str, str],
    label_columns: list[str],
) -> dict[str, Any]:
    """Describe one study completely enough to train on it without the source CSVs."""
    report = (source.get(config.report_column) or "").strip()
    source_language = (translation.get(config.language_column) or "und").strip().lower()
    translated = (translation.get(config.translated_report_column) or "").strip()
    training_report, report_language, report_provenance = _resolve_text(
        config, report, source_language, translated
    )
    return {
        "manifest_version": MANIFEST_VERSION,
        "study_uid": study_uid,
        "patient_group_id": _pseudonym(_patient_id(slots, study_uid)),
        # A real split is assigned once every row exists; see _assign_splits.
        "split": "excluded",
        "training_eligible": bool(training_report and slots),
        "report": training_report,
        "source_report": report,
        "source_language": source_language,
        "report_language": report_language,
        "report_provenance": report_provenance,
        "report_sha256": sha256_text(training_report),
        "source_report_sha256": sha256_text(report),
        # <study>/<series> only: where the tree is mounted is a host setting.
        "slots": {plane: slot_identity(path) for plane, path in (slots or {}).items()},
        # Fingerprints let a later audit prove the referenced series bytes are
        # the ones this cohort was frozen against.
        "series_fingerprints": {
            plane: series_fingerprint(path) for plane, path in (slots or {}).items()
        },
        "laterality": resolve_laterality(slots) if slots else None,
        "preprocessing_status": "not_validated" if slots else "missing_plane",
        "preprocessing_qc": {},
        # Non-report CSV columns are carried through untouched for later analysis.
        "labels": {name: source.get(name, "") for name in label_columns},
    }


def build_cohort_manifest(config: ManifestConfig) -> ManifestSummary:
    """Freeze a cohort to disk and return its summary counts.

    Refuses to overwrite an existing manifest unless asked: a manifest is the
    identity of every experiment that cites it, so replacing one in place would
    invalidate results that still reference the old checksum.
    """
    output_path = Path(config.output_path)
    metadata_path = output_path.with_suffix(".meta.json")
    if not config.overwrite and (output_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Manifest output already exists: {output_path}")
    if config.text_policy not in TEXT_POLICIES:
        raise ValueError(f"Supported text policies: {', '.join(TEXT_POLICIES)}")

    required = {"StudyInstanceUID", config.report_column}
    report_rows = read_csv_rows(Path(config.reports_csv), required)
    if not report_rows:
        raise ValueError("Reports CSV is empty")
    series_by_study = read_series_manifest(Path(config.series_csv))

    translations: dict[str, dict[str, str]] = {}
    if config.translations_csv is not None:
        translation_rows = read_csv_rows(
            Path(config.translations_csv),
            {"StudyInstanceUID", config.language_column, config.translated_report_column},
        )
        translations = {
            row["StudyInstanceUID"].strip(): row
            for row in translation_rows
            if row.get("StudyInstanceUID", "").strip()
        }

    label_columns = [name for name in report_rows[0] if name not in required]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in report_rows:
        study_uid = (source.get("StudyInstanceUID") or "").strip()
        if not study_uid or study_uid in seen:
            raise ValueError(f"Missing or duplicate StudyInstanceUID: {study_uid!r}")
        seen.add(study_uid)
        rows.append(
            _manifest_row(
                config,
                source,
                study_uid,
                pick_study_slots(study_uid, series_by_study.get(study_uid, []), Path(config.series_root)),
                translations.get(study_uid, {}),
                label_columns,
            )
        )

    if config.validate_pixels:
        if config.pixel_workers < 1:
            raise ValueError("pixel_workers must be at least one")
        # DICOM decoding is dominated by file IO and releases the GIL, so threads
        # are enough and avoid re-importing pydicom in worker processes.
        with ThreadPoolExecutor(max_workers=config.pixel_workers) as executor:
            results = executor.map(lambda row: _validate_row_pixels(row, config), rows)
            for row, (status, qc) in zip(rows, results, strict=True):
                row["preprocessing_status"] = status
                row["preprocessing_qc"] = qc
                row["training_eligible"] = bool(row["training_eligible"] and status == "passed")

    _assign_splits(rows, config.train_fraction, config.validation_fraction, config.seed)
    # Reports keep their original scripts, so the manifest is not ASCII-escaped.
    write_jsonl(output_path, rows, ensure_ascii=False)
    write_json(metadata_path, _cohort_metadata(config, output_path))

    counts: dict[str, dict[str, int]] = {
        "split": defaultdict(int),
        "language": defaultdict(int),
        "preprocessing": defaultdict(int),
    }
    for row in rows:
        counts["split"][row["split"]] += 1
        counts["language"][row["report_language"]] += 1
        counts["preprocessing"][row["preprocessing_status"]] += 1
    eligible = sum(row["training_eligible"] for row in rows)
    return ManifestSummary(
        total_studies=len(rows),
        eligible_studies=eligible,
        split_counts=dict(counts["split"]),
        language_counts=dict(counts["language"]),
        preprocessing_counts=dict(counts["preprocessing"]),
        output_path=str(output_path),
        metadata_path=str(metadata_path),
    )


def _cohort_metadata(config: ManifestConfig, output_path: Path) -> dict[str, Any]:
    """Sidecar describing how the manifest was produced, and hashing its inputs."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "text_policy": config.text_policy,
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "validation_fraction": config.validation_fraction,
        "test_fraction": 1 - config.train_fraction - config.validation_fraction,
        "source_hashes": {
            "reports_csv": sha256_file(Path(config.reports_csv)),
            "series_csv": sha256_file(Path(config.series_csv)),
            **(
                {"translations_csv": sha256_file(Path(config.translations_csv))}
                if config.translations_csv is not None
                else {}
            ),
        },
        "manifest_sha256": sha256_file(output_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_manifest_rows(path: Path, *, series_root: Path | None = None) -> list[dict[str, Any]]:
    """Every row of a manifest, with slots resolved to this host's series directories.

    For read-only audits, which inspect all rows -- ineligible and test included --
    rather than the verified training records :func:`read_cohort_manifest` returns.
    """
    rows = read_jsonl(Path(path))
    if any(row.get("slots") for row in rows) and series_root is None:
        series_root = paths.host_series_root()
    for row in rows:
        row["slots"] = {plane: resolve_slot(value, series_root)
                        for plane, value in (row.get("slots") or {}).items()}
    return rows


def read_cohort_manifest(
    path: Path, *, series_root: Path | None = None
) -> tuple[list[StudyRecord], dict[str, Any]]:
    """Load the training-eligible studies of a verified cohort, with its metadata.

    Slots resolve under ``series_root``, defaulting to this host's ``train_series``
    tree from the paths configuration. The manifest itself names no host, so the
    same bytes -- and the same checksum -- serve wherever the data is mounted.

    Every invariant a training run depends on is checked here, before any of it
    can affect a result: unique study identifiers, known splits, no patient group
    spanning two splits, complete eligible rows, and manifest bytes matching the
    checksum in the sidecar. A missing checksum is an error rather than a warning --
    legacy cohorts must be migrated by an audited, recorded step, never accepted
    silently, because accepting one would make the run unattributable.
    """
    path = Path(path)
    rows = read_jsonl(path)
    metadata = read_json(path.with_suffix(".meta.json"))
    if metadata.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported cohort manifest version")

    seen_uids: set[str] = set()
    group_splits: dict[str, str] = {}
    for row in rows:
        uid = row.get("study_uid", "")
        if not uid or uid in seen_uids:
            raise ValueError(f"Missing or duplicate manifest study_uid: {uid!r}")
        seen_uids.add(uid)
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError(f"Invalid manifest split: {split!r}")
        group = row.get("patient_group_id", "")
        if group_splits.setdefault(group, split) != split:
            raise ValueError(f"patient group spans multiple splits: {group}")
        if row.get("training_eligible") and (
            not row.get("report") or set(row.get("slots", {})) != set(PLANES)
        ):
            raise ValueError(f"Eligible manifest row is incomplete: {uid}")

    expected_hash = metadata.get("manifest_sha256")
    if not expected_hash:
        raise ValueError("Cohort manifest checksum is required; audit and migrate legacy metadata")
    if sha256_file(path) != expected_hash:
        raise ValueError("Cohort manifest checksum does not match its metadata")

    eligible = [row for row in rows if row["training_eligible"]]
    if eligible and series_root is None:
        series_root = paths.host_series_root()
    # Unchecked, a missing tree only surfaces later as planes masked as undecodable.
    if eligible and not Path(series_root).is_dir():
        raise FileNotFoundError(f"Series root not found; is the data disk mounted? {series_root}")
    records = [
        StudyRecord(
            study_uid=row["study_uid"],
            report=row["report"],
            slots={plane: resolve_slot(value, series_root) for plane, value in row["slots"].items()},
            laterality=row["laterality"],
            patient_group_id=row["patient_group_id"],
            split=row["split"],
            report_language=row["report_language"],
            report_provenance=row["report_provenance"],
        )
        for row in eligible
    ]
    return records, metadata
