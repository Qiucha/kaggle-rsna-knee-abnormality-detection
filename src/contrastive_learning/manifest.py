from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from contrastive_learning.data import StudyRecord, pick_study_slots, read_series_manifest
from contrastive_learning.preprocessing import (
    PLANES,
    PREPROCESSING_VERSION,
    middle_dicom,
    read_slot,
    resolve_laterality,
    series_fingerprint,
)


MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ManifestConfig:
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
    validate_pixels: bool = False
    pixel_workers: int = 4
    image_size: int = 224
    crop_mm: float = 130.0
    slices_per_plane: int = 3
    slice_band_low: float = 0.20
    slice_band_high: float = 0.80
    overwrite: bool = False


@dataclass(frozen=True)
class ManifestSummary:
    total_studies: int
    eligible_studies: int
    excluded_studies: int
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    preprocessing_counts: dict[str, int]
    output_path: str
    metadata_path: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pseudonym(value: str) -> str:
    digest = hashlib.sha256(f"novo-contrastive-patient-v1\0{value}".encode()).hexdigest()
    return f"patient-{digest[:20]}"


def _patient_id(slots: dict[str, Path], fallback: str) -> str:
    try:
        import pydicom

        path = middle_dicom(slots[PLANES[0]])
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _assign_splits(
    rows: list[dict[str, Any]], train_fraction: float, validation_fraction: float, seed: int
) -> None:
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
    validation_target = round(total * validation_fraction)
    assigned = 0
    for group in groups:
        if assigned < train_target:
            split = "train"
        elif assigned < train_target + validation_target:
            split = "validation"
        else:
            split = "test"
        for row in group:
            row["split"] = split
        assigned += len(group)


def _validate_row_pixels(row: dict[str, Any], config: ManifestConfig) -> tuple[str, dict[str, Any]]:
    if not row["slots"]:
        return "missing_plane", {}
    kwargs = {
        "image_size": config.image_size,
        "crop_mm": config.crop_mm,
        "slices_per_plane": config.slices_per_plane,
        "slice_band": (config.slice_band_low, config.slice_band_high),
    }
    tensors = [read_slot(Path(row["slots"][plane]), **kwargs) for plane in PLANES]
    if any(tensor is None for tensor in tensors):
        return "decode_failed", {}
    return "passed", {
        "shape": [len(PLANES), config.slices_per_plane, config.image_size, config.image_size],
        "finite": all(bool(tensor is not None and tensor.isfinite().all()) for tensor in tensors),
        "min": min(float(tensor.min()) for tensor in tensors if tensor is not None),
        "max": max(float(tensor.max()) for tensor in tensors if tensor is not None),
    }


def build_cohort_manifest(config: ManifestConfig) -> ManifestSummary:
    output_path = Path(config.output_path)
    metadata_path = output_path.with_suffix(".meta.json")
    if not config.overwrite and (output_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Manifest output already exists: {output_path}")
    if config.text_policy not in {"multilingual_raw", "translated_english"}:
        raise ValueError("Supported text policies: multilingual_raw, translated_english")

    report_rows = _read_csv_rows(Path(config.reports_csv))
    if not report_rows:
        raise ValueError("Reports CSV is empty")
    required = {"StudyInstanceUID", config.report_column}
    if not required.issubset(report_rows[0]):
        raise ValueError(f"Reports CSV must contain {sorted(required)}")
    manifest = read_series_manifest(Path(config.series_csv))
    translations: dict[str, dict[str, str]] = {}
    if config.translations_csv is not None:
        translation_rows = _read_csv_rows(Path(config.translations_csv))
        translation_required = {
            "StudyInstanceUID",
            config.language_column,
            config.translated_report_column,
        }
        if translation_rows and not translation_required.issubset(translation_rows[0]):
            raise ValueError(f"Translations CSV must contain {sorted(translation_required)}")
        translations = {
            row["StudyInstanceUID"].strip(): row
            for row in translation_rows
            if row.get("StudyInstanceUID", "").strip()
        }
    label_columns = [name for name in report_rows[0] if name not in required]
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in report_rows:
        study_uid = (source.get("StudyInstanceUID") or "").strip()
        report = (source.get(config.report_column) or "").strip()
        if not study_uid or study_uid in seen:
            raise ValueError(f"Missing or duplicate StudyInstanceUID: {study_uid!r}")
        seen.add(study_uid)
        slots = pick_study_slots(study_uid, manifest.get(study_uid, []), Path(config.series_root))
        preprocessing_status = "not_validated" if slots is not None else "missing_plane"
        patient_group_id = _pseudonym(_patient_id(slots, study_uid) if slots else study_uid)
        translation = translations.get(study_uid, {})
        source_language = (translation.get(config.language_column) or "und").strip().lower()
        translated_report = (translation.get(config.translated_report_column) or "").strip()
        if config.text_policy == "multilingual_raw":
            training_report = report
            report_language = source_language
            report_provenance = "raw_multilingual"
        elif source_language == "en":
            training_report = report
            report_language = "en"
            report_provenance = "source_english"
        elif source_language != "und" and translated_report:
            training_report = translated_report
            report_language = "en"
            report_provenance = "translated_english"
        else:
            training_report = ""
            report_language = "und"
            report_provenance = "missing_translation"
        output_rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "study_uid": study_uid,
                "patient_group_id": patient_group_id,
                "split": "excluded",
                "training_eligible": bool(training_report and slots),
                "report": training_report,
                "source_report": report,
                "source_language": source_language,
                "report_language": report_language,
                "report_provenance": report_provenance,
                "report_sha256": hashlib.sha256(training_report.encode()).hexdigest(),
                "source_report_sha256": hashlib.sha256(report.encode()).hexdigest(),
                "slots": {plane: str(path) for plane, path in (slots or {}).items()},
                "series_fingerprints": {
                    plane: series_fingerprint(path) for plane, path in (slots or {}).items()
                },
                "laterality": resolve_laterality(slots) if slots else None,
                "preprocessing_status": preprocessing_status,
                "preprocessing_qc": {},
                "labels": {name: source.get(name, "") for name in label_columns},
            }
        )
    if config.validate_pixels:
        if config.pixel_workers < 1:
            raise ValueError("pixel_workers must be at least one")
        with ThreadPoolExecutor(max_workers=config.pixel_workers) as executor:
            results = executor.map(lambda row: _validate_row_pixels(row, config), output_rows)
            for row, (status, qc) in zip(output_rows, results, strict=True):
                row["preprocessing_status"] = status
                row["preprocessing_qc"] = qc
                row["training_eligible"] = bool(row["training_eligible"] and status == "passed")
    _assign_splits(output_rows, config.train_fraction, config.validation_fraction, config.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    metadata = {
        "manifest_version": MANIFEST_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "text_policy": config.text_policy,
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "validation_fraction": config.validation_fraction,
        "test_fraction": 1 - config.train_fraction - config.validation_fraction,
        "source_hashes": {
            "reports_csv": _sha256_file(Path(config.reports_csv)),
            "series_csv": _sha256_file(Path(config.series_csv)),
            **(
                {"translations_csv": _sha256_file(Path(config.translations_csv))}
                if config.translations_csv is not None
                else {}
            ),
        },
        "manifest_sha256": _sha256_file(output_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_counts: dict[str, int] = defaultdict(int)
    language_counts: dict[str, int] = defaultdict(int)
    preprocessing_counts: dict[str, int] = defaultdict(int)
    for row in output_rows:
        split_counts[row["split"]] += 1
        language_counts[row["report_language"]] += 1
        preprocessing_counts[row["preprocessing_status"]] += 1
    eligible = sum(row["training_eligible"] for row in output_rows)
    return ManifestSummary(
        total_studies=len(output_rows),
        eligible_studies=eligible,
        excluded_studies=len(output_rows) - eligible,
        split_counts=dict(split_counts),
        language_counts=dict(language_counts),
        preprocessing_counts=dict(preprocessing_counts),
        output_path=str(output_path),
        metadata_path=str(metadata_path),
    )


def read_cohort_manifest(path: Path) -> tuple[list[StudyRecord], dict[str, Any]]:
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
        if split not in {"train", "validation", "test", "excluded"}:
            raise ValueError(f"Invalid manifest split: {split!r}")
        group = row.get("patient_group_id", "")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"patient group spans multiple splits: {group}")
        group_splits[group] = split
        if row.get("training_eligible") and (
            not row.get("report") or set(row.get("slots", {})) != set(PLANES)
        ):
            raise ValueError(f"Eligible manifest row is incomplete: {uid}")
    expected_hash = metadata.get("manifest_sha256")
    if not expected_hash:
        raise ValueError("Cohort manifest checksum is required; audit and migrate legacy metadata")
    if _sha256_file(path) != expected_hash:
        raise ValueError("Cohort manifest checksum does not match its metadata")
    records = [
        StudyRecord(
            study_uid=row["study_uid"],
            report=row["report"],
            slots={plane: Path(value) for plane, value in row["slots"].items()},
            laterality=row["laterality"],
            patient_group_id=row["patient_group_id"],
            split=row["split"],
            report_language=row["report_language"],
            report_provenance=row["report_provenance"],
        )
        for row in rows
        if row["training_eligible"]
    ]
    return records, metadata
