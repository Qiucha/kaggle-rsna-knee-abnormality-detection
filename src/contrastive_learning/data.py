from __future__ import annotations

import csv
import hashlib
import json
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from contrastive_learning.cache import TensorMemoryCache
from contrastive_learning.preprocessing import (
    PLANES,
    PREPROCESSING_VERSION,
    apply_left_knee,
    count_dicoms,
    read_slot,
    resolve_laterality,
    series_fingerprint,
)
from contrastive_learning.text import select_report_view


@dataclass(frozen=True)
class StudyRecord:
    study_uid: str
    report: str
    slots: dict[str, Path]
    laterality: str | None
    patient_group_id: str | None = None
    split: str | None = None
    report_language: str = "und"
    report_provenance: str = "raw"


@dataclass(frozen=True)
class DataSummary:
    report_rows: int
    reports_with_text: int
    studies_scanned: int
    paired_studies: int
    missing_plane: int
    decode_failed: int


def read_reports(path: Path, report_column: str = "Report") -> dict[str, str]:
    """Read one English report per study; language is trusted as configured by the caller."""
    reports: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"StudyInstanceUID", report_column}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            uid = row["StudyInstanceUID"].strip()
            report = (row[report_column] or "").strip()
            if uid and report:
                if uid in reports:
                    raise ValueError(f"Duplicate StudyInstanceUID in {path}: {uid}")
                reports[uid] = report
    return reports


def read_series_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            by_study[row["StudyInstanceUID"].strip()].append(row)
    return dict(by_study)


def pick_study_slots(
    study_uid: str,
    series_rows: Sequence[dict[str, str]],
    series_root: Path,
) -> dict[str, Path] | None:
    """Choose the largest fluid-sensitive series in every baseline plane."""
    selected: dict[str, Path] = {}
    for plane in PLANES:
        candidates = [row for row in series_rows if row["Anatomical_Plane"] == plane]
        if not candidates:
            return None
        fluid = [row for row in candidates if str(row["Fluid_Sensitive"]).strip() == "1"]
        pool = fluid or candidates
        best_count, best_path = -1, None
        for row in pool:
            path = Path(series_root) / study_uid / row["SeriesInstanceUID"]
            count = count_dicoms(path)
            if count > best_count:  # First manifest row wins a slice-count tie, as in the notebook.
                best_count, best_path = count, path
        if best_path is None or best_count <= 0:
            return None
        selected[plane] = best_path
    return selected


def _all_slots_decode(slots: dict[str, Path], image_kwargs: dict[str, Any]) -> bool:
    return all(read_slot(slots[plane], **image_kwargs) is not None for plane in PLANES)


def build_study_records(
    reports_csv: Path,
    series_csv: Path,
    series_root: Path,
    *,
    report_column: str = "Report",
    validate_pixels: bool,
    image_kwargs: dict[str, Any],
    max_studies: int | None = None,
) -> tuple[list[StudyRecord], DataSummary]:
    reports = read_reports(reports_csv, report_column)
    manifest = read_series_manifest(series_csv)
    records: list[StudyRecord] = []
    missing_plane = decode_failed = 0
    studies_scanned = 0
    for uid, report in reports.items():
        studies_scanned += 1
        slots = pick_study_slots(uid, manifest.get(uid, []), series_root)
        if slots is None:
            missing_plane += 1
            continue
        if validate_pixels and not _all_slots_decode(slots, image_kwargs):
            decode_failed += 1
            continue
        records.append(StudyRecord(uid, report, slots, resolve_laterality(slots)))
        if max_studies is not None and len(records) >= max_studies:
            break
    summary = DataSummary(
        report_rows=len(reports),
        reports_with_text=len(reports),
        studies_scanned=studies_scanned,
        paired_studies=len(records),
        missing_plane=missing_plane,
        decode_failed=decode_failed,
    )
    return records, summary


def split_records(
    records: Sequence[StudyRecord], val_fraction: float, seed: int
) -> tuple[list[StudyRecord], list[StudyRecord]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if len(records) < 3:
        raise ValueError("At least three paired studies are required")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = min(max(1, round(len(shuffled) * val_fraction)), len(shuffled) - 2)
    return shuffled[val_count:], shuffled[:val_count]


def split_records_fold(
    records: Sequence[StudyRecord], fold_count: int, fold_index: int, seed: int
) -> tuple[list[StudyRecord], list[StudyRecord]]:
    """Return one deterministic, mutually exclusive study-level validation fold."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if not 0 <= fold_index < fold_count:
        raise ValueError("fold_index must be in [0, fold_count)")
    if len(records) < fold_count + 1:
        raise ValueError("The dataset must contain more studies than folds")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_records = shuffled[fold_index::fold_count]
    val_uids = {record.study_uid for record in val_records}
    train_records = [record for record in shuffled if record.study_uid not in val_uids]
    return train_records, val_records


class KneeReportDataset(Dataset):
    """One three-plane MRI reading paired with one study report."""

    def __init__(
        self,
        records: Iterable[StudyRecord],
        image_kwargs: dict[str, Any],
        *,
        cache_images: bool = False,
        image_cache_dir: Path | None = None,
        memory_cache_size: int | None = None,
    ):
        self.records = list(records)
        self.image_kwargs = dict(image_kwargs)
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir is not None else None
        if self.image_cache_dir is not None:
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        capacity = (64 if cache_images else 0) if memory_cache_size is None else memory_cache_size
        self.image_cache = TensorMemoryCache(capacity)
        self.cache_paths = {record.study_uid: self._cache_path(record) for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def _load_images(self, record: StudyRecord) -> torch.Tensor:
        cache_path = self.cache_paths[record.study_uid]
        if cache_path is not None and cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                expected = (len(PLANES), 3, self.image_kwargs["image_size"], self.image_kwargs["image_size"])
                if tuple(cached.shape) == expected and torch.isfinite(cached).all():
                    return cached
            except (OSError, RuntimeError, TypeError):
                pass
        images = []
        for plane in PLANES:
            image = read_slot(record.slots[plane], **self.image_kwargs)
            if image is None:
                raise RuntimeError(f"DICOM decode failed after indexing: {record.study_uid} {plane}")
            images.append(apply_left_knee(image, plane, record.laterality))
        stacked = torch.stack(images)
        if cache_path is not None:
            temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                torch.save(stacked, temporary)
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return stacked

    def _cache_path(self, record: StudyRecord) -> Path | None:
        if self.image_cache_dir is None:
            return None
        fingerprint = json.dumps(
            {
                "version": 1,
                "preprocessing_version": PREPROCESSING_VERSION,
                "image_kwargs": self.image_kwargs,
                "laterality": record.laterality,
                "slots": {name: str(path) for name, path in sorted(record.slots.items())},
                "series_fingerprints": {
                    name: series_fingerprint(path) for name, path in sorted(record.slots.items())
                },
            },
            sort_keys=True,
        ).encode()
        suffix = hashlib.sha256(fingerprint).hexdigest()[:12]
        return self.image_cache_dir / f"{record.study_uid}-{suffix}.pt"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        images = self.image_cache.get(record.study_uid)
        if images is None:
            images = self._load_images(record)
            self.image_cache.put(record.study_uid, images)
        return {
            "images": images,
            "report": record.report,
            "study_uid": record.study_uid,
        }


def collate_image_reports(
    batch: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    max_text_length: int,
    text_prefix: str = "",
    report_view: str = "full",
) -> dict[str, Any]:
    tokens = tokenizer(
        [text_prefix + select_report_view(item["report"], report_view) for item in batch],
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "study_uid": [item["study_uid"] for item in batch],
    }


def make_collator(
    tokenizer: Any, max_text_length: int, text_prefix: str = "", report_view: str = "full"
):
    return partial(
        collate_image_reports,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        text_prefix=text_prefix,
        report_view=report_view,
    )
