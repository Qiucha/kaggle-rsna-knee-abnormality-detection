"""Pair MRI studies with reports, and turn a pair into a training batch.

The unit of data in this project is a *study*: three fluid-sensitive series, one
per anatomical plane, plus the radiology report written for them. This module owns
how a study is discovered (from CSVs, when no frozen cohort manifest is used), how
it becomes a tensor, and how a list of them becomes a batch.

The dataset deliberately produces raw preprocessed pixels, not augmented ones.
That separation is what makes the two caches sound: the disk cache is keyed by a
fingerprint of the exact DICOM bytes and preprocessing settings, so a cached
tensor can never correspond to different inputs, and the memory cache holds the
same deterministic tensor. Augmentation is applied later, per batch, in training.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from contrastive_learning.artifacts import sha256_text, write_atomic
from contrastive_learning.preprocessing import (
    PLANES,
    PREPROCESSING_VERSION,
    apply_left_knee,
    count_decode_event,
    count_dicoms,
    read_slot,
    resolve_laterality,
    series_fingerprint,
    series_transfer_syntax,
)
from contrastive_learning.text import select_report_view


CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class StudyRecord:
    """One trainable example: three plane series, its report, and its provenance.

    ``split`` and the report language/provenance fields are populated from a cohort
    manifest and left at their defaults on the legacy CSV path, which has no
    concept of a fixed split.
    """

    study_uid: str
    report: str
    slots: dict[str, Path]
    laterality: str | None
    patient_group_id: str | None = None
    split: str | None = None
    report_language: str = "und"
    report_provenance: str = "raw"
    # Inference only: per plane, the series to try, in order, when the selected one
    # does not decode at all. Training and manifests never set it, so they keep
    # failing loudly; it is not part of any cache key.
    fallbacks: dict[str, tuple[Path, ...]] | None = None


# Manifests frozen before ADR 0001 store slots as absolute paths under this mount.
# Only the <study>/<series> part after it is meaningful; it is never used to read.
LEGACY_SERIES_ROOT = "/mnt/d/train_series"


def slot_identity(series_dir: Path | str) -> str:
    """Name a series by ``<study>/<series>``, the part of its path no host decides.

    Cache keys and mirrors use this rather than the full path, so moving the DICOM
    tree to another disk leaves every cached tensor reachable.
    """
    path = Path(series_dir)
    return f"{path.parent.name}/{path.name}"


def resolve_slot(stored: str, series_root: Path) -> Path:
    """Locate a manifest slot under this host's ``train_series`` tree.

    A slot is either ``<study>/<series>`` or a legacy absolute path under
    :data:`LEGACY_SERIES_ROOT`. Anything else is rejected rather than guessed at:
    a path that silently resolved to the wrong tree would train on the wrong bytes.
    """
    path = Path(stored)
    relative = path.relative_to(LEGACY_SERIES_ROOT) if path.is_relative_to(LEGACY_SERIES_ROOT) else path
    if relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts:
        raise ValueError(f"Unrecognised manifest slot: {stored!r}")
    return Path(series_root) / relative


@dataclass(frozen=True)
class DataSummary:
    """Why studies were kept or dropped while pairing reports with series."""

    # read_reports keeps only rows that have both a UID and nonempty text, so this
    # counts usable reports, not lines in the CSV.
    report_rows: int
    studies_scanned: int
    paired_studies: int
    missing_plane: int
    decode_failed: int


class TensorMemoryCache:
    """A bounded least-recently-used cache for preprocessed study tensors.

    One instance lives per loader worker process, so the effective footprint is
    the capacity multiplied by the worker count. Capacity ``0`` disables caching
    entirely, which is the default because study tensors are large.
    """

    def __init__(self, capacity: int = 0):
        if capacity < 0:
            raise ValueError("cache capacity cannot be negative")
        self.capacity = capacity
        self._values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str) -> torch.Tensor | None:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.capacity == 0:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)


# --------------------------------------------------------------------------- #
# Discovering studies from CSVs
# --------------------------------------------------------------------------- #


def read_csv_rows(path: Path, required_columns: set[str] | None = None) -> list[dict[str, str]]:
    """Read a CSV tolerating the UTF-8 byte order mark that Excel exports add.

    Every CSV this project reads -- reports, the series inventory, translations --
    goes through here, so a cohort manifest and a legacy CSV run can never disagree
    about a file's dialect or about which columns a file is required to carry.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if required_columns and not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required_columns)}")
        return list(reader)


def read_reports(path: Path, report_column: str = "Report") -> dict[str, str]:
    """Read one nonempty report per study, rejecting duplicate study identifiers.

    A duplicate is an error rather than a last-wins overwrite: it would put the
    same anatomy in the candidate pool twice and quietly change retrieval metrics.
    """
    reports: dict[str, str] = {}
    for row in read_csv_rows(path, {"StudyInstanceUID", report_column}):
        uid = row["StudyInstanceUID"].strip()
        report = (row[report_column] or "").strip()
        if uid and report:
            if uid in reports:
                raise ValueError(f"Duplicate StudyInstanceUID in {path}: {uid}")
            reports[uid] = report
    return reports


def read_series_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    """Group the series inventory by study, preserving each study's row order."""
    by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"}
    for row in read_csv_rows(path, required):
        by_study[row["StudyInstanceUID"].strip()].append(row)
    return dict(by_study)


def pick_study_slots(
    study_uid: str,
    series_rows: Sequence[dict[str, str]],
    series_root: Path,
) -> dict[str, Path] | None:
    """Choose one series per plane, or ``None`` when any plane is unusable.

    Preference order matches the baseline notebook exactly: fluid-sensitive series
    if any exist for the plane, then the largest slice count, then the earliest
    manifest row. All-or-nothing selection keeps every study three-plane, so the
    image tower's input shape never varies.
    """
    selected: dict[str, Path] = {}
    for plane in PLANES:
        candidates = [row for row in series_rows if row["Anatomical_Plane"] == plane]
        if not candidates:
            return None
        fluid = [row for row in candidates if str(row["Fluid_Sensitive"]).strip() == "1"]
        best_count, best_path = -1, None
        for row in fluid or candidates:
            path = Path(series_root) / study_uid / row["SeriesInstanceUID"]
            count = count_dicoms(path)
            if count > best_count:  # Strict '>' lets the first row win a tie.
                best_count, best_path = count, path
        if best_path is None or best_count <= 0:
            return None
        selected[plane] = best_path
    return selected


def rank_plane_candidates(
    study_uid: str,
    series_rows: Sequence[dict[str, str]],
    series_root: Path,
) -> dict[str, list[Path]]:
    """Every nonempty series per plane, in the order :func:`pick_study_slots` prefers.

    Preferred-weighting series come first (fluid-sensitive when the plane has any),
    by slice count and then manifest row, exactly as the selection rule ranks them;
    the plane's other series follow as a last resort. Used only to find a fallback
    when the selected series will not decode: the selection itself always comes
    from :func:`pick_study_slots`, so this ranking can never change a first choice.
    """
    ranked: dict[str, list[Path]] = {}
    for plane in PLANES:
        candidates = [row for row in series_rows if row["Anatomical_Plane"] == plane]
        fluid = [row for row in candidates if str(row["Fluid_Sensitive"]).strip() == "1"]
        preferred = fluid or candidates
        ordered: list[Path] = []
        for pool in (preferred, [row for row in candidates if row not in preferred]):
            counted = [
                (count_dicoms(path), row_index, path)
                for row_index, path in enumerate(
                    Path(series_root) / study_uid / row["SeriesInstanceUID"] for row in pool
                )
            ]
            counted.sort(key=lambda item: (-item[0], item[1]))
            ordered += [path for count, _, path in counted if count > 0 and path not in ordered]
        ranked[plane] = ordered
    return ranked


def plane_fallbacks(
    slots: dict[str, Path],
    study_uid: str,
    series_rows: Sequence[dict[str, str]],
    series_root: Path,
) -> dict[str, tuple[Path, ...]]:
    """The fallback series per plane: the ranked candidates minus the selected one."""
    ranked = rank_plane_candidates(study_uid, series_rows, series_root)
    return {
        plane: tuple(path for path in ranked[plane] if path != slots[plane]) for plane in PLANES
    }


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
    """Pair reports with three-plane series, reporting why studies were dropped.

    With ``validate_pixels`` every plane is decoded now, so a study that would fail
    mid-epoch is excluded up front; the cost is a full decode pass over the cohort.
    """
    reports = read_reports(reports_csv, report_column)
    series_by_study = read_series_manifest(series_csv)
    records: list[StudyRecord] = []
    missing_plane = decode_failed = studies_scanned = 0
    for uid, report in reports.items():
        studies_scanned += 1
        slots = pick_study_slots(uid, series_by_study.get(uid, []), series_root)
        if slots is None:
            missing_plane += 1
            continue
        if validate_pixels and any(
            read_slot(slots[plane], **image_kwargs) is None for plane in PLANES
        ):
            decode_failed += 1
            continue
        records.append(StudyRecord(uid, report, slots, resolve_laterality(slots)))
        if max_studies is not None and len(records) >= max_studies:
            break
    summary = DataSummary(
        report_rows=len(reports),
        studies_scanned=studies_scanned,
        paired_studies=len(records),
        missing_plane=missing_plane,
        decode_failed=decode_failed,
    )
    return records, summary


# --------------------------------------------------------------------------- #
# Splitting (legacy CSV path only; a cohort manifest carries its own splits)
# --------------------------------------------------------------------------- #


def split_records(
    records: Sequence[StudyRecord], val_fraction: float, seed: int
) -> tuple[list[StudyRecord], list[StudyRecord]]:
    """Seeded holdout split, always leaving at least one study on each side."""
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
    """Seeded K-fold split; the same seed yields complementary, reusable folds."""
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
    return [record for record in shuffled if record.study_uid not in val_uids], val_records


# --------------------------------------------------------------------------- #
# Dataset and batching
# --------------------------------------------------------------------------- #


class KneeReportDataset(Dataset):
    """Serve one study as ``{images: [3 planes, 3 slices, H, W], report, study_uid}``.

    Images are deterministic: identical settings and DICOM bytes always yield the
    same tensor, which is what lets the optional disk and memory caches be safe.
    """

    def __init__(
        self,
        records: Iterable[StudyRecord],
        image_kwargs: dict[str, Any],
        *,
        image_cache_dir: Path | None = None,
        memory_cache_size: int = 0,
    ):
        self.records = list(records)
        self.image_kwargs = dict(image_kwargs)
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir is not None else None
        if self.image_cache_dir is not None:
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache = TensorMemoryCache(memory_cache_size)
        # Cache paths hash series fingerprints, which stat every DICOM file. Do it
        # once here rather than per __getitem__ call.
        self.cache_paths = {record.study_uid: self._cache_path(record) for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, record: StudyRecord) -> Path | None:
        """Name a study's cache file after everything that could change its pixels.

        Because the digest covers preprocessing version and settings, series
        identities, laterality, and per-file size and modification times, a stale
        entry is unreachable rather than wrong: changed inputs simply produce a new
        name. Where the series tree is mounted is deliberately not covered.
        """
        if self.image_cache_dir is None:
            return None
        fingerprint = json.dumps(
            {
                "version": CACHE_FORMAT_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "image_kwargs": self.image_kwargs,
                "laterality": record.laterality,
                "slots": {name: slot_identity(path) for name, path in sorted(record.slots.items())},
                "series_fingerprints": {
                    name: series_fingerprint(path) for name, path in sorted(record.slots.items())
                },
            },
            sort_keys=True,
        )
        return self.image_cache_dir / f"{record.study_uid}-{sha256_text(fingerprint)[:12]}.pt"

    def _load_images(self, record: StudyRecord) -> torch.Tensor:
        """Read a study's pixels from the disk cache, or preprocess and cache them."""
        cache_path = self.cache_paths[record.study_uid]
        if cache_path is not None and cache_path.exists():
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached
        images = []
        used_fallback = False
        for plane in PLANES:
            image = read_slot(record.slots[plane], **self.image_kwargs)
            failed = record.slots[plane]
            # Inference records carry ranked alternatives: a plane that will not
            # decode at all degrades to the next series instead of ending the run.
            # That series may be a contrast the model rarely saw in this slot, so
            # every use is logged and counted.
            for alternative in (record.fallbacks or {}).get(plane, ()):
                if image is not None:
                    break
                count_decode_event(series_transfer_syntax(failed), "fallback_series")
                image = read_slot(alternative, **self.image_kwargs)
                print(
                    f"decode fallback: study {record.study_uid} {plane}: {failed.name} -> "
                    f"{alternative.name} ({'ok' if image is not None else 'failed'})",
                    file=sys.stderr,
                    flush=True,
                )
                failed, used_fallback = alternative, True
            if image is None:
                # Indexing already proved every plane decodes, so reaching here
                # means the underlying files changed during the run, or (inference)
                # no candidate series of this plane decodes.
                raise RuntimeError(f"DICOM decode failed after indexing: {record.study_uid} {plane}")
            images.append(apply_left_knee(image, plane, record.laterality))
        stacked = torch.stack(images)
        # The cache key names the selected series, so pixels from a fallback series
        # must not be stored under it.
        if cache_path is not None and not used_fallback:
            self._write_cache(cache_path, stacked)
        return stacked

    def _read_cache(self, cache_path: Path) -> torch.Tensor | None:
        """Return a cached tensor only if it is loadable, correctly shaped, finite.

        A truncated or partially written file is treated as a miss, so a cache is
        never a source of silent corruption -- worst case the study is re-read.
        """
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError):
            return None
        size = self.image_kwargs["image_size"]
        expected = (len(PLANES), self.image_kwargs["slices_per_plane"], size, size)
        if tuple(cached.shape) != expected or not torch.isfinite(cached).all():
            return None
        return cached

    @staticmethod
    def _write_cache(cache_path: Path, images: torch.Tensor) -> None:
        """Publish a cache entry atomically; concurrent workers may race harmlessly."""
        write_atomic(cache_path, lambda temporary: torch.save(images, temporary))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        images = self.image_cache.get(record.study_uid)
        if images is None:
            images = self._load_images(record)
            self.image_cache.put(record.study_uid, images)
        return {"images": images, "report": record.report, "study_uid": record.study_uid}


def collate_image_reports(
    batch: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    max_text_length: int,
    text_prefix: str = "",
    report_view: str = "full",
) -> dict[str, Any]:
    """Stack studies and tokenize the reports to the batch's longest sequence.

    The report view is selected here rather than at indexing time so one cohort can
    be trained under several text framings without rebuilding it.
    """
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
    """Bind text settings into a picklable collate function for loader workers."""
    return partial(
        collate_image_reports,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        text_prefix=text_prefix,
        report_view=report_view,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Transfer the model inputs of a batch, leaving study identifiers on the host."""
    return (
        batch["images"].to(device, non_blocking=True),
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
    )
