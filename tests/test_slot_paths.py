"""Where the DICOM tree lives is a host setting, never part of a study's identity.

See docs/adr/0001-series-paths-are-host-independent.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from contrastive_learning.data import (
    LEGACY_SERIES_ROOT,
    KneeReportDataset,
    StudyRecord,
    resolve_slot,
    slot_identity,
)
from contrastive_learning.manifest import ManifestConfig, build_cohort_manifest, read_cohort_manifest
from test_manifest import _write_csv, _write_dicom_header


PLANES = ("Sagittal", "Coronal", "Axial")


def test_slot_identity_is_the_study_and_series_directory_names():
    assert slot_identity(Path("/anywhere/train_series/1.2.3/4.5.6")) == "1.2.3/4.5.6"
    assert slot_identity("1.2.3/4.5.6") == "1.2.3/4.5.6"


@pytest.mark.parametrize(
    "stored", [f"{LEGACY_SERIES_ROOT}/1.2.3/4.5.6", "1.2.3/4.5.6"], ids=["legacy-absolute", "relative"]
)
def test_stored_slots_resolve_under_this_hosts_series_root(tmp_path, stored):
    assert resolve_slot(stored, tmp_path) == tmp_path / "1.2.3" / "4.5.6"


@pytest.mark.parametrize(
    "stored",
    ["/elsewhere/train_series/1.2.3/4.5.6", f"{LEGACY_SERIES_ROOT}/1.2.3", "1.2.3", "a/b/c", "../4.5.6", ""],
)
def test_unrecognised_slots_are_rejected(tmp_path, stored):
    with pytest.raises(ValueError, match="slot"):
        resolve_slot(stored, tmp_path)


def _build_cohort(tmp_path: Path, series_root: Path) -> Path:
    reports, series_csv, output = tmp_path / "reports.csv", tmp_path / "series.csv", tmp_path / "cohort.jsonl"
    uids = [f"study-{i}" for i in range(4)]
    _write_csv(reports, ["StudyInstanceUID", "Report"],
               [{"StudyInstanceUID": uid, "Report": f"report {uid}"} for uid in uids])
    rows = []
    for uid in uids:
        for plane in PLANES:
            _write_dicom_header(series_root / uid / f"{uid}-{plane}" / "1.dcm", f"patient-{uid}")
            rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": f"{uid}-{plane}",
                         "Fluid_Sensitive": "1", "Anatomical_Plane": plane})
    _write_csv(series_csv, list(rows[0]), rows)
    build_cohort_manifest(ManifestConfig(reports_csv=reports, series_csv=series_csv,
                                         series_root=series_root, output_path=output,
                                         train_fraction=0.5, validation_fraction=0.25))
    return output


def test_manifests_store_host_independent_slots_and_survive_a_data_move(tmp_path):
    old_root, new_root = tmp_path / "old" / "train_series", tmp_path / "new" / "train_series"
    manifest = _build_cohort(tmp_path, old_root)
    assert str(tmp_path) not in manifest.read_text()

    shutil.copytree(old_root, new_root)  # copy2 keeps nanosecond modification times
    shutil.rmtree(old_root)
    records, _ = read_cohort_manifest(manifest, series_root=new_root)

    assert records
    assert all(path.parent.parent == new_root and path.is_dir()
               for record in records for path in record.slots.values())


def _record(root: Path) -> StudyRecord:
    return StudyRecord(study_uid="study", report="report", laterality="L",
                       slots={plane: root / "study" / plane for plane in PLANES})


def test_image_cache_keys_do_not_change_when_the_data_moves(tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    for plane in PLANES:
        _write_dicom_header(old_root / "study" / plane / "1.dcm", "patient")
    shutil.copytree(old_root, new_root)

    def key(root):
        return KneeReportDataset([_record(root)], {"image_size": 224},
                                 image_cache_dir=tmp_path / "cache").cache_paths["study"]

    assert key(old_root) == key(new_root)


def test_window_cache_keys_do_not_change_when_the_data_moves(tmp_path):
    from contrastive_learning.finetune import WindowDataset

    config = {"windows_per_plane": 2, "window_cache_dir": str(tmp_path / "cache"), "image_size": 224,
              "crop_mm": 130.0, "slices_per_plane": 3, "slice_band": [0.2, 0.8]}
    dataset = WindowDataset([], config, training=False)
    assert dataset.cache_path(_record(tmp_path / "old")) == dataset.cache_path(_record(tmp_path / "new"))


def test_an_unmounted_series_root_fails_loudly(tmp_path):
    manifest = _build_cohort(tmp_path, tmp_path / "train_series")
    with pytest.raises(FileNotFoundError, match="mounted"):
        read_cohort_manifest(manifest, series_root=tmp_path / "not-mounted")
