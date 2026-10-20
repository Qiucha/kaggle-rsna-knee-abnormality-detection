from __future__ import annotations

import csv
import json
from pathlib import Path

import pydicom
import pytest

from contrastive_learning.manifest import ManifestConfig, build_cohort_manifest, read_cohort_manifest


def _write_dicom_header(path: Path, patient_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = pydicom.Dataset()
    dataset.PatientID = patient_id
    dataset.Laterality = "L"
    pydicom.dcmwrite(path, dataset, implicit_vr=True, little_endian=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_keeps_patient_groups_together_and_preserves_provenance(tmp_path: Path) -> None:
    reports_path = tmp_path / "reports.csv"
    series_path = tmp_path / "series.csv"
    series_root = tmp_path / "series"
    output_path = tmp_path / "cohort.jsonl"
    studies = [("study-1", "patient-a"), ("study-2", "patient-a"), ("study-3", "patient-b"),
               ("study-4", "patient-c"), ("study-5", "patient-d")]
    _write_csv(
        reports_path,
        ["StudyInstanceUID", "Report", "ACL"],
        [{"StudyInstanceUID": uid, "Report": f"raw report {uid}", "ACL": "1"} for uid, _ in studies],
    )
    series_rows = []
    for uid, patient_id in studies:
        for plane in ("Sagittal", "Coronal", "Axial"):
            series_uid = f"{uid}-{plane.lower()}"
            _write_dicom_header(series_root / uid / series_uid / "1.dcm", patient_id)
            series_rows.append(
                {
                    "StudyInstanceUID": uid,
                    "SeriesInstanceUID": series_uid,
                    "Fluid_Sensitive": "1",
                    "Anatomical_Plane": plane,
                }
            )
    _write_csv(
        series_path,
        ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"],
        series_rows,
    )

    summary = build_cohort_manifest(
        ManifestConfig(
            reports_csv=reports_path,
            series_csv=series_path,
            series_root=series_root,
            output_path=output_path,
            text_policy="multilingual_raw",
            train_fraction=0.5,
            validation_fraction=0.25,
            seed=7,
        )
    )
    records, metadata = read_cohort_manifest(output_path, series_root=series_root)

    by_uid = {record.study_uid: record for record in records}
    assert summary.total_studies == 5
    assert summary.eligible_studies == 5
    assert by_uid["study-1"].patient_group_id == by_uid["study-2"].patient_group_id
    assert by_uid["study-1"].split == by_uid["study-2"].split
    assert by_uid["study-1"].report == "raw report study-1"
    assert by_uid["study-1"].report_provenance == "raw_multilingual"
    assert metadata["text_policy"] == "multilingual_raw"
    assert metadata["source_hashes"]["reports_csv"]
    assert json.loads(output_path.read_text().splitlines()[0])["patient_group_id"] != "patient-a"

    raw_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    first_group = raw_rows[0]["patient_group_id"]
    same_group = next(row for row in raw_rows[1:] if row["patient_group_id"] == first_group)
    same_group["split"] = "test" if raw_rows[0]["split"] != "test" else "train"
    output_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="patient group spans multiple splits"):
        read_cohort_manifest(output_path, series_root=series_root)


def test_english_manifest_uses_audited_translations_and_excludes_missing_text(tmp_path: Path) -> None:
    reports_path = tmp_path / "reports.csv"
    translations_path = tmp_path / "translations.csv"
    series_path = tmp_path / "series.csv"
    series_root = tmp_path / "series"
    output_path = tmp_path / "cohort.jsonl"
    studies = ["english-study", "spanish-study", "missing-study"]
    _write_csv(
        reports_path,
        ["StudyInstanceUID", "Report"],
        [
            {"StudyInstanceUID": "english-study", "Report": "normal knee"},
            {"StudyInstanceUID": "spanish-study", "Report": "rodilla normal"},
            {"StudyInstanceUID": "missing-study", "Report": "befund"},
        ],
    )
    _write_csv(
        translations_path,
        ["StudyInstanceUID", "Original Language", "Translated Report"],
        [
            {"StudyInstanceUID": "english-study", "Original Language": "en", "Translated Report": "normal knee"},
            {"StudyInstanceUID": "spanish-study", "Original Language": "es", "Translated Report": "normal knee"},
            {"StudyInstanceUID": "missing-study", "Original Language": "de", "Translated Report": ""},
        ],
    )
    series_rows = []
    for uid in studies:
        for plane in ("Sagittal", "Coronal", "Axial"):
            series_uid = f"{uid}-{plane.lower()}"
            _write_dicom_header(series_root / uid / series_uid / "1.dcm", uid)
            series_rows.append(
                {
                    "StudyInstanceUID": uid,
                    "SeriesInstanceUID": series_uid,
                    "Fluid_Sensitive": "1",
                    "Anatomical_Plane": plane,
                }
            )
    _write_csv(
        series_path,
        ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"],
        series_rows,
    )

    summary = build_cohort_manifest(
        ManifestConfig(
            reports_csv=reports_path,
            translations_csv=translations_path,
            series_csv=series_path,
            series_root=series_root,
            output_path=output_path,
            text_policy="translated_english",
            train_fraction=0.5,
            validation_fraction=0.25,
        )
    )
    records, _ = read_cohort_manifest(output_path, series_root=series_root)
    by_uid = {record.study_uid: record for record in records}

    assert summary.eligible_studies == 2
    assert by_uid["english-study"].report == "normal knee"
    assert by_uid["english-study"].report_provenance == "source_english"
    assert by_uid["spanish-study"].report == "normal knee"
    assert by_uid["spanish-study"].report_provenance == "translated_english"
    assert "missing-study" not in by_uid


@pytest.mark.parametrize('checksum', [None, '', '0' * 64])
def test_manifest_requires_matching_checksum(tmp_path, checksum):
    path = tmp_path / 'cohort.jsonl'
    path.write_text('')
    metadata = {'manifest_version': 1}
    if checksum is not None:
        metadata['manifest_sha256'] = checksum
    path.with_suffix('.meta.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='checksum'):
        read_cohort_manifest(path)
