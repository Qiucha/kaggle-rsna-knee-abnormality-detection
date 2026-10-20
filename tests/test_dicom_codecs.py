"""DICOM codec robustness: decoder pins, decode counters, strict mode, series fallback.

The positive JPEG Lossless / JPEG-LS decode tests need the vendored pylibjpeg wheels
(Python 3.12, see kaggle/codecs/) and are skipped where they are not installed.
Failures are made with genuinely truncated pixel data, so no decoder is mocked there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import FileMetaDataset
from pydicom.pixels import pixel_array
from pydicom.uid import ExplicitVRLittleEndian, JPEGLosslessSV1, generate_uid

from contrastive_learning import data, preprocessing, submission_training
from contrastive_learning.data import (
    KneeReportDataset,
    pick_study_slots,
    rank_plane_candidates,
)
from contrastive_learning.preprocessing import (
    PLANES,
    _decode_pixels,
    decode_counts,
    decoding_plugin,
    read_slot,
)
from contrastive_learning.submission import submission_records

ROOT = Path(__file__).resolve().parents[1]
CODECS = ROOT / "kaggle" / "codecs"
FIXTURES = CODECS / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
KWARGS = {"image_size": 16, "crop_mm": 130.0, "slices_per_plane": 3, "slice_band": (0.2, 0.8)}


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv(preprocessing.STRICT_DECODE_ENV, raising=False)
    decoding_plugin.cache_clear()
    decode_counts(reset=True)
    yield
    decoding_plugin.cache_clear()
    decode_counts(reset=True)


def _rescaled(path: Path, array: np.ndarray) -> np.ndarray:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return array.astype(np.float32) * slope + intercept


def _available(uid: str) -> bool:
    from pydicom.pixels import get_decoder

    return get_decoder(uid).is_available


# --------------------------------------------------------------------------- #
# Decoder pins
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "uid,available,expected",
    [
        ("1.2.840.10008.1.2.4.50", ["pillow", "pylibjpeg"], "pillow"),  # lossy JPEG stays on Pillow
        ("1.2.840.10008.1.2.4.51", ["pillow", "pylibjpeg"], "pillow"),
        ("1.2.840.10008.1.2.4.90", ["pillow", "pylibjpeg"], "pillow"),
        ("1.2.840.10008.1.2.4.91", ["pillow", "pylibjpeg"], "pillow"),
        ("1.2.840.10008.1.2.5", ["pydicom", "pylibjpeg"], "pydicom"),  # RLE keeps pydicom's own
        ("1.2.840.10008.1.2.4.57", ["pylibjpeg"], "pylibjpeg"),
        ("1.2.840.10008.1.2.4.70", ["pylibjpeg"], "pylibjpeg"),
        ("1.2.840.10008.1.2.4.80", ["pylibjpeg"], "pylibjpeg"),
        ("1.2.840.10008.1.2.4.81", ["pylibjpeg"], "pylibjpeg"),
        ("1.2.840.10008.1.2.4.70", [], ""),  # pinned plugin missing: pydicom default
        ("1.2.840.10008.1.2.4.50", ["pylibjpeg"], ""),
        ("1.2.840.10008.1.2.1", [], ""),  # uncompressed: never pinned
        ("1.2.840.10008.1.2.4.201", ["pylibjpeg"], ""),  # HTJ2K: no Pillow decoder, default
    ],
)
def test_decoder_pin_per_transfer_syntax(monkeypatch, uid, available, expected):
    decoder = type("Decoder", (), {})()
    decoder.available_plugins = set(available)
    monkeypatch.setattr("pydicom.pixels.get_decoder", lambda _: decoder)
    assert decoding_plugin(uid) == expected


def test_decoder_pin_in_this_environment():
    # Pillow decodes JPEG 2000 here, as on Kaggle; JPEG Lossless is pinned only when
    # pylibjpeg is installed.
    assert decoding_plugin("1.2.840.10008.1.2.4.90") == "pillow"
    expected = "pylibjpeg" if _available(JPEGLosslessSV1) else ""
    assert decoding_plugin(JPEGLosslessSV1) == expected
    assert decoding_plugin(ExplicitVRLittleEndian) == ""


# --------------------------------------------------------------------------- #
# Vendored fixtures through the pipeline decoder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(MANIFEST["lossless"]))
def test_lossless_fixtures_decode_bit_exact(name):
    path = FIXTURES / name
    uid = str(pydicom.dcmread(path, stop_before_pixels=True).file_meta.TransferSyntaxUID)
    if not _available(uid):
        pytest.skip(f"no decoder for {name} here (vendored wheels are cp312 only)")
    reference = np.load(FIXTURES / MANIFEST["lossless"][name])
    decoded = _decode_pixels(path)
    assert decoded is not None and decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, _rescaled(path, reference))


@pytest.mark.parametrize("name", sorted(MANIFEST["lossy"]["files"]))
def test_lossy_fixtures_decode_exactly_as_pillow(name):
    # The pin that keeps lossy pixels unchanged once pylibjpeg is installed: with
    # the wheels, pydicom's default decodes lossy JPEG baseline differently.
    path = FIXTURES / name
    pillow = pixel_array(path, decoding_plugin="pillow")
    decoded = _decode_pixels(path)
    assert decoded is not None
    np.testing.assert_array_equal(decoded, _rescaled(path, pillow))
    import PIL

    if PIL.__version__ == MANIFEST["lossy"]["pillow_version"]:
        import hashlib

        digest = hashlib.sha256(np.ascontiguousarray(pillow).tobytes()).hexdigest()
        assert digest == MANIFEST["lossy"]["files"][name]["pillow_sha256"]


def test_fixtures_are_synthetic():
    # Real competition slices must never be committed as fixtures.
    for path in sorted(FIXTURES.glob("*.dcm")):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        assert ds.PatientID == "SYNTHETIC", path.name
        assert str(ds.SeriesDescription).startswith("SYNTHETIC CODEC FIXTURE"), path.name


def test_lossy_jpeg_fixtures_detect_a_missing_pin():
    # The self-test can only catch an unpinned decoder if pylibjpeg decodes these
    # fixtures differently from Pillow.
    if "pylibjpeg" not in __import__("pydicom.pixels").pixels.get_decoder(
        "1.2.840.10008.1.2.4.50"
    ).available_plugins:
        pytest.skip("pylibjpeg not installed")
    for name in ("lossy_jpeg_baseline_0.dcm", "lossy_jpeg_baseline_1.dcm"):
        path = FIXTURES / name
        assert not np.array_equal(
            pixel_array(path, decoding_plugin="pillow"), pixel_array(path, decoding_plugin="pylibjpeg")
        ), name


def test_jpeg_lossless_counts_as_failed_without_codecs():
    path = FIXTURES / "jpegll_0.dcm"
    decoded = _decode_pixels(path)
    outcome = "decoded" if _available(JPEGLosslessSV1) else "failed"
    assert (decoded is not None) == (outcome == "decoded")
    assert decode_counts() == {"JPEGLosslessSV1": {outcome: 1}}


def test_pinned_plugin_failure_retries_the_default_chain(monkeypatch):
    path = FIXTURES / "lossy_jpeg_baseline_0.dcm"
    expected = _decode_pixels(path)
    decode_counts(reset=True)
    # Make the pinned decode fail by swapping the pin for a plugin JPEG does not
    # have; the retry then runs pydicom's default chain, which is Pillow alone here.
    original = pydicom.dataset.Dataset.pixel_array_options
    calls = []

    def fail_first(self, **kwargs):
        calls.append(kwargs.get("decoding_plugin", ""))
        if kwargs.get("decoding_plugin"):
            return original(self, decoding_plugin="pydicom")
        return original(self, **kwargs)

    monkeypatch.setattr(pydicom.dataset.Dataset, "pixel_array_options", fail_first)
    decoded = _decode_pixels(path)
    assert calls == ["pillow", ""]
    if pixel_array(path).tobytes() == pixel_array(path, decoding_plugin="pillow").tobytes():
        np.testing.assert_array_equal(decoded, expected)
    assert decode_counts()["JPEGBaseline8Bit"] == {"decoded": 1, "pin_retry_decoded": 1}


# --------------------------------------------------------------------------- #
# Synthetic series: partial failure, strict mode, full-plane fallback
# --------------------------------------------------------------------------- #


def write_slice(path: Path, z: float, value: int, *, broken: bool = False, size: int = 24) -> None:
    """An uncompressed 16-bit slice; ``broken`` truncates its pixel data (undecodable)."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = ds.Columns = size
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [5.0, 5.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.InstanceNumber = int(z)
    pixels = (np.arange(size * size, dtype=np.uint16).reshape(size, size) % 97) * 7 + value
    ds.PixelData = pixels.tobytes()[: size] if broken else pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def write_series(directory: Path, values: list[int], broken: set[int] = frozenset()) -> Path:
    for index, value in enumerate(values):
        write_slice(directory / f"{index:03d}.dcm", float(index), value, broken=index in broken)
    return directory


# Five slices select indices [0, 1, 3]; channel 1 (slice 1) is nearest channel 0.
VALUES = [100, 300, 500, 700, 900]


def test_partial_failure_counts_and_substitutes_nearest_slice(tmp_path):
    mixed = write_series(tmp_path / "mixed", VALUES, broken={1})
    # What substitution must produce: slice 1 carrying slice 0's pixels.
    reference = write_series(tmp_path / "reference", [100, 100, 500, 700, 900])
    image = read_slot(mixed, **KWARGS)
    counts = decode_counts()
    assert image is not None
    assert torch.equal(image, read_slot(reference, **KWARGS))
    assert counts["ExplicitVRLittleEndian"]["failed"] == 1
    assert counts["ExplicitVRLittleEndian"]["substituted_slices"] == 1


def test_strict_mode_raises_on_partial_failure(tmp_path, monkeypatch):
    mixed = write_series(tmp_path / "mixed", VALUES, broken={1})
    monkeypatch.setenv(preprocessing.STRICT_DECODE_ENV, "1")
    with pytest.raises(RuntimeError, match="1 of 3 selected slices"):
        read_slot(mixed, **KWARGS)
    monkeypatch.setenv(preprocessing.STRICT_DECODE_ENV, "0")
    assert read_slot(mixed, **KWARGS) is not None


def test_undecodable_plane_returns_none_and_is_counted(tmp_path):
    dead = write_series(tmp_path / "dead", VALUES, broken=set(range(5)))
    assert read_slot(dead, **KWARGS) is None
    assert decode_counts()["ExplicitVRLittleEndian"]["undecodable_planes"] == 1


def make_study(root: Path, uid: str = "study") -> list[dict[str, str]]:
    """A three-plane study whose selected sagittal series is wholly undecodable."""
    rows = []

    def add(series: str, plane: str, fluid: str, values: list[int], broken=frozenset()):
        write_series(root / uid / series, values, broken)
        rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": series,
                     "Anatomical_Plane": plane, "Fluid_Sensitive": fluid})

    add("sag-dead", "Sagittal", "1", VALUES + [950], broken=set(range(6)))  # most slices: selected
    add("sag-t1", "Sagittal", "0", VALUES + [950, 990])  # other weighting: last resort
    add("sag-empty", "Sagittal", "1", [])
    add("sag-good", "Sagittal", "1", VALUES)  # the fallback
    add("cor", "Coronal", "1", VALUES)
    add("ax", "Axial", "1", VALUES)
    return rows


def test_ranking_keeps_first_choice_and_orders_fallbacks(tmp_path):
    rows = make_study(tmp_path)
    ranked = rank_plane_candidates("study", rows, tmp_path)
    picked = pick_study_slots("study", rows, tmp_path)
    assert {plane: ranked[plane][0] for plane in PLANES} == picked
    assert [path.name for path in ranked["Sagittal"]] == ["sag-dead", "sag-good", "sag-t1"]


def test_full_plane_failure_falls_back_to_next_series(tmp_path, capsys):
    rows = make_study(tmp_path)
    (record,) = submission_records(["study"], {"study": rows}, tmp_path)
    assert record.slots["Sagittal"].name == "sag-dead"
    assert [p.name for p in record.fallbacks["Sagittal"]] == ["sag-good", "sag-t1"]
    assert record.fallbacks["Coronal"] == ()

    cache = tmp_path / "cache"
    images = KneeReportDataset([record], KWARGS, image_cache_dir=cache)[0]["images"]
    assert torch.equal(images[0], read_slot(tmp_path / "study" / "sag-good", **KWARGS))
    assert "decode fallback: study study Sagittal: sag-dead -> sag-good (ok)" in capsys.readouterr().err
    assert decode_counts()["ExplicitVRLittleEndian"]["fallback_series"] == 1
    # Fallback pixels are never cached under the selected series' key.
    assert not list(cache.glob("*.pt"))

    # Training records carry no fallbacks and still fail loudly.
    bare = data.StudyRecord("study", "", record.slots, record.laterality)
    with pytest.raises(RuntimeError, match="DICOM decode failed after indexing: study Sagittal"):
        KneeReportDataset([bare], KWARGS)[0]


def test_empty_preferred_series_are_not_replaced_by_other_weightings(tmp_path):
    # pick_study_slots rejects a plane whose fluid series are all empty; the fallback
    # ranking must not turn that into a non-fluid first choice.
    rows = []
    for series, plane, fluid in [("sag", "Sagittal", "1"), ("cor-empty", "Coronal", "1"),
                                 ("cor-t1", "Coronal", "0"), ("ax", "Axial", "1")]:
        if series != "cor-empty":
            write_series(tmp_path / "s" / series, VALUES)
        rows.append({"StudyInstanceUID": "s", "SeriesInstanceUID": series,
                     "Anatomical_Plane": plane, "Fluid_Sensitive": fluid})
    assert pick_study_slots("s", rows, tmp_path) is None
    with pytest.raises(ValueError, match="lacks a usable three-plane input"):
        submission_records(["s"], {"s": rows}, tmp_path)


# --------------------------------------------------------------------------- #
# Bundle contents
# --------------------------------------------------------------------------- #


def test_bundle_ships_codecs_with_hashes(tmp_path):
    hashes = submission_training.copy_codecs(ROOT, tmp_path)
    wheels = [name for name in hashes if name.startswith("codecs/wheels/")]
    assert sorted(Path(name).name.split("-")[0] for name in wheels) == [
        "pylibjpeg", "pylibjpeg_libjpeg", "pylibjpeg_openjpeg", "pylibjpeg_rle",
    ]
    assert "codecs/install_and_selftest.py" in hashes
    assert "codecs/fixtures/manifest.json" in hashes
    shipped = {name for name in hashes if name.startswith("codecs/fixtures/")}
    listed = {*MANIFEST["lossless"], *MANIFEST["lossy"]["files"], *MANIFEST["lossless"].values()}
    assert {f"codecs/fixtures/{name}" for name in listed} <= shipped
    for name, digest in hashes.items():
        assert (tmp_path / name).is_file() and len(digest) == 64


def test_bundle_refuses_incomplete_codecs(tmp_path):
    fake_root = tmp_path / "repo"
    (fake_root / "kaggle" / "codecs" / "wheels").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Incomplete vendored codecs"):
        submission_training.copy_codecs(fake_root, tmp_path / "bundle")
