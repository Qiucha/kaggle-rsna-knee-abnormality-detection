import hashlib
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from contrastive_learning.preprocessing import ordered_slice_paths
from experiment_tools.dicom_series import plane_digest


def write_slice(path: Path, z: float, value: int, *, broken: bool = False, size: int = 8) -> None:
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
    pixels = (np.arange(size * size, dtype=np.uint16).reshape(size, size) % 31) * 3 + value
    ds.PixelData = pixels.tobytes()[:size] if broken else pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def write_series(directory: Path, values: list[int], broken: frozenset[int] = frozenset()) -> Path:
    for index, value in enumerate(values):
        write_slice(directory / f"{index:03d}.dcm", float(index), value, broken=index in broken)
    return directory


def test_plane_digest_hashes_every_slice_in_anatomical_order(tmp_path):
    series = write_series(tmp_path / "series", [10, 20, 30])
    expected = hashlib.sha256()
    for path in ordered_slice_paths(series):
        array = np.ascontiguousarray(pydicom.dcmread(path).pixel_array.astype(np.float32))
        expected.update(str(array.shape).encode())
        expected.update(str(array.dtype).encode())
        expected.update(array.tobytes())
    digest = plane_digest(str(series))
    assert digest["ordered_decoded_pixel_sha256"] == expected.hexdigest()
    assert digest["slices"] == 3
    assert digest["slice_shapes"] == ["(8, 8)"]
    assert digest["dtypes"] == ["float32"]


def test_plane_digest_tells_copies_from_changed_pixels(tmp_path):
    first = plane_digest(str(write_series(tmp_path / "a", [10, 20, 30])))
    copy = plane_digest(str(write_series(tmp_path / "b", [10, 20, 30])))
    changed = plane_digest(str(write_series(tmp_path / "c", [10, 20, 31])))
    assert first["ordered_decoded_pixel_sha256"] == copy["ordered_decoded_pixel_sha256"]
    assert first["ordered_decoded_pixel_sha256"] != changed["ordered_decoded_pixel_sha256"]


def test_plane_digest_marks_an_undecodable_slice_instead_of_failing(tmp_path):
    whole = plane_digest(str(write_series(tmp_path / "whole", [10, 20, 30])))
    broken = plane_digest(str(write_series(tmp_path / "broken", [10, 20, 30], broken=frozenset({1}))))
    assert broken["slices"] == 3
    assert broken["ordered_decoded_pixel_sha256"] != whole["ordered_decoded_pixel_sha256"]
