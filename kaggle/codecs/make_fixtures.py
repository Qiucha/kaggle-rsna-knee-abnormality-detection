"""Regenerate the synthetic codec self-test fixtures in ``fixtures/``.

Every fixture is a synthetic phantom (ellipses, a gradient and seeded noise) in a
freshly built DICOM dataset with clearly synthetic headers: PatientID ``SYNTHETIC``,
and UIDs derived deterministically from the fixture name. No real image or header
is involved.

Run in a throwaway CPython 3.12 venv that has the encoders but NOT pylibjpeg, so the
recorded Pillow hashes are what Kaggle's image decodes before the codec wheels:

    python -m venv /tmp/fixgen && /tmp/fixgen/bin/pip install \
        numpy==2.0.2 pydicom==3.0.2 pillow==11.3.0 python-gdcm==3.2.6
    /tmp/fixgen/bin/python kaggle/codecs/make_fixtures.py

Outputs (all rewritten):
* ``ref_0.npy`` (160x160 uint16, 12-bit) and ``ref_1.npy`` (160x160 int16, signed);
* ``{jpegll,j2kll,jpegls}_{0,1}.dcm``: JPEG Lossless SV1, JPEG 2000 lossless and
  JPEG-LS lossless encodings of those refs, via GDCM;
* ``lossy_jpeg_baseline_{0,1}.dcm`` (8-bit, quality 75) and ``lossy_j2k_{0,1}.dcm``
  (16-bit irreversible), via Pillow, from 256x256 textured phantoms;
* ``manifest.json`` with the lossless refs and the Pillow-decoded lossy hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.pixels import pixel_array
from pydicom.uid import (
    JPEG2000,
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    MRImageStorage,
    generate_uid,
)

OUT = Path(__file__).resolve().parent / "fixtures"
SEED = 20260927


def phantom(size: int, low: float, high: float, rng: np.random.Generator, texture: float) -> np.ndarray:
    """A smooth knee-sized phantom: nested ellipses on a gradient, plus noise."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1) * 2 - 1
    image = 0.15 + 0.2 * (x + 1) / 2  # left-to-right gradient
    for cx, cy, rx, ry, level in [(0.0, 0.0, 0.8, 0.9, 0.35), (-0.25, 0.1, 0.3, 0.45, 0.25),
                                  (0.3, -0.2, 0.25, 0.2, -0.2), (0.1, 0.45, 0.15, 0.1, 0.3)]:
        image += level * ((((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) <= 1)
    image += texture * rng.standard_normal((size, size))
    # Fine periodic texture gives the lossy codecs high-frequency content to round.
    image += 0.5 * texture * np.sin(x * 37) * np.cos(y * 23)
    image = (image - image.min()) / (image.max() - image.min())
    return low + image * (high - low)


def dataset(name: str, pixels: np.ndarray, bits_stored: int, signed: bool, rescale: bool) -> Dataset:
    """A minimal, obviously synthetic MR slice holding ``pixels`` uncompressed."""
    uid = lambda part: generate_uid(entropy_srcs=["knee-codec-fixture", name, part])
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = uid("sop")
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = uid("study"), uid("series")
    ds.PatientID, ds.PatientName = "SYNTHETIC", "SYNTHETIC^PHANTOM"
    ds.Modality, ds.SeriesDescription = "MR", f"SYNTHETIC CODEC FIXTURE {name}"
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, bits_stored, bits_stored - 1
    ds.PixelRepresentation = int(signed)
    ds.PixelSpacing = [0.8, 0.8]
    if rescale:
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    ds.PixelData = pixels.astype(np.int16 if signed else np.uint16).tobytes()
    return ds


def gdcm_encode(ds: Dataset, syntax: str, path: Path) -> None:
    import gdcm

    target = {
        "jpegll": gdcm.TransferSyntax.JPEGLosslessProcess14_1,
        "j2kll": gdcm.TransferSyntax.JPEG2000Lossless,
        "jpegls": gdcm.TransferSyntax.JPEGLSLossless,
    }[syntax]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "raw.dcm"
        ds.save_as(source, enforce_file_format=True)
        reader = gdcm.ImageReader()
        reader.SetFileName(str(source))
        if not reader.Read():
            raise RuntimeError(f"GDCM could not read {source}")
        change = gdcm.ImageChangeTransferSyntax()
        change.SetTransferSyntax(gdcm.TransferSyntax(target))
        change.SetInput(reader.GetImage())
        if not change.Change():
            raise RuntimeError(f"GDCM could not encode {path.name}")
        writer = gdcm.ImageWriter()
        writer.SetFileName(str(path))
        writer.SetFile(reader.GetFile())
        writer.SetImage(change.GetOutput())
        if not writer.Write():
            raise RuntimeError(f"GDCM could not write {path}")
    # GDCM stamps the current date and time; pin them so reruns are byte-identical.
    encoded = pydicom.dcmread(path)
    encoded.StudyDate, encoded.StudyTime = "20000101", "000000"
    encoded.save_as(path, enforce_file_format=True)


def pillow_encode(ds: Dataset, pixels: np.ndarray, kind: str, path: Path) -> None:
    from PIL import Image

    buffer = io.BytesIO()
    if kind == "jpeg_baseline":
        Image.fromarray(pixels.astype(np.uint8)).save(buffer, "JPEG", quality=75)
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
    else:
        Image.fromarray(pixels.astype(np.uint16)).save(
            buffer, "JPEG2000", irreversible=True, quality_mode="rates", quality_layers=[20], no_jp2=True
        )
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.file_meta.TransferSyntaxUID = JPEG2000
    ds.PixelData = encapsulate([buffer.getvalue()])
    ds["PixelData"].VR = "OB"
    ds.save_as(path, enforce_file_format=True)


def main() -> int:
    try:
        import pylibjpeg  # noqa: F401

        print("Refusing to run with pylibjpeg installed: the Pillow hashes must be "
              "recorded without it (see the module docstring).", file=sys.stderr)
        return 1
    except ImportError:
        pass
    import PIL

    rng = np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True)
    refs = [
        # 12-bit unsigned with a rescale intercept, and 16-bit signed without one.
        (np.round(phantom(160, 0, 4095, rng, 0.02)).astype(np.uint16), 12, False, True),
        (np.round(phantom(160, -3000, 12000, rng, 0.02)).astype(np.int16), 16, True, False),
    ]
    manifest = {
        "about": "Synthetic codec self-test fixtures, generated by make_fixtures.py (no real "
        "images or headers). Lossless: must decode bit-exactly to the ref array under any "
        "decoder. Lossy: the pipeline must decode them exactly as Pillow does; pillow_sha256 "
        "is sha256(np.ascontiguousarray(array).tobytes()) of the Pillow decode, recorded "
        "without pylibjpeg installed.",
        "lossless": {},
        "lossy": {"pillow_version": PIL.__version__, "files": {}},
    }
    for index, (ref, bits, signed, rescale) in enumerate(refs):
        np.save(OUT / f"ref_{index}.npy", ref)
        for syntax in ("jpegll", "j2kll", "jpegls"):
            name = f"{syntax}_{index}.dcm"
            gdcm_encode(dataset(name, ref, bits, signed, rescale), syntax, OUT / name)
            decoded = pydicom.dcmread(OUT / name)
            assert decoded.file_meta.TransferSyntaxUID != ExplicitVRLittleEndian, name
            # GDCM also decodes what it encoded: prove the encoding is lossless.
            if not np.array_equal(decoded.pixel_array.astype(np.int64), ref.astype(np.int64)):
                raise RuntimeError(f"{name} does not round-trip to ref_{index}.npy")
            manifest["lossless"][name] = f"ref_{index}.npy"
    for index in range(2):
        for kind, high in (("jpeg_baseline", 255), ("j2k", 4095)):
            name = f"lossy_{kind}_{index}.dcm"
            pixels = np.round(phantom(256, 0, high, rng, 0.08)).astype(np.uint16)
            pillow_encode(dataset(name, pixels, 16, False, False), pixels, kind, OUT / name)
            array = pixel_array(OUT / name, decoding_plugin="pillow")
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            manifest["lossy"]["files"][name] = {"pillow_sha256": digest}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['lossless'])} lossless and {len(manifest['lossy']['files'])} "
          f"lossy fixtures to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
