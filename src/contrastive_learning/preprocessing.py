"""Turn a DICOM series into a tensor, exactly as the baseline notebook does.

This module is deliberately conservative: it mirrors the image path in
``minimal-baseline:train.ipynb`` step for step, so that any difference in results
between the baseline and this pipeline is attributable to the model rather than to
preprocessing. That is why the parameters are fixed rather than tuned: one
fluid-sensitive series per plane, geometrically ordered slices, three samples from
the central 20-80%, a 130 mm physical center crop, series-level 1st-99th percentile
normalization, resize to 224, and a left-knee orientation convention.

Three of those choices carry medical rather than engineering reasoning:

* **Geometric slice ordering.** File names and instance numbers do not reliably
  reflect anatomical order. Slices are projected onto the series normal so "the
  middle of the knee" means the anatomical middle.
* **Physical rather than pixel cropping.** Pixel spacing varies between scanners,
  so a fixed pixel crop would frame different amounts of anatomy per study; a
  millimetre crop keeps the field of view comparable.
* **Left-knee convention.** Right knees are mirrored so that "lateral" and "medial"
  land on consistent sides of the image, which halves what the model must learn.
  Augmentation therefore never flips, since a flip would destroy that convention.

Every read is failure-tolerant by design: a corrupt slice or an unreadable tag
returns ``None`` or a fallback rather than raising, so one bad study is excluded at
indexing time instead of ending a run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pydicom
except ModuleNotFoundError:  # Allows non-DICOM unit tests to run before dependencies are installed.
    pydicom = None  # type: ignore[assignment]

# Plane order is fixed and load-bearing: it is the channel order of every study
# tensor, and the image tower's per-plane embeddings are indexed by it.
PLANES = ("Sagittal", "Coronal", "Axial")
PLANE_KEYS = {"Sagittal": "sagittal", "Coronal": "coronal", "Axial": "axial"}

# Bumping this invalidates every cached tensor, because it is hashed into their
# cache keys and into cohort metadata. Change it whenever pixel output changes.
PREPROCESSING_VERSION = "2026-09-04.1"


def _require_pydicom() -> Any:
    if pydicom is None:
        raise ModuleNotFoundError("pydicom is required for DICOM preprocessing; install requirements.txt")
    return pydicom


# --------------------------------------------------------------------------- #
# Slice ordering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SliceMeta:
    """The header fields needed to place one slice in anatomical order."""

    path: Path
    position: float | None
    slice_location: float | None
    instance_number: Any
    sop_uid: str


def _as_floats(value: Any, length: int) -> tuple[float, ...] | None:
    """Parse a DICOM multi-value field, rejecting wrong lengths and non-finite values."""
    try:
        values = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return values if len(values) == length and all(np.isfinite(values)) else None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _instance_key(value: Any) -> tuple[int, float | str]:
    """Sort numeric instance numbers before unparsable ones, deterministically."""
    number = _optional_float(value)
    return (0, number) if number is not None else (1, str(value or ""))


def _slice_meta(path: Path) -> SliceMeta | None:
    """Read one slice's ordering metadata; ``None`` if the file is unreadable.

    ``position`` is the slice's signed distance along the series normal, computed
    from the patient position and orientation. That is a true anatomical coordinate,
    unlike ``SliceLocation``, which is scanner-defined and sometimes absent.
    """
    dicom = _require_pydicom()
    try:
        ds = dicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None
    ipp = _as_floats(ds.get("ImagePositionPatient"), 3)
    iop = _as_floats(ds.get("ImageOrientationPatient"), 6)
    position = None
    if ipp is not None and iop is not None:
        normal = np.cross(np.asarray(iop[:3]), np.asarray(iop[3:]))
        norm = float(np.linalg.norm(normal))
        if norm > 0:
            position = float(np.dot(np.asarray(ipp), normal / norm))
    sop = ds.get("SOPInstanceUID")
    return SliceMeta(
        path=path,
        position=position,
        slice_location=_optional_float(ds.get("SliceLocation")),
        instance_number=ds.get("InstanceNumber"),
        sop_uid=str(sop) if sop not in (None, "") else path.name,
    )


def _tie_key(meta: SliceMeta) -> tuple:
    """Total order for slices that share a coordinate, so ordering never varies."""
    return (_instance_key(meta.instance_number), meta.sop_uid, meta.path.name)


def ordered_slice_paths(series_dir: Path) -> list[Path]:
    """Order a series' slices anatomically, with the baseline's fallback chain.

    Geometric position is preferred, then ``SliceLocation``, then instance number,
    then identifiers. Each fallback requires the field on *every* slice: a partially
    populated field would interleave two incompatible orderings.
    """
    paths = sorted(p for p in Path(series_dir).iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    metas = [meta for path in paths if (meta := _slice_meta(path)) is not None]
    if not metas:
        raise FileNotFoundError(f"No readable DICOM slices in {series_dir}")
    if all(meta.position is not None for meta in metas):
        metas.sort(key=lambda meta: (meta.position, *_tie_key(meta)))
    elif all(meta.slice_location is not None for meta in metas):
        metas.sort(key=lambda meta: (meta.slice_location, *_tie_key(meta)))
    elif all(meta.instance_number not in (None, "") for meta in metas):
        metas.sort(key=_tie_key)
    else:
        metas.sort(key=lambda meta: (meta.sop_uid, meta.path.name))
    return [meta.path for meta in metas]


# --------------------------------------------------------------------------- #
# Series inspection
# --------------------------------------------------------------------------- #


def middle_dicom(series_dir: Path) -> Path | None:
    """A representative slice by file name, for reading series-level header tags.

    Name order is enough here: any slice of the series carries the same spacing,
    laterality, and patient tags, so full geometric ordering is not worth its cost.
    """
    files = sorted(p for p in Path(series_dir).iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    return files[len(files) // 2] if files else None


def count_dicoms(series_dir: Path) -> int:
    """Count slices without reading any of them; used to rank candidate series."""
    directory = Path(series_dir)
    if not directory.is_dir():
        return 0
    with os.scandir(directory) as entries:
        return sum(entry.is_file() and entry.name.lower().endswith(".dcm") for entry in entries)


def series_fingerprint(series_dir: Path) -> str:
    """Identify a series by file names, sizes, and modification times.

    Stat metadata rather than content: hashing hundreds of megabytes per study on
    every index would dominate runtime, and this is sufficient to notice replaced or
    re-exported data. The preprocessing version is included so a settings change
    invalidates the fingerprint too.
    """
    digest = hashlib.sha256()
    digest.update(PREPROCESSING_VERSION.encode())
    for path in sorted(Path(series_dir).glob("*.dcm")):
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _row_spacing(series_dir: Path) -> float | None:
    """Millimetres per pixel row, or ``None`` when the tag is missing or invalid."""
    path = middle_dicom(series_dir)
    if path is None:
        return None
    dicom = _require_pydicom()
    try:
        ds = dicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=["PixelSpacing"])
        spacing = getattr(ds, "PixelSpacing", None)
        value = float(spacing[0]) if spacing is not None and len(spacing) else float("nan")
        return value if np.isfinite(value) and value > 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Pixels
# --------------------------------------------------------------------------- #


def _decode_pixels(path: Path) -> np.ndarray | None:
    """Decode one slice into rescaled float values, or ``None`` if it cannot be read.

    The rescale slope and intercept are applied so values are in the units the
    scanner intended, which is what makes percentile normalization meaningful.
    """
    dicom = _require_pydicom()
    try:
        ds = dicom.dcmread(path, force=True)
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        return pixels * slope + intercept
    except Exception:
        return None


def _slice_indices(n_slices: int, slice_band: tuple[float, float], count: int) -> np.ndarray:
    """Pick ``count`` slice indices spread across the central band of a series.

    The band excludes the outer slices, which are mostly air and soft tissue rather
    than joint. Degenerate series collapse to their middle slice, and indices are
    padded by repetition so the output channel count is always ``count``.
    """
    low = int(slice_band[0] * (n_slices - 1))
    high = int(slice_band[1] * (n_slices - 1))
    if high > low:
        indices = np.unique(np.linspace(low, high, count).astype(int))
    else:
        indices = np.asarray([n_slices // 2], dtype=int)
    while len(indices) < count:
        indices = np.append(indices, indices[-1])
    return indices[:count]


def read_slot(
    series_dir: Path,
    *,
    image_size: int = 224,
    crop_mm: float = 130.0,
    slices_per_plane: int = 3,
    slice_band: tuple[float, float] = (0.20, 0.80),
) -> torch.Tensor | None:
    """Read one plane's series into a float tensor ``[slices, H, W]`` in ``[0, 1]``.

    Returns ``None`` when the series is unreadable, so callers can exclude the study
    rather than handle an exception. Undecodable individual slices are substituted
    with their nearest decoded neighbour, which keeps the channel count fixed
    without inventing pixel values.
    """
    try:
        paths = ordered_slice_paths(series_dir)
    except (FileNotFoundError, OSError):
        return None
    indices = _slice_indices(len(paths), slice_band, slices_per_plane)

    slices: list[np.ndarray | None] = [_decode_pixels(paths[int(index)]) for index in indices]
    decoded = [index for index, image in enumerate(slices) if image is not None]
    if not decoded:
        return None
    for index, image in enumerate(slices):
        if image is None:
            slices[index] = slices[min(decoded, key=lambda good: abs(good - index))]

    # Mismatched in-series dimensions cannot be stacked; a zero plane keeps the
    # study usable while contributing no signal.
    shape = slices[0].shape  # type: ignore[union-attr]
    volume = np.stack(
        [image if image.shape == shape else np.zeros(shape, np.float32) for image in slices]  # type: ignore[union-attr]
    ).astype(np.float32, copy=False)

    spacing = _row_spacing(series_dir)
    if spacing is not None:
        wanted = int(round(crop_mm / spacing))
        height, width = shape
        # Skip the crop when it would be degenerate or larger than the image.
        if 16 < wanted < min(height, width):
            cy, cx, half = height // 2, width // 2, wanted // 2
            volume = volume[:, max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]

    # Percentiles over the whole slot, not per slice, so relative slice intensity
    # is preserved; they also clip the bright outliers that min/max scaling ruins.
    low_value, high_value = np.percentile(volume, [1, 99])
    volume = np.clip((volume - low_value) / max(high_value - low_value, 1e-6), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).float()


# --------------------------------------------------------------------------- #
# Laterality and orientation
# --------------------------------------------------------------------------- #


def _lr_char(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text[0] if text and text[0] in ("L", "R") else None


def _center_x(ds: Any) -> float | None:
    """Patient-space x coordinate of an image's center, from position and orientation.

    In the DICOM patient coordinate system negative x is the patient's right, which
    is what makes this usable as a laterality signal of last resort.
    """
    try:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
        center = ipp[:3] + iop[:3] * spacing[1] * float(ds.Columns) / 2
        center += iop[3:6] * spacing[0] * float(ds.Rows) / 2
        return float(center[0])
    except Exception:
        return None


def resolve_laterality(slots: dict[str, Path], min_offset_mm: float = 20.0) -> str | None:
    """Resolve which knee was imaged: Series tag, then Image tag, then geometry.

    Returns ``None`` rather than guessing when no source is conclusive -- including
    when the geometric center sits within ``min_offset_mm`` of the midline, where the
    sign carries no information. An unresolved study is simply left unmirrored.
    """
    dicom = _require_pydicom()
    series_tags: list[str] = []
    image_tags: list[str] = []
    center_xs: list[float] = []
    for plane in PLANES:
        path = middle_dicom(slots[plane])
        if path is None:
            continue
        try:
            ds = dicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if tag := _lr_char(getattr(ds, "Laterality", None)):
            series_tags.append(tag)
        if tag := _lr_char(getattr(ds, "ImageLaterality", None)):
            image_tags.append(tag)
        if (center_x := _center_x(ds)) is not None:
            center_xs.append(center_x)
    if series_tags:
        return series_tags[0]
    if image_tags:
        return image_tags[0]
    if not center_xs:
        return None
    median = float(np.median(center_xs))
    if abs(median) < min_offset_mm:
        return None
    return "R" if median < 0 else "L"


def apply_left_knee(image: torch.Tensor, plane: str, laterality: str | None) -> torch.Tensor:
    """Mirror a right knee into the left-knee convention; left knees pass through.

    Coronal and axial planes mirror across image width. Sagittal images already view
    the knee from the side, so left/right is encoded in *which* side the viewer is
    on: those flip along the slice axis instead of within a slice.
    """
    if laterality != "R":
        return image
    return torch.flip(image, dims=[-1] if plane in ("Coronal", "Axial") else [0])


def augment_study_images(
    images: torch.Tensor,
    *,
    rotation_degrees: float = 8.0,
    zoom: float = 0.08,
    shift: float = 0.05,
    intensity: float = 0.10,
) -> torch.Tensor:
    """Randomly rotate, zoom, shift, and rescale intensity -- never flip.

    Accepts any leading batch dimensions and restores them, so it applies equally to
    one study or a batch of them. Every plane of every study draws its own transform.
    Magnitudes are small: knee anatomy is consistently positioned, and larger
    warping would move findings out of frame. Flipping is excluded because
    :func:`apply_left_knee` has already established a laterality convention.
    """
    leading = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:]).float()
    count, device = flat.shape[0], flat.device
    # Zoom is folded into the rotation matrix by scaling its basis vectors, so one
    # grid sample performs rotation, zoom, and translation together.
    scale = 1.0 + torch.rand(count, device=device) * zoom
    radians = (torch.rand(count, device=device) - 0.5) * 2 * rotation_degrees * np.pi / 180
    cosine, sine = torch.cos(radians) / scale, torch.sin(radians) / scale
    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    theta[:, 0, 2] = (torch.rand(count, device=device) - 0.5) * 2 * shift
    theta[:, 1, 2] = (torch.rand(count, device=device) - 0.5) * 2 * shift
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    # Border padding avoids the black wedges that zero padding would rotate in.
    flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    gain = 1.0 + (torch.rand(count, 1, 1, 1, device=device) - 0.5) * 2 * intensity
    return (flat * gain).clamp(0, 1).reshape(*leading, *flat.shape[-3:]).to(images.dtype)
