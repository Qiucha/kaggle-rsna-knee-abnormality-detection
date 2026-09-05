"""Baseline-compatible study-to-tensor preprocessing.

This module intentionally mirrors the image path in ``minimal-baseline:train.ipynb``:
one fluid-sensitive series per anatomical plane, geometrically ordered slices,
three samples in the central 20--80%, a 130 mm center crop, series-level 1st--99th
percentile normalization, resize to 224, and a left-knee orientation convention.
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

PLANES = ("Sagittal", "Coronal", "Axial")
PLANE_KEYS = {"Sagittal": "sagittal", "Coronal": "coronal", "Axial": "axial"}
PREPROCESSING_VERSION = "2026-09-04.1"


def _require_pydicom() -> Any:
    if pydicom is None:
        raise ModuleNotFoundError("pydicom is required for DICOM preprocessing; install requirements.txt")
    return pydicom


@dataclass(frozen=True)
class SliceMeta:
    path: Path
    position: float | None
    slice_location: float | None
    instance_number: Any
    sop_uid: str


def _as_floats(value: Any, length: int) -> tuple[float, ...] | None:
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
    number = _optional_float(value)
    return (0, number) if number is not None else (1, str(value or ""))


def _slice_meta(path: Path) -> SliceMeta | None:
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
    return (_instance_key(meta.instance_number), meta.sop_uid, meta.path.name)


def ordered_slice_paths(series_dir: Path) -> list[Path]:
    """Order DICOM slices by geometry with the baseline's deterministic fallbacks."""
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


def middle_dicom(series_dir: Path) -> Path | None:
    files = sorted(p for p in Path(series_dir).iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    return files[len(files) // 2] if files else None


def count_dicoms(series_dir: Path) -> int:
    directory = Path(series_dir)
    if not directory.is_dir():
        return 0
    with os.scandir(directory) as entries:
        return sum(entry.is_file() and entry.name.lower().endswith(".dcm") for entry in entries)


def series_fingerprint(series_dir: Path) -> str:
    """Hash DICOM file identity and stat metadata for cache invalidation."""
    digest = hashlib.sha256()
    digest.update(PREPROCESSING_VERSION.encode())
    for path in sorted(Path(series_dir).glob("*.dcm")):
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _decode_pixels(path: Path) -> np.ndarray | None:
    dicom = _require_pydicom()
    try:
        ds = dicom.dcmread(path, force=True)
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        return pixels * slope + intercept
    except Exception:
        return None


def _row_spacing(series_dir: Path) -> float | None:
    path = middle_dicom(series_dir)
    if path is None:
        return None
    dicom = _require_pydicom()
    try:
        ds = dicom.dcmread(
            path, stop_before_pixels=True, force=True, specific_tags=["PixelSpacing"]
        )
        spacing = getattr(ds, "PixelSpacing", None)
        value = float(spacing[0]) if spacing is not None and len(spacing) else float("nan")
        return value if np.isfinite(value) and value > 0 else None
    except Exception:
        return None


def read_slot(
    series_dir: Path,
    *,
    image_size: int = 224,
    crop_mm: float = 130.0,
    slices_per_plane: int = 3,
    slice_band: tuple[float, float] = (0.20, 0.80),
) -> torch.Tensor | None:
    """Read one plane/series into a float tensor shaped ``[3, H, W]`` in ``[0, 1]``."""
    try:
        paths = ordered_slice_paths(series_dir)
    except (FileNotFoundError, OSError):
        return None
    n_slices = len(paths)
    low = int(slice_band[0] * (n_slices - 1))
    high = int(slice_band[1] * (n_slices - 1))
    if high > low:
        indices = np.unique(np.linspace(low, high, slices_per_plane).astype(int))
    else:
        indices = np.asarray([n_slices // 2], dtype=int)
    while len(indices) < slices_per_plane:
        indices = np.append(indices, indices[-1])

    planes: list[np.ndarray | None] = [_decode_pixels(paths[int(index)]) for index in indices[:slices_per_plane]]
    decoded = [index for index, plane in enumerate(planes) if plane is not None]
    if not decoded:
        return None
    for index, plane in enumerate(planes):
        if plane is None:
            planes[index] = planes[min(decoded, key=lambda good: abs(good - index))]

    assert all(plane is not None for plane in planes)
    shape = planes[0].shape  # type: ignore[union-attr]
    fixed = [plane if plane.shape == shape else np.zeros(shape, np.float32) for plane in planes]  # type: ignore[union-attr]
    volume = np.stack(fixed).astype(np.float32, copy=False)

    spacing = _row_spacing(series_dir)
    if spacing is not None:
        wanted = int(round(crop_mm / spacing))
        height, width = shape
        if 16 < wanted < min(height, width):
            cy, cx, half = height // 2, width // 2, wanted // 2
            volume = volume[:, max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]

    low_value, high_value = np.percentile(volume, [1, 99])
    volume = np.clip((volume - low_value) / max(high_value - low_value, 1e-6), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).float()


def _lr_char(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text[0] if text and text[0] in ("L", "R") else None


def _center_x(ds: Any) -> float | None:
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
    """Resolve study laterality from Series tag, Image tag, then patient geometry."""
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
    """Baseline spatial/intensity augmentation without anatomical flips."""
    leading = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:]).float()
    count, device = flat.shape[0], flat.device
    scale = 1.0 + torch.rand(count, device=device) * zoom
    radians = (torch.rand(count, device=device) - 0.5) * 2 * rotation_degrees * np.pi / 180
    cosine, sine = torch.cos(radians) / scale, torch.sin(radians) / scale
    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    theta[:, 0, 2] = (torch.rand(count, device=device) - 0.5) * 2 * shift
    theta[:, 1, 2] = (torch.rand(count, device=device) - 0.5) * 2 * shift
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    gain = 1.0 + (torch.rand(count, 1, 1, 1, device=device) - 0.5) * 2 * intensity
    return (flat * gain).clamp(0, 1).reshape(*leading, *flat.shape[-3:]).to(images.dtype)
