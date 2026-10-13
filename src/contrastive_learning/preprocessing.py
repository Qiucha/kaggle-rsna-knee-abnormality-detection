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
indexing time instead of ending a run. Tolerance is not silence, though: every
decode outcome is counted per transfer syntax (:func:`decode_counts`), and
``KNEE_STRICT_DECODE=1`` turns a partially decoded plane into an error.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
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

# Bumping this invalidates every cached tensor, because it is hashed into their
# cache keys and into cohort metadata. Change it whenever pixel output changes.
# The per-syntax decoder pins below deliberately did not bump it: they keep every
# file that decoded before on the decoder it used before, so no pixel changed.
PREPROCESSING_VERSION = "2026-09-04.1"

# Which pydicom decoding plugin each compressed transfer syntax is pinned to.
# Without a pin, installing the vendored pylibjpeg wheels would silently move lossy
# JPEG baseline/extended, JPEG 2000 and RLE onto pylibjpeg, because pydicom tries it
# ahead of Pillow and its own RLE decoder; for lossy JPEG baseline that changes
# pixels (verified on Kaggle, see kaggle/codecs/README.md). So each syntax stays
# on the plugin that decodes it today, and pylibjpeg is used only where nothing did:
# JPEG Lossless (.57, .70) and JPEG-LS. Uncompressed syntaxes have no plugins.
DECODING_PLUGINS = {
    "1.2.840.10008.1.2.4.50": "pillow",  # JPEG Baseline (8-bit)
    "1.2.840.10008.1.2.4.51": "pillow",  # JPEG Extended (12-bit)
    "1.2.840.10008.1.2.4.90": "pillow",  # JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.91": "pillow",  # JPEG 2000
    "1.2.840.10008.1.2.5": "pydicom",  # RLE Lossless
    "1.2.840.10008.1.2.4.57": "pylibjpeg",  # JPEG Lossless
    "1.2.840.10008.1.2.4.70": "pylibjpeg",  # JPEG Lossless SV1
    "1.2.840.10008.1.2.4.80": "pylibjpeg",  # JPEG-LS Lossless
    "1.2.840.10008.1.2.4.81": "pylibjpeg",  # JPEG-LS Near-Lossless
}

# Set to "1" for local validation: a plane with any undecodable selected slice then
# raises instead of being patched from its neighbours.
STRICT_DECODE_ENV = "KNEE_STRICT_DECODE"

# Per-process decode telemetry keyed by (transfer syntax, event). Submission
# inference decodes in-process, so one counter sees every read; DataLoader workers
# each keep their own. Events: decoded, failed, pin_retry_decoded,
# substituted_slices, undecodable_planes, fallback_series.
_DECODE_COUNTS: Counter[tuple[str, str]] = Counter()


def image_kwargs(settings: Any) -> dict[str, Any]:
    """The :func:`read_slot` settings carried by a training or manifest config.

    Both configs declare the same five fields, and these values are hashed into
    image cache keys and recorded as manifest pixel QC. Deriving the kwargs here,
    from whichever config is at hand, keeps a manifest's validated pixels and a
    run's decoded pixels from drifting apart silently.
    """
    return {
        "image_size": settings.image_size,
        "crop_mm": settings.crop_mm,
        "slices_per_plane": settings.slices_per_plane,
        "slice_band": (settings.slice_band_low, settings.slice_band_high),
    }


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


def series_fov_mm(series_dir: Path) -> tuple[float, float] | None:
    """A slice's true physical extent ``(rows_mm, cols_mm)``, read from the DICOM.

    ``ReconstructionDiameter`` would state this directly but is absent from every
    series in this cohort, so it is derived as ``PixelSpacing * (Rows, Columns)``.
    Returns ``None`` when the tags are missing or invalid, so callers can fall back
    rather than handle an exception.
    """
    path = middle_dicom(series_dir)
    if path is None:
        return None
    dicom = _require_pydicom()
    try:
        ds = dicom.dcmread(
            path, stop_before_pixels=True, force=True,
            specific_tags=["PixelSpacing", "Rows", "Columns"],
        )
        spacing = getattr(ds, "PixelSpacing", None)
        rows, columns = getattr(ds, "Rows", None), getattr(ds, "Columns", None)
        if spacing is None or len(spacing) < 2 or not rows or not columns:
            return None
        extent = (float(spacing[0]) * int(rows), float(spacing[1]) * int(columns))
        return extent if all(np.isfinite(v) and v > 0 for v in extent) else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Pixels
# --------------------------------------------------------------------------- #


def count_decode_event(syntax: str, event: str, count: int = 1) -> None:
    """Record ``count`` decode events of one kind against one transfer syntax."""
    _DECODE_COUNTS[(syntax, event)] += count


def decode_counts(*, reset: bool = False) -> dict[str, dict[str, int]]:
    """This process's decode counters as ``{syntax: {event: count}}``.

    With ``reset`` the counters are cleared after reading, so a caller can scope
    them to one run (submission inference does this at its start).
    """
    counts: dict[str, dict[str, int]] = {}
    for (syntax, event), value in sorted(_DECODE_COUNTS.items()):
        counts.setdefault(syntax, {})[event] = value
    if reset:
        _DECODE_COUNTS.clear()
    return counts


def _syntax_name(uid: str) -> str:
    """A short readable transfer syntax name, e.g. ``JPEGLosslessSV1``."""
    if not uid:
        return "unknown"
    return _require_pydicom().uid.UID(uid).keyword or uid


def _transfer_syntax(ds: Any) -> str:
    return str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "") or "")


def slice_transfer_syntax(path: Path) -> str:
    """One file's transfer syntax name, from its header only; ``unknown`` if unreadable."""
    try:
        ds = _require_pydicom().dcmread(path, stop_before_pixels=True, force=True)
        return _syntax_name(_transfer_syntax(ds))
    except Exception:
        return "unknown"


def series_transfer_syntax(series_dir: Path) -> str:
    """The transfer syntax name of a series' middle file, used to label its counters."""
    path = middle_dicom(series_dir)
    return slice_transfer_syntax(path) if path is not None else "unknown"


@cache
def decoding_plugin(transfer_syntax: str) -> str:
    """The plugin :func:`_decode_pixels` pins for a syntax, or ``""`` for the default.

    A pin whose plugin is not installed is dropped, so an environment without
    Pillow or pylibjpeg decodes exactly as it did before pins existed. pydicom fixes
    plugin availability when it is imported, which is what makes caching safe.
    """
    wanted = DECODING_PLUGINS.get(transfer_syntax, "")
    if not wanted:
        return ""
    try:
        from pydicom.pixels import get_decoder

        available = get_decoder(transfer_syntax).available_plugins
    except Exception:
        return ""
    return wanted if wanted in available else ""


def _decode_pixels(path: Path) -> np.ndarray | None:
    """Decode one slice into rescaled float values, or ``None`` if it cannot be read.

    The rescale slope and intercept are applied so values are in the units the
    scanner intended, which is what makes percentile normalization meaningful.

    The decoder is pinned per transfer syntax (:data:`DECODING_PLUGINS`). When a
    pinned plugin raises, pydicom's default chain gets one more try: the pinned
    plugin was the only one available before the codec wheels, so a file it cannot
    decode also failed before, and the retry can rescue such a file (12-bit JPEG
    extended, say) but never change one that already decoded.
    """
    dicom = _require_pydicom()
    syntax = "unknown"
    try:
        ds = dicom.dcmread(path, force=True)
        uid = _transfer_syntax(ds)
        syntax = _syntax_name(uid)
        plugin = decoding_plugin(uid)
        if plugin:
            ds.pixel_array_options(decoding_plugin=plugin)
        try:
            array = ds.pixel_array
        except Exception:
            if not plugin:
                raise
            ds.pixel_array_options()
            array = ds.pixel_array
            count_decode_event(syntax, "pin_retry_decoded")
        pixels = array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        rescaled = pixels * slope + intercept
    except Exception:
        count_decode_event(syntax, "failed")
        return None
    count_decode_event(syntax, "decoded")
    return rescaled


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


def jittered_slice_indices(
    n_slices: int, slice_band: tuple[float, float], count: int, rng: np.random.Generator
) -> np.ndarray:
    """One random slice from each of ``count`` equal strata of the band, in order.

    The training-time counterpart of :func:`_slice_indices`: each draw stays in the
    stratum around the fixed sampler's pick, so channel order and spacing look like
    what evaluation serves, while the exact slices change every epoch.
    """
    low = int(slice_band[0] * (n_slices - 1))
    high = int(slice_band[1] * (n_slices - 1))
    if high - low + 1 < count:
        return _slice_indices(n_slices, slice_band, count)
    edges = np.linspace(low, high + 1, count + 1).astype(int)
    return np.asarray([rng.integers(a, b) for a, b in pairwise(edges)], dtype=int)


def slice_group_indices(
    n_slices: int, slice_band: tuple[float, float], group_size: int, groups: int
) -> np.ndarray:
    """Indices of ``groups`` local stacks of ``group_size`` slices, tiling the band.

    Returns shape ``(groups, group_size)``. Two differences from :func:`_slice_indices`,
    which returns one spread-out set:

    * **Each group is contiguous**, so the channels of one input are a coherent local
      mini-volume. The existing sampler puts its three channels a median 31mm apart,
      which is further than most findings are wide.
    * **The groups tile the band**, so the volume is covered rather than sampled at
      three fixed depths.

    Group starts are spread evenly and clamped into the band; short series repeat
    slices rather than fail, matching the existing sampler's padding behaviour.
    """
    if groups < 1 or group_size < 1:
        raise ValueError("groups and group_size must be >= 1")
    low = int(slice_band[0] * (n_slices - 1))
    high = int(slice_band[1] * (n_slices - 1))
    if high <= low:  # degenerate series: every group is the middle slice
        return np.full((groups, group_size), n_slices // 2, dtype=int)
    # Last legal start keeps a whole group inside the series.
    last_start = max(low, min(high, n_slices - group_size))
    starts = np.unique(np.linspace(low, last_start, groups).astype(int))
    while len(starts) < groups:  # too few distinct starts to tile: repeat the last
        starts = np.append(starts, starts[-1])
    return np.stack([np.arange(s, s + group_size).clip(0, n_slices - 1) for s in starts[:groups]])


def window_indices(
    n_slices: int,
    slice_band: tuple[float, float],
    windows: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``windows`` contiguous 3-slice windows, one per equal stratum of the band.

    Returns indices ``(windows, 3)`` and a validity mask ``(windows,)``. Each window is
    centred on one slice of its stratum: the stratum centre when ``rng`` is ``None``
    (evaluation) and a uniform draw from it otherwise (training), so the two only
    ever disagree inside a stratum. A stratum with no slice (a stack too short for
    ``windows`` distinct centres) is marked invalid instead of repeating a neighbour;
    its indices copy a valid window's so nothing extra is decoded.
    """
    if windows < 1:
        raise ValueError("windows must be >= 1")
    low = int(slice_band[0] * (n_slices - 1))
    high = max(low, int(slice_band[1] * (n_slices - 1)))
    edges = np.linspace(low, high + 1, windows + 1).astype(int)
    valid = edges[1:] > edges[:-1]
    centres = np.asarray([
        (rng.integers(a, b) if rng is not None else (a + b - 1) // 2) if b > a else -1
        for a, b in pairwise(edges)
    ])
    centres[~valid] = centres[valid][0]
    indices = (centres[:, None] + np.arange(-1, 2)).clip(0, n_slices - 1)
    return indices, valid


def read_windows(
    series_dir: Path,
    *,
    windows: int,
    image_size: int = 224,
    crop_mm: float = 130.0,
    slice_band: tuple[float, float] = (0.20, 0.80),
    slice_rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One plane as ``windows`` 3-slice images ``[windows, 3, H, W]`` plus a validity mask.

    Every selected slice is decoded once and normalised together, as :func:`read_slot`
    does for its three, so intensity stays comparable across depth. Invalid windows
    are zero. Returns ``None`` when the series is unreadable.
    """
    try:
        paths = ordered_slice_paths(series_dir)
    except (FileNotFoundError, OSError):
        return None
    indices, valid = window_indices(len(paths), slice_band, windows, slice_rng)
    unique = np.unique(indices[valid])
    volume = _read_volume(series_dir, paths, unique, image_size=image_size, crop_mm=crop_mm)
    if volume is None:
        return None
    images = volume[torch.as_tensor(np.searchsorted(unique, indices))]
    mask = torch.as_tensor(valid)
    images[~mask] = 0
    return images, mask


def apply_left_knee_windows(
    images: torch.Tensor, valid: torch.Tensor, plane: str, laterality: str | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`apply_left_knee` for ``[windows, 3, H, W]``: sagittal depth is both axes."""
    if laterality != "R":
        return images, valid
    if plane in ("Coronal", "Axial"):
        return torch.flip(images, dims=[-1]), valid
    return torch.flip(images, dims=[0, 1]), torch.flip(valid, dims=[0])


def _centre_to_extent(volume: np.ndarray, wanted: int) -> np.ndarray:
    """Centre-crop or zero-pad every slice to exactly ``wanted`` pixels square.

    Cropping alone cannot normalise scale: a fixed physical box only fits inside the
    image when it is smaller than the acquired field of view, so any constant above
    the smallest FOV in the cohort would be silently skipped for some series and the
    two would land at different millimetres per pixel. Padding the short side instead
    makes the physical extent exact for every series. Pad value is zero, which is
    what lies outside the acquired volume anyway.
    """
    slices, height, width = volume.shape
    cy, cx, half = height // 2, width // 2, wanted // 2
    out = np.zeros((slices, 2 * half, 2 * half), dtype=volume.dtype)
    top, left = max(0, cy - half), max(0, cx - half)
    patch = volume[:, top : cy + half, left : cx + half]
    y = (out.shape[1] - patch.shape[1]) // 2
    x = (out.shape[2] - patch.shape[2]) // 2
    out[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
    return out


def read_slot(
    series_dir: Path,
    *,
    image_size: int = 224,
    crop_mm: float = 130.0,
    slices_per_plane: int = 3,
    slice_band: tuple[float, float] = (0.20, 0.80),
    slice_rng: np.random.Generator | None = None,
) -> torch.Tensor | None:
    """Read one plane's series into a float tensor ``[slices, H, W]`` in ``[0, 1]``.

    Returns ``None`` when the series is unreadable, so callers can exclude the study
    rather than handle an exception. Undecodable individual slices are substituted
    with their nearest decoded neighbour, which keeps the channel count fixed
    without inventing pixel values. Both outcomes are counted per transfer syntax;
    with ``KNEE_STRICT_DECODE=1`` a substitution raises instead. ``slice_rng``
    switches to :func:`jittered_slice_indices`, for training only.
    """
    try:
        paths = ordered_slice_paths(series_dir)
    except (FileNotFoundError, OSError):
        return None
    if slice_rng is None:
        indices = _slice_indices(len(paths), slice_band, slices_per_plane)
    else:
        indices = jittered_slice_indices(len(paths), slice_band, slices_per_plane, slice_rng)
    return _read_volume(series_dir, paths, indices, image_size=image_size, crop_mm=crop_mm)


def _read_volume(
    series_dir: Path, paths: list[Path], indices: np.ndarray, *, image_size: int, crop_mm: float
) -> torch.Tensor | None:
    """Decode ``indices`` of one series and crop, normalise and resize them jointly."""
    slices: list[np.ndarray | None] = [_decode_pixels(paths[int(index)]) for index in indices]
    decoded = [index for index, image in enumerate(slices) if image is not None]
    failed = [paths[int(indices[index])] for index, image in enumerate(slices) if image is None]
    if not decoded:
        count_decode_event(slice_transfer_syntax(failed[0]), "undecodable_planes")
        return None
    if failed:
        syntaxes = sorted({slice_transfer_syntax(path) for path in failed})
        if os.environ.get(STRICT_DECODE_ENV) == "1":
            raise RuntimeError(
                f"{len(failed)} of {len(slices)} selected slices failed to decode in "
                f"{series_dir} ({', '.join(syntaxes)}); unset {STRICT_DECODE_ENV} to substitute"
            )
        for path in failed:
            count_decode_event(slice_transfer_syntax(path), "substituted_slices")
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
        # Degenerate requests are still skipped; everything else is made exact, so
        # every series leaves here at the same crop_mm / image_size scale.
        if wanted > 16:
            volume = _centre_to_extent(volume, wanted)

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
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly rotate, zoom, shift, and rescale intensity -- never flip.

    Accepts any leading batch dimensions and restores them, so it applies equally to
    one study or a batch of them. Every plane of every study draws its own transform.
    Magnitudes are small: knee anatomy is consistently positioned, and larger
    warping would move findings out of frame. Flipping is excluded because
    :func:`apply_left_knee` has already established a laterality convention.
    ``generator`` keeps augmentation draws separate from sampling and dropout.
    """
    leading = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:]).float()
    count, device = flat.shape[0], flat.device
    # Zoom is folded into the rotation matrix by scaling its basis vectors, so one
    # grid sample performs rotation, zoom, and translation together.
    scale = 1.0 + torch.rand(count, device=device, generator=generator) * zoom
    radians = (torch.rand(count, device=device, generator=generator) - 0.5) * 2 * rotation_degrees * np.pi / 180
    cosine, sine = torch.cos(radians) / scale, torch.sin(radians) / scale
    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    theta[:, 0, 2] = (torch.rand(count, device=device, generator=generator) - 0.5) * 2 * shift
    theta[:, 1, 2] = (torch.rand(count, device=device, generator=generator) - 0.5) * 2 * shift
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    # Border padding avoids the black wedges that zero padding would rotate in.
    flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    gain = 1.0 + (torch.rand(count, 1, 1, 1, device=device, generator=generator) - 0.5) * 2 * intensity
    return (flat * gain).clamp(0, 1).reshape(*leading, *flat.shape[-3:]).to(images.dtype)
