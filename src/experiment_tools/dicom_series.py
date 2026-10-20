"""DICOM series readers shared by the audits under scripts/audits/.

Reading only: nothing here writes to the cohort, trains, or opens a label.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from contrastive_learning.preprocessing import _decode_pixels, ordered_slice_paths


def plane_digest(directory: str) -> dict:
    """Ordered decoded-pixel hash for one series, plus shape facts for the record."""
    digest = hashlib.sha256()
    paths = ordered_slice_paths(Path(directory))
    shapes, dtypes = set(), set()
    for path in paths:
        pixels = _decode_pixels(path)
        if pixels is None:
            digest.update(b"<undecodable>")
            continue
        array = np.ascontiguousarray(pixels)
        shapes.add(tuple(array.shape))
        dtypes.add(str(array.dtype))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return {
        "ordered_decoded_pixel_sha256": digest.hexdigest(),
        "slices": len(paths),
        "slice_shapes": sorted(str(s) for s in shapes),
        "dtypes": sorted(dtypes),
    }


def middle_slice_header(series_dir: Path | str):
    """``(dataset, slice_count)`` of a series' middle file by name, headers only, or ``None``.

    Name order is arbitrary for this dataset (file names are randomised SOP UIDs), so
    the "middle" slice is a representative one for series-level tags, not a central
    one. Every file in the directory counts as a slice. ``None`` when the directory
    cannot be listed, is empty, or the chosen file is not readable as DICOM.
    """
    import pydicom  # outside the try: a missing package must raise, not read as "unreadable"

    try:
        files = sorted(os.listdir(series_dir))
        dataset = pydicom.dcmread(Path(series_dir) / files[len(files) // 2],
                                  stop_before_pixels=True, force=True)
    except Exception:
        return None
    return dataset, len(files)
