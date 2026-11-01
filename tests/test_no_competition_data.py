"""Regression guard: no competition data in tracked files.

The competition rules (section 4.b) and the RSNA MIRA license forbid publishing any
portion of the data. This scans every tracked text file for real study UIDs and for
embedded raster images (which in this repo could only be MRI slices).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from contrastive_learning.manifest import study_pseudonym

REPO = Path(__file__).resolve().parents[1]

REAL_UID = re.compile(rb"1\.2\.826\.0\.1\.3680043\.8\.498\.[0-9]+")
# A UID abbreviated to a run of 8+ of its digits next to an ellipsis.
TRUNCATED_UID = re.compile(rb"(?:\xe2\x80\xa6|\.\.\.)[0-9]{8,}|[0-9]{8,}(?:\xe2\x80\xa6|\.\.\.)")
EMBEDDED_IMAGE = re.compile(rb"data:image/(?:png|jpeg);base64|\"image/(?:png|jpeg)\"")
BINARY_SUFFIXES = {
    ".dcm", ".npy", ".npz", ".whl", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    ".pt", ".pth", ".safetensors", ".bin", ".zip", ".gz", ".so", ".pyc",
}


def _tracked_text_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    files = []
    for name in out.decode("utf-8").split("\0"):
        path = REPO / name
        if name and path.suffix.lower() not in BINARY_SUFFIXES and path.is_file():
            files.append(path)
    return files


def test_no_real_uids_or_embedded_images_in_tracked_files() -> None:
    offenders = []
    for path in _tracked_text_files():
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary despite its suffix
        rel = path.relative_to(REPO)
        if REAL_UID.search(data):
            offenders.append(f"{rel}: real competition study UID")
        if TRUNCATED_UID.search(data):
            offenders.append(f"{rel}: abbreviated study UID")
        if EMBEDDED_IMAGE.search(data):
            offenders.append(f"{rel}: embedded raster image")
    assert not offenders, "competition data in tracked files:\n" + "\n".join(offenders)


def test_study_pseudonym_is_deterministic_and_well_formed() -> None:
    uid = "1.2.3.4"
    value = study_pseudonym(uid)
    assert value == study_pseudonym(uid)
    assert value == "study-6694f83c9f476da3"  # sha256("1.2.3.4")[:16]
    assert re.fullmatch(r"study-[0-9a-f]{16}", value)
    assert study_pseudonym("1.2.3.5") != value
