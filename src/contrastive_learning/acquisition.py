"""Recover what the series CSV cannot say: pulse-sequence weighting and fat suppression.

`train_series.csv` carries `Fluid_Sensitive` and `Fat_Suppression`, which name physically
independent properties -- weighting is set by TR and TE, fat suppression is a preparation
applied on top of any weighting -- but the two columns are **byte-identical on all 24,371
rows**, so between them they carry one bit rather than two. Both axes therefore have to be
read back out of the DICOM header. See `digest/acquisition-metadata-trust.md`.

The recovery rules follow the public RSNA-knee work synthesised in `prev/rsna_knee_101.ipynb`,
which established them and named the two string-matching traps guarded below.

Everything here is a pure function of header values, so it is testable without DICOM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Separators are normalised to spaces before any token test. Underscore is a word
# character, so `\bwe\b` would never fire inside `t2_de3d_we_tra` without this.
_SEPARATORS = re.compile(r"[_\-.,/]+")

# Matched as exact tokens, never as substrings: GE writes SAT_GEMS for *spatial*
# saturation, so a substring test for "SAT" marks non-suppressed series as suppressed.
_FAT_SAT_OPTIONS = frozenset({"FS", "FATSAT", "FAT_SAT", "FSAT"})

_FAT_SAT = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|water excit|"
    r"\btirm\b|\bfatsup\b"
)
_T1 = re.compile(r"\bt1\b|\bt1w\b")
_T2 = re.compile(r"\bt2\b|\bt2w\b")
_PD = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

WEIGHTINGS = ("T1", "T2", "PD", "GRE", "UNK")

# TR/TE thresholds, used only when the description says nothing. Spin-echo convention:
# short TR is T1; long TR with long TE is T2; long TR with short TE is proton density.
_SHORT_TR_MS = 800.0
_LONG_TE_MS = 60.0


@dataclass(frozen=True)
class Acquisition:
    """What a series actually is, as opposed to what the CSV says it is."""

    weighting: str          # one of WEIGHTINGS
    fat_suppressed: bool
    fluid_sensitive: bool   # PD and T2 show fluid bright; T1 does not
    source: str             # "description" or "timing" -- which rule decided weighting


def _normalise(*values: str | None) -> str:
    """Lowercase, with separators turned into spaces so token tests can fire."""
    joined = " ".join(value for value in values if value)
    return _SEPARATORS.sub(" ", joined.lower())


def _as_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # NaN is not a reading


def is_fat_suppressed(description: str | None, sequence_name: str | None,
                      scan_options: str | list | None) -> bool:
    """True when a fat-cancelling preparation was applied.

    ``scan_options`` is matched token-wise; everything else by word-boundary regex on
    the separator-normalised description.
    """
    if _FAT_SAT.search(_normalise(description, sequence_name)):
        return True
    if scan_options is None:
        return False
    tokens = scan_options if isinstance(scan_options, (list, tuple)) else str(scan_options).split("\\")
    return any(str(token).strip().upper() in _FAT_SAT_OPTIONS for token in tokens)


def recover(
    description: str | None = None,
    sequence_name: str | None = None,
    scan_options: str | list | None = None,
    scanning_sequence: str | list | None = None,
    repetition_time=None,
    echo_time=None,
) -> Acquisition:
    """Read weighting and fat suppression out of one series' header fields.

    The description wins when it names a weighting, because a radiographer naming the
    sequence is a more direct statement than inferring it from timing. TR/TE decide only
    when the description is silent, and gradient-echo is reported as its own class rather
    than forced into the spin-echo TR/TE convention.
    """
    text = _normalise(description, sequence_name)
    says_t1, says_t2, says_pd = bool(_T1.search(text)), bool(_T2.search(text)), bool(_PD.search(text))

    weighting, source = None, "description"
    if says_pd:                       # "PD" is checked first: pd_tse_fs names both pd and t2
        weighting = "PD"
    elif says_t2 and not says_t1:
        weighting = "T2"
    elif says_t1 and not says_t2:
        weighting = "T1"

    if weighting is None:
        source = "timing"
        sequences = scanning_sequence or ""
        if not isinstance(sequences, str):
            sequences = "\\".join(str(value) for value in sequences)
        if "GR" in sequences.upper():
            weighting = "GRE"
        else:
            tr, te = _as_float(repetition_time), _as_float(echo_time)
            if tr is None:
                weighting = "UNK"
            elif tr < _SHORT_TR_MS:
                weighting = "T1"
            elif te is not None and te > _LONG_TE_MS:
                weighting = "T2"
            else:
                weighting = "PD"

    return Acquisition(
        weighting=weighting,
        fat_suppressed=is_fat_suppressed(description, sequence_name, scan_options),
        fluid_sensitive=weighting in ("PD", "T2"),
        source=source,
    )
