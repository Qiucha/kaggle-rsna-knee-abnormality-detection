"""Hardlink cache entries from absolute-path keys to ``<study>/<series>`` keys (ADR 0001).

Image and window cache names used to hash each slot's absolute path under
``/mnt/d/train_series``. They now hash the slot identity, so moving the DICOM tree
no longer orphans them. This links every existing entry under its new name, leaving
the old name in place: it costs no space, it can be undone by deleting the new
names, and code that predates the change still finds its entries.

The settings each entry was written with are not recorded, so every settings
combination found in ``configs/`` and ``checkpoints/`` is tried and the matches
counted. An entry no combination explains is reported, never guessed at; a miss
only costs a rebuild.

Usage: migrate_cache_keys.py [--series-root DIR] [--apply] CACHE_DIR...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning import paths  # noqa: E402
from contrastive_learning.artifacts import sha256_text  # noqa: E402
from contrastive_learning.data import CACHE_FORMAT_VERSION, LEGACY_SERIES_ROOT  # noqa: E402
from contrastive_learning.manifest import read_manifest_rows  # noqa: E402
from contrastive_learning.preprocessing import PLANES, PREPROCESSING_VERSION, series_fingerprint  # noqa: E402

SETTING_KEYS = ("image_size", "crop_mm", "slices_per_plane")


def candidate_settings() -> tuple[list[dict], list[int]]:
    """Every distinct image setting combination, and window count, any config records."""
    found, windows = {}, set()

    def visit(value):
        if isinstance(value, dict):
            if all(key in value for key in SETTING_KEYS):
                band = value.get("slice_band") or [value.get("slice_band_low"), value.get("slice_band_high")]
                if None not in band:
                    kwargs = {key: value[key] for key in SETTING_KEYS}
                    kwargs["slice_band"] = [float(band[0]), float(band[1])]
                    found[json.dumps(kwargs, sort_keys=True)] = kwargs
            if isinstance(value.get("windows_per_plane"), int):
                windows.add(value["windows_per_plane"])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for top in ("configs", "checkpoints"):
        for path in (ROOT / top).rglob("*.json"):
            if path.stat().st_size < 5_000_000:
                try:
                    visit(json.loads(path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
    return list(found.values()), sorted(windows)


def studies(series_root: Path) -> list[dict]:
    """One row per study with all three planes, across every cohort manifest."""
    by_uid = {}
    for manifest in sorted((ROOT / "data/processed").glob("*.jsonl")):
        for row in read_manifest_rows(manifest, series_root=series_root):
            if set(row["slots"]) == set(PLANES):
                by_uid.setdefault(row["study_uid"], row)
    return list(by_uid.values())


def names(row, payload_for) -> tuple[str, str]:
    """The (legacy, current) cache file names of one study under one setting."""
    def name(slot_value):
        slots = {plane: slot_value(path) for plane, path in sorted(row["slots"].items())}
        return f"{row['study_uid']}-{sha256_text(json.dumps(payload_for(row, slots), sort_keys=True))[:12]}.pt"

    return (name(lambda path: f"{LEGACY_SERIES_ROOT}/{path.parent.name}/{path.name}"),
            name(lambda path: f"{path.parent.name}/{path.name}"))


def image_payload(kwargs, fingerprints):
    return lambda row, slots: {
        "version": CACHE_FORMAT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "image_kwargs": kwargs,
        "laterality": row["laterality"],
        "slots": slots,
        "series_fingerprints": fingerprints[row["study_uid"]],
    }


def window_payload(kwargs, windows):
    settings = {"preprocessing_version": PREPROCESSING_VERSION, "windows": windows,
                **{key: value for key, value in kwargs.items() if key != "slices_per_plane"}}
    return lambda row, slots: {**settings, "laterality": row["laterality"],
                               "slots": {plane: slots[plane] for plane in PLANES}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dirs", nargs="+", type=Path)
    parser.add_argument("--series-root", type=Path, help="default: this host's train_series")
    parser.add_argument("--apply", action="store_true", help="create the links; otherwise only count")
    args = parser.parse_args()

    series_root = args.series_root or paths.host_series_root()
    rows = studies(series_root)
    settings, window_counts = candidate_settings()
    print(f"{len(rows)} studies, {len(settings)} image settings, windows {window_counts}")
    fingerprints = {}

    for cache_dir in args.cache_dirs:
        existing = {entry.name for entry in os.scandir(cache_dir) if entry.name.endswith(".pt")}
        window = "window-cache" in cache_dir.name
        if not window and not fingerprints:
            fingerprints = {row["study_uid"]: {plane: series_fingerprint(path)
                                               for plane, path in sorted(row["slots"].items())}
                            for row in rows}
        payloads = ([window_payload(kwargs, count) for kwargs in settings for count in window_counts]
                    if window else [image_payload(kwargs, fingerprints) for kwargs in settings])
        mapping = {}
        for payload_for in payloads:
            for row in rows:
                old, new = names(row, payload_for)
                if old in existing:
                    mapping[old] = new
        linked = 0
        for old, new in mapping.items():
            if args.apply and not (cache_dir / new).exists():
                os.link(cache_dir / old, cache_dir / new)
                linked += 1
        unexplained = existing - set(mapping) - set(mapping.values())
        print(f"{cache_dir}: {len(existing)} entries, {len(mapping)} matched, "
              f"{linked} linked, {len(unexplained)} unexplained")


if __name__ == "__main__":
    main()
