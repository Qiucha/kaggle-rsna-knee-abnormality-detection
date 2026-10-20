"""Frozen K8 features from unadapted OmniRad-base and its CLIP-adapted image tower.

Question: does plain CLIP contrastive learning (``omnirad-clip-run5-v1``) improve a
radiology-pretrained backbone? Both towers read the *same* decoded windows in one
pass, so the only difference between the two feature matrices is the encoder weights.
The feature is run 5's own: CLS + patch-mean per 3-slice window, masked mean over the
8 windows of a plane, then over present planes (``submission.window_features``).

Identity checks run first and abort the extraction if they fail:

* the baseline tower rebuilt through ``load_encoder(baseline=True)`` equals a fresh
  pinned OmniRad load, so "unadapted" really is OmniRad and not ``facebook/dinov2-base``;
* the CLIP tower equals the baseline in the embeddings and blocks 0..(12-unfreeze-1)
  and differs in the unfrozen blocks, so the CLIP checkpoint is the adaptation of that
  baseline and nothing else.

Code, configs and checkpoints are read from this repository; only ``--output`` is written.

Usage: extract_omnirad_probe_features.py --output DIR [--limit N] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contrastive_learning.artifacts import sha256_file, write_atomic, write_json  # noqa: E402
from contrastive_learning.labels import read_lm_labels  # noqa: E402
from contrastive_learning.lm_proxy import lm_test_population  # noqa: E402
from contrastive_learning.lm_supervision import load_population, state_digest  # noqa: E402
from contrastive_learning.manifest import read_cohort_manifest  # noqa: E402
from contrastive_learning.submission import (  # noqa: E402
    LABELS,
    UID,
    _WindowReads,
    load_encoder,
    validate_ids,
    window_features,
)
from contrastive_learning.timm_dinov2 import load_timm_dinov2  # noqa: E402

CLIP_CHECKPOINT = (
    "checkpoints/contrastive/omnirad-clip-run5-v1/"
    "omnirad-base-e5-base-clip-k8-336-u6-bs32-seed2026/best.pt"
)
CLIP_SHA256 = "574b5048ebd7d9e74ca748fa84f9e17b859af91fc1caa7183716a67e1ab6f30a"
CONFIG = {
    "cohort_manifest": "data/processed/contrastive-cohort-gold-holdout-v1.jsonl",
    "manifest_sha256": "d104337696a19d52d12480c1cfaae30c937916102e36a67e3124755fa20123f5",
    "lm_labels_csv": "data/lm_label/lm_labels_kaggle.csv",
    "gold_labels_csv": "data/gold_label.csv",
    "gold_splits": ["excluded", "test"],
    "expected_gold_studies": 58,
}
CHUNK = 64
ARMS = ("base", "clip")
HEADER_TAGS = ("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
               "StationName", "InstitutionName", "SoftwareVersions")


def check_identity(clip, base, run_config, unfreeze: int) -> dict:
    """Abort unless base is pinned OmniRad and clip differs from it only where unfrozen."""
    fresh = load_timm_dinov2(
        run_config.image_model, run_config.image_model_revision,
        base_model=run_config.image_model_base,
        base_revision=run_config.image_model_base_revision, local_files_only=True,
    ).eval()
    fresh_state = fresh.state_dict()
    if state_digest(fresh) != state_digest(base):
        raise AssertionError("Baseline tower is not the pinned OmniRad weights")
    depth = len(base.encoder.layer)
    frozen_blocks = tuple(range(depth - unfreeze))
    changed, unchanged = [], []
    for name, value in clip.state_dict().items():
        (unchanged if torch.equal(value, fresh_state[name]) else changed).append(name)
    leaked = [n for n in changed if n.startswith("embeddings.")
              or any(n.startswith(f"encoder.layer.{i}.") for i in frozen_blocks)]
    if leaked:
        raise AssertionError(f"CLIP tower differs from OmniRad in frozen parameters: {leaked[:5]}")
    if not changed:
        raise AssertionError("CLIP tower is byte-identical to OmniRad; nothing was adapted")
    return {
        "base_equals_fresh_pinned_omnirad": True,
        "frozen_blocks_identical": list(frozen_blocks),
        "changed_tensors": len(changed),
        "unchanged_tensors": len(unchanged),
        "changed_blocks": sorted({int(n.split(".")[2]) for n in changed if n.startswith("encoder.layer.")}),
        "layernorm_changed": any(n.startswith("layernorm.") for n in changed),
        "parameter_relative_shift": {
            "mean": float(np.mean([
                (clip.state_dict()[n].float() - fresh_state[n].float()).norm().item()
                / max(fresh_state[n].float().norm().item(), 1e-12) for n in changed
            ])),
        },
    }


def read_header(record) -> dict:
    """Acquisition fields from one DICOM header of the study (no pixels)."""
    import pydicom

    for plane in ("Sagittal", "Coronal", "Axial"):
        series = record.slots.get(plane)
        if series is None:
            continue
        try:
            files = sorted(p for p in Path(series).iterdir() if p.suffix == ".dcm")
            ds = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True, force=True)
        except Exception:  # unreadable header just means no metadata for this study
            continue
        out = {tag: str(getattr(ds, tag, "")) for tag in HEADER_TAGS}
        out["plane_used"] = plane
        return out
    return {}


def _savez(**arrays):
    """A write_atomic writer that closes its handle, so the bytes are flushed before the rename."""
    def write(target: Path) -> None:
        with target.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
    return write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, help="first N studies per split (smoke test)")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--no-headers", action="store_true")
    args = parser.parse_args()
    out = args.output
    (out / "parts").mkdir(parents=True, exist_ok=True)

    checkpoint = ROOT / CLIP_CHECKPOINT
    sha = sha256_file(checkpoint)
    if sha != CLIP_SHA256:
        raise ValueError(f"CLIP checkpoint changed: {sha}")
    clip, run_config, _ = load_encoder(checkpoint)
    base, base_config, _ = load_encoder(checkpoint, baseline=True)
    for tower in (clip, base):
        tower.requires_grad_(False).eval()
    identity = check_identity(clip, base, run_config, run_config.image_unfreeze_last)
    print(json.dumps(identity, indent=2), flush=True)
    write_json(out / "identity.json", identity)
    if args.identity_only:
        return

    groups, targets, masks, population = load_population(CONFIG, ROOT)
    records, _ = read_cohort_manifest(ROOT / CONFIG["cohort_manifest"])
    table = read_lm_labels(ROOT / CONFIG["lm_labels_csv"], LABELS)
    gold_frame = pd.read_csv(ROOT / CONFIG["gold_labels_csv"], dtype={UID: str})
    test_rows, test_y, test_m = lm_test_population(records, table, validate_ids(gold_frame))
    splits = {
        "train": (groups["train"], targets["train"], masks["train"]),
        "validation": (groups["validation"], targets["validation"], masks["validation"]),
        "test": (test_rows, test_y, test_m),
        "gold": (groups["gold"], targets["gold"], masks["gold"]),
    }
    if args.limit:
        splits = {k: (r[: args.limit], y[: args.limit], m[: args.limit]) for k, (r, y, m) in splits.items()}
    population_arrays = {}
    for split, (rows, y, m) in splits.items():
        population_arrays[f"{split}_uids"] = np.asarray([r.study_uid for r in rows])
        population_arrays[f"{split}_targets"] = y
        population_arrays[f"{split}_masks"] = m
    write_atomic(out / "population.npz",
                 _savez(labels=np.asarray(LABELS), **population_arrays))

    device = args.device
    clip.to(device)
    base.to(device)
    image_kwargs = run_config.image_kwargs
    if base_config.image_kwargs != image_kwargs:
        raise AssertionError("Arms must share one preprocessing")
    for split, (rows, _, _) in splits.items():
        for start in range(0, len(rows), CHUNK):
            part = out / "parts" / f"{split}-{start:05d}.npz"
            chunk = rows[start : start + CHUNK]
            if part.exists():
                try:
                    with np.load(part, allow_pickle=False) as cached:
                        if cached["base"].shape[0] == len(chunk) and cached["clip"].shape[0] == len(chunk):
                            continue
                except Exception:  # truncated by a hard stop; rebuild
                    pass
            loader = torch.utils.data.DataLoader(
                _WindowReads(chunk, image_kwargs), batch_size=args.batch_size,
                num_workers=min(4, os.cpu_count() or 1),
            )
            parts = {arm: [] for arm in ARMS}
            for images, valid in loader:
                images, valid = images.to(device), valid.to(device)
                for arm, tower in zip(ARMS, (base, clip)):
                    parts[arm].append(window_features(tower, images, valid).cpu().numpy())
            arrays = {arm: np.concatenate(v) for arm, v in parts.items()}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite features")
            write_atomic(part, _savez(**arrays))
            print(f"{split}: {min(start + CHUNK, len(rows))}/{len(rows)}", flush=True)

    for arm, tower in zip(ARMS, (base, clip)):
        if any(p.requires_grad for p in tower.parameters()):
            raise AssertionError("Tower became trainable")
    assembled = {}
    for split, (rows, _, _) in splits.items():
        for arm in ARMS:
            assembled[f"{split}_{arm}"] = np.concatenate([
                np.load(out / "parts" / f"{split}-{start:05d}.npz")[arm]
                for start in range(0, len(rows), CHUNK)
            ])
            if len(assembled[f"{split}_{arm}"]) != len(rows):
                raise ValueError(f"Feature count mismatch for {split}/{arm}")
    write_atomic(out / "features.npz", _savez(**assembled))
    if not (out / "features.npz").stat().st_size:
        raise ValueError("features.npz is empty")

    if not args.no_headers:
        meta = {r.study_uid: {**read_header(r), "laterality": r.laterality, "split": split}
                for split, (rows, _, _) in splits.items() for r in rows}
        write_json(out / "acquisition.json", meta)
    write_json(out / "run.json", {
        "clip_checkpoint": CLIP_CHECKPOINT, "clip_sha256": sha, "image_kwargs": image_kwargs,
        "code_root": str(ROOT), "limit": args.limit,
        "studies": {s: len(r) for s, (r, _, _) in splits.items()},
    })
    print("done", flush=True)


if __name__ == "__main__":
    main()
