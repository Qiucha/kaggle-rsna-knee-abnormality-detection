"""LM-verdict AUROC on the locked cohort test split, reported for any exported bundle.

The test split is used by no contrastive checkpoint selection, no LM-head fitting or
epoch selection, and no gold-head fitting, so every bundle can be scored on it with
one metric. It ranked all three early Kaggle submissions in public-score order; see
``digest/kaggle-submission-pipeline.md``. Heads trained on LM verdicts are scored
against the same kind of target, so the metric favors them over heads trained on other
labels. It is most comparable between encoders sharing one LM-head recipe.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from .artifacts import read_json, sha256_file, write_atomic, write_json
from .labels import read_lm_labels
from .manifest import read_cohort_manifest
from .submission import LABELS, UID, extract_features, load_encoder, predict_head, validate_ids

METRIC = "lm_test_macro_auroc"
FEATURE = "uniform_mean_cls_patch_v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "lm_labels_csv": ROOT / "data/lm_label/lm_labels_kaggle.csv",
    "gold_labels_csv": ROOT / "data/cleaned.csv",
    "cache_dir": ROOT / "checkpoints/kaggle/lm-proxy-cache",
    "image_cache_dir": ROOT / "checkpoints/contrastive/full-multilingual-v1/image-cache",
}


def lm_test_population(records, table, gold_ids):
    """Non-gold locked-test studies with at least one LM YES/NO verdict.

    Gold studies and every study sharing a gold patient group are removed, so the
    population never overlaps the gold benchmark.
    """
    gold_ids = set(gold_ids)
    gold_groups = {
        r.patient_group_id
        for r in records
        if r.study_uid in gold_ids and r.patient_group_id is not None
    }
    rows, targets, masks = [], [], []
    for record in records:
        if (
            record.split != "test"
            or record.study_uid in gold_ids
            or (record.patient_group_id is not None and record.patient_group_id in gold_groups)
        ):
            continue
        value, mask = table.get(record.study_uid)
        mask = mask.numpy()
        if mask.sum() == 0:
            continue
        rows.append(record)
        target = (value.numpy() + 1) / 2
        target[mask == 0] = 0
        targets.append(target)
        masks.append(mask)
    if not rows:
        raise ValueError("Locked test split has no LM-labelled non-gold studies")
    return rows, np.stack(targets).astype(np.float32), np.stack(masks).astype(np.float32)


def _auroc(scores, positives):
    ranks = rankdata(scores)
    n_pos = positives.sum()
    n_neg = len(positives) - n_pos
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def masked_macro_auroc(predictions, targets, masks):
    """Macro AUROC over labels with both classes observed; UNK cells are ignored."""
    per_label = {}
    for j, label in enumerate(LABELS):
        keep = masks[:, j] > 0
        positives = targets[keep, j] > 0.5
        if 0 < positives.sum() < len(positives):
            per_label[label] = float(_auroc(predictions[keep, j], positives))
        else:
            per_label[label] = None
    scored = [value for value in per_label.values() if value is not None]
    if not scored:
        raise ValueError("No label has both LM classes observed")
    return {"macro_auroc": float(np.mean(scored)), "labels_scored": len(scored), "per_label": per_label}


def paired_intervals(predictions, targets, masks, *, resamples=2000, seed=2030):
    """Study-bootstrap 95% intervals for every pair, sharing the same resamples."""
    rng = np.random.default_rng(seed)
    names = list(predictions)
    draws = {name: [] for name in names}
    for _ in range(resamples):
        index = rng.integers(0, len(targets), len(targets))
        for name in names:
            draws[name].append(
                masked_macro_auroc(predictions[name][index], targets[index], masks[index])["macro_auroc"]
            )
    point = {name: masked_macro_auroc(p, targets, masks)["macro_auroc"] for name, p in predictions.items()}
    result = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            delta = np.asarray(draws[first]) - np.asarray(draws[second])
            low, high = np.percentile(delta, [2.5, 97.5])
            result[f"{first} - {second}"] = {
                "difference": point[first] - point[second],
                "ci95": [float(low), float(high)],
            }
    return result


def run_checkpoints(run_dir):
    """Map each bundle member to its encoder checkpoint, verified by checksum."""
    config = read_json(run_dir / "benchmark.json")["config"]
    paths = config["checkpoints"] if "checkpoints" in config else [config["checkpoint"]]
    by_sha = {sha256_file(ROOT / p): ROOT / p for p in paths}
    members = read_json(run_dir / "bundle" / "bundle.json")["members"]
    pairs = []
    for member in members:
        if member["checkpoint_sha256"] not in by_sha:
            raise ValueError(f"No configured checkpoint matches bundle member {member['file']}")
        pairs.append((by_sha[member["checkpoint_sha256"]], run_dir / "bundle" / member["file"], member["checkpoint_sha256"]))
    return pairs


def load_test_population(manifest_path, lm_labels_csv, gold_labels_csv):
    records, metadata = read_cohort_manifest(manifest_path)
    gold_frame = pd.read_csv(gold_labels_csv, dtype={UID: str})
    rows, targets, masks = lm_test_population(
        records, read_lm_labels(lm_labels_csv, LABELS), validate_ids(gold_frame)
    )
    identity = {
        "manifest_sha256": metadata["manifest_sha256"],
        "lm_labels_sha256": sha256_file(lm_labels_csv),
        "gold_labels_sha256": sha256_file(gold_labels_csv),
        "study_uids": [r.study_uid for r in rows],
    }
    return rows, targets, masks, identity


def encoder_test_features(checkpoint, checkpoint_sha, *, device, batch_size, lm_labels_csv, gold_labels_csv, cache_dir, image_cache_dir):
    """Frozen-encoder features on that encoder's own locked test split, cached by checksum."""
    backbone, run_config, _ = load_encoder(checkpoint)
    manifest = Path(run_config.cohort_manifest)
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    rows, targets, masks, population = load_test_population(manifest, lm_labels_csv, gold_labels_csv)
    identity = json.loads(json.dumps({
        "checkpoint_sha256": checkpoint_sha,
        "feature": FEATURE,
        "image_kwargs": run_config.image_kwargs,
        "manifest_sha256": population["manifest_sha256"],
        "study_uids": population["study_uids"],
    }, sort_keys=True))
    cache = Path(cache_dir) / f"{checkpoint_sha[:16]}-test.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as cached:
            if json.loads(str(cached["identity"])) != identity:
                raise ValueError(f"Feature cache identity mismatch: {cache}")
            features = cached["features"]
    else:
        backbone.requires_grad_(False).eval()
        parts = []
        for start in range(0, len(rows), 32):
            parts.append(extract_features(backbone, rows[start : start + 32], run_config.image_kwargs, device, batch_size, image_cache_dir))
            print(f"lm-proxy features {checkpoint_sha[:12]}: {min(start + 32, len(rows))}/{len(rows)}", flush=True)
        features = np.concatenate(parts)
        cache.parent.mkdir(parents=True, exist_ok=True)

        def save(temporary):
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, identity=json.dumps(identity, sort_keys=True), features=features)

        write_atomic(cache, save)
    return features, targets, masks, population


def score_run(run_dir, *, device="cpu", batch_size=8, **paths):
    """Write ``<run_dir>/lm-proxy.json`` and return (report, predictions, targets, masks)."""
    paths = {**DEFAULTS, **paths}
    run_dir = Path(run_dir)
    member_predictions, population, targets, masks = [], None, None, None
    members = run_checkpoints(run_dir)
    for checkpoint, member_path, checkpoint_sha in members:
        features, y, m, member_population = encoder_test_features(
            checkpoint, checkpoint_sha, device=device, batch_size=batch_size,
            lm_labels_csv=paths["lm_labels_csv"], gold_labels_csv=paths["gold_labels_csv"],
            cache_dir=paths["cache_dir"], image_cache_dir=paths["image_cache_dir"],
        )
        if population is not None and member_population != population:
            raise ValueError("Ensemble members have different locked test populations")
        population, targets, masks = member_population, y, m
        head = torch.load(member_path, map_location="cpu", weights_only=False)["head"]
        member_predictions.append(predict_head(features, head))
    predictions = np.mean(member_predictions, axis=0)
    report = {
        "metric": METRIC,
        "definition": "masked macro AUROC vs LM YES/NO verdicts on non-gold locked-test studies; UNK ignored",
        **masked_macro_auroc(predictions, targets, masks),
        "studies": len(targets),
        "observed": masks.sum(0).astype(int).tolist(),
        "positive": (targets * masks).sum(0).astype(int).tolist(),
        "population_identity": {k: v for k, v in population.items() if k != "study_uids"},
        "study_uids_sha256": hashlib.sha256("\n".join(population["study_uids"]).encode()).hexdigest(),
        "members": [sha for _, _, sha in members],
        "caveat": "Favors heads trained on LM verdicts over heads trained on other labels; compare encoders under one head recipe.",
    }
    write_json(run_dir / "lm-proxy.json", report)
    return report, predictions, targets, masks
