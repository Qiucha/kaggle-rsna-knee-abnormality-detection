#!/usr/bin/env python3
"""Compare the four pinned tower-size arms on identical gold OOF folds."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contrastive_learning.artifacts import sha256_file
from contrastive_learning.checkpoints import verified_best_epoch
from experiment_tools.gold_probe import (
    fast_auroc,
    load_gold_labels,
    paired_bootstrap,
    resolve_population,
    score_arm,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "configs/downstream/gold-probe-size-matched-holdout-v1.json"
OUTPUT = ROOT / "digest/tower-size-factorial-v1.json"
INACTIVE_NEW_DEFAULTS = {
    "declip_image_ss_weight",
    "declip_mv_weight",
    "declip_nn_weight",
    "declip_queue_size",
    "declip_text_mask_prob",
    "freeze_temperature",
    "max_inverse_temperature",
    "use_declip",
}
ALLOWED_DIFFERENCES = INACTIVE_NEW_DEFAULTS | {
    "image_model",
    "image_model_revision",
    "text_model",
    "text_model_revision",
    "output_dir",
}

ARMS = {
    "small_image_base_text": {
        "run": ROOT
        / "checkpoints/contrastive/both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026",
        "features": ROOT / "digest/lora-holdout-features.npz",
        "key": "unexposed_lora_6blk_r64/holdout-both-lora-depth6-r64-seed2026|uniform_mean_768",
        "image": "facebook/dinov2-small",
        "text": "intfloat/multilingual-e5-base",
    },
    "small_image_small_text": {
        "run": ROOT
        / "checkpoints/contrastive/tower-size-factorial-v1/dinov2-small-e5-small-depth6-r64-seed2026",
        "features": ROOT / "digest/tower-size-text-small-features.npz",
        "key": "text_small/dinov2-small-e5-small-depth6-r64-seed2026|uniform_mean_768",
        "image": "facebook/dinov2-small",
        "text": "intfloat/multilingual-e5-small",
    },
    "base_image_base_text": {
        "run": ROOT
        / "checkpoints/contrastive/tower-size-factorial-v1/dinov2-base-e5-base-depth6-r64-seed2026",
        "features": ROOT / "digest/tower-size-image-base-features.npz",
        "key": "image_base/dinov2-base-e5-base-depth6-r64-seed2026|uniform_mean_1536",
        "image": "facebook/dinov2-base",
        "text": "intfloat/multilingual-e5-base",
    },
    "base_image_small_text": {
        "run": ROOT
        / "checkpoints/contrastive/size-matched-holdout-v1/dinov2-base-e5-small-depth6-r64-seed2026",
        "features": ROOT / "digest/size-matched-holdout-features.npz",
        "key": "size_matched_lora_6blk_r64/dinov2-base-e5-small-depth6-r64-seed2026|uniform_mean_1536",
        "image": "facebook/dinov2-base",
        "text": "intfloat/multilingual-e5-small",
    },
}

CONTRASTS = {
    "text_small_minus_base_at_small_image": (
        "small_image_base_text",
        "small_image_small_text",
    ),
    "text_small_minus_base_at_base_image": (
        "base_image_base_text",
        "base_image_small_text",
    ),
    "image_base_minus_small_at_base_text": (
        "small_image_base_text",
        "base_image_base_text",
    ),
    "image_base_minus_small_at_small_text": (
        "small_image_small_text",
        "base_image_small_text",
    ),
}


def _history(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [row["epoch"] for row in rows] != list(range(1, 21)):
        raise ValueError(f"Incomplete 20-epoch run: {path}")
    return rows


def _interaction_bootstrap(gold: np.ndarray, scores: dict, settings: dict) -> dict:
    rng = np.random.default_rng(settings["seed"])
    values = []
    names = list(ARMS)
    for _ in range(settings["resamples"]):
        index = rng.integers(0, len(gold), len(gold))
        per_arm = {name: [] for name in names}
        for column in range(gold.shape[1]):
            target = gold[index, column]
            if np.unique(target).size < 2:
                continue
            for name in names:
                per_arm[name].append(fast_auroc(target, scores[name][index, column]))
        mean = {name: float(np.mean(values)) for name, values in per_arm.items()}
        values.append(
            (mean["base_image_small_text"] - mean["base_image_base_text"])
            - (mean["small_image_small_text"] - mean["small_image_base_text"])
        )
    return {
        "ci95": [float(v) for v in np.percentile(values, [2.5, 97.5])],
        "bootstrap_mean": float(np.mean(values)),
        "resamples": len(values),
        "unit": "study",
    }


def main() -> None:
    config = json.loads(PROBE.read_text())
    reference_config = json.loads(
        (ARMS["small_image_base_text"]["run"] / "config.json").read_text()
    )
    reference_active = {
        key: value
        for key, value in reference_config.items()
        if key not in ALLOWED_DIFFERENCES
    }
    uids, matrix = load_gold_labels(ROOT / config["gold_labels_csv"], config["labels"])
    records, gold, population = resolve_population(
        ROOT / config["cohort_manifest"], uids, matrix, ("excluded",)
    )
    expected = [record.study_uid for record in records]
    results = {}
    oof = {}
    for name, spec in ARMS.items():
        run_config = json.loads((spec["run"] / "config.json").read_text())
        active = {
            key: value
            for key, value in run_config.items()
            if key not in ALLOWED_DIFFERENCES
        }
        if (
            active != reference_active
            or run_config["image_model"] != spec["image"]
            or run_config["text_model"] != spec["text"]
        ):
            raise ValueError(
                f"{name}: active training config differs beyond the tower choice"
            )
        stored = np.load(spec["features"], allow_pickle=True)
        if list(stored["study_uids"]) != expected:
            raise ValueError(f"{name}: gold study order differs")
        features = stored[spec["key"]]
        if features.shape != (56, 768 if "small_image" in name else 1536):
            raise ValueError(f"{name}: unexpected feature shape {features.shape}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            scored = score_arm(features, gold, config["labels"], config["probe"])
        oof[name] = scored.pop("_oof_scores")
        rows = _history(spec["run"] / "metrics.jsonl")
        best = verified_best_epoch(
            spec["run"] / "metrics.jsonl", spec["run"] / "best.pt"
        )
        low = min(rows, key=lambda row: row["validation_loss"])
        results[name] = {
            "image": spec["image"],
            "text": spec["text"],
            "feature_dimensions": features.shape[1],
            "checkpoint": str(spec["run"] / "best.pt"),
            "checkpoint_sha256": sha256_file(spec["run"] / "best.pt"),
            "features": str(spec["features"]),
            "best_epoch": best["epoch"],
            "best_mean_r1": best["mean_r@1"],
            "best_image_to_text_r1": best["image_to_text_r@1"],
            "best_text_to_image_r1": best["text_to_image_r@1"],
            "best_image_to_text_r5": best["image_to_text_r@5"],
            "best_text_to_image_r5": best["text_to_image_r@5"],
            "best_image_to_text_r10": best["image_to_text_r@10"],
            "best_text_to_image_r10": best["text_to_image_r@10"],
            "minimum_validation_loss": low["validation_loss"],
            "minimum_validation_loss_epoch": low["epoch"],
            "final_validation_loss": rows[-1]["validation_loss"],
            "final_training_loss": rows[-1]["train_loss"],
            "gold_macro_auroc": scored["macro_auroc"],
            "gold_macro_auprc": scored["macro_auprc"],
            "gold_per_label": scored["per_label"],
        }
    if (
        abs(results["small_image_base_text"]["gold_macro_auroc"] - 0.7593615004455258)
        > 1e-9
    ):
        raise ValueError("Selected reference failed to reproduce")
    if (
        abs(results["base_image_small_text"]["gold_macro_auroc"] - 0.6968687165569286)
        > 1e-9
    ):
        raise ValueError("Two-tower swap failed to reproduce")
    contrasts = {}
    for title, (left, right) in CONTRASTS.items():
        contrasts[title] = {
            "left": left,
            "right": right,
            "point_difference": results[right]["gold_macro_auroc"]
            - results[left]["gold_macro_auroc"],
            "paired_bootstrap": paired_bootstrap(
                gold, config["labels"], oof[left], oof[right], config["bootstrap"]
            ),
        }
    interaction_point = (
        contrasts["text_small_minus_base_at_base_image"]["point_difference"]
        - contrasts["text_small_minus_base_at_small_image"]["point_difference"]
    )
    payload = {
        "schema_version": 1,
        "population": population,
        "labels": config["labels"],
        "probe": config["probe"],
        "bootstrap": config["bootstrap"],
        "arms": results,
        "contrasts": contrasts,
        "interaction": {
            "definition": "(E5-small minus E5-base at DINOv2-base) minus (E5-small minus E5-base at DINOv2-small)",
            "point_difference": interaction_point,
            **_interaction_bootstrap(gold, oof, config["bootstrap"]),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {name: value["gold_macro_auroc"] for name, value in results.items()},
            indent=2,
        )
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
