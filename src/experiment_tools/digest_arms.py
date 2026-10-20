"""Helpers shared by the arm-comparison digest builders (next-model-pairs, codec-declip).

Each arm is one contrastive run under ``checkpoints/contrastive`` plus its 58-case LM heads under ``checkpoints/kaggle``
(MLP and linear heads, three head seeds). These functions read a run's metrics and selection history, re-score the LM-test
split with the saved heads, and format table cells.
"""

import json

import numpy as np
import torch

from contrastive_learning.artifacts import sha256_file
from contrastive_learning.lm_proxy import DEFAULTS, encoder_test_features
from contrastive_learning.submission import predict_head

SEEDS = (2026, 2027, 2028)


def f4(value):
    return "—" if value is None else f"{value:.4f}"


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def delta(interval):
    if interval is None:
        return "—"
    lo, hi = interval["ci95"]
    return f"{interval['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def contrastive(run_dir):
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    best = max(rows, key=lambda r: (r["mean_r@1"], -r["validation_loss"], -r["epoch"]))
    telemetry = run_dir.parent / f"{run_dir.name}.telemetry.json"
    record = json.loads(telemetry.read_text()) if telemetry.exists() else {}
    return {
        "epochs": len(rows),
        "selected_epoch": best["epoch"],
        "train_loss": best["train_loss"],
        "validation_loss": best["validation_loss"],
        "validation_loss_min": min(r["validation_loss"] for r in rows),
        "validation_loss_final": rows[-1]["validation_loss"],
        "image_to_text_r1": best["image_to_text_r@1"],
        "text_to_image_r1": best["text_to_image_r@1"],
        "mean_r1": best["mean_r@1"],
        "elapsed_hours": record["elapsed_seconds"] / 3600 if "elapsed_seconds" in record else None,
        "peak_allocated_gb": record["peak_allocated_bytes"] / 1e9 if "peak_allocated_bytes" in record else None,
        "history": rows,
    }


def head_histories(head_dir, kind):
    return [json.loads((head_dir / f"{kind}-seed{seed}-history.json").read_text()) for seed in SEEDS]


def lm_test_predictions(run_dir, head_dir, kind):
    checkpoint = run_dir / "best.pt"
    features, targets, masks, population = encoder_test_features(
        checkpoint, sha256_file(checkpoint), device="cpu", batch_size=8,
        lm_labels_csv=DEFAULTS["lm_labels_csv"], gold_labels_csv=DEFAULTS["gold_labels_csv"],
        cache_dir=DEFAULTS["cache_dir"], image_cache_dir=DEFAULTS["image_cache_dir"],
    )
    heads = [torch.load(head_dir / f"{kind}-seed{seed}.pt", map_location="cpu", weights_only=False) for seed in SEEDS]
    return np.mean([predict_head(features, head) for head in heads], axis=0), targets, masks, population
