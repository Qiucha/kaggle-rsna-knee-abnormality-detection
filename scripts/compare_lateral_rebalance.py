#!/usr/bin/env python3
"""Paired LM-test and gold-58 deltas for the lateral/synovitis rebalancing arms vs u9-b1.

Both arms train the u9-b1 recipe with one change each: (A) one extra draw of every study
positive for Lateral Meniscus, Lateral OA or Synovitis, other labels masked on the extra
draws, plus fresh N(0, 0.1) noise on the two lateral soft targets; (B) 2x loss weight on
the three labels, renormalised to mean 1. "A ep30" reruns A with a 30-epoch cosine budget,
patience off and a divergence stop; "A ep50" resumes it from last.pt with a warm restart at 0.5x LR
over epochs 31-50. Every run is read from its own ``predictions.npz``
and must match run 5 study for study. Intervals are study bootstraps shared across runs
(:func:`lm_proxy.paired_intervals`), so each delta is paired. The run-5 seed replicate
gives the training-noise reference.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contrastive_learning.lm_proxy import (  # noqa: E402
    load_test_population,
    masked_macro_auroc,
    paired_intervals,
)
from contrastive_learning.submission import LABELS  # noqa: E402

FINETUNE = ROOT / "checkpoints/finetune"
BASE = "u9-b1"
RUNS = {  # key -> (run directory, title)
    "run5": (FINETUNE / "r4a-k8-omnirad-base-soft-336", "Run 5 (u3, seed 2026)"),
    "run5 seed2027": (FINETUNE / "r4a-k8-omnirad-base-soft-336-seed2027", "Run 5 (u3, seed 2027)"),
    BASE: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1", "u9-b1 (base)"),
    "oversample+noise": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise",
                         "A · oversample + lateral noise"),
    "loss weight 2x": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-w2", "B · 2× loss weight"),
    "oversample+noise ep30": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30",
                              "A · 30-epoch budget"),
    "oversample+noise ep50": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep50",
                              "A · extended to 50 (warm restart)"),
}
ARMS = ["oversample+noise", "loss weight 2x", "oversample+noise ep30", "oversample+noise ep50"]
TARGETS = ["Lateral Meniscus", "Lateral OA", "Synovitis"]
PAIRS = [("run5 seed2027", "run5"), ("oversample+noise", BASE), ("loss weight 2x", BASE),
         ("oversample+noise", "loss weight 2x"), ("oversample+noise ep30", BASE),
         ("oversample+noise ep30", "oversample+noise"), ("oversample+noise ep50", BASE),
         ("oversample+noise ep50", "oversample+noise ep30")]


def _paired(predictions, y, m):
    raw = paired_intervals(predictions, y, m)
    out = {}
    for first, second in PAIRS:
        if first not in predictions or second not in predictions:
            continue
        key = f"{first} - {second}"
        if key in raw:
            out[key] = raw[key]
        else:
            flipped = raw[f"{second} - {first}"]
            out[key] = {"difference": -flipped["difference"], "ci95": [-flipped["ci95"][1], -flipped["ci95"][0]]}
    return out


def _history(run_dir):
    path = run_dir / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    available = {k: v for k, v in RUNS.items() if (v[0] / "predictions.npz").exists()}
    missing = [k for k in ARMS if k not in available]
    saved = {}
    for name, (path, _) in available.items():
        with np.load(path / "predictions.npz", allow_pickle=False) as data:
            saved[name] = {key: data[key].copy() for key in data.files}
    base = saved["run5"]
    for name, data in saved.items():
        for key in ("lm_test_uids", "gold_uids", "gold_targets"):
            if not np.array_equal(data[key], base[key]):
                raise ValueError(f"{name}: {key} differs from run 5")
    config = json.loads((RUNS["run5"][0] / "config.json").read_text())
    _, targets, masks, identity = load_test_population(
        ROOT / config["cohort_manifest"], ROOT / config["lm_labels_csv"], ROOT / config["gold_labels_csv"])
    if identity["study_uids"] != base["lm_test_uids"].tolist():
        raise ValueError("LM-test population differs from saved predictions")
    gold_y = base["gold_targets"]
    splits = {"lm_test": (targets, masks), "gold": (gold_y, np.ones_like(gold_y))}
    result = {"point": {}, "paired": {}, "per_label": {}, "paired_target_labels": {},
              "bootstrap": {"resamples": 2000, "seed": 2030}, "base": BASE, "arms": ARMS,
              "target_labels": TARGETS, "missing_arms": missing}
    for split, (y, m) in splits.items():
        predictions = {name: data[split] for name, data in saved.items()}
        scores = {name: masked_macro_auroc(p, y, m) for name, p in predictions.items()}
        result["point"][split] = {name: s["macro_auroc"] for name, s in scores.items()}
        result["per_label"][split] = {name: {label: s["per_label"].get(label) for label in LABELS}
                                      for name, s in scores.items()}
        result["paired"][split] = _paired(predictions, y, m)
        # Per target label: mask every other label so the "macro" is that label alone.
        result["paired_target_labels"][split] = {}
        for label in TARGETS:
            only = np.zeros_like(m)
            only[:, LABELS.index(label)] = m[:, LABELS.index(label)]
            result["paired_target_labels"][split][label] = _paired(predictions, y, only)
    result["runs"] = []
    for name, (path, title) in available.items():
        report = json.loads((path / "report.json").read_text())
        run_config = json.loads((path / "config.json").read_text())
        history = _history(path)
        result["runs"].append({
            "key": name, "title": title, "dir": str(path.relative_to(ROOT)),
            "best_epoch": report["best_epoch"], "validation_masked_bce": report["validation_masked_bce"],
            "epochs_completed": len(history), "history": history,
            **{k: run_config[k] for k in ("oversample", "label_noise", "label_weights") if k in run_config},
            **{k: report[k] for k in ("oversample", "label_weights", "stopped") if k in report},
            **{k: run_config[k] for k in ("epochs", "patience", "divergence_stop", "resume_from") if k in run_config},
        })
    result["recipe"] = {k: json.loads((RUNS[BASE][0] / "config.json").read_text())[k]
                        for k in ("image_model", "image_unfreeze_last", "image_size", "windows_per_plane",
                                  "targets", "epochs", "patience", "seed")}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for split in splits:
        print(f"\n{split}")
        for name, value in result["point"][split].items():
            print(f"  {name:20s} {value:.4f}")
        for key, value in result["paired"][split].items():
            low, high = value["ci95"]
            print(f"  {key:40s} {value['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
        for label, pairs in result["paired_target_labels"][split].items():
            for key, value in pairs.items():
                if key.endswith(f" - {BASE}"):
                    low, high = value["ci95"]
                    print(f"  {label:18s} {key:32s} {value['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
    if missing:
        print(f"\nmissing arms: {missing}")


if __name__ == "__main__":
    main()
