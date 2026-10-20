#!/usr/bin/env python3
"""Paired LM-test and gold-58 deltas for the CLIP-init u9 run, as a single model and as an ensemble member.

The run copies A30 (u9-b1 + lateral oversample and target noise, cosine over 30 epochs, divergence stop) and
changes only the starting weights: the CLIP-adapted OmniRad-base (omnirad-clip-run5-v1 u6 best) instead of the
pretrained one. A30 is the one-factor reference; the shallow from-clip run (run 5 recipe, u3, same CLIP init) is
the ensemble member it was built to replace, and the run-5 seed replicate gives the training-noise floor.
Ensembles are per-label rank means of the members' saved predictions, the submitted bundle's rule; the submitted
4-member ensemble (A30 + u9-b1 + from-clip + DINOv2-S 336, 0.894 public) is their reference.
Every run is read from its own ``predictions.npz`` and must match run 5 study for study. Intervals are study
bootstraps shared across all entries (:func:`lm_proxy.paired_intervals`), so each delta is paired.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contrastive_learning.lm_proxy import (  # noqa: E402
    load_test_population,
    masked_macro_auroc,
    paired_intervals,
)
from contrastive_learning.submission import LABELS  # noqa: E402

FINETUNE = ROOT / "checkpoints/finetune"
BASE, REF, ARM, SHALLOW = "u9-b1", "A30", "CLIP-init u9", "from-clip u3"
RUNS = {  # key -> (run directory, title)
    "run5": (FINETUNE / "r4a-k8-omnirad-base-soft-336", "Run 5 (u3, seed 2026)"),
    "run5 seed2027": (FINETUNE / "r4a-k8-omnirad-base-soft-336-seed2027", "Run 5 (u3, seed 2027)"),
    BASE: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1", "u9-b1"),
    REF: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30", "A30 · pretrained init"),
    SHALLOW: (FINETUNE / "r4a-k8-omnirad-base-soft-336-from-clip", "From-clip · run-5 recipe, u3"),
    ARM: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-from-clip", "A30 recipe · CLIP init"),
    "dS336": (FINETUNE / "r4a-k8-dinov2-small-soft-336", "DINOv2-small soft 336"),
}
ENS4, SWAP, DEEP = "ens4 (submitted)", "ens4 swap", "deep three"
ENSEMBLES = {  # key -> (members, title)
    ENS4: ([REF, BASE, SHALLOW, "dS336"], "A30 + u9-b1 + from-clip u3 + DINOv2-S (submitted, 0.894)"),
    SWAP: ([REF, BASE, ARM, "dS336"], "A30 + u9-b1 + CLIP-init u9 + DINOv2-S"),
    DEEP: ([REF, BASE, ARM], "A30 + u9-b1 + CLIP-init u9"),
}
# A30's oversampled labels; the shallow members dilute gold Lateral Meniscus in ensembles.
TARGETS = ["Lateral Meniscus", "Lateral OA", "Synovitis"]
PAIRS = [("run5 seed2027", "run5"), (ARM, REF), (ARM, SHALLOW), (REF, BASE),
         (SWAP, ENS4), (DEEP, ENS4), (ENS4, REF)]


def _ranks(p):
    return np.column_stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])])


def _paired(predictions, y, m):
    raw = paired_intervals(predictions, y, m)
    out = {}
    for first, second in PAIRS:
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
    saved = {}
    for name, (path, _) in RUNS.items():
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
              "bootstrap": {"resamples": 2000, "seed": 2030}, "base": BASE, "reference": REF, "arm": ARM,
              "shallow": SHALLOW, "target_labels": TARGETS, "ensemble_rule": "per-label rank mean",
              "ensembles": [{"key": k, "title": t, "members": ms} for k, (ms, t) in ENSEMBLES.items()]}
    for split, (y, m) in splits.items():
        predictions = {name: data[split] for name, data in saved.items()}
        for name, (members, _) in ENSEMBLES.items():
            predictions[name] = np.mean([_ranks(predictions[k]) for k in members], axis=0)
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
    for name, (path, title) in RUNS.items():
        report = json.loads((path / "report.json").read_text())
        run_config = json.loads((path / "config.json").read_text())
        history = _history(path)
        result["runs"].append({
            "key": name, "title": title, "dir": str(path.relative_to(ROOT)),
            "best_epoch": report["best_epoch"], "validation_masked_bce": report["validation_masked_bce"],
            "epochs_completed": len(history), "history": history, "seconds": report.get("seconds"),
            **{k: report[k] for k in ("stopped",) if k in report},
            **{k: run_config[k] for k in ("epochs", "patience", "divergence_stop", "init_checkpoint",
                                          "init_unfreeze_last", "unfreeze_last") if k in run_config},
        })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for split in splits:
        print(f"\n{split}")
        for name, value in result["point"][split].items():
            print(f"  {name:28s} {value:.4f}")
        for key, value in result["paired"][split].items():
            low, high = value["ci95"]
            print(f"  {key:50s} {value['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
        for label, pairs in result["paired_target_labels"][split].items():
            for key in (f"{ARM} - {REF}", f"{SWAP} - {ENS4}", f"{DEEP} - {ENS4}"):
                low, high = pairs[key]["ci95"]
                print(f"  {label:18s} {key:50s} {pairs[key]['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
