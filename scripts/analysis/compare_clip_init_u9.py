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

Reads: ``predictions.npz``, ``report.json``, ``config.json`` and ``history.jsonl`` of each run under
``checkpoints/finetune``, and the cohort files named in run 5's config.
Writes: only ``--output``.
Run: ``.venv/bin/python scripts/analysis/compare_clip_init_u9.py --output FILE.json``
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiment_tools.paired_runs import (  # noqa: E402
    load_predictions,
    pick,
    print_report,
    run_records,
    score_splits,
    split_populations,
)

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


def _ranks(p):
    return np.column_stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])])


def run_extras(report, run_config):
    return {**pick(report, ("stopped",)),
            **pick(run_config, ("epochs", "patience", "divergence_stop", "init_checkpoint",
                                "init_unfreeze_last", "unfreeze_last"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = load_predictions(RUNS)
    populations = split_populations(ROOT, RUNS, saved)
    predictions = {split: {name: data[split] for name, data in saved.items()} for split in populations}
    for split_predictions in predictions.values():
        for name, (members, _) in ENSEMBLES.items():
            split_predictions[name] = np.mean([_ranks(split_predictions[k]) for k in members], axis=0)
    result = {**score_splits(predictions, populations, PAIRS, TARGETS),
              "bootstrap": {"resamples": 2000, "seed": 2030}, "base": BASE, "reference": REF, "arm": ARM,
              "shallow": SHALLOW, "target_labels": TARGETS, "ensemble_rule": "per-label rank mean",
              "ensembles": [{"key": k, "title": t, "members": ms} for k, (ms, t) in ENSEMBLES.items()]}
    result["runs"] = run_records(ROOT, RUNS, run_extras)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_report(result, label_pair=lambda key: key in (f"{ARM} - {REF}", f"{SWAP} - {ENS4}", f"{DEEP} - {ENS4}"))


if __name__ == "__main__":
    main()
