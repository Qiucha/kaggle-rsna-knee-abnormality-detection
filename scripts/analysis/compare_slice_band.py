#!/usr/bin/env python3
"""Paired LM-test and gold-58 deltas for the wide slice-band trial vs its matched run.

The trial copies the 30-epoch lateral oversample + noise arm (A30: u9-b1, one extra draw of every study
positive for Lateral Meniscus, Lateral OA or Synovitis, N(0, 0.1) lateral target noise, cosine over 30
epochs, patience off, divergence stop) and changes only the slice band, [0.2, 0.8] -> [0.05, 0.95], in
training and evaluation alike. A30 is the one-factor reference; u9-b1 and the run-5 seed replicate give
context and the training-noise floor.
Every run is read from its own ``predictions.npz`` and must match run 5 study for study. Intervals are
study bootstraps shared across runs (:func:`lm_proxy.paired_intervals`), so each delta is paired.

Reads: ``predictions.npz``, ``report.json``, ``config.json`` and ``history.jsonl`` of each run under
``checkpoints/finetune``, and the cohort files named in run 5's config.
Writes: only ``--output``.
Run: ``.venv/bin/python scripts/analysis/compare_slice_band.py --output FILE.json``
"""

import argparse
import json
import sys
from pathlib import Path

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
BASE, REF, AUG = "u9-b1", "oversample+noise ep30", "oversample+noise ep30 band05"
RUNS = {  # key -> (run directory, title)
    "run5": (FINETUNE / "r4a-k8-omnirad-base-soft-336", "Run 5 (u3, seed 2026)"),
    "run5 seed2027": (FINETUNE / "r4a-k8-omnirad-base-soft-336-seed2027", "Run 5 (u3, seed 2027)"),
    BASE: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1", "u9-b1"),
    REF: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30", "A30 · oversample + noise, 30 epochs"),
    AUG: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-band05", "A30 · slice band [0.05, 0.95]"),
}
# Structures at the sagittal edges that the [0.2, 0.8] band cuts: the meniscal bodies and the collateral ligaments.
TARGETS = ["Lateral Meniscus", "Medial Meniscus", "MCL"]
PAIRS = [("run5 seed2027", "run5"), (AUG, REF), (AUG, BASE), (REF, BASE)]


def run_extras(report, run_config):
    return {**pick(report, ("image_augmentation", "stopped")),
            **pick(run_config, ("epochs", "patience", "divergence_stop", "slice_band"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = load_predictions(RUNS)
    populations = split_populations(ROOT, RUNS, saved)
    predictions = {split: {name: data[split] for name, data in saved.items()} for split in populations}
    result = {**score_splits(predictions, populations, PAIRS, TARGETS),
              "bootstrap": {"resamples": 2000, "seed": 2030}, "base": BASE, "reference": REF, "arm": AUG,
              "target_labels": TARGETS}
    result["runs"] = run_records(ROOT, RUNS, run_extras)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_report(result)


if __name__ == "__main__":
    main()
