#!/usr/bin/env python3
"""Paired LM-test and gold-58 deltas for the lateral/synovitis rebalancing arms vs u9-b1.

Both arms train the u9-b1 recipe with one change each: (A) one extra draw of every study
positive for Lateral Meniscus, Lateral OA or Synovitis, other labels masked on the extra
draws, plus fresh N(0, 0.1) noise on the two lateral soft targets; (B) 2x loss weight on
the three labels, renormalised to mean 1. "A ep30" reruns A with a 30-epoch cosine budget,
patience off and a divergence stop; "A ep50" resumes it from last.pt with a warm restart at 0.5x LR
over epochs 31-50. "u9-b1 ep30" is u9-b1 under A ep30's budget and stopping rule, the matched
control for epochs. "u9-b1 ep43" matches A ep30's training draws (43 x 3,438 vs 30 x 4,946). Every run is read from its own ``predictions.npz``
and must match run 5 study for study. Intervals are study bootstraps shared across runs
(:func:`lm_proxy.paired_intervals`), so each delta is paired. The run-5 seed replicate
gives the training-noise reference.

Reads: ``predictions.npz``, ``report.json``, ``config.json`` and ``history.jsonl`` of each run under
``checkpoints/finetune`` (arms whose run is not there yet are listed as missing, not an error), and
the cohort files named in run 5's config.
Writes: only ``--output``.
Run: ``.venv/bin/python scripts/analysis/compare_lateral_rebalance.py --output FILE.json``
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
BASE = "u9-b1"
RUNS = {  # key -> (run directory, title)
    "run5": (FINETUNE / "r4a-k8-omnirad-base-soft-336", "Run 5 (u3, seed 2026)"),
    "run5 seed2027": (FINETUNE / "r4a-k8-omnirad-base-soft-336-seed2027", "Run 5 (u3, seed 2027)"),
    BASE: (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1", "u9-b1 (base)"),
    "u9-b1 ep30": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-ep30", "u9-b1 · 30 epochs (matched epochs)"),
    "u9-b1 ep43": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-ep43", "u9-b1 · 43 epochs (matched draws)"),
    "oversample+noise": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise",
                         "A · oversample + lateral noise"),
    "loss weight 2x": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-w2", "B · 2× loss weight"),
    "oversample+noise ep30": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30",
                              "A · 30-epoch budget"),
    "oversample+noise ep50": (FINETUNE / "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep50",
                              "A · extended to 50 (warm restart)"),
}
ARMS = ["oversample+noise", "loss weight 2x", "u9-b1 ep30", "u9-b1 ep43", "oversample+noise ep30",
        "oversample+noise ep50"]
TARGETS = ["Lateral Meniscus", "Lateral OA", "Synovitis"]
PAIRS = [("run5 seed2027", "run5"), ("oversample+noise", BASE), ("loss weight 2x", BASE),
         ("oversample+noise", "loss weight 2x"), ("oversample+noise ep30", BASE),
         ("oversample+noise ep30", "oversample+noise"), ("oversample+noise ep50", BASE),
         ("oversample+noise ep50", "oversample+noise ep30"), ("u9-b1 ep30", BASE),
         ("oversample+noise ep30", "u9-b1 ep30"), ("oversample+noise ep50", "u9-b1 ep30"),
         ("u9-b1 ep43", BASE), ("oversample+noise ep30", "u9-b1 ep43"), ("oversample+noise ep50", "u9-b1 ep43")]


def run_extras(report, run_config):
    return {**pick(run_config, ("oversample", "label_noise", "label_weights")),
            **pick(report, ("oversample", "label_weights", "stopped")),
            **pick(run_config, ("epochs", "patience", "divergence_stop", "resume_from"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    available = {k: v for k, v in RUNS.items() if (v[0] / "predictions.npz").exists()}
    missing = [k for k in ARMS if k not in available]
    saved = load_predictions(available)
    populations = split_populations(ROOT, available, saved)
    predictions = {split: {name: data[split] for name, data in saved.items()} for split in populations}
    result = {**score_splits(predictions, populations, PAIRS, TARGETS, available=saved),
              "bootstrap": {"resamples": 2000, "seed": 2030}, "base": BASE, "arms": ARMS,
              "target_labels": TARGETS, "missing_arms": missing}
    result["runs"] = run_records(ROOT, available, run_extras, seconds=False)
    result["recipe"] = {k: json.loads((RUNS[BASE][0] / "config.json").read_text())[k]
                        for k in ("image_model", "image_unfreeze_last", "image_size", "windows_per_plane",
                                  "targets", "epochs", "patience", "seed")}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_report(result, widths=(20, 40, 32),
                 label_pair=lambda key: key.endswith((f" - {BASE}", " - u9-b1 ep30", " - u9-b1 ep43")))
    if missing:
        print(f"\nmissing arms: {missing}")


if __name__ == "__main__":
    main()
