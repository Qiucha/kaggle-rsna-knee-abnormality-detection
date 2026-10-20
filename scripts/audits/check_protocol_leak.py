#!/usr/bin/env python
"""Does the acquisition protocol leak the label?

A knee scanned with extra sequences was scanned by someone looking for something, so
the number of series in a study may carry diagnostic information on its own. If it
does, a model fed variable-length slots can read the label off the protocol rather
than the pixels -- re-inflating exactly the probe figures the gold-holdout work
deflated.

Two readouts, because the direct one is underpowered:
  PRIMARY   series count -> each of the 12 gold labels, on the 58 labelled studies.
            AUROC with a permutation test. 9-35 positives per label, so absence of
            evidence here is weak evidence of absence; a LARGE effect is still
            detectable and that is what would disqualify the design.
  PROXY     series count -> report length, on all 4,407 studies. Well powered. Reports
            describe findings, so if protocol tracked pathology at all, longer reports
            should accompany richer protocols.

Series counts come from DISK, not from train_series.csv -- the CSV's acquisition flags
are known unreliable (digest/acquisition-metadata-trust.md).

Read-only. No training, no test-split evaluation; the test split is included in the
proxy only as an unlabelled count of series and reports.

Usage: check_protocol_leak.py [--out PATH]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning import paths  # noqa: E402
from experiment_tools.cohort_audit import load_studies  # noqa: E402

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RESAMPLES = 10000
RNG_SEED = 20260914


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC, ties given half credit."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    positives, negatives = labels == 1, labels == 0
    n_pos, n_neg = positives.sum(), negatives.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def series_count(study_dir: Path) -> int | None:
    try:
        return sum(1 for d in os.scandir(study_dir) if d.is_dir())
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/contrastive-cohort.jsonl")
    parser.add_argument("--train-csv", type=Path, help="default: train.csv on this host's data_root")
    parser.add_argument("--out", default="digest/protocol-leak-check.json")
    args = parser.parse_args()

    rows = load_studies(args.manifest, None)
    counts, reports = {}, {}
    for row in rows:
        study_dir = Path(next(iter(row["slots"].values()))).parent
        n = series_count(study_dir)
        if n:
            counts[row["study_uid"]] = n
            reports[row["study_uid"]] = row.get("report") or ""

    labels = pd.read_csv(args.train_csv or paths.host_data_root() / "train.csv").set_index("StudyInstanceUID")
    gold = labels[labels[TARGETS].notna().all(axis=1)]
    gold = gold[gold.index.isin(counts)]

    rng = np.random.default_rng(RNG_SEED)
    x = np.array([counts[uid] for uid in gold.index], dtype=float)
    result = {"n_gold": int(len(gold)), "n_all": len(counts), "resamples": RESAMPLES,
              "series_count": {"median": float(np.median(list(counts.values()))),
                               "min": int(min(counts.values())), "max": int(max(counts.values()))},
              "primary": {}, "proxy": {}}

    for target in TARGETS:
        y = gold[target].to_numpy(dtype=float)
        observed = auroc(x, y)
        null = np.array([auroc(x, rng.permutation(y)) for _ in range(RESAMPLES)])
        # two-sided: how often does a shuffled label separate at least this strongly?
        p = float((np.abs(null - 0.5) >= abs(observed - 0.5)).mean())
        result["primary"][target] = {"auroc": observed, "n_pos": int((y == 1).sum()),
                                     "permutation_p": p}

    lengths = np.array([len(reports[uid]) for uid in counts], dtype=float)
    series = np.array([counts[uid] for uid in counts], dtype=float)
    def spearman(a, b):
        ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])
    rho = spearman(series, lengths)
    null = np.array([spearman(series, rng.permutation(lengths)) for _ in range(1000)])
    result["proxy"] = {"spearman_series_vs_report_length": rho,
                       "permutation_p": float((np.abs(null) >= abs(rho)).mean()),
                       "n": len(series)}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print(f"series per study: median {result['series_count']['median']:.0f} "
          f"(range {result['series_count']['min']}-{result['series_count']['max']}), "
          f"{result['n_all']:,} studies\n")
    print(f"PRIMARY  series count -> gold label, n={result['n_gold']}")
    print(f"{'label':<18}{'n_pos':>6}{'AUROC':>8}{'perm p':>9}")
    flagged = []
    for target in TARGETS:
        e = result["primary"][target]
        mark = "  <-- flag" if e["permutation_p"] < 0.05 else ""
        if mark:
            flagged.append(target)
        print(f"{target:<18}{e['n_pos']:>6}{e['auroc']:>8.3f}{e['permutation_p']:>9.3f}{mark}")
    aurocs = [result["primary"][t]["auroc"] for t in TARGETS]
    print(f"\nmean |AUROC - 0.5| = {np.mean([abs(a - 0.5) for a in aurocs]):.3f}; "
          f"labels flagged at p<0.05: {len(flagged)} of 12 "
          f"(expect ~0.6 by chance){' -> ' + ', '.join(flagged) if flagged else ''}")
    pr = result["proxy"]
    print(f"\nPROXY    series count vs report length, n={pr['n']:,}: "
          f"Spearman {pr['spearman_series_vs_report_length']:+.3f}, perm p {pr['permutation_p']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
