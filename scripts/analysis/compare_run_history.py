#!/usr/bin/env python
"""Compare a rerun of a fine-tuning rung with its recorded run, epoch by epoch.

Reads ``history.jsonl`` (and ``report.json`` once both runs have one) from the two run
directories; writes nothing. Prints, per epoch, the recorded value, the rerun's value
and their difference for every training and validation metric both histories log,
then the selection and the scores.

No CUDA determinism is pinned, so two runs of the same code drift apart in the low
decimals; two A30 runs of identical code differed by up to 0.0022 in training loss
and 0.018 in validation loss per epoch. The check therefore gates only the training
loss, at ``--train-tolerance`` (0.01, provisional), and fails when no epoch overlaps,
when a loss is not finite, when an epoch has no training loss both runs logged, or,
once both runs are finished, when they ran different epochs. Validation is reported,
not gated.

A rung resumed from a parent copies the parent's history; pass ``--first-epoch`` to
skip the inherited epochs, which match trivially.

    python scripts/analysis/compare_run_history.py \\
        --record checkpoints/finetune/<rung> --rerun /path/to/rerun
"""
import argparse
import json
import math
import sys
from pathlib import Path

TRAIN_KEYS = ("train_masked_bce", "train_objective")
VALIDATION_KEYS = ("validation_masked_bce", "validation_macro_auroc")


def history(run: Path) -> dict[int, dict]:
    rows = {}
    lines = [line for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number < len(lines):
                raise
            print(f"{run}: skipping a partial last history line")  # a writer may be midway through it
            continue
        rows[row["epoch"]] = row
    return rows


def epoch_table(record: dict[int, dict], rerun: dict[int, dict], tolerance: float) -> list[str]:
    """Print one row per shared epoch; return the training-loss failures.

    A metric is tabulated when both runs log it at some shared epoch. Every shared epoch
    must carry at least one training loss logged by both, or that epoch fails.
    """
    epochs = sorted(set(record) & set(rerun))
    keys = [k for k in TRAIN_KEYS + VALIDATION_KEYS if any(k in record[e] and k in rerun[e] for e in epochs)]
    print("epoch | " + " | ".join(f"{k}: record / rerun / diff" for k in keys) + " | seconds: record / rerun")
    worst, failures = dict.fromkeys(keys, 0.0), []
    for epoch in epochs:
        a, b = record[epoch], rerun[epoch]
        cells = []
        if not any(k in a and k in b for k in TRAIN_KEYS):
            failures.append(f"epoch {epoch} has no training loss logged by both runs")
        for key in keys:
            if key not in a or key not in b:
                cells.append("— / — / —")
                continue
            difference = b[key] - a[key]
            worst[key] = max(worst[key], abs(difference)) if math.isfinite(difference) else math.inf
            if key in TRAIN_KEYS and not abs(difference) <= tolerance:  # NaN fails too
                failures.append(f"epoch {epoch} {key} {difference:+.4f}")
            cells.append(f"{a[key]:.4f} / {b[key]:.4f} / {difference:+.4f}")
        print(f"{epoch:>5} | " + " | ".join(cells) + f" | {a.get('seconds', 0):.0f} / {b.get('seconds', 0):.0f}")
    print("largest |diff|: " + ", ".join(f"{k} {v:.4f}" for k, v in worst.items()))
    return failures


def report_comparison(record: Path, rerun: Path) -> None:
    old, new = (json.loads((run / "report.json").read_text()) for run in (record, rerun))
    print(f"best_epoch: record {old['best_epoch']} / rerun {new['best_epoch']}")
    print(f"validation_masked_bce: record {old['validation_masked_bce']:.4f} / rerun {new['validation_masked_bce']:.4f}")
    print(f"stopped: record {old.get('stopped', 'absent')} / rerun {new.get('stopped', 'absent')}")
    for key in ("lm_test", "gold"):
        a, b = old[key]["macro_auroc"], new[key]["macro_auroc"]
        print(f"{key} macro AUROC: record {a:.4f} / rerun {b:.4f} / diff {b - a:+.4f}")
    for key in ("window_counts", "oversample"):
        if key in old or key in new:
            print(f"{key} identical: {old.get(key) == new.get(key)}")


def compare(record_dir: Path, rerun_dir: Path, tolerance: float = 0.01, first_epoch: int = 1) -> list[str]:
    """Print the comparison; return why the rerun fails to reproduce the record (empty: it does)."""
    record, rerun = ({e: row for e, row in history(d).items() if e >= first_epoch} for d in (record_dir, rerun_dir))
    failures = []
    if not set(record) & set(rerun):
        return ["no epoch in common"]
    failures += epoch_table(record, rerun, tolerance)
    if (record_dir / "report.json").exists() and (rerun_dir / "report.json").exists():
        report_comparison(record_dir, rerun_dir)
        if set(record) != set(rerun):
            failures.append(f"finished runs differ in epochs: record only {sorted(set(record) - set(rerun))}, "
                            f"rerun only {sorted(set(rerun) - set(record))}")
    else:
        unfinished = [name for name, d in (("record", record_dir), ("rerun", rerun_dir)) if not (d / "report.json").exists()]
        print(f"no report.json yet in the {' and '.join(unfinished)}; compared epochs "
              f"{min(set(record) & set(rerun))}-{max(set(record) & set(rerun))}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, required=True, help="the recorded run directory")
    parser.add_argument("--rerun", type=Path, required=True, help="the rerun's directory")
    parser.add_argument("--train-tolerance", type=float, default=0.01,
                        help="largest per-epoch training-loss difference that still passes")
    parser.add_argument("--first-epoch", type=int, default=1,
                        help="first epoch to compare; skips epochs a resumed rung inherited")
    args = parser.parse_args()
    for run in (args.record, args.rerun):
        if not (run / "history.jsonl").exists():
            parser.error(f"{run} has no history.jsonl")

    failures = compare(args.record, args.rerun, args.train_tolerance, args.first_epoch)
    print("FAIL: " + "; ".join(failures) if failures else f"PASS: training loss within {args.train_tolerance} every epoch")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
