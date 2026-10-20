"""Summarize the label-similarity-v1 arms into one evidence JSON.

Reads each arm's ``metrics.jsonl`` and ``config.json``, pairs the arms epoch by
epoch, and records the three curves the experiment is read on: training loss,
validation loss, and validation Recall@1. Nothing here refits or re-scores
anything -- it reads what the runs wrote, so the numbers cannot drift from the
run record.

One comparability note is load-bearing and is repeated into the output. The two
arms **do not optimize the same objective**, so their ``train_loss`` columns are
not on one scale: the weighted arm's cross entropy is taken against a softened
target whose entropy is nonzero, which raises the achievable floor regardless of
how well the model fits. ``validation_loss`` and ``mean_r@1`` are computed by
:mod:`contrastive_learning.evaluation` against the unmodified hard diagonal for
both arms, and those two are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# The fields that define the comparison. Any difference outside this set means the
# arms differ in something the experiment did not intend to vary.
_COMPARED_FIELDS = (
    "cohort_manifest", "text_model", "text_pooling", "text_prefix", "max_text_length",
    "image_model", "image_unfreeze_last", "text_unfreeze_last", "image_plane_pooling",
    "embedding_dim", "epochs", "batch_size", "lr_image", "lr_text", "lr_projection",
    "weight_decay", "warmup_fraction", "grad_clip_norm", "seed", "crop_mm", "image_size",
    "report_view", "image_drop_path_rate",
)

_CURVES = ("train_loss", "validation_loss", "mean_r@1")


def read_arm(directory: Path) -> dict[str, Any]:
    """One arm's configuration, per-epoch curves, and selected checkpoint."""
    config = json.loads((directory / "config.json").read_text())
    epochs = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    if not epochs:
        raise ValueError(f"No recorded epochs in {directory}")
    # Mirrors the project's selection policy: best mean_r@1, ties broken by lower
    # validation loss, then by the earliest epoch that earned it.
    selected = min(epochs, key=lambda m: (-m["mean_r@1"], m["validation_loss"], m["epoch"]))
    best_validation_loss = min(epochs, key=lambda m: (m["validation_loss"], m["epoch"]))
    return {
        "directory": _relative(directory),
        "label_similarity_weight": config.get("label_similarity_weight", 0.0),
        "lm_label_csv": config.get("lm_label_csv"),
        "epochs_recorded": len(epochs),
        "curves": {name: [round(float(m[name]), 6) for m in epochs] for name in _CURVES},
        "inverse_temperature": [round(float(m["inverse_temperature"]), 4) for m in epochs],
        "selected_epoch": selected["epoch"],
        "selected": {name: round(float(selected[name]), 6) for name in _CURVES},
        "final": {name: round(float(epochs[-1][name]), 6) for name in _CURVES},
        "best_validation_loss": {
            "epoch": best_validation_loss["epoch"],
            "value": round(float(best_validation_loss["validation_loss"]), 6),
        },
        "overfitting_gap": round(
            float(epochs[-1]["validation_loss"]) - float(best_validation_loss["validation_loss"]), 6
        ),
        "_config": config,
    }


def compare(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    """Paired epoch-by-epoch differences, plus a check that nothing else varied."""
    unintended = {
        field: [baseline["_config"].get(field), treatment["_config"].get(field)]
        for field in _COMPARED_FIELDS
        if baseline["_config"].get(field) != treatment["_config"].get(field)
    }
    shared = min(baseline["epochs_recorded"], treatment["epochs_recorded"])
    per_epoch = {
        name: [
            round(treatment["curves"][name][i] - baseline["curves"][name][i], 6)
            for i in range(shared)
        ]
        # train_loss is deliberately excluded: the two arms minimize different
        # objectives, so their difference is not a quantity with a meaning.
        for name in ("validation_loss", "mean_r@1")
    }
    return {
        "arms_differ_only_in_label_similarity": not unintended,
        "unintended_differences": unintended,
        "epochs_compared": shared,
        "per_epoch_difference": per_epoch,
        "selected_mean_r@1_difference": round(
            treatment["selected"]["mean_r@1"] - baseline["selected"]["mean_r@1"], 6
        ),
        "best_validation_loss_difference": round(
            treatment["best_validation_loss"]["value"] - baseline["best_validation_loss"]["value"], 6
        ),
        "epochs_treatment_ahead_on_mean_r@1": sum(1 for d in per_epoch["mean_r@1"] if d > 0),
        "epochs_treatment_ahead_on_validation_loss": sum(
            1 for d in per_epoch["validation_loss"] if d < 0
        ),
    }


def _spread(left: list[float], right: list[float]) -> dict[str, float]:
    """Summary of the per-epoch absolute gap between two runs' curves."""
    gaps = sorted(abs(x - y) for x, y in zip(left, right))
    return {
        "epochs": len(gaps),
        "median": round(statistics.median(gaps), 6),
        "mean": round(statistics.mean(gaps), 6),
        "max": round(gaps[-1], 6),
    }


def _relative(path: Path) -> str:
    """Repository-relative when it lies inside the tree, absolute otherwise.

    Arguments arrive however the caller typed them, so resolving first is what
    lets a relative --probe path be recorded the same way an absolute one is.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints/contrastive/label-similarity-v1",
    )
    parser.add_argument("--experiment-version", default="label-similarity-v1")
    parser.add_argument("--experiment-config", default="configs/contrastive/label-similarity-v1.json")
    parser.add_argument("--baseline", default="lambda00-seed2026")
    parser.add_argument(
        "--treatment",
        nargs="+",
        default=["lambda03-seed2026"],
        help="One or more treatment arms, each compared against --baseline",
    )
    parser.add_argument(
        "--probe", type=Path, help="Optional downstream probe JSON to fold in for the AUROC table"
    )
    parser.add_argument(
        "--paired",
        type=Path,
        help="Optional compare_downstream_arms.py JSON carrying the paired AUROC interval",
    )
    parser.add_argument(
        "--control",
        type=Path,
        help=(
            "A second run of the BASELINE configuration, used to measure the run-to-run noise "
            "floor. Nothing in this repository pins CUDA determinism, so two runs of one config "
            "at one seed diverge; without this the treatment difference has no yardstick."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY_ROOT / "digest/label-similarity-v1.json"
    )
    arguments = parser.parse_args()

    baseline = read_arm(Path(arguments.baseline) if Path(arguments.baseline).exists()
                        else arguments.run_root / arguments.baseline)
    treatments = {
        name: read_arm(Path(name) if Path(name).exists() else arguments.run_root / name)
        for name in arguments.treatment
    }
    # The first treatment drives the single-arm fields kept for compatibility with
    # the v1 report shape; every arm appears in "comparisons".
    treatment = next(iter(treatments.values()))
    report: dict[str, Any] = {
        "experiment_version": arguments.experiment_version,
        "config": arguments.experiment_config,
        "comparability": {
            "train_loss": (
                "NOT comparable between arms: the weighted arm's cross entropy is taken against a "
                "softened target with nonzero entropy, which raises its floor irrespective of fit. "
                "Read each arm's train_loss only against its own earlier epochs."
            ),
            "validation_loss": (
                "Comparable: computed against the unmodified hard diagonal for both arms, over the "
                "full 433-study validation split. Not comparable to train_loss in either arm."
            ),
            "mean_r@1": "Comparable: full-split retrieval, identical candidate pool for both arms.",
        },
        "arms": {
            arguments.baseline: {k: v for k, v in baseline.items() if k != "_config"},
            **{
                name: {k: v for k, v in arm.items() if k != "_config"}
                for name, arm in treatments.items()
            },
        },
        "comparison": compare(baseline, treatment),
        "comparisons": {
            name: compare(baseline, arm) for name, arm in treatments.items()
        },
    }
    if arguments.control is not None and arguments.control.exists():
        control = read_arm(arguments.control)
        treatment_effect = compare(baseline, treatment)["selected_mean_r@1_difference"]
        noise = round(control["selected"]["mean_r@1"] - baseline["selected"]["mean_r@1"], 6)
        report["noise_floor"] = {
            "control_run": _relative(arguments.control),
            "what_it_is": (
                "A second run of the BASELINE configuration at the same seed, differing from it "
                "only by CUDA nondeterminism. Its distance from the baseline is how far two "
                "identical runs drift, which is the smallest difference this experiment can resolve."
            ),
            "baseline_selected_mean_r@1": baseline["selected"]["mean_r@1"],
            "control_selected_mean_r@1": control["selected"]["mean_r@1"],
            "treatment_selected_mean_r@1": treatment["selected"]["mean_r@1"],
            "noise_floor_mean_r@1": abs(noise),
            "treatment_effect_mean_r@1": treatment_effect,
            "treatment_effect_exceeds_noise_floor": abs(treatment_effect) > abs(noise),
            "per_epoch_absolute_difference": {
                # The selection maxima above overstate the floor, because taking the
                # best of 20 noisy epochs inflates the gap. These two distributions
                # are the like-for-like comparison: how far the treatment moves each
                # epoch, against how far pure nondeterminism moves it.
                "nondeterminism_only": _spread(baseline["curves"]["mean_r@1"],
                                               control["curves"]["mean_r@1"]),
                "treatment_versus_baseline": _spread(baseline["curves"]["mean_r@1"],
                                                     treatment["curves"]["mean_r@1"]),
            },
            "treatment_within_control_spread": (
                min(baseline["selected"]["mean_r@1"], control["selected"]["mean_r@1"])
                <= treatment["selected"]["mean_r@1"]
                <= max(baseline["selected"]["mean_r@1"], control["selected"]["mean_r@1"])
            ),
        }

    if arguments.probe is not None and arguments.probe.exists():
        probe = json.loads(arguments.probe.read_text())
        results = probe.get("results", {})
        report["downstream_probe"] = {
            "source": _relative(arguments.probe),
            "population": probe.get("population", {}).get("population_size"),
            "macro_auroc": {
                arm: round(float(values["macro_auroc"]), 6)
                for arm, values in sorted(results.items())
                if isinstance(values, dict) and "macro_auroc" in values
            },
            "macro_auprc": {
                arm: round(float(values["macro_auprc"]), 6)
                for arm, values in sorted(results.items())
                if isinstance(values, dict) and "macro_auprc" in values
            },
        }
    if arguments.paired is not None and arguments.paired.exists():
        paired = json.loads(arguments.paired.read_text())
        report["paired_auroc"] = {
            "source": _relative(arguments.paired),
            "comparisons": {
                name: {
                    "macro_auroc_baseline": values["macro_auroc_baseline"],
                    "macro_auroc_candidate": values["macro_auroc_candidate"],
                    "difference": values["macro_auroc_difference_mean"],
                    "ci95": values["macro_auroc_difference_ci95"],
                    "fraction_favouring_candidate": values["fraction_of_replicates_favouring_right"],
                }
                for name, values in paired.get("paired_comparisons", {}).items()
            },
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison"], indent=2))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
