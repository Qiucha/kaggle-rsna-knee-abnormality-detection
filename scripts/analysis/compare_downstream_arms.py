#!/usr/bin/env python3
"""Pair two downstream-probe arms against each other instead of against unadapted DINOv2.

``evaluate_downstream_probe.py`` answers "did contrastive learning move the image
representation at all", so every comparison it builds is against the unadapted
backbone. The question here is narrower and needs a different baseline: at a matched
seed, does unfreezing the final TEXT block on top of the final image block change what
the image tower encodes?

Nothing is recomputed from scratch and nothing is re-tuned. The fold assignment, the
estimator, and the bootstrap all come from the same frozen probe config, and the
features come from the cache that probe already wrote, so the numbers here sit on the
same folds and resamples as its table and are directly comparable to it. This script
is read-only: it never loads a checkpoint, never touches the locked test split, and
never writes into a run directory.

Reads: the frozen probe config, the ``features.npz`` it produced, and the gold labels.
Writes: only ``--output``.
Run: ``.venv/bin/python scripts/analysis/compare_downstream_arms.py --config CONFIG --features FEATURES
--output FILE.json --pair BASELINE_KEY::CANDIDATE_KEY [--pair ...]``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from contrastive_learning.artifacts import sha256_file, write_json  # noqa: E402
from experiment_tools.gold_probe import (  # noqa: E402
    load_feature_cache,
    load_gold_labels,
    paired_bootstrap,
    per_label_auroc_difference,
    resolve_population,
    score_arm,
    without_oof_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="The frozen probe config the features were built under")
    parser.add_argument("--features", type=Path, required=True, help="features.npz written by evaluate_downstream_probe.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="BASELINE::CANDIDATE",
        help="Two feature keys, e.g. 'after_cl/image-block-seed2026/uniform_mean_768::after_cl/both-block-seed2026/uniform_mean_768'",
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    if arguments.output.exists() and not arguments.overwrite:
        raise SystemExit(f"{arguments.output} exists; pass --overwrite to replace it")

    config = json.loads(arguments.config.read_text())
    labels = config["labels"]
    gold_uids, gold_matrix = load_gold_labels(REPOSITORY_ROOT / config["gold_labels_csv"], labels)
    records, gold, population = resolve_population(
        REPOSITORY_ROOT / config["cohort_manifest"], gold_uids, gold_matrix,
        # Read from the same frozen config the features were built under, so this
        # script cannot resolve a different population than the probe did.
        tuple(config.get("population_splits", ("train", "validation"))),
    )
    if population["manifest_sha256"] != config["manifest_sha256"]:
        raise SystemExit("Cohort manifest checksum does not match the frozen probe config")

    features, provenance, order = load_feature_cache(arguments.features)
    if order != [record.study_uid for record in records]:
        raise SystemExit("Cached features describe a different study order than the cohort resolves to")

    flat: dict[str, np.ndarray] = {
        f"{name}/{key}": matrix for name, sets in features.items() for key, matrix in sets.items()
    }

    pairs = []
    for raw in arguments.pair:
        if "::" not in raw:
            raise SystemExit(f"--pair must be BASELINE::CANDIDATE, got {raw!r}")
        baseline_key, candidate_key = raw.split("::", 1)
        for key in (baseline_key, candidate_key):
            if key not in flat:
                raise SystemExit(f"Unknown feature key {key!r}. Available:\n  " + "\n  ".join(sorted(flat)))
        pairs.append((baseline_key, candidate_key))

    # One scoring pass per distinct arm, reused across every pair that mentions it.
    scored: dict[str, dict] = {}
    for key in sorted({k for pair in pairs for k in pair}):
        print(f"probing {key} ({flat[key].shape[1]} features)", flush=True)
        scored[key] = score_arm(flat[key], gold, labels, config["probe"])

    comparisons = {}
    for baseline_key, candidate_key in pairs:
        baseline_scores = scored[baseline_key]["_oof_scores"]
        candidate_scores = scored[candidate_key]["_oof_scores"]
        comparison = paired_bootstrap(
            gold, labels, baseline_scores, candidate_scores, config["bootstrap"]
        )
        comparison["baseline_arm"] = baseline_key
        comparison["candidate_arm"] = candidate_key
        comparison["macro_auroc_baseline"] = scored[baseline_key]["macro_auroc"]
        comparison["macro_auroc_candidate"] = scored[candidate_key]["macro_auroc"]
        comparison["per_label_auroc_difference"] = per_label_auroc_difference(
            gold, labels, scored[baseline_key], scored[candidate_key], config["bootstrap"]
        )
        comparisons[f"{candidate_key} vs {baseline_key}"] = comparison

    payload = {
        "schema_version": 1,
        "comparison_version": config["probe_version"] + "-paired-arms",
        "purpose": "Seed-matched arm-versus-arm contrast, not arm-versus-unadapted.",
        "config": config,
        "config_sha256": sha256_file(arguments.config),
        "features_sha256": sha256_file(arguments.features),
        "population": population,
        "study_uids": [record.study_uid for record in records],
        "provenance": provenance,
        "results": {key: without_oof_scores(value) for key, value in scored.items()},
        "paired_comparisons": comparisons,
    }
    write_json(arguments.output, payload)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
