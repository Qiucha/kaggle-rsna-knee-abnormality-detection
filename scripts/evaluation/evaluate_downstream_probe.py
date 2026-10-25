"""Score representations on the downstream label-extraction task, before and after contrastive learning.

Retrieval Recall@1 is the intermediate check this project selects on. This script
answers the question that actually matters for ROADMAP M5: do the learned image
representations read out the twelve pathology labels better than the unadapted
DINOv2 features they started from, measured with the AUROC suite the label work
already uses?

The comparison is only meaningful against a checkpoint that actually moved the
image backbone. Runs with ``--image-unfreeze-last 0`` leave every DINOv2 weight
byte-identical, so their pre-projection features cannot differ from the unadapted
ones; the frozen-tower arm is carried here precisely to assert that identity.

Nothing here is generalisation evidence. Fifty-six of the fifty-eight gold studies
were image/report training pairs or members of the checkpoint-selection split, so
every number below is a readout statistic on a cohort the representation has seen.
The locked test split is never loaded.

Reads: the frozen probe config (``configs/downstream/gold-probe-v1.json`` by default), the image
caches and checkpoints it names, the gold labels and the LM label files. Extracts features unless
``--features`` already holds them (``--from-features`` forces reuse; no GPU needed then).
Writes: the features cache (``--features``) and the results JSON (``--output``).
Run: ``.venv/bin/python scripts/evaluation/evaluate_downstream_probe.py [--config CONFIG] [--overwrite]``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from contrastive_learning.artifacts import sha256_file, write_json
from experiment_tools.gold_probe import (
    load_gold_labels,
    resolve_population,
    build_towers,
    extract_features_by_preprocessing,
    load_feature_cache,
    score_arm,
    paired_bootstrap,
    per_label_auroc_difference,
    without_oof_scores,
    lm_extraction_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/downstream/gold-probe-v1.json")
    parser.add_argument("--image-cache-dir", type=Path, default=REPOSITORY_ROOT / "checkpoints/contrastive/full-multilingual-v1/image-cache")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "digest/downstream-probe-v1.json")
    parser.add_argument("--features", type=Path, default=REPOSITORY_ROOT / "checkpoints/contrastive/gold-probe-v1/features.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--features-only", action="store_true", help="Extract and save features only, then exit")
    parser.add_argument("--from-features", action="store_true", help="Load features from --features cache without rebuilding towers")
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text())
    labels = config["labels"]
    if not arguments.features_only and arguments.output.exists() and not arguments.overwrite:
        raise SystemExit(f"{arguments.output} exists; pass --overwrite to replace it")

    gold_uids, gold_matrix = load_gold_labels(REPOSITORY_ROOT / config["gold_labels_csv"], labels)
    records, gold, population = resolve_population(
        REPOSITORY_ROOT / config["cohort_manifest"], gold_uids, gold_matrix,
        tuple(config.get("population_splits", ("train", "validation"))),
    )
    if population["manifest_sha256"] != config["manifest_sha256"]:
        raise SystemExit("Cohort manifest checksum does not match the frozen probe config")
    print(json.dumps(population, indent=2))

    device = torch.device(arguments.device)
    if arguments.features.exists() and (arguments.from_features or not arguments.overwrite):
        features, provenance, order = load_feature_cache(arguments.features)
        if order != [record.study_uid for record in records]:
            raise SystemExit("Cached features describe a different study order; rerun with --overwrite")
    else:
        towers, provenance = build_towers(config, device)
        arm_cache_dirs = {
            f"{arm}/{Path(path).parent.name}": REPOSITORY_ROOT / spec["image_cache_dir"]
            for arm, spec in config["arms"].items()
            if arm != "before_cl" and "image_cache_dir" in spec
            for path in spec["checkpoints"]
        }
        features = extract_features_by_preprocessing(
            records, towers, provenance, arguments.image_cache_dir, arm_cache_dirs
        )
        arguments.features.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arguments.features,
            index=json.dumps({name: sorted(sets) for name, sets in features.items()}),
            provenance=json.dumps(provenance),
            study_uids=np.array([record.study_uid for record in records]),
            **{f"{name}|{key}": value for name, sets in features.items() for key, value in sets.items()},
        )

    if arguments.features_only:
        print(f"Features successfully saved to {arguments.features}")
        return

    # Each arm's floor is the unadapted backbone at that arm's own preprocessing. For a
    # single-preprocessing config this resolves to "before_cl/dinov2-unadapted" for every
    # arm, exactly as before; for a mixed one it keeps a 130mm arm off a 150mm floor.
    unadapted_for = provenance.get("_unadapted_for", {})

    def floor_of(arm_name: str) -> str:
        return unadapted_for.get(arm_name, "before_cl/dinov2-unadapted")

    # Declared in advance: a frozen-backbone run cannot move uniform_mean_768.
    identity_check = {}
    for name, sets in features.items():
        if name.startswith("frozen_tower_null/"):
            uniform_key = next(key for key in sets if key.startswith("uniform_mean_"))
            baseline_features = features[floor_of(name)][uniform_key]
            deviation = float(np.abs(sets[uniform_key] - baseline_features).max())
            identity_check[name] = {
                "max_absolute_deviation_from_unadapted": deviation,
                "identical": bool(deviation == 0.0),
                "compared_against": floor_of(name),
            }

    probe = config["probe"]
    results: dict[str, Any] = {}
    for name, sets in features.items():
        for key, matrix in sorted(sets.items()):
            print(f"probing {name}/{key} ({matrix.shape[1]} features)", flush=True)
            results[f"{name}/{key}"] = score_arm(matrix, gold, labels, probe)

    comparisons = {}
    for key, value in results.items():
        name, _feature_set = key.rsplit("/", 1)
        # The floor is ALWAYS uniform_mean_768, whatever feature set the candidate uses:
        # the unadapted backbone has no learned pooling and no projection head, so those
        # feature sets simply do not exist for it. Only the preprocessing GROUP varies
        # here -- matching the candidate's feature set instead would ask for
        # "before_cl/.../plane_attention_768", which is never extracted, and would break
        # every single-preprocessing config as well as this one.
        floor_keys = [key for key in features[floor_of(name)] if key.startswith("uniform_mean_")]
        if len(floor_keys) != 1:
            raise SystemExit(f"{name}: expected one uniform-mean floor, got {floor_keys}")
        baseline_key = f"{floor_of(name)}/{floor_keys[0]}"
        if key == baseline_key:
            continue
        if baseline_key not in results:
            raise SystemExit(f"{key}: its floor {baseline_key} was not scored")
        baseline_scores = results[baseline_key]["_oof_scores"]
        comparison = paired_bootstrap(
            gold, labels, baseline_scores, value["_oof_scores"], config["bootstrap"]
        )
        # Named explicitly: with more than one preprocessing, "before" is no longer a
        # single number and a reader must be able to see which floor was subtracted.
        comparison["baseline_key"] = baseline_key
        comparison["macro_auroc_before"] = results[baseline_key]["macro_auroc"]
        comparison["macro_auroc_after"] = value["macro_auroc"]
        comparison["per_label_auroc_difference"] = per_label_auroc_difference(
            gold, labels, results[baseline_key], value, config["bootstrap"]
        )
        comparisons[key] = comparison

    payload = {
        "schema_version": 1,
        "probe_version": config["probe_version"],
        "config": config,
        "config_sha256": sha256_file(arguments.config),
        "population": population,
        "study_uids": [record.study_uid for record in records],
        "provenance": provenance,
        "frozen_backbone_identity_check": identity_check,
        "results": {key: without_oof_scores(value) for key, value in results.items()},
        "comparisons_against_unadapted_dinov2": comparisons,
        "lm_label_extraction_reference": lm_extraction_reference(
            REPOSITORY_ROOT / "data/lm_label", [record.study_uid for record in records], gold, labels
        ),
    }
    write_json(arguments.output, payload)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
