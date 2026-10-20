"""Why the contrastive model's learned plane pooling reads labels worse than averaging.

The downstream probe found that the image tower's attention-pooled representation
scores below a plain mean of the same three plane vectors, on the same checkpoint.
That is a strange result on its face -- the attention is strictly more expressive
than the mean -- so this script takes it apart.

Three candidate explanations are separable from saved artifacts:

1. *Re-weighting.* The attention concentrates mass on planes that identify a report
   but do not show a given finding. Isolated by probing ``sum_p w_p f_p`` and by
   probing each plane on its own.
2. *The plane-embedding term.* ``ImageTower`` adds a learned per-plane vector before
   pooling, so the pooled feature carries ``sum_p w_p e_p`` -- a study-dependent
   nuisance in the span of three vectors, unrelated to the pixels. Isolated by
   probing the same weights without it.
3. *Information loss.* Any pooling is a 3-to-1 reduction. Bounded by probing the
   three planes concatenated.

Every arm is paired against ``uniform_mean_768`` of the same checkpoint, under the
probe contract copied by value from the frozen downstream-probe config.

Output: digest/plane-pooling-analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from experiment_tools.gold_probe import (  # the identical probe contract, not a copy
    label_bootstrap,
    load_gold_labels,
    paired_bootstrap,
    resolve_population,
    score_arm,
)

from contrastive_learning.artifacts import sha256_file, write_json
from contrastive_learning.checkpoints import (
    checkpoint_config,
    has_full_weights,
    load_model_weights,
    read_checkpoint,
)
from contrastive_learning.data import KneeReportDataset
from contrastive_learning.model import build_model
from contrastive_learning.preprocessing import PLANES


@torch.inference_mode()
def plane_features_and_weights(
    tower: Any, records: list[Any], image_kwargs: dict[str, Any], cache: Path, batch: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-plane features, plane-marked features, and the attention weights.

    ``raw`` is the backbone output per plane; ``marked`` is that plus the learned
    plane embedding, which is what the attention actually scores and pools. Keeping
    both is what lets the embedding term be isolated downstream.
    """
    dataset = KneeReportDataset(records, image_kwargs, image_cache_dir=cache)
    raw_parts, marked_parts, weight_parts = [], [], []
    for start in range(0, len(dataset), batch):
        images = torch.stack(
            [dataset[index]["images"] for index in range(start, min(start + batch, len(dataset)))]
        )
        count, planes = images.shape[:2]
        flat = images.reshape(count * planes, *images.shape[2:]).float()
        flat = (flat - tower.mean) / tower.std
        tokens = tower.backbone(pixel_values=flat).last_hidden_state
        raw = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1).reshape(count, planes, -1)
        marked = raw + tower.plane_embeddings[:planes]
        raw_parts.append(raw)
        marked_parts.append(marked)
        weight_parts.append(tower.plane_attention(marked).squeeze(-1).softmax(dim=1))
    return (
        torch.cat(raw_parts).float().numpy(),
        torch.cat(marked_parts).float().numpy(),
        torch.cat(weight_parts).float().numpy(),
    )


def feature_sets(raw: np.ndarray, marked: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """The seven representations under comparison, all from one backbone pass."""
    expanded = weights[:, :, None]
    sets = {
        "uniform_mean_768": raw.mean(axis=1),
        "attention_pooled_768": (expanded * marked).sum(axis=1),
        "attention_no_embeddings_768": (expanded * raw).sum(axis=1),
        "concat_planes_2304": raw.reshape(len(raw), -1),
    }
    for index, plane in enumerate(PLANES):
        sets[f"plane_{plane.lower()}_768"] = raw[:, index]
    return sets


def weight_statistics(weights: np.ndarray) -> dict[str, Any]:
    """How concentrated the pooling is, and how much of it depends on the study.

    Entropy is reported against ``log 3``, the uniform mean's own entropy, so the
    distance between the learned pooling and the arm it is being compared to is
    readable directly rather than inferred from three means.
    """
    entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(axis=1)
    return {
        "plane_order": list(PLANES),
        "mean_weight": [float(value) for value in weights.mean(axis=0)],
        "sd_weight_across_studies": [float(value) for value in weights.std(axis=0)],
        "min_weight": [float(value) for value in weights.min(axis=0)],
        "max_weight": [float(value) for value in weights.max(axis=0)],
        "uniform_weight": 1.0 / weights.shape[1],
        "mean_entropy": float(entropy.mean()),
        "uniform_entropy": float(np.log(weights.shape[1])),
        "mean_largest_weight": float(weights.max(axis=1).mean()),
        "argmax_plane_counts": [int(value) for value in np.bincount(weights.argmax(axis=1), minlength=weights.shape[1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/downstream/gold-probe-planes-v1.json")
    parser.add_argument("--image-cache-dir", type=Path, default=REPOSITORY_ROOT / "checkpoints/contrastive/full-multilingual-v1/image-cache")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "digest/plane-pooling-analysis.json")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text())
    labels, probe, bootstrap = config["labels"], config["probe"], config["bootstrap"]

    gold_uids, gold_matrix = load_gold_labels(REPOSITORY_ROOT / config["gold_labels_csv"], labels)
    records, gold, population = resolve_population(
        REPOSITORY_ROOT / config["cohort_manifest"], gold_uids, gold_matrix
    )
    if population["manifest_sha256"] != config["manifest_sha256"]:
        raise SystemExit("Cohort manifest checksum does not match the frozen config")

    device = torch.device(arguments.device)
    towers: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    image_kwargs: dict[str, Any] | None = None
    for path in config["checkpoints"]:
        checkpoint_path = REPOSITORY_ROOT / path
        checkpoint = read_checkpoint(checkpoint_path)
        run_config = checkpoint_config(checkpoint)
        model, _, _ = build_model(
            run_config.image_model,
            run_config.text_model,
            run_config.embedding_dim,
            run_config.image_unfreeze_last,
            run_config.text_unfreeze_last,
            image_model_revision=run_config.image_model_revision,
            text_model_revision=run_config.text_model_revision,
            text_pooling=run_config.text_pooling,
            image_plane_pooling=run_config.image_plane_pooling,
            pretrained_backbones=not has_full_weights(checkpoint),
            local_files_only=True,
        )
        load_model_weights(model, checkpoint)
        name = checkpoint_path.parent.name
        if run_config.image_plane_pooling != "attention":
            # This script exists to read out learned plane-attention weights, which a
            # uniform-pooling run does not have. Say so, rather than failing later on a
            # None subscript inside the extraction loop.
            raise ValueError(
                f"{path} was trained with --image-plane-pooling "
                f"{run_config.image_plane_pooling!r}, so it has no plane attention to "
                "analyse; this script applies only to attention-pooling runs"
            )
        towers[name] = model.image_tower.eval().to(device)
        provenance[name] = {
            "checkpoint": path,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "image_unfreeze_last": run_config.image_unfreeze_last,
            "seed": run_config.seed,
        }
        image_kwargs = image_kwargs or run_config.image_kwargs

    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    weights_report: dict[str, Any] = {}
    for name, tower in towers.items():
        raw, marked, weights = plane_features_and_weights(
            tower, records, image_kwargs, arguments.image_cache_dir
        )
        weights_report[name] = weight_statistics(weights)
        sets = feature_sets(raw, marked, weights)
        scored = {}
        for key, matrix in sets.items():
            print(f"probing {name}/{key} ({matrix.shape[1]} features)", flush=True)
            scored[key] = score_arm(matrix, gold, labels, probe)
        reference = scored["uniform_mean_768"]
        for key, value in scored.items():
            results[f"{name}/{key}"] = {
                part: content for part, content in value.items() if part != "_oof_scores"
            }
            if key == "uniform_mean_768":
                continue
            comparison = paired_bootstrap(
                gold, labels, reference["_oof_scores"], value["_oof_scores"], bootstrap
            )
            comparison["macro_auroc_uniform_mean"] = reference["macro_auroc"]
            comparison["macro_auroc_this_arm"] = value["macro_auroc"]
            # Per-label intervals only for the arm the finding is about; the rest
            # would be thousands of intervals nobody reads.
            if key == "attention_pooled_768":
                comparison["per_label_auroc_difference"] = {
                    label: {
                        "difference": value["per_label"][label]["auroc"]
                        - reference["per_label"][label]["auroc"],
                        "ci95": label_bootstrap(
                            gold[:, index],
                            reference["_oof_scores"][:, index],
                            value["_oof_scores"][:, index],
                            bootstrap,
                        ),
                    }
                    for index, label in enumerate(labels)
                }
            comparisons[f"{name}/{key}"] = comparison

    write_json(
        arguments.output,
        {
            "schema_version": 1,
            "probe_version": config["probe_version"],
            "config": config,
            "config_sha256": sha256_file(arguments.config),
            "population": population,
            "study_uids": [record.study_uid for record in records],
            "provenance": provenance,
            "plane_attention_weights": weights_report,
            "results": results,
            "comparisons_against_uniform_mean_same_checkpoint": comparisons,
        },
    )
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
