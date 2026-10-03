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
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from contrastive_learning.artifacts import sha256_file, write_json
from contrastive_learning.checkpoints import (
    checkpoint_config,
    has_full_weights,
    load_model_weights,
    read_checkpoint,
)
from contrastive_learning.data import KneeReportDataset
from contrastive_learning.model import build_model

# --------------------------------------------------------------------------- #
# Population and labels
# --------------------------------------------------------------------------- #


def load_gold_labels(csv_path: Path, labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Read the binary gold label matrix, refusing the blank annotation template.

    ``data/drop.csv`` carries the same 58 studies and the same columns with every
    label cell empty. Read as labels it yields a silent all-negative target, so a
    non-zero positive count per label is asserted rather than assumed.
    """
    import pandas as pd

    frame = pd.read_csv(csv_path)
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing label columns: {missing}")
    uids = frame["StudyInstanceUID"].astype(str).tolist()
    if len(set(uids)) != len(uids):
        raise ValueError(f"Duplicate StudyInstanceUID in {csv_path}")
    matrix = frame[labels].to_numpy(dtype=float)
    if not np.isin(matrix, (0.0, 1.0)).all():
        raise ValueError(f"Gold labels in {csv_path} must be binary 0/1")
    positives = matrix.sum(axis=0)
    empty = [label for label, count in zip(labels, positives) if count == 0]
    if empty:
        raise ValueError(
            f"{csv_path} has zero positives for {empty}; this is the blank annotation "
            "template, not a label set"
        )
    return uids, matrix


def resolve_population(
    manifest: Path, gold_uids: list[str], gold_matrix: np.ndarray,
    population_splits: tuple[str, ...] = ("train", "validation"),
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    """Intersect the gold studies with the cohort, keeping ``population_splits`` only.

    Locked-test gold studies are counted and then dropped before any pixel is read,
    so the test gate is enforced by construction rather than by discipline. ``test``
    can never be part of the population regardless of what a config asks for.

    The default is the exposed population every published probe used. A cohort that
    holds the gold studies out of representation training parks them in ``excluded``
    instead, and passes ``("excluded",)`` -- see
    ``digest/gold-holdout-probe-protocol.md``. The split is counted explicitly rather
    than defaulted, so an unrecognised value raises instead of silently producing an
    empty population, which is this project's characteristic failure shape.
    """
    from contrastive_learning.manifest import read_cohort_manifest

    if "test" in population_splits:
        raise SystemExit("The locked test split can never be part of a probe population")

    records, metadata = read_cohort_manifest(manifest)
    by_uid = {record.study_uid: record for record in records}
    index_of = {uid: index for index, uid in enumerate(gold_uids)}

    kept_records, kept_rows = [], []
    counts = {"train": 0, "validation": 0, "test": 0, "excluded": 0, "absent": 0}
    for uid in gold_uids:
        record = by_uid.get(uid)
        if record is None:
            counts["absent"] += 1
            continue
        if record.split not in counts:
            raise SystemExit(f"Unhandled cohort split in probe population: {record.split!r}")
        counts[record.split] += 1
        if record.split not in population_splits:
            continue
        kept_records.append(record)
        kept_rows.append(index_of[uid])
    if not kept_records:
        raise SystemExit(
            f"Probe population is empty for splits {population_splits}; counts={counts}"
        )
    return (
        kept_records,
        gold_matrix[kept_rows],
        {
            "manifest_sha256": metadata["manifest_sha256"],
            "gold_studies": len(gold_uids),
            "split_counts": counts,
            "population_splits": list(population_splits),
            "population_size": len(kept_records),
        },
    )


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def _plane_features(backbone: torch.nn.Module, images: torch.Tensor, mean, std) -> torch.Tensor:
    """Per-plane DINOv2 features, exactly as ``ImageTower.forward`` computes them.

    Returns ``[B, planes, 2 * hidden]``: each plane's CLS token concatenated with
    the mean over its patch tokens, before any pooling across planes.
    """
    batch, planes = images.shape[:2]
    flat = images.reshape(batch * planes, *images.shape[2:]).float()
    if flat.device != mean.device:
        flat = flat.to(mean.device)
    flat = (flat - mean) / std
    tokens = backbone(pixel_values=flat).last_hidden_state
    features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1)
    return features.reshape(batch, planes, -1)


@torch.inference_mode()
def extract_features(
    records: list[Any],
    image_kwargs: dict[str, Any],
    image_cache_dir: Path,
    towers: dict[str, Any],
    batch_size: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """Encode every study once per arm, sharing one pass over the cached pixels.

    Both poolings and the projection are derived from the same per-plane features,
    so an arm costs one backbone pass rather than three.
    """
    dataset = KneeReportDataset(records, image_kwargs, image_cache_dir=image_cache_dir)
    dimensions = {
        name: 2 * (tower["backbone"] if isinstance(tower, dict) else tower.backbone).config.hidden_size
        for name, tower in towers.items()
    }
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {
            f"uniform_mean_{dimensions[name]}": [],
            f"plane_attention_{dimensions[name]}": [],
            "projection_256": [],
        }
        for name in towers
    }
    for start in range(0, len(dataset), batch_size):
        items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([item["images"] for item in items])
        for name, tower in towers.items():
            if isinstance(tower, dict):  # unadapted backbone: no learned parts exist
                planes = _plane_features(tower["backbone"], images, tower["mean"], tower["std"])
                collected[name][f"uniform_mean_{dimensions[name]}"].append(planes.mean(dim=1))
                continue
            planes = _plane_features(tower.backbone, images, tower.mean, tower.std)
            collected[name][f"uniform_mean_{dimensions[name]}"].append(planes.mean(dim=1))
            if getattr(tower, "plane_pooling", "attention") == "attention":
                marked = planes + tower.plane_embeddings[: planes.shape[1]]
                weights = tower.plane_attention(marked).squeeze(-1).softmax(dim=1)
                pooled = torch.einsum("bp,bpd->bd", weights, marked)
                collected[name][f"plane_attention_{dimensions[name]}"].append(pooled)
            else:
                # A uniform-pooling tower has no attention parameters, so
                # plane_attention_768 does not exist for it and is left empty; the
                # return below drops feature sets with no parts. projection_256 must
                # read the pooling the tower actually uses, which is the plane mean.
                pooled = planes.mean(dim=1)
            collected[name]["projection_256"].append(tower.projection(pooled))
    return {
        name: {key: torch.cat(parts).float().cpu().numpy() for key, parts in sets.items() if parts}
        for name, sets in collected.items()
    }


def build_towers(config: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    """Instantiate the unadapted backbone and every checkpoint's image tower."""
    from transformers import AutoModel

    from contrastive_learning.model import _IMAGENET_MEAN, _IMAGENET_STD

    towers: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    before = config["arms"]["before_cl"]
    backbone = AutoModel.from_pretrained(
        before["image_model"], revision=before["image_model_revision"], local_files_only=True
    ).eval().to(device)
    towers["before_cl/dinov2-unadapted"] = {
        "backbone": backbone,
        "mean": torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device),
        "std": torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device),
    }
    provenance["before_cl/dinov2-unadapted"] = {
        "kind": "unadapted pretrained backbone",
        "image_model": before["image_model"],
        "image_model_revision": before["image_model_revision"],
    }

    image_kwargs = None
    # Every arm the config declares except the unadapted one, rather than a hardcoded
    # pair: an arm added to a config must not be silently skipped here, which is what
    # a fixed list did. Order follows the config so feature keys stay reproducible.
    checkpoint_arms = [name for name in config["arms"] if name != "before_cl"]
    for arm in checkpoint_arms:
        for path in config["arms"][arm]["checkpoints"]:
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
                use_lora=getattr(run_config, "use_lora", False),
                lora_rank=getattr(run_config, "lora_rank", 16),
                lora_alpha=getattr(run_config, "lora_alpha", 32.0),
                lora_dropout=getattr(run_config, "lora_dropout", 0.0),
                image_drop_path_rate=getattr(run_config, "image_drop_path_rate", 0.0),
                pretrained_backbones=not has_full_weights(checkpoint),
                local_files_only=True,
            )
            load_model_weights(model, checkpoint)
            name = f"{arm}/{checkpoint_path.parent.name}"
            towers[name] = model.image_tower.eval().to(device)
            provenance[name] = {
                "kind": "contrastive checkpoint",
                "checkpoint": path,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "image_unfreeze_last": run_config.image_unfreeze_last,
                "text_unfreeze_last": run_config.text_unfreeze_last,
                "use_lora": getattr(run_config, "use_lora", False),
                "lora_rank": getattr(run_config, "lora_rank", None) if getattr(run_config, "use_lora", False) else None,
                "seed": run_config.seed,
                "epoch": checkpoint.get("epoch"),
            }
            # Only the image tower is ever read below, and provenance is already
            # recorded. Releasing the text tower and the checkpoint keeps peak memory
            # proportional to the image towers retained rather than to the text
            # backbone, which is an order of magnitude larger and is rebuilt for every
            # checkpoint in this loop.
            model.text_tower = None
            del model, checkpoint
            gc.collect()
            # Each arm carries its OWN preprocessing. Recording only the first
            # checkpoint's was a silent defect: a config mixing crops (e.g. the 130mm
            # baselines against the 150mm re-baseline) extracted every arm at whichever
            # crop happened to load first, feeding an encoder pixels it never trained on
            # and producing a plausible, wrong number with no error.
            provenance[name]["image_kwargs"] = dict(run_config.image_kwargs)
            if image_kwargs is None:
                image_kwargs = run_config.image_kwargs
                provenance["_image_kwargs"] = dict(image_kwargs)
                provenance["_cohort_manifest"] = str(run_config.cohort_manifest)
    return towers, provenance


def _preprocessing_key(kwargs: dict[str, Any]) -> str:
    """A hashable identity for one preprocessing, used to group arms for extraction."""
    return json.dumps(kwargs, sort_keys=True, default=list)


def _preprocessing_tag(kwargs: dict[str, Any], varying: list[str]) -> str:
    """A short human-readable label built only from the fields that actually differ."""
    parts = []
    for field in sorted(varying):
        value = kwargs[field]
        if field == "crop_mm":
            parts.append(f"crop{value:g}mm")
        elif field == "image_size":
            parts.append(f"size{value}")
        elif field == "slices_per_plane":
            parts.append(f"slices{value}")
        elif field == "slice_band":
            low, high = value
            parts.append(f"band{low:g}-{high:g}")
        else:
            parts.append(f"{field}{value}")
    return "-".join(parts)


def extract_features_by_preprocessing(
    records: list[Any],
    towers: dict[str, Any],
    provenance: dict[str, Any],
    default_cache_dir: Path,
    arm_cache_dirs: dict[str, Path] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract features once per DISTINCT preprocessing, not once globally.

    When every arm shares one preprocessing -- the case for every probe config written
    before 2026-09-15 -- this makes exactly one extraction pass over all towers and
    returns byte-identical output under byte-identical names, so existing artifacts
    reproduce. Only a config that genuinely mixes preprocessings takes the new path.

    In that mixed case the unadapted backbone is extracted once per preprocessing and
    its name is suffixed with a tag, because "unadapted DINOv2 at 130mm" and "at 150mm"
    are different feature sets and must not share one key.

    Cache safety: the image cache fingerprint includes ``image_kwargs``
    (``data.py`` ``_cache_path``), so pointing a group at the wrong cache directory can
    only cause a miss and a re-decode -- never the wrong pixels.
    """
    unadapted_name = "before_cl/dinov2-unadapted"
    unadapted = towers.get(unadapted_name)
    checkpoint_towers = {name: tower for name, tower in towers.items() if name != unadapted_name}

    groups: dict[str, dict[str, Any]] = {}
    for name, tower in checkpoint_towers.items():
        kwargs = provenance[name]["image_kwargs"]
        key = _preprocessing_key(kwargs)
        groups.setdefault(key, {"image_kwargs": kwargs, "towers": {}})
        groups[key]["towers"][name] = tower

    if not groups:
        raise SystemExit("No checkpoint arms to extract; a probe config must declare at least one")

    # Only the fields that actually vary become part of a tag, so a two-crop config
    # reads "crop130mm"/"crop150mm" rather than repeating the shared settings.
    all_kwargs = [g["image_kwargs"] for g in groups.values()]
    varying = [f for f in all_kwargs[0] if len({_preprocessing_key({f: k[f]}) for k in all_kwargs}) > 1]

    features: dict[str, dict[str, np.ndarray]] = {}
    # Every arm's floor is the unadapted backbone extracted at ITS OWN preprocessing.
    # Recorded here and carried in provenance (which round-trips through the features
    # npz) so the scoring code below never has to guess which floor an arm belongs to.
    unadapted_for: dict[str, str] = {}
    for key, group in groups.items():
        tag = _preprocessing_tag(group["image_kwargs"], varying) if len(groups) > 1 else ""
        pass_towers = dict(group["towers"])
        floor_name = f"{unadapted_name}@{tag}" if tag else unadapted_name
        if unadapted is not None:
            pass_towers[floor_name] = unadapted
            for member in pass_towers:
                unadapted_for[member] = floor_name
        cache_dir = default_cache_dir
        if arm_cache_dirs:
            named = {arm_cache_dirs[n] for n in group["towers"] if n in arm_cache_dirs}
            if len(named) > 1:
                raise SystemExit(
                    f"Arms sharing one preprocessing declare different image_cache_dir values: {named}"
                )
            if named:
                cache_dir = named.pop()
        print(f"extracting {len(pass_towers)} towers at {group['image_kwargs']} cache={cache_dir}")
        features.update(extract_features(records, group["image_kwargs"], cache_dir, pass_towers))
    provenance["_unadapted_for"] = unadapted_for
    return features


# --------------------------------------------------------------------------- #
# Probe and metrics
# --------------------------------------------------------------------------- #


def out_of_fold_scores(features: np.ndarray, targets: np.ndarray, probe: dict[str, Any]) -> np.ndarray:
    """One held-out decision-function score per study, from stratified folds.

    Standardisation is fitted inside each training fold. Fitting it once over the
    whole population would leak the held-out studies' feature distribution into
    their own predictions, which is the usual silent failure of probe code.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    scores = np.full(len(targets), np.nan, dtype=float)
    splitter = StratifiedKFold(
        n_splits=probe["folds"], shuffle=True, random_state=probe["fold_seed"]
    )
    for train_index, test_index in splitter.split(features, targets):
        scaler = StandardScaler().fit(features[train_index])
        model = LogisticRegression(
            penalty=probe["penalty"],
            C=probe["C"],
            solver=probe["solver"],
            max_iter=probe["max_iter"],
            class_weight=probe["class_weight"],
        ).fit(scaler.transform(features[train_index]), targets[train_index])
        scores[test_index] = model.decision_function(scaler.transform(features[test_index]))
    if np.isnan(scores).any():
        raise RuntimeError("Every study must receive exactly one held-out score")
    return scores


def _auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Midrank AUROC, or NaN when the target has a single class.

    Every reported point estimate goes through this function, so the headline
    numbers are sklearn's. The bootstrap uses :func:`_fast_auroc` instead, which
    computes the same quantity several times faster; :func:`score_arm` asserts
    the two agree on every label before any interval is built.
    """
    from sklearn.metrics import roc_auc_score

    if np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets, scores))


def _fast_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    """The same midrank AUROC, as a rank-sum, for the millions of bootstrap fits.

    AUROC is the Mann-Whitney statistic: the mean rank of the positives, corrected
    for their own ordering and scaled by the number of comparable pairs. Average
    ranks are what makes it midrank, so tied scores share credit exactly as
    ``roc_auc_score`` gives it -- this is not the notebook's ``argsort`` shortcut.
    """
    from scipy.stats import rankdata

    positives = targets == 1
    n_positive = float(positives.sum())
    n_negative = float(len(targets) - n_positive)
    if n_positive == 0.0 or n_negative == 0.0:
        return float("nan")
    ranks = rankdata(scores)
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1.0) / 2.0)
        / (n_positive * n_negative)
    )


def score_arm(
    features: np.ndarray, gold: np.ndarray, labels: list[str], probe: dict[str, Any]
) -> dict[str, Any]:
    """Pooled out-of-fold AUROC and AUPRC for every label of one feature set."""
    from sklearn.metrics import average_precision_score

    scores = np.column_stack(
        [out_of_fold_scores(features, gold[:, index], probe) for index in range(len(labels))]
    )
    per_label = {}
    for index, label in enumerate(labels):
        targets = gold[:, index]
        auroc = _auroc(targets, scores[:, index])
        # The bootstrap below runs on the rank-sum implementation. Prove here, on
        # real scores, that it is the same number before trusting any interval.
        if abs(auroc - _fast_auroc(targets, scores[:, index])) > 1e-9:
            raise RuntimeError(f"Midrank AUROC implementations disagree on {label!r}")
        per_label[label] = {
            "positives": int(targets.sum()),
            "negatives": int(len(targets) - targets.sum()),
            "auroc": auroc,
            "auprc": float(average_precision_score(targets, scores[:, index])),
        }
    aurocs = [value["auroc"] for value in per_label.values()]
    return {
        "per_label": per_label,
        "macro_auroc": float(np.nanmean(aurocs)),
        "macro_auprc": float(np.mean([value["auprc"] for value in per_label.values()])),
        "_oof_scores": scores,
    }


def paired_bootstrap(
    gold: np.ndarray,
    labels: list[str],
    left: np.ndarray,
    right: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Bootstrap the macro-AUROC difference over studies, resampling both arms together.

    The same resampled study indices score both arms in every replicate, so the
    interval describes the paired difference rather than the difference of two
    independent intervals. A replicate that leaves a label single-class drops that
    label from that replicate's macro average, and the count is reported.
    """
    generator = np.random.default_rng(settings["seed"])
    size = gold.shape[0]
    differences, left_values, right_values = [], [], []
    degenerate = 0
    for _ in range(settings["resamples"]):
        index = generator.integers(0, size, size=size)
        resampled_gold = gold[index]
        left_scores, right_scores = [], []
        for column in range(len(labels)):
            targets = resampled_gold[:, column]
            if np.unique(targets).size < 2:
                degenerate += 1
                continue
            left_scores.append(_fast_auroc(targets, left[index, column]))
            right_scores.append(_fast_auroc(targets, right[index, column]))
        if not left_scores:
            continue
        left_values.append(float(np.mean(left_scores)))
        right_values.append(float(np.mean(right_scores)))
        differences.append(right_values[-1] - left_values[-1])
    array = np.asarray(differences)
    return {
        "macro_auroc_difference_mean": float(array.mean()),
        "macro_auroc_difference_ci95": [
            float(np.percentile(array, 2.5)),
            float(np.percentile(array, 97.5)),
        ],
        "fraction_of_replicates_favouring_right": float((array > 0).mean()),
        "single_class_label_replicates_dropped": int(degenerate),
        "resamples": len(array),
        "unit": settings["unit"],
        "limitation": settings["limitation"],
    }


def label_bootstrap(
    targets: np.ndarray, left: np.ndarray, right: np.ndarray, settings: dict[str, Any]
) -> list[float]:
    """Paired per-label AUROC difference interval, same resampling contract."""
    generator = np.random.default_rng(settings["seed"])
    size = len(targets)
    differences = []
    for _ in range(settings["resamples"]):
        index = generator.integers(0, size, size=size)
        resampled = targets[index]
        if np.unique(resampled).size < 2:
            continue
        differences.append(
            _fast_auroc(resampled, right[index]) - _fast_auroc(resampled, left[index])
        )
    if not differences:
        return [float("nan"), float("nan")]
    array = np.asarray(differences)
    return [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]


def _notebook_argsort_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Section 6 of the EDA notebook's AUROC, reproduced verbatim for comparison.

    The difference from midrank is the two ``argsort`` lines: they hand every tied
    score a distinct consecutive rank in whatever order the sort happened to
    produce, instead of the average rank the tied block shares. Carried here only
    so the size of that difference is a measured number in the artifact rather
    than an assertion in prose.
    """
    if np.unique(targets).size < 2:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_positive = targets.sum()
    n_negative = len(targets) - n_positive
    return float(
        (ranks[targets == 1].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def lm_extraction_reference(
    lm_dir: Path, uids: list[str], gold: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """Score the LM report-reading labels against gold on the same studies.

    This is the text-side ceiling for the same task: those models read the report
    that states the findings, so what they reach bounds label noise rather than
    image difficulty.

    Reported twice, deliberately. LM label outputs are heavily tied -- one column
    carries 41 identical ``0.0`` values over 56 studies -- and the notebook's
    argsort AUROC resolves those ties by sort order. Scoring both ways puts the
    size of that difference in the artifact.
    """
    import pandas as pd

    reference = {}
    for path in sorted(lm_dir.glob("*.csv")):
        frame = pd.read_csv(path).set_index("StudyInstanceUID")
        if not set(uids).issubset(frame.index):
            continue
        aligned = frame.loc[uids, labels].to_numpy(dtype=float)
        per_label = {
            label: _auroc(gold[:, index], aligned[:, index]) for index, label in enumerate(labels)
        }
        argsort = {
            label: _notebook_argsort_auroc(gold[:, index], aligned[:, index])
            for index, label in enumerate(labels)
        }
        differences = [argsort[label] - per_label[label] for label in labels]
        reference[path.name] = {
            "sha256": sha256_file(path),
            "per_label_auroc": per_label,
            "macro_auroc": float(np.nanmean(list(per_label.values()))),
            "notebook_argsort_per_label_auroc": argsort,
            "notebook_argsort_macro_auroc": float(np.nanmean(list(argsort.values()))),
            "notebook_argsort_minus_midrank": {
                "macro": float(np.nanmean(differences)),
                "max_absolute_per_label": float(np.nanmax(np.abs(differences))),
                "labels_affected": int(sum(abs(value) > 1e-12 for value in differences)),
                "labels_total": len(labels),
            },
        }
    return reference


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


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
        stored = np.load(arguments.features, allow_pickle=True)
        features = json.loads(str(stored["index"]))
        features = {name: {key: stored[f"{name}|{key}"] for key in keys} for name, keys in features.items()}
        provenance = json.loads(str(stored["provenance"]))
        order = [str(uid) for uid in stored["study_uids"]]
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
        comparison["per_label_auroc_difference"] = {
            label: {
                "difference": value["per_label"][label]["auroc"]
                - results[baseline_key]["per_label"][label]["auroc"],
                "ci95": label_bootstrap(
                    gold[:, index], baseline_scores[:, index], value["_oof_scores"][:, index],
                    config["bootstrap"],
                ),
            }
            for index, label in enumerate(labels)
        }
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
        "results": {
            key: {name: part for name, part in value.items() if name != "_oof_scores"}
            for key, value in results.items()
        },
        "comparisons_against_unadapted_dinov2": comparisons,
        "lm_label_extraction_reference": lm_extraction_reference(
            REPOSITORY_ROOT / "data/lm_label", [record.study_uid for record in records], gold, labels
        ),
    }
    write_json(arguments.output, payload)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
