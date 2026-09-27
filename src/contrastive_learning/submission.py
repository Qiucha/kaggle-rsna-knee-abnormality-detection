"""Offline image-only deployment of contrastive encoders and supervised probes.

The deployment representation is the mean of per-plane CLS+mean-patch features,
matching the project's strongest existing readout. No report encoder is needed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import read_json, sha256_file, write_atomic, write_json
from .checkpoints import checkpoint_config, has_full_weights, read_checkpoint
from .data import (
    KneeReportDataset,
    StudyRecord,
    pick_study_slots,
    plane_fallbacks,
    read_series_manifest,
)
from .model import _IMAGENET_MEAN, _IMAGENET_STD, freeze_then_unfreeze_last
from .preprocessing import PREPROCESSING_VERSION, decode_counts, resolve_laterality

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
UID = "StudyInstanceUID"


def validate_ids(frame: pd.DataFrame) -> list[str]:
    if UID not in frame or frame[UID].isna().any():
        raise ValueError("Missing StudyInstanceUID")
    ids = frame[UID].astype(str).tolist()
    if not ids or any(not uid.strip() for uid in ids) or len(set(ids)) != len(ids):
        raise ValueError("StudyInstanceUID must be nonempty and unique")
    return ids


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected = validate_ids(sample)
    if list(sample.columns) != [UID, *LABELS] or list(frame.columns) != list(
        sample.columns
    ):
        raise ValueError(
            "Submission columns must match the twelve-label competition schema"
        )
    if validate_ids(frame) != expected:
        raise ValueError("Submission IDs/order do not match sample_submission.csv")
    values = frame[LABELS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")


def _adapt(backbone, config: dict) -> None:
    if config["use_lora"]:
        from .lora import apply_lora_to_backbone

        apply_lora_to_backbone(
            backbone,
            config["image_unfreeze_last"],
            config["lora_rank"],
            config["lora_alpha"],
            dropout=config["lora_dropout"],
        )
    else:
        freeze_then_unfreeze_last(backbone, config["image_unfreeze_last"])


def load_encoder(checkpoint_path: Path, *, baseline: bool = False):
    """Restore just the image backbone; compact deltas require pinned local weights."""
    from transformers import AutoConfig, AutoModel

    checkpoint = read_checkpoint(checkpoint_path)
    config = checkpoint_config(checkpoint)
    architecture = AutoConfig.from_pretrained(
        config.image_model,
        revision=config.image_model_revision,
        local_files_only=True,
    )
    architecture.drop_path_rate = config.image_drop_path_rate
    if baseline or not has_full_weights(checkpoint):
        if not config.image_model_revision:
            raise ValueError(
                "Compact checkpoints must pin their pretrained image revision"
            )
        backbone = AutoModel.from_pretrained(
            config.image_model,
            revision=config.image_model_revision,
            config=architecture,
            local_files_only=True,
        )
    else:
        backbone = AutoModel.from_config(architecture)
    adaptation = {
        key: getattr(config, key)
        for key in (
            "use_lora",
            "image_unfreeze_last",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
        )
    }
    if not baseline:
        _adapt(backbone, adaptation)
        prefix = "image_tower.backbone."
        state = checkpoint.get(
            "model_state_dict", checkpoint.get("trainable_state_dict")
        )
        weights = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if has_full_weights(checkpoint):
            backbone.load_state_dict(weights, strict=True)
        else:
            required = {
                key for key, value in backbone.named_parameters() if value.requires_grad
            }
            if required != set(weights):
                raise ValueError(
                    f"Incomplete image delta: missing={required - set(weights)}, extra={set(weights) - required}"
                )
            result = backbone.load_state_dict(weights, strict=False)
            if result.unexpected_keys:
                raise ValueError(f"Unexpected image weights: {result.unexpected_keys}")
    return backbone.eval(), config, adaptation


@torch.inference_mode()
def image_features(backbone, images: torch.Tensor) -> torch.Tensor:
    batch, planes = images.shape[:2]
    images = images.reshape(batch * planes, *images.shape[2:]).float()
    mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    tokens = backbone(pixel_values=(images - mean) / std).last_hidden_state
    first_patch = 1 + int(getattr(backbone.config, "num_register_tokens", 0))
    return (
        torch.cat((tokens[:, 0], tokens[:, first_patch:].mean(1)), dim=-1)
        .reshape(batch, planes, -1)
        .mean(1)
    )


def extract_features(
    backbone, records, image_kwargs, device, batch_size=8, cache_dir=None
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = KneeReportDataset(records, image_kwargs, image_cache_dir=cache_dir)
    backbone.to(device).eval()
    parts = []
    for start in range(0, len(dataset), batch_size):
        images = torch.stack(
            [
                dataset[i]["images"]
                for i in range(start, min(start + batch_size, len(dataset)))
            ]
        )
        parts.append(image_features(backbone, images.to(device)).cpu().numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("Encoder produced nonfinite features")
    return result


def fit_head(features: np.ndarray, targets: np.ndarray, *, c=1.0):
    """Fit a standardized binary logistic probe per label, export portable tensors."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    coefficients, intercepts = [], []
    for column in targets.T:
        if set(np.unique(column)) != {0, 1}:
            raise ValueError("Every training label must contain both binary classes")
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000).fit(
            scaled, column
        )
        coefficients.append(model.coef_[0])
        intercepts.append(model.intercept_[0])
    return {
        key: torch.tensor(value, dtype=torch.float64)
        for key, value in {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "coef": np.asarray(coefficients),
            "intercept": np.asarray(intercepts),
        }.items()
    }


def predict_head(features: np.ndarray, head: dict) -> np.ndarray:
    kind = head.get("kind", "logistic_v1")
    if kind == "mean_heads_v1":
        if not head["heads"]:
            raise ValueError("Empty head ensemble")
        return np.mean(
            [predict_head(features, member) for member in head["heads"]], axis=0
        )
    if kind == "dense_v1":
        values = torch.as_tensor(features, dtype=torch.float32)
        values = (values - head["mean"]) / head["scale"]
        for i, layer in enumerate(head["layers"]):
            values = torch.nn.functional.linear(values, layer["weight"], layer["bias"])
            if i < len(head["layers"]) - 1:
                values = values.relu()
        return values.sigmoid().numpy()
    if kind != "logistic_v1":
        raise ValueError(f"Unsupported head kind: {kind}")
    values = torch.as_tensor(features, dtype=torch.float64)
    logits = ((values - head["mean"]) / head["scale"]) @ head["coef"].T + head[
        "intercept"
    ]
    return logits.sigmoid().numpy()


def benchmark_head(features, targets, *, folds=8, seed=2026, c=1.0):
    """Per-label stratified OOF probabilities; scalers see only their training fold."""
    from sklearn.model_selection import StratifiedKFold

    oof = np.empty_like(targets, dtype=float)
    assignments = np.empty_like(targets, dtype=int)
    for label in range(targets.shape[1]):
        column = targets[:, label]
        if (
            not np.isin(column, [0, 1]).all()
            or min((column == 0).sum(), (column == 1).sum()) < folds
            or folds < 2
        ):
            raise ValueError(
                f"Label {label} needs at least {folds} examples of each class"
            )
        for fold, (train, val) in enumerate(
            StratifiedKFold(folds, shuffle=True, random_state=seed).split(
                features, column
            )
        ):
            head = fit_head(features[train], targets[train, label : label + 1], c=c)
            oof[val, label] = predict_head(features[val], head)[:, 0]
            assignments[val, label] = fold
    return oof, assignments


def score_predictions(targets, probabilities):
    from sklearn.metrics import average_precision_score, roc_auc_score

    per_label = {
        label: {
            "auroc": float(roc_auc_score(targets[:, i], probabilities[:, i])),
            "average_precision": float(
                average_precision_score(targets[:, i], probabilities[:, i])
            ),
        }
        for i, label in enumerate(LABELS)
    }
    return {
        "macro_auroc": float(np.mean([row["auroc"] for row in per_label.values()])),
        "per_label": per_label,
    }


def export_member(path: Path, backbone, config, adaptation, head) -> None:
    payload = {
        "architecture": backbone.config.to_dict(),
        "adaptation": adaptation,
        "backbone": {
            key: value.detach().cpu() for key, value in backbone.state_dict().items()
        },
        "head": head,
        "image_kwargs": config.image_kwargs,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    write_atomic(path, lambda temporary: torch.save(payload, temporary))


def load_member(path: Path, device):
    from transformers import AutoConfig, AutoModel

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["preprocessing_version"] != PREPROCESSING_VERSION:
        raise ValueError("Bundle preprocessing version differs from runtime")
    config = dict(payload["architecture"])
    model_type = config.pop("model_type")
    backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **config))
    _adapt(backbone, payload["adaptation"])
    backbone.load_state_dict(payload["backbone"], strict=True)
    return backbone.eval().to(device), payload


def predict_submission(
    bundle_dir: Path, data_root: Path, output: Path, *, device="cpu", batch_size=8
):
    """Predict in sample order, fail explicitly on missing planes/decoding errors.

    A selected plane that does not decode at all falls back to the next ranked
    series of that plane; the run fails only when none decodes. Decode counters
    are reset here and reported (stderr and the benchmark JSON) at the end.
    """
    started = time.perf_counter()
    bundle = read_json(bundle_dir / "bundle.json")
    if (
        bundle["schema_version"] != 1
        or bundle["labels"] != LABELS
        or not bundle["members"]
    ):
        raise ValueError("Unsupported or empty submission bundle")
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={UID: str})
    ids = validate_ids(sample)
    validate_submission(sample, sample)
    test = pd.read_csv(data_root / "test.csv", dtype={UID: str})
    if set(validate_ids(test)) != set(ids):
        raise ValueError("test.csv and sample_submission.csv IDs differ")
    records = submission_records(
        ids, read_series_manifest(data_root / "test_series.csv"), data_root / "test_series"
    )
    decode_counts(reset=True)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    predictions = []
    for member in bundle["members"]:
        path = bundle_dir / member["file"]
        if sha256_file(path) != member["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {path.name}")
        backbone, payload = load_member(path, device)
        features = extract_features(
            backbone, records, payload["image_kwargs"], device, batch_size
        )
        predictions.append(predict_head(features, payload["head"]))
        del backbone, payload
    submission = pd.DataFrame(np.mean(predictions, axis=0), columns=LABELS)
    submission.insert(0, UID, ids)
    validate_submission(submission, sample)
    write_atomic(output, lambda temporary: submission.to_csv(temporary, index=False))
    report = {
        "studies": len(ids),
        "members": len(predictions),
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
        "submission_sha256": sha256_file(output),
        "bundle_sha256": sha256_file(bundle_dir / "bundle.json"),
        # Summed over members: every member re-reads every study.
        "decode_counts": decode_counts(),
    }
    print(
        "decode_counts", json.dumps(report["decode_counts"], sort_keys=True),
        file=sys.stderr, flush=True,
    )
    write_json(output.with_suffix(".benchmark.json"), report)
    return report


def submission_records(ids, series, series_root: Path) -> list[StudyRecord]:
    """One inference record per study, with ranked fallback series per plane.

    The selected series are exactly the training rule's (:func:`pick_study_slots`);
    fallbacks only matter when a selected plane does not decode at all, which
    would otherwise abort the whole submission.
    """
    records = []
    for uid in ids:
        rows = series.get(uid, [])
        slots = pick_study_slots(uid, rows, series_root)
        if slots is None:
            raise ValueError(
                f"Study {uid} lacks a usable three-plane input; submission not written"
            )
        records.append(
            StudyRecord(
                uid, "", slots, resolve_laterality(slots),
                fallbacks=plane_fallbacks(slots, uid, rows, series_root),
            )
        )
    return records
