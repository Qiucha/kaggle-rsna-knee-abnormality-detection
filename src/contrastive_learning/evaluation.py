from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_learning.config import TrainConfig
from contrastive_learning.data import KneeReportDataset, build_study_records, make_collator, split_records, split_records_fold
from contrastive_learning.manifest import read_cohort_manifest
from contrastive_learning.model import build_model
from contrastive_learning.training import (
    _image_kwargs,
    _move_batch,
    embedding_health_metrics,
    retrieval_diagnostics,
)


def _load_config(value: dict[str, Any]) -> TrainConfig:
    names = {field.name for field in fields(TrainConfig)}
    values = {key: item for key, item in value.items() if key in names}
    for name in ("data_root", "reports_csv", "cohort_manifest", "output_dir", "image_cache_dir", "resume"):
        if values.get(name) is not None:
            values[name] = Path(values[name])
    values["resume"] = None
    return TrainConfig(**values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _bootstrap_interval(values: list[float], seed: int = 2026) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(5000, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _records_for_split(config: TrainConfig, split: str):
    if config.cohort_manifest is not None:
        records, _ = read_cohort_manifest(config.cohort_manifest)
        return [record for record in records if record.split == split]
    records, _ = build_study_records(
        config.reports_csv,
        config.series_csv,
        config.series_root,
        report_column=config.report_column,
        validate_pixels=False,
        image_kwargs=_image_kwargs(config),
        max_studies=config.max_studies,
    )
    if config.fold_count is None:
        train, validation = split_records(records, config.val_fraction, config.seed)
    else:
        train, validation = split_records_fold(
            records, config.fold_count, config.fold_index, config.seed
        )
    if split == "train":
        return train
    if split == "validation":
        return validation
    raise ValueError("Legacy checkpoints only have train and validation splits")


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    split: str = "validation",
    output_path: Path | None = None,
    rankings_path: Path | None = None,
    image_cache_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = _load_config(checkpoint["config"])
    if image_cache_dir is not None:
        config.image_cache_dir = Path(image_cache_dir)
    records = _records_for_split(config, split)
    if not records:
        raise ValueError(f"No eligible records in split {split!r}")
    model, tokenizer, model_info = build_model(
        config.image_model,
        config.text_model,
        config.embedding_dim,
        config.image_unfreeze_last,
        config.text_unfreeze_last,
        image_model_revision=config.image_model_revision,
        text_model_revision=config.text_model_revision,
        text_pooling=config.text_pooling,
        pretrained_backbones="model_state_dict" not in checkpoint,
        local_files_only=True,
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(
        KneeReportDataset(
            records,
            _image_kwargs(config),
            image_cache_dir=config.image_cache_dir,
            memory_cache_size=(
                config.image_memory_cache_size or (64 if config.cache_images else 0)
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=make_collator(
            tokenizer, config.max_text_length, config.text_prefix, config.report_view
        ),
        pin_memory=torch.cuda.is_available(),
    )
    image_parts, text_parts, study_uids = [], [], []
    for batch in loader:
        images, input_ids, attention_mask = _move_batch(batch, device)
        image_parts.append(model.encode_images(images).cpu())
        text_parts.append(model.encode_text(input_ids, attention_mask).cpu())
        study_uids.extend(batch["study_uid"])
    metrics, rankings = retrieval_diagnostics(
        torch.cat(image_parts), torch.cat(text_parts), study_uids
    )
    metrics.update(
        embedding_health_metrics(
            torch.cat(image_parts), torch.cat(text_parts), model.logit_scale
        )
    )
    stored_metrics = checkpoint.get("metrics", {})
    count = len(study_uids)
    image_rows = rankings[:count]
    text_rows = rankings[count:]
    paired_r1 = [
        (image_row["expected_r@1"] + text_row["expected_r@1"]) / 2
        for image_row, text_row in zip(image_rows, text_rows, strict=True)
    ]
    result = {
        "schema_version": 1,
        "metric_tie_policy": "uniform_expected_credit",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": split,
        "studies": len(study_uids),
        "metrics": metrics,
        "stored_metrics": stored_metrics,
        "metric_deltas": {
            key: metrics[key] - float(stored_metrics[key])
            for key in metrics.keys() & stored_metrics.keys()
        },
        "queries_with_tied_candidates": sum(row["tied_candidates"] > 1 for row in rankings),
        "random_ranking_baseline": {
            "r@1": 1 / count,
            "r@5": min(5, count) / count,
            "r@10": min(10, count) / count,
            "mean_rank": (count + 1) / 2,
        },
        "bootstrap_95_ci": {
            "image_to_text_r@1": _bootstrap_interval(
                [row["expected_r@1"] for row in image_rows]
            ),
            "text_to_image_r@1": _bootstrap_interval(
                [row["expected_r@1"] for row in text_rows], seed=2027
            ),
            "mean_r@1": _bootstrap_interval(paired_r1, seed=2028),
        },
        "model_info": model_info,
    }
    output_path = output_path or checkpoint_path.parent / f"evaluation-{split}-{checkpoint_path.stem}.json"
    rankings_path = rankings_path or checkpoint_path.parent / f"rankings-{split}-{checkpoint_path.stem}.jsonl"
    if not overwrite and (output_path.exists() or rankings_path.exists()):
        raise FileExistsError("Evaluation output already exists; pass overwrite=True to replace it")
    _write_atomic(output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_atomic(
        rankings_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rankings),
    )
    return result
