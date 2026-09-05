from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_learning.config import MULTILINGUAL_TEXT_MODELS, TrainConfig
from contrastive_learning.data import (
    KneeReportDataset,
    build_study_records,
    make_collator,
    split_records,
    split_records_fold,
)
from contrastive_learning.model import build_model, symmetric_contrastive_loss
from contrastive_learning.manifest import read_cohort_manifest
from contrastive_learning.preprocessing import augment_study_images
from contrastive_learning.run_state import initialize_run, update_run_manifest
from contrastive_learning.text import REPORT_VIEWS
from contrastive_learning.selection import is_better


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _image_kwargs(config: TrainConfig) -> dict[str, Any]:
    return {
        "image_size": config.image_size,
        "crop_mm": config.crop_mm,
        "slices_per_plane": config.slices_per_plane,
        "slice_band": (config.slice_band_low, config.slice_band_high),
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["images"].to(device, non_blocking=True),
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
    )


def retrieval_metrics(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> dict[str, float]:
    if image_embeddings.ndim != 2 or image_embeddings.shape != text_embeddings.shape:
        raise ValueError("Retrieval embeddings must be paired rank-two tensors with matching shapes")
    if not torch.isfinite(image_embeddings).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("Retrieval embeddings must be finite")
    similarities = image_embeddings.float() @ text_embeddings.float().T

    def direction(matrix: torch.Tensor, prefix: str) -> dict[str, float]:
        greater, tied, ranks = _tie_aware_ranks(matrix)
        out = {
            f"{prefix}_median_rank": float(ranks.float().median().item()),
            f"{prefix}_mean_rank": float(ranks.float().mean().item()),
        }
        for k in (1, 5, 10):
            expected_hits = (k - greater).clamp(min=0)
            expected_hits = torch.minimum(expected_hits, tied) / tied
            out[f"{prefix}_r@{k}"] = float(expected_hits.float().mean().item())
        return out

    metrics = direction(similarities, "image_to_text")
    metrics.update(direction(similarities.T, "text_to_image"))
    metrics["mean_r@1"] = (metrics["image_to_text_r@1"] + metrics["text_to_image_r@1"]) / 2
    return metrics


def _tie_aware_ranks(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diagonal = matrix.diagonal().unsqueeze(1)
    greater = (matrix > diagonal).sum(dim=1)
    tied = (matrix == diagonal).sum(dim=1)
    ranks = 1 + greater + (tied - 1).float() / 2
    return greater, tied, ranks


def retrieval_diagnostics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    study_uids: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return aggregate retrieval metrics and auditable per-query tie/rank records."""
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Retrieval embeddings must be rank-two tensors")
    if image_embeddings.shape != text_embeddings.shape or len(study_uids) != len(image_embeddings):
        raise ValueError("Retrieval embeddings and study_uids must describe paired examples")
    similarities = image_embeddings.float() @ text_embeddings.float().T
    rows: list[dict[str, Any]] = []
    for matrix, direction in (
        (similarities, "image_to_text"),
        (similarities.T, "text_to_image"),
    ):
        greater, tied, ranks = _tie_aware_ranks(matrix)
        top_indices = torch.argsort(matrix, dim=1, descending=True, stable=True)[:, 0]
        for index, uid in enumerate(study_uids):
            row: dict[str, Any] = {
                "direction": direction,
                "query_uid": uid,
                "positive_uid": uid,
                "top_uid": study_uids[int(top_indices[index])],
                "positive_similarity": float(matrix[index, index]),
                "top_similarity": float(matrix[index, top_indices[index]]),
                "greater_candidates": int(greater[index]),
                "tied_candidates": int(tied[index]),
                "expected_rank": float(ranks[index]),
            }
            for k in (1, 5, 10):
                row[f"expected_r@{k}"] = min(
                    max(k - int(greater[index]), 0) / int(tied[index]), 1.0
                )
            rows.append(row)
    return retrieval_metrics(image_embeddings, text_embeddings), rows


def embedding_health_metrics(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, logit_scale: torch.Tensor
) -> dict[str, float]:
    similarities = image_embeddings.float() @ text_embeddings.float().T
    diagonal = similarities.diagonal()
    off_diagonal = similarities[
        ~torch.eye(len(similarities), dtype=torch.bool, device=similarities.device)
    ]
    scaled = logit_scale.detach().float().exp().clamp(max=100) * similarities
    return {
        "validation_loss": float(symmetric_contrastive_loss(scaled)),
        "positive_similarity_mean": float(diagonal.mean()),
        "negative_similarity_mean": float(off_diagonal.mean()) if len(off_diagonal) else 0.0,
        "image_embedding_dimension_std_mean": float(image_embeddings.float().std(dim=0).mean()),
        "text_embedding_dimension_std_mean": float(text_embeddings.float().std(dim=0).mean()),
        "inverse_temperature": float(logit_scale.detach().float().exp().clamp(max=100)),
    }


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    image_parts, text_parts = [], []
    for batch in loader:
        images, input_ids, attention_mask = _move_batch(batch, device)
        image_parts.append(model.encode_images(images).cpu())
        text_parts.append(model.encode_text(input_ids, attention_mask).cpu())
    if not image_parts:
        raise RuntimeError("Validation loader produced no batches")
    image_embeddings = torch.cat(image_parts)
    text_embeddings = torch.cat(text_parts)
    metrics = retrieval_metrics(image_embeddings, text_embeddings)
    metrics.update(embedding_health_metrics(image_embeddings, text_embeddings, model.logit_scale))
    return metrics


def _optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("image_tower.backbone"):
            tower = "image"
        elif name.startswith("text_tower.backbone"):
            tower = "text"
        else:
            tower = "projection"
        use_decay = parameter.ndim >= 2 and "norm" not in name.casefold()
        groups.setdefault((tower, use_decay), []).append(parameter)
    learning_rates = {
        "image": config.lr_image,
        "text": config.lr_text,
        "projection": config.lr_projection,
    }
    parameter_groups = [
        {
            "params": parameters,
            "lr": learning_rates[tower],
            "weight_decay": config.weight_decay if use_decay else 0.0,
            "group_name": f"{tower}_{'decay' if use_decay else 'no_decay'}",
        }
        for (tower, use_decay), parameters in groups.items()
    ]
    return torch.optim.AdamW(parameter_groups)


def _scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    config: TrainConfig,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    best_score: float,
    run_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
            "epoch": epoch,
            "metrics": metrics,
            "config": config.as_dict(),
            "run_id": run_id,
            "best_score": best_score,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    if config.trainable_checkpoints:
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        payload["state_dict_format"] = "trainable_delta_v1"
        payload["trainable_state_dict"] = {
            name: value for name, value in model.state_dict().items() if name in trainable
        }
    else:
        payload["state_dict_format"] = "full_v1"
        payload["model_state_dict"] = model.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(config: TrainConfig) -> dict[str, Any]:
    if config.slices_per_plane != 3:
        raise ValueError("DINOv2 requires exactly three sampled slices as input channels")
    if config.image_size % 14:
        raise ValueError("DINOv2 image_size must be divisible by its 14-pixel patch size")
    if config.batch_size < 2:
        raise ValueError("Contrastive training requires batch_size >= 2")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.image_unfreeze_last < 0 or config.text_unfreeze_last < 0:
        raise ValueError("unfreeze counts cannot be negative")
    if (config.fold_count is None) != (config.fold_index is None):
        raise ValueError("fold_count and fold_index must be provided together")
    if config.max_studies is not None and config.max_studies < 1:
        raise ValueError("max_studies must be positive")
    if config.image_memory_cache_size < 0:
        raise ValueError("image_memory_cache_size cannot be negative")
    if config.text_pooling not in {"cls", "mean"}:
        raise ValueError("text_pooling must be 'cls' or 'mean'")
    if config.report_view not in REPORT_VIEWS:
        raise ValueError(f"report_view must be one of {REPORT_VIEWS}")
    if config.trainable_checkpoints and (
        config.image_model_revision is None or config.text_model_revision is None
    ):
        raise ValueError("trainable checkpoints require pinned image and text model revisions")
    seed_everything(config.seed)
    image_kwargs = _image_kwargs(config)
    if config.cohort_manifest is not None:
        records, manifest_metadata = read_cohort_manifest(config.cohort_manifest)
        text_policy = manifest_metadata["text_policy"]
        if (
            text_policy == "multilingual_raw"
            and config.text_model not in MULTILINGUAL_TEXT_MODELS
            and not config.allow_text_policy_mismatch
        ):
            raise ValueError(
                "multilingual_raw manifest requires a registered multilingual text model; "
                "use multilingual-e5 or explicitly pass --allow-text-policy-mismatch"
            )
        if config.max_studies is not None:
            records = records[: config.max_studies]
        train_records = [record for record in records if record.split == "train"]
        val_records = [record for record in records if record.split == "validation"]
        data_summary = {
            "manifest": str(config.cohort_manifest),
            "text_policy": text_policy,
            "selected": len(records),
            "train": len(train_records),
            "validation": len(val_records),
            "test": sum(record.split == "test" for record in records),
        }
        print(json.dumps({"data": data_summary}, indent=2))
        if config.prepare_only:
            return {"data": data_summary, "selected": len(records)}
    else:
        print(f"Indexing reports from {config.reports_csv} and images from {config.series_root}")
        records, summary = build_study_records(
            config.reports_csv,
            config.series_csv,
            config.series_root,
            report_column=config.report_column,
            validate_pixels=config.validate_pixels,
            image_kwargs=image_kwargs,
            max_studies=config.max_studies,
        )
        print(json.dumps({"data": asdict(summary), "selected": len(records)}, indent=2))
        if config.prepare_only:
            return {"data": asdict(summary), "selected": len(records)}
        if config.fold_count is None:
            train_records, val_records = split_records(records, config.val_fraction, config.seed)
        else:
            train_records, val_records = split_records_fold(
                records, config.fold_count, config.fold_index, config.seed
            )
    if len(train_records) < 2 or not val_records:
        raise ValueError("Manifest/split must contain at least two training and one validation study")
    print(f"Split by StudyInstanceUID: train={len(train_records)} validation={len(val_records)}")
    run_context = initialize_run(config)
    model, tokenizer, block_info = build_model(
        config.image_model,
        config.text_model,
        config.embedding_dim,
        config.image_unfreeze_last,
        config.text_unfreeze_last,
        image_model_revision=config.image_model_revision,
        text_model_revision=config.text_model_revision,
        text_pooling=config.text_pooling,
        local_files_only=config.local_files_only,
    )
    print(f"Transformer blocks: {block_info}; text={config.text_model}; image={config.image_model}")

    update_run_manifest(config.output_dir, {"resolved_models": block_info})
    collator = make_collator(
        tokenizer, config.max_text_length, config.text_prefix, config.report_view
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        KneeReportDataset(
            train_records,
            image_kwargs,
            cache_images=config.cache_images,
            image_cache_dir=config.image_cache_dir,
            memory_cache_size=(
                config.image_memory_cache_size or (64 if config.cache_images else 0)
            ),
        ),
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        KneeReportDataset(
            val_records,
            image_kwargs,
            cache_images=config.cache_images,
            image_cache_dir=config.image_cache_dir,
            memory_cache_size=(
                config.image_memory_cache_size or (64 if config.cache_images else 0)
            ),
        ),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    if len(train_loader) == 0:
        raise ValueError("Training split is smaller than batch_size; lower --batch-size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config.epochs * len(train_loader), config.warmup_fraction)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if run_context.checkpoint is not None:
        checkpoint = run_context.checkpoint
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("loader_generator_state") is not None:
            generator.set_state(checkpoint["loader_generator_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    history_path = config.output_dir / "metrics.jsonl"
    best_score = run_context.best_score
    best_metrics = run_context.best_metrics
    for epoch in range(run_context.start_epoch, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        for batch in train_loader:
            images, input_ids, attention_mask = _move_batch(batch, device)
            images = augment_study_images(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, _, _ = model(images, input_ids, attention_mask)
                loss = symmetric_contrastive_loss(logits)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))
            scheduler.step()
            total_loss += float(loss.detach())
            total_grad_norm += float(grad_norm)

        metrics = evaluate(model, val_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_loader),
                "train_grad_norm": total_grad_norm / len(train_loader),
                **{
                    f"lr_{group['group_name']}": float(group["lr"])
                    for group in optimizer.param_groups
                },
            }
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True))
        is_best = is_better(metrics, best_metrics)
        if is_best:
            best_metrics = metrics
            best_score = metrics["mean_r@1"]
        checkpoint_kwargs = dict(
            path=config.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            config=config,
            scaler=scaler,
            loader_generator=generator,
            best_score=best_score,
            run_id=run_context.run_id,
        )
        _save_checkpoint(**checkpoint_kwargs)
        if is_best:
            checkpoint_kwargs["path"] = config.output_dir / "best.pt"
            _save_checkpoint(
                **checkpoint_kwargs,
            )

    return {
        "best_mean_r@1": best_score,
        "output_dir": str(config.output_dir),
        "train_studies": len(train_records),
        "validation_studies": len(val_records),
    }
