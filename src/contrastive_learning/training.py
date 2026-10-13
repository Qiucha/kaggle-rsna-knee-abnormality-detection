"""Run one training job: prepare studies, optimize, validate, and checkpoint.

:func:`run_training` is the whole public surface. Given a :class:`TrainConfig` it
either creates a new run directory or continues an existing one, trains to the
configured epoch count, and leaves behind everything needed to audit and reproduce
what happened -- ``config.json``, ``run_manifest.json``, ``metrics.jsonl``,
``last.pt``, and ``best.pt``.

The design commitments worth knowing before reading the code:

* **Every epoch is validated and checkpointed.** ``best.pt`` is chosen by the
  policy in :mod:`contrastive_learning.checkpoints`, never by training loss, and
  never later than the epoch that earned it.
* **A resume is not an approximation.** Optimizer, scheduler, loss scaler, loader
  shuffle, and global random state are all restored, and the saved configuration
  must match on every field but the epoch target.
* **The three towers learn at different rates.** Pretrained backbones move slowly,
  the randomly initialized projections quickly; the parameter groups below encode
  that, along with excluding norms and biases from weight decay.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_learning.artifacts import append_jsonl, jsonl_line
from contrastive_learning.checkpoints import (
    initialize_run,
    is_better,
    restore_training_state,
    save_checkpoint,
    update_run_manifest,
)
from contrastive_learning.config import TrainConfig
from contrastive_learning.data import move_batch
from contrastive_learning.evaluation import evaluate
from contrastive_learning.inputs import load_cohort, make_loader
from contrastive_learning.labels import read_lm_labels
from contrastive_learning.declip import (
    ReportEmbeddingQueue,
    augment_report_tokens,
    compute_declip_loss,
)
from contrastive_learning.model import (
    build_model,
    label_similarity_targets,
    soft_target_contrastive_loss,
    symmetric_contrastive_loss,
)
from contrastive_learning.preprocessing import augment_study_images


# Upper bound on the learnable inverse temperature, as in CLIP. Without it the
# scale runs away early, saturating the softmax and stalling the gradient.
MAX_LOGIT_SCALE = math.log(100.0)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch so a run is reproducible from its config."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Group trainable parameters by tower and decay eligibility.

    Six groups at most: image/text/projection crossed with decay/no-decay. Weight
    decay applies only to matrix-shaped parameters that are not normalization
    weights, the standard exclusion for biases, norms, and learned embeddings.
    Group names are recorded so ``metrics.jsonl`` can report each learning rate.
    """
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
    return torch.optim.AdamW(
        [
            {
                "params": parameters,
                "lr": learning_rates[tower],
                "weight_decay": config.weight_decay if use_decay else 0.0,
                "group_name": f"{tower}_{'decay' if use_decay else 'no_decay'}",
            }
            for (tower, use_decay), parameters in groups.items()
        ]
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup into cosine decay to zero, stepped once per batch.

    Warmup protects the pretrained backbones from the large early updates that a
    freshly initialized projection head provokes.
    """
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    *,
    declip_queue: ReportEmbeddingQueue | None = None,
    mask_token_id: int | None = None,
) -> dict[str, float]:
    """Optimize for one epoch and return its loss, gradient norm, and learning rates.

    Augmentation happens on device, after transfer, so the cached study tensors stay
    reusable. Gradients are unscaled before clipping, otherwise the clip threshold
    would apply to loss-scaled values and effectively vary with the scaler.
    """
    use_amp = config.amp and device.type == "cuda"
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_loss_mv = 0.0
    total_loss_nn = 0.0
    total_loss_img_ss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if config.contrastive_microbatch_size:
            from .gradient_cache import backward_cached_clip

            loss = backward_cached_clip(model, batch, device, scaler, config)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(config.max_inverse_temperature))
            scheduler.step()
            total_loss += float(loss)
            total_grad_norm += float(grad_norm)
            continue
        images, input_ids, attention_mask = move_batch(batch, device)
        save_ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if (config.save_on_cpu and device.type == "cuda")
            else contextlib.nullcontext()
        )
        if config.use_declip and declip_queue is not None:
            images_aug = augment_study_images(images)
            input_ids_aug = augment_report_tokens(
                input_ids,
                attention_mask,
                mask_token_id=mask_token_id,
                mask_prob=config.declip_text_mask_prob,
            )
            with torch.autocast(device_type=device.type, enabled=use_amp):
                with save_ctx:
                    z_I = model.encode_images(images)
                    z_I_aug = model.encode_images(images_aug)
                    z_T = model.encode_text(input_ids, attention_mask)
                    z_T_aug = model.encode_text(input_ids_aug, attention_mask)

                z_T_nn = declip_queue.get_nearest_neighbours(z_T)
                targets = None
                if config.label_similarity_weight > 0.0:
                    targets = label_similarity_targets(
                        batch["label_values"].to(device, non_blocking=True),
                        batch["label_mask"].to(device, non_blocking=True),
                        config.label_similarity_weight,
                    )

                image_ss_weight = (
                    config.declip_image_ss_weight
                    if (config.image_unfreeze_last > 0 or config.use_lora)
                    else 0.0
                )
                loss, breakdown = compute_declip_loss(
                    z_I=z_I,
                    z_I_aug=z_I_aug,
                    z_T=z_T,
                    z_T_aug=z_T_aug,
                    z_T_nn=z_T_nn,
                    logit_scale=model.logit_scale,
                    targets=targets,
                    mv_weight=config.declip_mv_weight,
                    nn_weight=config.declip_nn_weight,
                    image_ss_weight=image_ss_weight,
                )
            declip_queue.enqueue(z_T)
            total_loss_mv += breakdown["loss_mv"]
            total_loss_nn += breakdown["loss_nn"]
            total_loss_img_ss += breakdown["loss_img_ss"]
        else:
            images = augment_study_images(images)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                with save_ctx:
                    logits, _, _ = model(images, input_ids, attention_mask)
                if config.label_similarity_weight > 0.0:
                    # Built in float32 outside the autocast dtype's reach: the target
                    # is a normalized probability row, and halving its precision would
                    # distort small off-diagonal mass more than it costs to compute.
                    targets = label_similarity_targets(
                        batch["label_values"].to(device, non_blocking=True),
                        batch["label_mask"].to(device, non_blocking=True),
                        config.label_similarity_weight,
                    )
                    loss = soft_target_contrastive_loss(logits.float(), targets)
                else:
                    loss = symmetric_contrastive_loss(logits)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            max_scale = math.log(config.max_inverse_temperature)
            model.logit_scale.clamp_(max=max_scale)
        scheduler.step()
        total_loss += float(loss.detach())
        total_grad_norm += float(grad_norm)

    out = {
        "train_loss": total_loss / len(loader),
        "train_grad_norm": total_grad_norm / len(loader),
        **{f"lr_{group['group_name']}": float(group["lr"]) for group in optimizer.param_groups},
    }
    if config.use_declip and declip_queue is not None:
        out["train_loss_mv"] = total_loss_mv / len(loader)
        out["train_loss_nn"] = total_loss_nn / len(loader)
        out["train_loss_img_ss"] = total_loss_img_ss / len(loader)
    return out


def run_training(config: TrainConfig) -> dict[str, Any]:
    """Prepare studies, train or resume, and persist validation-selected checkpoints.

    Returns the best validation score and the split sizes. Progress is printed as
    JSON per epoch and appended to ``metrics.jsonl``, so a run can be monitored and
    audited from the same records.
    """
    config.validate_training()
    seed_everything(config.seed)

    cohort = load_cohort(config, for_training=True)
    preparation = {"data": cohort.summary, "selected": len(cohort.records)}
    print(json.dumps(preparation, indent=2))
    if config.prepare_only:
        return preparation

    train_records = cohort.split("train")
    val_records = cohort.split("validation")
    if len(train_records) < 2 or not val_records:
        raise ValueError("Manifest/split must contain at least two training and one validation study")
    print(f"Study split: train={len(train_records)} validation={len(val_records)}")

    # Claims the output directory (or validates the resume) before any model is
    # built, so a configuration mistake fails in seconds rather than minutes.
    run_context = initialize_run(config)
    model, tokenizer, block_info = build_model(
        config.image_model,
        config.text_model,
        config.embedding_dim,
        config.image_unfreeze_last,
        config.text_unfreeze_last,
        image_model_revision=config.image_model_revision,
        image_model_format=config.image_model_format,
        image_model_base=config.image_model_base,
        image_model_base_revision=config.image_model_base_revision,
        text_model_revision=config.text_model_revision,
        text_pooling=config.text_pooling,
        image_plane_pooling=config.image_plane_pooling,
        windows_per_plane=config.windows_per_plane,
        image_drop_path_rate=config.image_drop_path_rate,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        gradient_checkpointing=config.gradient_checkpointing,
        local_files_only=config.local_files_only,
    )
    print(f"Transformer blocks: {block_info}; text={config.text_model}; image={config.image_model}")
    # Records the backbone revisions actually resolved, which is what a later
    # reload of a compact checkpoint needs.
    update_run_manifest(config.output_dir, {"resolved_models": block_info})

    label_table = None
    if config.label_similarity_weight > 0.0:
        label_table = read_lm_labels(config.lm_label_csv, config.label_findings)
        coverage = label_table.coverage([record.study_uid for record in train_records])
        # Recorded rather than enforced: an unlabelled study falls back to the
        # one-hot target, so partial coverage weakens the experiment instead of
        # invalidating it, and the manifest is where that gets noticed.
        coverage["finding_names"] = list(label_table.findings)
        print(json.dumps({"label_similarity": {**coverage, "source": str(config.lm_label_csv)}}))
        update_run_manifest(config.output_dir, {"label_similarity_coverage": coverage})

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = make_loader(
        train_records, tokenizer, config, training=True, generator=generator, label_table=label_table
    )
    val_loader = make_loader(val_records, tokenizer, config)
    if len(train_loader) == 0:
        raise ValueError("Training split is smaller than batch_size; lower --batch-size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.freeze_temperature:
        model.logit_scale.requires_grad = False
    model.to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config.epochs * len(train_loader), config.warmup_fraction)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    if run_context.checkpoint is not None:
        restore_training_state(
            run_context.checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=generator,
        )

    history_path = config.output_dir / "metrics.jsonl"
    best_score = run_context.best_score
    best_metrics = run_context.best_metrics
    declip_queue = None
    mask_token_id = None
    if config.use_declip:
        declip_queue = ReportEmbeddingQueue(
            capacity=config.declip_queue_size,
            embedding_dim=config.embedding_dim,
            device=device,
        )
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
        if mask_token_id is None or mask_token_id == getattr(tokenizer, "unk_token_id", None):
            mask_token_id = getattr(tokenizer, "pad_token_id", 0)

    for epoch in range(run_context.start_epoch, config.epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scheduler,
            scaler,
            config,
            declip_queue=declip_queue,
            mask_token_id=mask_token_id,
        )
        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, **train_metrics)
        # Append before checkpointing: history must never be missing an epoch that
        # a checkpoint claims, which is what resume verification checks.
        append_jsonl(history_path, metrics)
        print(jsonl_line(metrics), end="")

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
        save_checkpoint(**checkpoint_kwargs)
        if is_best:
            # Same payload under a second name; best.pt therefore stays fully
            # resumable rather than being weights alone.
            save_checkpoint(**{**checkpoint_kwargs, "path": config.output_dir / "best.pt"})

    return {
        "best_mean_r@1": best_score,
        "output_dir": str(config.output_dir),
        "train_studies": len(train_records),
        "validation_studies": len(val_records),
    }
