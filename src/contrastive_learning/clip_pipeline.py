"""An auditable CLIP pre-training → matched run-5 fine-tuning experiment."""

from __future__ import annotations

import gc
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .artifacts import read_json, read_jsonl, sha256_file, write_json
from .config import TrainConfig
from .finetune import ROOT, WindowModel, image_kwargs, load_backbone, run_finetune, validate_config
from .inputs import load_cohort
from .lm_proxy import load_test_population, masked_macro_auroc, paired_intervals
from .lm_supervision import load_population
from .model import build_model
from .training import _optimizer, _scheduler, _train_epoch, run_training, seed_everything


def validate_pipeline(fine: dict, pre_value: dict) -> TrainConfig:
    """Reject changes that would invalidate the promised plain-CLIP comparison."""
    pre = TrainConfig.from_dict(pre_value)
    pre.validate_training()
    validate_config(fine)
    if pre.use_declip or pre.use_lora or pre.label_similarity_weight or pre.lm_label_csv:
        raise ValueError("This experiment requires plain CLIP with full-block adaptation")
    if (pre.batch_size, pre.image_unfreeze_last, pre.text_unfreeze_last) != (32, 6, 6):
        raise ValueError("The approved pre-training recipe is batch 32, last 6 blocks in both towers")
    if pre.image_kwargs != {**image_kwargs(fine), "windows": fine["windows_per_plane"]}:
        raise ValueError("Pre-training and fine-tuning image preprocessing differ")
    if pre.windows_per_plane != 8 or pre.image_plane_pooling != "uniform" or not pre.contrastive_microbatch_size:
        raise ValueError("CLIP must use run 5's K8 masked mean pipeline with 32-study gradient replay")
    if Path(fine["init_checkpoint"]) != pre.output_dir / "best.pt":
        raise ValueError("Fine-tuning must start from the selected CLIP checkpoint")
    if fine.get("init_unfreeze_last") != 3:
        raise ValueError("The supervised stage must match run 5's last-three-block adaptation")
    for key in ("image_model", "image_model_revision", "image_model_format",
                "image_model_base", "image_model_base_revision", "cohort_manifest"):
        if str(getattr(pre, key)) != str(fine[key]):
            raise ValueError(f"Pre-training and fine-tuning {key} differ")
    manifest_sha = sha256_file(ROOT / pre.cohort_manifest)
    if manifest_sha != fine["manifest_sha256"] or manifest_sha != pre_value["manifest_sha256"]:
        raise ValueError("The frozen gold-holdout manifest checksum differs")
    baseline = read_json(ROOT / fine["comparison_baseline"] / "config.json")
    if any(fine.get(key) != value for key, value in baseline.items() if key != "name"):
        raise ValueError("The supervised recipe differs from run 5 beyond its initialization")
    # This validates the gold exclusion, patient groups, and supervised populations.
    groups, _, _, _ = load_population(fine, ROOT)
    cohort = load_cohort(pre, for_training=True)
    gold_uids = {r.study_uid for r in groups["gold"]}
    seen = cohort.split("train") + cohort.split("validation")
    if gold_uids & {r.study_uid for r in seen}:
        raise ValueError("Gold cases enter contrastive training or checkpoint selection")
    gold_groups = {r.patient_group_id for r in groups["gold"] if r.patient_group_id}
    if gold_groups & {r.patient_group_id for r in seen if r.patient_group_id}:
        raise ValueError("Gold patient groups enter contrastive training or selection")
    return pre


def preflight(pre: TrainConfig, fine: dict, status_dir: Path) -> dict:
    """Profile real optimizer steps at the specified batch sizes, before committing hours.

    Pixels are synthetic; the 32 reports are from the training split. These weights
    are discarded. Training starts again from the pinned pretrained weights and seed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("The approved experiment requires CUDA")
    seed_everything(pre.seed)
    device = torch.device("cuda")
    model, tokenizer, info = build_model(
        pre.image_model, pre.text_model, pre.embedding_dim, pre.image_unfreeze_last,
        pre.text_unfreeze_last, image_model_revision=pre.image_model_revision,
        image_model_format=pre.image_model_format, image_model_base=pre.image_model_base,
        image_model_base_revision=pre.image_model_base_revision,
        text_model_revision=pre.text_model_revision, text_pooling=pre.text_pooling,
        image_plane_pooling=pre.image_plane_pooling,
        windows_per_plane=pre.windows_per_plane,
        gradient_checkpointing=pre.gradient_checkpointing, local_files_only=True,
    )
    model.to(device)
    reports = load_cohort(pre).split("train")[:pre.batch_size]
    tokens = tokenizer([pre.text_prefix + r.report for r in reports], padding="max_length",
                       truncation=True, max_length=pre.max_text_length, return_tensors="pt")
    batch = {**tokens, "images": torch.rand(pre.batch_size, 3, pre.windows_per_plane, 3,
                                            pre.image_size, pre.image_size),
             "valid": torch.ones(pre.batch_size, 3, pre.windows_per_plane, dtype=torch.bool)}
    optimizer = _optimizer(model, pre)
    scheduler = _scheduler(optimizer, 2, pre.warmup_fraction)
    scaler = torch.amp.GradScaler("cuda", enabled=pre.amp)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = _train_epoch(model, [batch, batch], device, optimizer, scheduler, scaler, pre)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Nonfinite CLIP preflight metrics: {metrics}")
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise RuntimeError(f"Missing gradients during CLIP preflight: {missing}")
    clip = {"seconds": time.perf_counter() - started, "metrics": metrics,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "batch_size": pre.batch_size, "image_shape": list(batch["images"].shape),
            "study_microbatch_size": pre.contrastive_microbatch_size,
            "images_per_study": 3 * pre.windows_per_plane,
            "text_shape": list(tokens["input_ids"].shape), "resolved_models": info,
            "all_trainable_parameters_received_gradients": True}
    # Check the compact checkpoint handoff with the exact image architecture.
    checkpoint_path = status_dir / "preflight-image-delta.pt"
    torch.save({"config": pre.as_dict(), "epoch": 0, "trainable_state_dict": {
        n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
    }}, checkpoint_path)
    clip["projected_pretrain_compute_hours"] = (clip["seconds"] / 2 *
        (len(load_cohort(pre).split("train")) // pre.batch_size) * pre.epochs / 3600)
    clip["timing_projection_excludes"] = ["DICOM reads", "validation", "checkpoint writes"]
    del model, optimizer, scheduler, scaler, tokens, batch
    gc.collect()
    torch.cuda.empty_cache()
    backbone, adaptation = load_backbone({**fine, "init_checkpoint": str(checkpoint_path)})
    if adaptation["image_unfreeze_last"] != 3:
        raise RuntimeError("Checkpoint handoff retained the contrastive stage's depth")
    window = WindowModel(backbone, fine["head_hidden"], fine["dropout"], fine["pooling"]).to(device)
    optimizer = torch.optim.AdamW([p for p in window.parameters() if p.requires_grad],
                                 lr=fine["lr_backbone"])
    scaler = torch.amp.GradScaler("cuda", enabled=fine["amp"])
    images = torch.rand(fine["batch_size"], 3, fine["windows_per_plane"], 3,
                        fine["image_size"], fine["image_size"], device=device)
    valid = torch.ones(images.shape[:3], dtype=torch.bool, device=device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=fine["amp"]):
            logits = window(images, valid)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits) * 0.5)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    supervised = {"batch_size": fine["batch_size"], "image_shape": list(images.shape),
                  "image_unfreeze_last": adaptation["image_unfreeze_last"],
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    del backbone, window, optimizer, scaler, images, valid, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint_path.unlink()
    result = {"synthetic_pixels": True, "discarded_profile_weights": True,
              "clip": clip, "supervised": supervised}
    write_json(status_dir / "preflight.json", result)
    return result


def compare_to_run5(fine: dict, output: Path, status_dir: Path) -> dict:
    """Apply the predeclared pass rule to shared study-bootstrap resamples."""
    directories = {"candidate": output, "run5": ROOT / fine["comparison_baseline"],
                   "seed2027": ROOT / fine["comparison_seed"]}
    seed_pending = not (directories["seed2027"] / "predictions.npz").exists()
    if seed_pending and fine.get("allow_pending_seed_comparison"):
        del directories["seed2027"]
    predictions = {}
    for name, directory in directories.items():
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            predictions[name] = {key: saved[key].copy() for key in saved.files}
    rows, targets, masks, identity = load_test_population(
        ROOT / fine["cohort_manifest"], ROOT / fine["lm_labels_csv"], ROOT / fine["gold_labels_csv"])
    reference = predictions["run5"]
    for name, saved in predictions.items():
        if saved["lm_test_uids"].tolist() != identity["study_uids"]:
            raise ValueError(f"{name}: LM-test population differs")
        for key in ("gold_uids", "gold_targets"):
            if not np.array_equal(saved[key], reference[key]):
                raise ValueError(f"{name}: gold population differs")
    report = {}
    for split, y, mask in (("lm_test", targets, masks),
                           ("gold", reference["gold_targets"], np.ones_like(reference["gold_targets"]))):
        scores = {name: saved[split] for name, saved in predictions.items()}
        report[split] = {
            "macro_auroc": {name: masked_macro_auroc(p, y, mask)["macro_auroc"] for name, p in scores.items()},
            "paired": paired_intervals(scores, y, mask),
        }
    lm_delta = report["lm_test"]["paired"]["candidate - run5"]
    gold_delta = report["gold"]["paired"]["candidate - run5"]
    noise = (abs(report["lm_test"]["paired"]["run5 - seed2027"]["difference"])
             if "seed2027" in predictions else None)
    rule = fine["pass_rule"]
    checks = {
        "lm_test_ci_excludes_zero_in_positive_direction": lm_delta["ci95"][0] > rule["lm_test_ci_lower_gt"],
        "gold_drop_less_than_0_024": gold_delta["difference"] > rule["gold_delta_gt"],
        "lm_test_gain_exceeds_absolute_seed_difference": lm_delta["difference"] > noise if noise is not None else None,
    }
    report.update(pass_checks=checks, passed=all(checks.values()) if noise is not None else None,
                  seed_comparison_pending=noise is None, pass_rule=rule,
                  absolute_lm_test_seed_difference=noise, bootstrap={"resamples": 2000, "seed": 2030},
                  predictions_sha256={n: sha256_file(p / "predictions.npz") for n, p in directories.items()})
    write_json(status_dir / "comparison.json", report)
    supervised = read_json(output / "report.json")
    table = "| Run | LM-test AUROC | Gold-58 AUROC |\n| --- | ---: | ---: |\n"
    for name in directories:
        table += (f"| {name} | {report['lm_test']['macro_auroc'][name]:.6f} | "
                  f"{report['gold']['macro_auroc'][name]:.6f} |\n")
    verdict = "PENDING SEED COMPARISON" if noise is None else "PASS" if report["passed"] else "DOES NOT PASS"
    noise_text = "pending the postponed seed-2027 run" if noise is None else f"{noise:.6f}"
    text = (f"OmniRad K8 plain CLIP → run-5 fine-tuning: {verdict}\n\n"
            + table + "\n"
            f"Supervised selected epoch: {supervised['best_epoch']}; "
            f"supervised training and scoring: {supervised['seconds'] / 3600:.2f} hours.\n\n"
            f"LM-test Δ: {lm_delta['difference']:+.6f}, 95% CI {lm_delta['ci95']}.\n"
            f"Gold-58 Δ: {gold_delta['difference']:+.6f}, 95% CI {gold_delta['ci95']}.\n"
            f"Absolute run-5 seed difference: {noise_text}.\n\n"
            "Gold-58 is exploratory because these cases have been examined repeatedly.\n"
            "No Kaggle submission or queue reprioritization is automatic.\n")
    (status_dir / "completion-report.md").write_text(text)
    return report


def reuse_pretraining(pre: TrainConfig, expected_sha: str) -> dict:
    """Require the original complete recipe and selected checkpoint before reuse."""
    saved = TrainConfig.from_dict(read_json(pre.output_dir / "config.json"))
    if saved.as_dict() != pre.as_dict():
        raise ValueError("Saved contrastive pre-training recipe differs from the approved recipe")
    metrics = read_jsonl(pre.output_dir / "metrics.jsonl")
    if [row["epoch"] for row in metrics] != list(range(1, pre.epochs + 1)):
        raise ValueError("Contrastive pre-training is incomplete; cannot skip it")
    if not expected_sha or sha256_file(pre.output_dir / "best.pt") != expected_sha:
        raise ValueError("Selected contrastive checkpoint checksum differs")
    return {"output_dir": str(pre.output_dir),
            "best_mean_r@1": max(row["mean_r@1"] for row in metrics),
            "pretraining_reused": True}


def run_pipeline(fine: dict, output: Path, *, device: str = "cuda") -> dict:
    status_dir = ROOT / fine["pipeline_status_dir"]
    status_dir.mkdir(parents=True, exist_ok=True)
    status = {"pipeline": fine["name"], "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
              "output": str(output), "stage": "validation"}

    def update(stage: str, **extra):
        status.update(stage=stage, updated_at=time.strftime("%Y-%m-%d %H:%M:%S %z"), **extra)
        write_json(status_dir / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        update("validation")
        pre_value = read_json(ROOT / fine["contrastive_pretrain_config"])
        pre = validate_pipeline(fine, pre_value)
        reuse = fine.get("reuse_completed_contrastive_checkpoint", False)
        if Path(output).exists() or (pre.output_dir.exists() and not reuse):
            raise FileExistsError("Experiment output exists; inspect it before resuming or replacing it")
        if device != "cuda":
            raise ValueError("This pipeline requires device=cuda")
        if reuse:
            result = reuse_pretraining(pre, fine.get("reuse_checkpoint_sha256", ""))
        else:
            update("preflight")
            profile = preflight(pre, fine, status_dir)
            update("contrastive_pretraining", gpu_preflight=profile)
            result = run_training(pre)
        update("supervised_finetuning", contrastive=result,
               selected_checkpoint_sha256=sha256_file(ROOT / fine["init_checkpoint"]))
        gc.collect()
        torch.cuda.empty_cache()
        result = run_finetune(fine, output, device=device)
        update("comparison", supervised=result)
        comparison = compare_to_run5(fine, Path(output), status_dir)
        update("completed_pending_seed" if comparison["seed_comparison_pending"] else "completed",
               passed=comparison["passed"], seed_comparison_pending=comparison["seed_comparison_pending"],
               completion_report=str(status_dir / "completion-report.md"))
        return comparison
    except Exception as error:
        update("failed", failed_stage=status["stage"], error=str(error), traceback=traceback.format_exc())
        (status_dir / "completion-report.md").write_text(
            f"OmniRad CLIP pipeline failed during {status['failed_stage']}:\n\n{error}\n")
        raise
