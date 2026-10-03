#!/usr/bin/env python3
"""Profile GPU memory usage of LoRA fine-tuning across ranks and layer depths.

Tests ranks {16, 32, 64} and backward layer depths {2, 4, 6, 8, 10, 12}
on the RTX 3070 (8GB VRAM) for the contrastive knee MRI architecture:
  - Image tower: DINOv2-small (12 ViT blocks, 3 planes folded -> 96 ViT passes per step)
  - Text tower: Multilingual-E5-base (12 transformer blocks, seq len 512)
  - Projection heads + temperature parameter

Measures:
  - Peak allocated VRAM (MiB)
  - Peak reserved VRAM (MiB)
  - Trainable parameters
  - LoRA parameters
  - Whether training fits in the 8GB memory envelope or throws OOM
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.lora import apply_lora_to_backbone, get_lora_parameter_summary
from contrastive_learning.model import (
    ImageReportContrastiveModel,
    ImageTower,
    TextTower,
    symmetric_contrastive_loss,
)


def profile_single_configuration(
    rank: int,
    layer_count: int,
    batch_size: int = 32,
    seq_len: int = 512,
    use_lora: bool = True,
    device: torch.device = torch.device("cuda:0"),
) -> dict[str, Any]:
    """Profile a single configuration for peak memory and parameter counts."""
    # Aggressively clean up any previous GPU allocations
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)

    try:
        # 1. Load frozen backbones from local cache
        image_backbone = AutoModel.from_pretrained(
            "facebook/dinov2-small",
            revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
            local_files_only=True,
        )
        text_backbone = AutoModel.from_pretrained(
            "intfloat/multilingual-e5-base",
            revision="d128750597153bb5987e10b1c3493a34e5a4502a",
            local_files_only=True,
        )

        # Freeze everything by default
        for p in image_backbone.parameters():
            p.requires_grad = False
        for p in text_backbone.parameters():
            p.requires_grad = False

        if use_lora:
            # Apply LoRA to the last `layer_count` blocks of both image and text backbones
            img_replaced = apply_lora_to_backbone(
                image_backbone, count=layer_count, rank=rank, alpha=float(rank * 2)
            )
            txt_replaced = apply_lora_to_backbone(
                text_backbone, count=layer_count, rank=rank, alpha=float(rank * 2)
            )
        else:
            # Full fine-tuning baseline: unfreeze last `layer_count` blocks
            from contrastive_learning.model import freeze_then_unfreeze_last

            freeze_then_unfreeze_last(image_backbone, layer_count)
            freeze_then_unfreeze_last(text_backbone, layer_count)
            img_replaced = 0
            txt_replaced = 0

        # 2. Assemble contrastive model
        image_tower = ImageTower(image_backbone, embedding_dim=256, plane_pooling="attention")
        text_tower = TextTower(text_backbone, embedding_dim=256, pooling="mean")
        model = ImageReportContrastiveModel(image_tower, text_tower)
        model.to(device)
        model.train()

        # Parameter counts
        param_summary = get_lora_parameter_summary(model)

        # Optimizer: AdamW on all trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        # 3. Create realistic training batch
        images = torch.randn(batch_size, 3, 3, 224, 224, device=device)
        input_ids = torch.randint(0, 20000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

        # Reset peak stats right before the training step
        torch.cuda.reset_peak_memory_stats(device)

        # 4. Perform full training step: forward -> loss -> backward -> optimizer step
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, _, _ = model(images, input_ids, attention_mask)
            loss = symmetric_contrastive_loss(logits)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

        # Clean up
        del images, input_ids, attention_mask, logits, loss, model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "status": "SUCCESS",
            "mode": f"LoRA (r={rank})" if use_lora else "Full FT",
            "rank": rank if use_lora else None,
            "layer_count": layer_count,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "peak_allocated_mb": round(peak_alloc_mb, 2),
            "peak_reserved_mb": round(peak_res_mb, 2),
            "total_vram_mb": round(total_vram_mb, 2),
            "memory_headroom_mb": round(total_vram_mb - peak_res_mb, 2),
            "fits_in_8gb": peak_res_mb <= total_vram_mb,
            "trainable_parameters": param_summary["trainable_parameters"],
            "lora_parameters": param_summary["lora_parameters"],
            "total_parameters": param_summary["total_parameters"],
            "trainable_fraction_pct": round(param_summary["trainable_fraction"] * 100, 3),
            "image_layers_adapted": img_replaced,
            "text_layers_adapted": txt_replaced,
        }

    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "mode": f"LoRA (r={rank})" if use_lora else "Full FT",
            "rank": rank if use_lora else None,
            "layer_count": layer_count,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "error": "CUDA Out Of Memory",
            "fits_in_8gb": False,
        }
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "ERROR",
            "mode": f"LoRA (r={rank})" if use_lora else "Full FT",
            "rank": rank if use_lora else None,
            "layer_count": layer_count,
            "error": str(e),
            "fits_in_8gb": False,
        }


def main():
    parser = argparse.ArgumentParser(description="Profile LoRA memory across ranks and layer depths")
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--include-full-ft", action="store_true", help="Also profile full fine-tuning")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "digest" / "lora-memory-profile.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Error: CUDA GPU required for memory profiling.")
        sys.exit(1)

    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    print(f"=== GPU Memory Profiler: {torch.cuda.get_device_name(device)} ({total_mem:.1f} MiB) ===")
    print(f"Target ranks: {args.ranks}")
    print(f"Target layer depths: {args.layers} (applied from backward)")
    print(f"Batch size: {args.batch_size} (3 planes -> {args.batch_size * 3} images)")
    print(f"Text sequence length: {args.seq_len}\n")

    results = []

    # 1. Profile Full Fine-Tuning baseline if requested
    if args.include_full_ft:
        print("--- Baseline: Full Fine-Tuning (without LoRA) ---")
        for layers in args.layers:
            print(f"Testing Full FT with last {layers} blocks...", end=" ", flush=True)
            res = profile_single_configuration(
                rank=0,
                layer_count=layers,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                use_lora=False,
                device=device,
            )
            results.append(res)
            if res["status"] == "SUCCESS":
                print(f"OK | Alloc: {res['peak_allocated_mb']} MiB | Res: {res['peak_reserved_mb']} MiB | Headroom: {res['memory_headroom_mb']} MiB | Trainable: {res['trainable_parameters']:,}")
            else:
                print(f"FAILED ({res['status']}) - {res.get('error', '')}")

    # 2. Profile LoRA across ranks and layer depths
    for rank in args.ranks:
        print(f"\n--- LoRA Rank {rank} ---")
        oom_encountered = False
        for layers in args.layers:
            print(f"Testing LoRA r={rank}, last {layers} blocks...", end=" ", flush=True)
            res = profile_single_configuration(
                rank=rank,
                layer_count=layers,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                use_lora=True,
                device=device,
            )
            results.append(res)
            if res["status"] == "SUCCESS":
                print(f"OK | Alloc: {res['peak_allocated_mb']} MiB | Res: {res['peak_reserved_mb']} MiB | Headroom: {res['memory_headroom_mb']} MiB | Trainable: {res['trainable_parameters']:,} ({res['trainable_fraction_pct']}%)")
            else:
                print(f"FAILED ({res['status']}) - {res.get('error', '')}")
                oom_encountered = True
                break

    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully written to {args.output_json}")


if __name__ == "__main__":
    main()
