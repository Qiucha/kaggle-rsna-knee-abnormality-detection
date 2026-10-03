#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo "Starting DeCLIP Remaining Experiments Pipeline"
echo "Timestamp: $(date -u)"
echo "Host / Python: $(.venv-cuda/bin/python --version)"
echo "========================================================"

# Helper function to update probe config with existing checkpoints
update_probe_config() {
  .venv-cuda/bin/python -c '
import json
from pathlib import Path

cfg_file = Path("configs/downstream/gold-probe-declip-v1.json")
cfg = json.loads(cfg_file.read_text())
all_checkpoints = [
    "checkpoints/contrastive/both-block-depth-v1/both-unfreeze6-seed2026/best.pt",
    "checkpoints/contrastive/both-lora-v1/both-lora-depth6-r64-seed2026/best.pt",
    "checkpoints/contrastive/both-lora-v1/both-lora-depth6-r64-bs64-reg-seed2026/best.pt",
    "checkpoints/contrastive/declip-v1/declip-frozen-seed2026/best.pt",
    "checkpoints/contrastive/declip-v1/declip-mv-only-frozen-seed2026/best.pt",
    "checkpoints/contrastive/declip-v1/declip-lora-depth6-r64-seed2026/best.pt",
]
valid = [p for p in all_checkpoints if Path(p).exists()]
cfg["arms"]["after_cl"]["checkpoints"] = valid
cfg_file.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Downstream probe config synced with {len(valid)} available checkpoints.")
'
}

# -----------------------------------------------------------------------------
# 1. DeCLIP Ablation: Multi-View InfoNCE Only (lambda_NN=0, frozen backbones)
# -----------------------------------------------------------------------------
MV_ONLY_DIR="checkpoints/contrastive/declip-v1/declip-mv-only-frozen-seed2026"
if [ ! -f "$MV_ONLY_DIR/best.pt" ]; then
  echo ""
  echo ">>> [1/2] Launching declip-mv-only-frozen-seed2026 (BS=32, 20ep) <<<"
  echo "Start: $(date -u)"
  PYTHONPATH=src .venv-cuda/bin/python -m contrastive_learning.cli \
    --cohort-manifest data/processed/contrastive-cohort.jsonl \
    --text-model multilingual-e5 \
    --text-model-revision d128750597153bb5987e10b1c3493a34e5a4502a \
    --text-pooling mean \
    --text-prefix "query: " \
    --image-model facebook/dinov2-small \
    --image-model-revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
    --max-text-length 512 \
    --report-view full \
    --epochs 20 \
    --batch-size 32 \
    --num-workers 4 \
    --image-cache-dir checkpoints/contrastive/full-multilingual-v1/image-cache \
    --image-memory-cache-size 16 \
    --trainable-checkpoints \
    --local-files-only \
    --lr-projection 1e-3 \
    --seed 2026 \
    --image-unfreeze-last 0 \
    --text-unfreeze-last 0 \
    --use-declip \
    --declip-mv-weight 1.0 \
    --declip-nn-weight 0.0 \
    --declip-text-mask-prob 0.15 \
    --output-dir "$MV_ONLY_DIR"
  echo "Completed declip-mv-only-frozen-seed2026 at $(date -u)"

  # Probe & update digest for intermediate results
  update_probe_config
  echo "Evaluating intermediate downstream probing..."
  PYTHONPATH=src .venv-cuda/bin/python scripts/evaluate_downstream_probe.py \
    --config configs/downstream/gold-probe-declip-v1.json \
    --image-cache-dir checkpoints/contrastive/full-multilingual-v1/image-cache \
    --output digest/downstream-probe-declip-v1.json \
    --features digest/declip-features.npz \
    --overwrite
  python3 scripts/build_declip_digest.py
  echo "Intermediate digest updated."
else
  echo ">>> [1/2] declip-mv-only-frozen-seed2026 already complete, skipping. <<<"
fi

# -----------------------------------------------------------------------------
# 2. DeCLIP LoRA 6-Block: Multi-View + NN Queue + SimCLR Self-Supervision
# -----------------------------------------------------------------------------
LORA_DIR="checkpoints/contrastive/declip-v1/declip-lora-depth6-r64-seed2026"
if [ ! -f "$LORA_DIR/best.pt" ]; then
  echo ""
  echo ">>> [2/2] Launching declip-lora-depth6-r64-seed2026 (BS=32, 20ep, LoRA r=64) <<<"
  echo "Start: $(date -u)"
  PYTHONPATH=src .venv-cuda/bin/python -m contrastive_learning.cli \
    --cohort-manifest data/processed/contrastive-cohort.jsonl \
    --text-model multilingual-e5 \
    --text-model-revision d128750597153bb5987e10b1c3493a34e5a4502a \
    --text-pooling mean \
    --text-prefix "query: " \
    --image-model facebook/dinov2-small \
    --image-model-revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
    --max-text-length 512 \
    --report-view full \
    --epochs 20 \
    --batch-size 32 \
    --num-workers 4 \
    --image-cache-dir checkpoints/contrastive/full-multilingual-v1/image-cache \
    --image-memory-cache-size 16 \
    --trainable-checkpoints \
    --local-files-only \
    --lr-image 1e-4 \
    --lr-text 1e-4 \
    --lr-projection 1e-3 \
    --seed 2026 \
    --use-lora \
    --image-unfreeze-last 6 \
    --text-unfreeze-last 6 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --gradient-checkpointing \
    --use-declip \
    --declip-mv-weight 1.0 \
    --declip-nn-weight 0.5 \
    --declip-image-ss-weight 0.5 \
    --declip-queue-size 512 \
    --declip-text-mask-prob 0.15 \
    --output-dir "$LORA_DIR"
  echo "Completed declip-lora-depth6-r64-seed2026 at $(date -u)"
else
  echo ">>> [2/2] declip-lora-depth6-r64-seed2026 already complete, skipping. <<<"
fi

# -----------------------------------------------------------------------------
# 3. Final Probe Evaluation & Digest Generation
# -----------------------------------------------------------------------------
echo ""
echo ">>> Running Final Downstream Linear Probing & Digest Regeneration <<<"
update_probe_config

PYTHONPATH=src .venv-cuda/bin/python scripts/evaluate_downstream_probe.py \
  --config configs/downstream/gold-probe-declip-v1.json \
  --image-cache-dir checkpoints/contrastive/full-multilingual-v1/image-cache \
  --output digest/downstream-probe-declip-v1.json \
  --features digest/declip-features.npz \
  --overwrite

python3 scripts/build_declip_digest.py
python3 scripts/build_both_block_digest.py

echo "========================================================"
echo "All remaining DeCLIP experiments and digests complete!"
echo "End Timestamp: $(date -u)"
echo "========================================================"
