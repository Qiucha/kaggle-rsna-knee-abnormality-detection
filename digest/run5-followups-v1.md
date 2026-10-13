# Run-5 follow-ups

Updated 2026-10-09 22:33:49 +0800

| Completed run | LM-test | Δ vs run 5 [95% CI] | Gold-58 | Best epoch |
|---|---:|---|---:|---:|
| Run 5: soft 336, seed 2026 | 0.8775 | baseline | 0.8267 | 17 |
| Hard labels at 336 | 0.8584 | -0.0191 [-0.0346, -0.0043] | 0.8055 | 5 |
| K8 plain CLIP → run-5 fine-tuning | 0.8801 | +0.0026 [-0.0089, +0.0142] | 0.8248 | 11 |
| Run 5: seed 2027 | 0.8746 | -0.0029 [-0.0110, +0.0061] | 0.8333 | 16 |
| 30-epoch limit | 0.8751 | -0.0024 [-0.0087, +0.0040] | 0.8444 | 16 |
| Unfreeze 6 image blocks | 0.8782 | +0.0007 [-0.0088, +0.0098] | 0.8618 | 17 |
| 384 px | 0.8794 | +0.0019 [-0.0051, +0.0087] | 0.8393 | 16 |
| DINOv2-small soft 336 | 0.8714 | -0.0061 [-0.0197, +0.0080] | 0.8150 | 20 |
| CoAtNet stages 1–3 | 0.8675 | -0.0100 [-0.0228, +0.0033] | 0.8394 | 16 |
| Unfreeze 1 image block | 0.8685 | -0.0090 [-0.0194, +0.0005] | 0.8245 | 18 |
| Unfreeze 9 image blocks | 0.8834 | +0.0059 [-0.0045, +0.0153] | 0.8775 | 12 |
| Unfreeze all 12 image blocks | 0.8700 | -0.0075 [-0.0171, +0.0025] | 0.8788 | 17 |
| K8 336 CLIP + LoRA → run-5 fine-tuning | 0.8741 | -0.0034 [-0.0161, +0.0093] | 0.8178 | 7 |

Absolute run-5 seed difference: 0.002871.

K8 CLIP pre-training: 20/20 epochs; selected epoch 18; mean retrieval Recall@1 0.19207081198692322. Retrieval Recall@1 is distinct from pathology AUROC.

| Unfinished run | Status | Completed epochs |
|---|---|---:|

LM-test uses LM labels; gold-58 is exploratory. No interrupted run has a final AUROC.

[Phase 2 digest page](site/pages/phase2.html#run5-followups)
[Machine-readable snapshot](run5-followups-v1.json)
