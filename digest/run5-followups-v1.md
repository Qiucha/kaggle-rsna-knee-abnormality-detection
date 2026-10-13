# Run-5 follow-ups

Updated 2026-10-05 09:02:56 +0800

| Completed run | LM-test | Δ vs run 5 [95% CI] | Gold-58 | Best epoch |
|---|---:|---|---:|---:|
| Run 5: soft 336, seed 2026 | 0.8775 | baseline | 0.8267 | 17 |
| Hard labels at 336 | 0.8584 | -0.0191 [-0.0346, -0.0043] | 0.8055 | 5 |
| K8 plain CLIP → run-5 fine-tuning | 0.8801 | +0.0026 [-0.0089, +0.0142] | 0.8248 | 11 |
| Run 5: seed 2027 | 0.8746 | -0.0029 [-0.0110, +0.0061] | 0.8333 | 16 |
| 30-epoch limit | 0.8751 | -0.0024 [-0.0087, +0.0040] | 0.8444 | 16 |

Absolute run-5 seed difference: 0.002871.

K8 CLIP pre-training: 20/20 epochs; selected epoch 18; mean retrieval Recall@1 0.19207081198692322. Retrieval Recall@1 is distinct from pathology AUROC.

| Unfinished run | Status | Completed epochs |
|---|---|---:|
| Unfreeze 6 image blocks | running | 12/20 |
| 384 px | queued | 0/20 |
| DINOv2-small soft 336 | queued | 0/20 |
| CoAtNet stages 1–3 | queued | 0/20 |

LM-test uses LM labels; gold-58 is exploratory. No interrupted run has a final AUROC.

[Fine-tuning digest page](site/pages/finetune.html#run5-followups)
[Machine-readable snapshot](run5-followups-v1.json)
