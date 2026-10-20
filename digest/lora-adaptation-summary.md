# LoRA adaptation findings

The [LoRA depth matrix](../configs/contrastive/both-lora-v1.json) adapts the last
six transformer blocks of both towers with ranks 16, 32, and 64. The backbones
remain frozen; the adapters and contrastive projection heads train. The separate
[gold-holdout matrix](../configs/contrastive/both-lora-holdout-v1.json) excludes
the 56 eligible gold-labeled studies from encoder training and checkpoint selection.

| Six-block setting | Gold-label macro AUROC | Evaluation population |
| --- | ---: | --- |
| Full block fine-tuning | 0.7262 | Exposed development studies |
| LoRA rank 16 | 0.7235 | Exposed development studies |
| LoRA rank 32 | 0.7456 | Exposed development studies |
| LoRA rank 64 | 0.7524 | Exposed development studies |
| LoRA rank 64, gold holdout | 0.7594 | Gold studies excluded from encoder training and selection |

All figures are twelve-label macro AUROC from an eight-fold logistic probe on
56 gold studies. The exposed rows compare representations on cases used during
encoder development; they are useful for screening, not an independent clinical
estimate. The holdout row removes direct encoder exposure, but repeated model
selection on this small gold cohort still makes it development evidence. The
two populations must not be read as a paired generalization comparison.

At rank 64, the adapters train 8.79 million parameters (2.85% of the combined
model) versus 54.01 million (17.95%) for six-block full fine-tuning. The
rank-64 holdout checkpoint is the encoder selected for the first submission;
the fitted pathology heads and Kaggle result are documented on the `submission`
branch.
