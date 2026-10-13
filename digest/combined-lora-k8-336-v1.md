# Combined CLIP + LoRA + K8 + 336

Updated 2026-10-07 17:07:26 +0800

| Image tower | Contrastive status | Epochs | Selected epoch | Mean R@1 | LM-test MLP AUROC | Gold-58 MLP AUROC |
|---|---|---:|---:|---:|---:|---:|
| OmniRad-base | completed | 20/20 | 19 | 0.1663 | 0.8812 | 0.7844 |
| DINOv2-small | completed | 20/20 | 15 | 0.1724 | 0.8670 | 0.7652 |

Retrieval uses validation image-report pairs. LM-test AUROC uses LM labels; gold-58 AUROC is exploratory. Pending arms have no final score. No Kaggle submission was made.

Recipe: LoRA rank 64 / alpha 128 on the last six blocks of both towers; eight windows per plane at 336 px; batch size 32; 20 contrastive epochs; seed 2026; checkpoint selected by mean validation Recall@1.

The OmniRad label fine-tune from this contrastive checkpoint is listed separately in [run-5 follow-ups](run5-followups-v1.md). Its AUROC is from an adapted image tower, whereas the MLP scores above use a frozen tower.

The frozen-head exporter verifies checkpoint/bundle provenance and recomputes AUROC from existing feature and prediction caches. It does not extract images or train models.
