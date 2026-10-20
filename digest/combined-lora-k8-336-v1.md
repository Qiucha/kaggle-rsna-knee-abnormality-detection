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

## Reproduce

From the repository root on a CUDA host, with `paths.json` naming `data_root` and, optionally, `train_series_mirror`:

1. Train one arm: `python scripts/training/run_contrastive_matrix.py --matrix configs/contrastive/combined-lora-k8-336-v1.json --experiment <arm>`.
2. Fit its heads: `python scripts/training/train_lm_head.py --config configs/kaggle/lm-mlp-combined-58-<arm>.json --features checkpoints/kaggle/lm-mlp-combined-58-<arm>-features.npz --output checkpoints/kaggle/lm-mlp-combined-58-<arm> --device cuda`. These head configs set `reload_tolerance` to 1e-3 for the fp16 K8 features.
3. Optionally fine-tune from the OmniRad checkpoint: `python scripts/training/finetune_supervised.py --config configs/finetune/r4a-k8-omnirad-base-soft-336-from-clip-lora.json --output checkpoints/finetune/<name>`.
4. Rebuild this digest: `python scripts/digests/build_combined_lora_digest.py`.

For a seed variant, add a matrix entry whose `args` set `--seed`. Copy the head config and point `checkpoint` at the new run's `best.pt`.
Not in git: the cohort manifest (checked against `manifest_sha256`), `data/lm_label/lm_labels_kaggle.csv`, and the pinned Hugging Face snapshots.
