# One-stage label + contrastive loss

Result recorded 2026-10-10. **The tested joint-loss recipe does not improve on label-only u9.** LM-test is statistically unresolved; exploratory gold-58 is lower. Keep label-only u9 as the reference.

## What was combined

The same OmniRad-base image encoder feeds a 12-label classifier and an image–report projection in each training step. The objective is **masked label BCE + 0.1 × image-to-report contrastive loss**. This replaces sequential CLIP pre-training followed by label fine-tuning with one 20-epoch stage.

The run uses 9 unfrozen image blocks, K=8 three-slice windows per plane, uniform pooling, 336 px, soft training labels, seed 2026, batch size 1 and gradient accumulation 16. Frozen multilingual-E5-base features form a bank of 3,357 distinct reports from 3,438 training studies; identical reports share a bank entry. The text projection trains. This is image-to-report cross-entropy over the bank, rather than symmetric in-batch InfoNCE. Gradient accumulation changes the optimizer batch size; the bank supplies contrastive negatives.

Validation hard-verdict masked BCE selects the checkpoint, with mean retrieval Recall@1 breaking ties and earliest epoch breaking remaining ties. All 20 epochs completed; epoch **17** was selected at BCE **0.429490**. The comparison uses the existing supervised populations: 3,438 train, 426 validation, 432 LM-test and 58 gold studies. Gold is excluded from training and selection.

## Scores and paired differences

| Training | LM-test AUROC | Gold-58 AUROC |
|---|---:|---:|
| Run 5: label only, u3 | 0.877506 | 0.826671 |
| Run 5: seed 2027 | 0.874635 | 0.833274 |
| Two stages: CLIP → labels, u3 | 0.880097 | 0.824806 |
| Label only: u9, batch 1 | **0.883406** | **0.877467** |
| One stage: joint loss, u9 | 0.880864 | 0.849366 |

| Candidate − reference | Δ LM-test [95% CI] | Δ gold-58 [95% CI] |
|---|---:|---:|
| **Joint u9 − label-only u9** | **−0.002543 [−0.013547, +0.008423]** | **−0.028101 [−0.054232, −0.003503]** |
| CLIP → labels u3 − label-only u3 | +0.002591 [−0.008917, +0.014214] | −0.001865 [−0.030166, +0.025990] |
| Run-5 seed 2027 − seed 2026 | −0.002871 [−0.011024, +0.006146] | +0.006603 [−0.015292, +0.029129] |
| Joint u9 − CLIP → labels u3 | +0.000766 [−0.009996, +0.011957] | +0.024560 [−0.006469, +0.054914] |

Intervals are paired study bootstraps (2,000 resamples, bootstrap seed 2030). The scoring script checks study order and gold targets across saved predictions. The primary comparator holds the image depth, resolution, windows and batch size fixed. Joint vs CLIP → labels changes depth from u3 to u9 and cannot isolate the effect of combining losses.

On gold-58, the largest losses against label-only u9 are lateral meniscus (−0.1168), medial meniscus (−0.1100) and lateral OA (−0.0986); effusion gains +0.0522. The [snapshot](joint-loss-v1.json) contains all per-label scores and epoch histories, displayed on the [Phase-2 page](site/pages/phase2.html#joint-loss).

## Interpretation and reproducibility

Only one joint seed and λ=0.1 were tested. LM-test measures agreement with LM verdicts. Gold-58 has been inspected repeatedly and is exploratory; the study-bootstrap intervals do not measure training-seed uncertainty. This result supports keeping the label-only reference for this recipe, without ruling out other auxiliary weights or bank designs. No Kaggle score is recorded for the joint run.

The source run is `checkpoints/joint/omnirad-base-k8-soft-336-u9-b1-bank-l0.1/`. Its `comparison.json`, `report.json`, `config.json`, `run_manifest.json` and `metrics.jsonl`, plus each comparator's report/history, are hashed in [joint-loss-v1.json](joint-loss-v1.json). The recorded training source was a dirty `joint/text-bank` tree; the snapshot preserves its commit and source-tree hash.

Refresh with `.venv/bin/python scripts/build_joint_loss_digest.py`, then `.venv/bin/python scripts/build_digest_site.py`. The site's `export` command also refreshes this snapshot when the source comparison exists. A regular build needs only the saved digest snapshot.
