# Frozen LoRA encoder with an LM-supervised MLP

Protocol fixed before training the heads on 2026-09-21. Configuration:
[`configs/kaggle/lm-mlp-v1.json`](../configs/kaggle/lm-mlp-v1.json).

## Results — 2026-09-21

**The LM-supervised MLP improves over the matched LM linear head, but does not establish an improvement over the previous gold-trained logistic strategy.**

| Strategy | Gold macro AUROC | Gold labels used for head fitting? |
| --- | ---: | --- |
| LM linear, three-seed ensemble | 0.727514 | No |
| **LM MLP, three-seed ensemble** | **0.760040** | **No** |
| Previous gold-only logistic OOF | 0.759362 | Other gold folds |

Paired macro-AUROC differences (95% study-bootstrap intervals):

- MLP − LM linear: **+0.032525**, [+0.005992, +0.058915].
- MLP − gold-only logistic OOF: **+0.000678**, [-0.050266, +0.050137].

1,999 of 2,000 resamples retained both classes for every finding. These are conditional descriptive intervals on a repeatedly used development cohort.

MLP seed AUROCs: 2026: 0.754965, 2027: 0.759653, 2028: 0.760570.
Selected MLP epochs were **3, 2, 2**, compared with **15, 18, 14** for the linear heads. Selection used only LM validation loss; no gold-directed tuning or seed selection followed these results.

### Per-label gold AUROC

| Finding | LM linear | LM MLP | Gold-only logistic OOF |
| --- | ---: | ---: | ---: |
| ACL | 0.6706 | 0.6759 | 0.7404 |
| MCL | 0.4401 | 0.5547 | 0.7005 |
| Medial Meniscus | 0.7577 | 0.7679 | 0.7372 |
| Lateral Meniscus | 0.6537 | 0.7032 | 0.6858 |
| Medial OA | 0.9320 | 0.9439 | 0.9422 |
| Lateral OA | 0.6747 | 0.7010 | 0.8283 |
| PF OA | 0.8097 | 0.7861 | 0.6639 |
| Effusion | 0.8986 | 0.9249 | 0.9051 |
| Synovitis | 0.7705 | 0.7897 | 0.5949 |
| Baker's | 0.8750 | 0.8352 | 0.7708 |
| Contusion | 0.7226 | 0.7923 | 0.8720 |
| Fracture | 0.5249 | 0.6456 | 0.6712 |

### Verification and artifacts

- Encoder state hashes agreed before extraction and after training; all encoder parameters remained frozen.
- Export/reload maximum probability difference: 7.15e-07 on real DICOM input.
- Offline public-example inference produced a valid three-study `submission.csv` in **7.00 seconds**, with **183.6 MB** peak GPU allocation. This does not estimate hidden-test runtime.
- Full test suite: **146 passed, 1 skipped**. Ruff passed.
- Head fitting, gold benchmarking, and export took **9.19 seconds** after the features were prepared.
- MLP bundle is about **101.8 MB**, including one encoder and all three heads.
- Local artifacts: `checkpoints/kaggle/lm-mlp-v1/benchmark.json`, `selection.json`, per-seed histories/heads, `gold-predictions.npz`, and `bundle/`.
- Frozen-feature cache: `checkpoints/kaggle/lm-mlp-features-v1.npz`.

The MLP is a valid alternative submission candidate and is better than its matched LM-linear control. The current evidence does **not** justify calling it a leaderboard improvement over the existing logistic bundle. Both candidates are retained; nothing has been uploaded.

## Question and design

Does supervised learning from the report-derived LM labels improve the current
best frozen image representation, and does a nonlinear head improve over a linear
head trained on the same labels?

The encoder is the six-block rank-64 gold-holdout LoRA checkpoint used by the
[existing submission pipeline](kaggle-submission-pipeline.md). It produces 768
features: CLS concatenated with mean patch tokens, averaged over the three MRI
planes. The encoder and LoRA adapters are frozen and in evaluation mode. Features
are extracted once, cached with checkpoint/manifest/label hashes and study order,
and used by every head. Byte-level encoder state hashes must agree before and after
extraction and supervised training.

The actual input is `data/lm_label/lm_labels_kaggle.csv` (4,406 studies). Its bare
numeric label columns are verdict lookup scores, not calibrated probabilities.
Training uses `__verdict`: **YES=1, NO=0; UNK contributes no loss**. The `__conf`
columns are not used. No unknown finding becomes an assumed negative.

## Populations and selection

The encoder's frozen gold-holdout manifest defines the partitions:

| Partition | Studies used | Purpose |
| --- | ---: | --- |
| Train | 3,438 | Fit feature standardization and head weights |
| Validation | 426 | Select each seed's epoch by minimum macro masked BCE |
| Gold | 56 | Score the selected heads only after all epoch choices are saved |

47 train/validation studies with no observed labels are dropped. The 440 locked-test
studies (including two gold cases) are excluded. Gold identities and their known
study groups are excluded from both head-fitting partitions even though the LM CSV
contains their labels. Patient identity beyond those study groups is unavailable.

The validation images previously participated in contrastive checkpoint selection;
head-validation statistics are development statistics. The 56 gold images and
reports were excluded from encoder training and selection, and their labels do not
enter head training, standardization, early stopping, or model selection here.
This gold set has nevertheless been examined repeatedly in the broader project.

## Heads and loss

- Linear control: `768 → 12`.
- MLP: `768 → 256 → ReLU → Dropout(0.2) → 12`.
- Standardization: mean and population standard deviation from training features only.
- AdamW: learning rate 0.001, weight decay 0.01, batch size 128.
- Up to 100 epochs, early-stop patience 12, minimum meaningful improvement 1e-5.
- Seeds 2026, 2027, 2028 for each architecture. Report individual seeds and their
  mean-probability ensemble; no seed is chosen using gold performance.

The objective averages binary cross-entropy over the observed cells **within each
label**, then equally across the twelve labels. Fixed training observation rates
normalize minibatch losses so sparsely mentioned findings still contribute their
share. Validation uses its own complete split's observation rates. There is no
positive-class weighting or confidence weighting.

The deployed MLP ensemble retains the selected train-only heads; there is no refit
on validation or gold labels. One shared encoder feeds all three small heads, so
inference does not repeat image encoding for each seed.

## Benchmark interpretation

Report twelve-label macro and per-label AUROC, average precision, per-seed scores,
and paired study-bootstrap 95% intervals for differences (2,000 draws, seed 2029).
The primary architecture comparison is **LM MLP versus LM linear**, with identical
supervision and feature partitions.

Also reproduce the original gold-only logistic OOF reference (8 per-label stratified
folds, seed 2026, C=1). That reference learns from other gold cases in each fold;
the LM heads learn from thousands of report-derived labels and zero gold labels.
This comparison is between complete training strategies, not an architecture-only
comparison or a fresh Kaggle leaderboard estimate. Bootstrap intervals condition
on the fitted models and do not capture all training/development uncertainty.

## Run and inspect

```bash
.venv-cuda/bin/python scripts/train_lm_head.py \
  --config configs/kaggle/lm-mlp-v1.json \
  --features checkpoints/kaggle/lm-mlp-features-v1.npz \
  --output checkpoints/kaggle/lm-mlp-v1 \
  --device cuda
```

`--features-only` prepares the frozen-feature cache without fitting any head.
A repeated run reuses features only when their identity matches. DICOM data are
assumed immutable under the pinned manifest; use a new cache if source pixels change.
Existing experiment output directories are refused. Change config/output paths to
record a different experiment; do not overwrite the first result.

Outputs include per-seed heads and training histories, `selection.json` saved before
gold evaluation, `gold-predictions.npz`, `benchmark.json`, and a self-contained
`bundle/` for the existing Kaggle notebook. Benchmark artifacts contain study IDs
and targets; upload only `bundle/` if this candidate is chosen later.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-cuda/bin/python checkpoints/kaggle/lm-mlp-v1/bundle/predict_kaggle.py \
  --bundle checkpoints/kaggle/lm-mlp-v1/bundle \
  --data-root /mnt/d \
  --output checkpoints/kaggle/lm-mlp-v1/submission.csv \
  --device cuda
```

No Kaggle upload is part of this experiment. The existing logistic-head bundle stays
available for comparison.
