# RSNA Knee Abnormality Detection: Fine-Tuning on Language-Model Labels + Rank Ensemble of Deep Models

**Private 0.889 / Public 0.902** (Deep Three ensemble, submission 57296356). The private score sets the final ranking.
*Second pick: A30 single model, Private 0.878 / Public 0.890 (submission 57094902)*

> **Team:** Ouch! My Knee Hurts!
> **Authors:** Yun-Hsuan Ku, Yeshwanth Patil, Zihan Zhou, Raymond Chou, Bowen Liu
> **Code:** [github.com/Qiucha/novo-project](https://github.com/Qiucha/novo-project)
> **Submission kernels:** `a30-deep3-rank` (v2), `r4a-k8-omb-soft-336-u9-a30` (v1)

Thanks to RSNA, the annotators and Kaggle for hosting this competition, and to everyone who shared notebooks and ideas. Special thanks to **@pilkwang** for the CC0 dataset [`rsna-knee-llm-labels`](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels). All our supervised models train on those labels.

---

## Summary

* **No external images.** We trained only on the competition DICOMs plus the labels that a language model (LM) extracted from the radiology reports. The only external asset is the pretrained backbone `Snarcy/OmniRad-base` (ViT-B/14, DINOv2 pretrained on RadImageNet).
* **Scores:**

  | Model | Private | Public | LM-test | Gold-58 |
  |---|---:|---:|---:|---:|
  | Deep Three ensemble | 0.889 | 0.902 | 0.9017 | 0.8897 |
  | A30 single model | 0.878 | 0.890 | 0.8917 | 0.8814 |

* **Recipe:**
  1. **Backbone:** OmniRad-base, with the last 9 of 12 transformer blocks trained.
  2. **Input:** 8 windows per plane. Each window is 3 adjacent slices stacked as an RGB image at 336 px. That gives 24 images per study.
  3. **Targets:** the LM soft values (`YES` = 0.68 / 0.82 / 0.94, `NO` = 0.08) instead of 0/1 labels.
  4. **A30 only:** studies positive for Lateral Meniscus, Lateral OA or Synovitis are sampled twice per epoch, small noise is added to the lateral targets, and training runs 30 epochs.
  5. **Ensemble:** per-label rank average of three models, all with 9 trained blocks.
* **Did not help:** image–report matching losses, image augmentation, learned window weights, weight averaging, test-time shifts. See [What Did Not Work](#what-did-not-work).

---

## Labels and Validation

### Targets

We predict 12 findings, scored by macro ROC-AUC:

* **Ligaments:** ACL, MCL
* **Menisci:** Medial Meniscus, Lateral Meniscus
* **Osteoarthritis (OA):** Medial OA, Lateral OA, Patellofemoral (PF) OA
* **Fluid:** Effusion, Synovitis, Baker's Cyst
* **Bone:** Contusion, Fracture

Only 58 studies have human labels. For training we used the LM labels from `@pilkwang`'s dataset:

* Each finding is `YES`, `NO` or `UNK` (not mentioned in the report).
* Each verdict also has a soft value. `YES` has one of three levels (0.68, 0.82, 0.94), `NO` is 0.08 and `UNK` is 0.28.
* `UNK` is common, for example 84% of studies for Synovitis and 56% for Fracture. We mask `UNK` out of the loss.

### Splits

One fixed split of 4,354 studies:

| Split | Studies | Used for |
|---|---:|---|
| Train | 3,438 | training |
| Validation | 426 | picking the checkpoint (lowest validation loss) |
| LM-test | 432 | comparing models: macro AUROC against the LM labels |
| Gold-58 | 58 | sanity check against the human labels |

* **Comparisons** use a paired bootstrap on LM-test (2,000 resamples, 95% CI).
* **Our bar:** a change should have an LM-test CI above zero and must not drop Gold-58 by more than 0.024. Most single-model steps did not clear the first part, so we adopted them on the point estimate plus the Gold-58 tiebreak (see Takeaways).
* **Ties:** when the LM-test CI crossed zero, we used Gold-58 as the tiebreaker.

---

## Preprocessing

```
DICOM study (sagittal, coronal, axial)
  ├─ pick the fat-suppressed series in each plane, if there is one
  ├─ flip right knees so all knees face the same way
  ├─ crop 130 mm, resize to 336 × 336
  ├─ keep the central 60% of slices (positions 0.2–0.8)
  └─ split those slices into 8 equal parts; take 3 adjacent slices per part as one RGB image
       • training: random position within each part
       • inference: centre of each part
→ 24 images per study (3 planes × 8)
```

Missing or unreadable planes are filled with zeros and masked, so they don't affect the average.

---

## Model

```
24 images (336 × 336)
  → OmniRad-base ViT-B/14 (blocks 1–3 frozen, blocks 4–12 trained)
  → per image: [CLS token, mean of patch tokens] concatenated (1536-d)
  → average over the valid images
  → MLP: 1536 → 256 → ReLU → Dropout(0.2) → 12 logits
```

**Loss:** binary cross-entropy on the soft targets. `UNK` cells are ignored, and each label counts equally.

**Training (A30):**
* Backbone LR 2e-5, head LR 1e-3, weight decay 0.02.
* 10% warmup, then cosine decay over 30 epochs.
* Batch size 1 with 16 gradient accumulation steps.
* Seed 2026; we kept epoch 22.
* About 26 min per epoch.

**Lateral oversampling (A30):**
* **Extra samples:** studies positive for Lateral Meniscus, Lateral OA or Synovitis are seen twice per epoch. That makes 4,946 samples per epoch instead of 3,438 (+44%).
* **Masking:** on the extra copies, only those three labels count in the loss.
* **Noise:** Gaussian noise (σ = 0.1) is added to the Lateral Meniscus and Lateral OA targets.

---

## Step-by-Step Progress

Each row is one Kaggle submission. Δ is measured against the row above unless noted. **u9** (9 trained blocks), **A30** (u9 + lateral oversampling, 30 epochs) and **A30-CLIP** (the A30 recipe started from the CLIP-trained weights) are the three models in the final ensemble. The other steps are named by what they changed.

| Stage | Step | Change | Δ LM-test [95% CI] | Δ Gold-58 [95% CI] | Private | Public |
|:---|:---|:---|---:|---:|---:|---:|
| 1 | CLIP + gold heads | First submission: CLIP-trained encoder (DINOv2-S with LoRA + E5-base), frozen, with logistic heads trained on the 58 human-labelled studies | — | — | 0.690 | 0.717 |
| 1 | CLIP + LM heads | Same idea, but MLP heads trained on the LM labels instead of the 58 human labels | — | — | 0.756 | 0.788 |
| 2 | Fine-tune DINOv2-S | Train the last 3 blocks of DINOv2-S on the LM labels; 8 windows per plane; 0/1 labels | +0.036 | +0.018 | 0.801 | 0.826 |
| *2* | *DINOv2-S + soft* | *Side branch: soft targets on DINOv2-S (vs Fine-tune DINOv2-S)* | *+0.0001 [−0.0104, +0.0103]* | *+0.0389 [+0.0151, +0.0637]* | *0.813* | *0.834* |
| 2 | OmniRad backbone | Swap DINOv2-S for OmniRad-base (vs Fine-tune DINOv2-S) | +0.0003 | +0.030 | 0.829 | 0.839 |
| 2 | Soft + 336 px | OmniRad-base with soft targets and 336 px input | +0.0073 [−0.0033, +0.0180] | +0.0130 [−0.0168, +0.0412] | 0.865 | 0.879 |
| 2 | **u9** | Train 9 blocks instead of 3 | +0.0059 [−0.0045, +0.0153] | +0.0508 [+0.0217, +0.0819] | 0.869 | 0.888 |
| 2 | **A30** | u9 + lateral oversampling + target noise, 30 epochs | +0.0083 [−0.0025, +0.0202] | +0.0040 [−0.0149, +0.0218] | **0.878** | **0.890** |
| 2 | 4-model ensemble | Rank ensemble: A30 + u9 + 3-block CLIP-start model + DINOv2-S at 336 px (vs A30) | +0.0130 [+0.0062, +0.0203] | −0.0091 [−0.0245, +0.0059] | 0.880 | 0.894 |
| 2 | Swap in A30-CLIP | Replace the 3-block CLIP-start model with **A30-CLIP** (vs 4-model ensemble) | −0.0001 [−0.0031, +0.0029] | +0.0135 [+0.0042, +0.0230] | 0.887 | 0.899 |
| 2 | **Deep Three** | Drop DINOv2-S too: A30 + u9 + A30-CLIP (vs 4-model ensemble) | −0.0031 [−0.0084, +0.0019] | +0.0173 [+0.0051, +0.0301] | **0.889** | **0.902** |

**Stage 1** trained a CLIP-style image–report model and put small heads on its frozen features. **Stage 2** fine-tuned the image encoder directly on the LM labels, then ensembled the best models. The table shows no LM-test change for the stage-1 rows. A third stage-1 submission, with the smaller E5-small text encoder, scored 0.745 private / 0.774 public. The deltas for Fine-tune DINOv2-S (against CLIP + LM heads) and OmniRad backbone come from the submission records, which store no CI.

### Takeaways

1. **Fine-tuning gave the largest gain.** Going from frozen CLIP features (stage 1) to fine-tuning on the LM labels (stage 2) added +0.036 on LM-test, +0.045 on private and +0.038 on public.
2. **Most single-model steps were within noise on LM-test.** For the OmniRad backbone, Soft + 336 px, u9 and A30, the LM-test CI crosses zero. We adopted them on the point estimate plus Gold-58, and public went up each time.
3. **Gold-58 was a useful but imperfect tiebreaker.**
   * It agreed with public on the last two ensemble steps.
   * It disagreed twice. From A30 to the 4-model ensemble, Gold-58 fell while public rose. Gold-58 also preferred DINOv2-S + soft over the OmniRad backbone (0.8223 vs 0.8137), but public preferred the OmniRad backbone (0.839 vs 0.834).
4. **Training more blocks helped Gold-58 most.**

   | Blocks trained | 1 | 3 | 6 | 9 | 12 |
   |---|---:|---:|---:|---:|---:|
   | Δ Gold-58 vs 3 blocks | −0.002 | 0 | +0.035 | +0.051 | +0.052 |

   On LM-test, no depth's CI excluded zero. We chose 9 blocks because it had the highest LM-test point estimate.

---

## Ensembling

We average the per-label ranks of each model's predictions:

$$\hat{y}_{i,c} = \frac{1}{M} \sum_{m=1}^{M} \text{rank}(p_{m,i,c})$$

* **LM-test liked mixed ensembles.** Adding the shallower models (DINOv2-S, the 3-block CLIP-start model) gave the best LM-test score (0.905).
* **Gold-58 did not.** The mixed ensembles lowered Gold-58 Lateral Meniscus AUROC:

  | Model | A30 alone | 4-model ensemble | Swap in A30-CLIP | Deep Three |
  |---|---:|---:|---:|---:|
  | Gold-58 Lateral Meniscus AUROC | 0.834 | 0.755 | 0.778 | 0.831 |

* **The leaderboard agreed with Gold-58.** Deep Three is also better than A30 alone on LM-test: +0.0100 [+0.0048, +0.0155]. It scored best on both leaderboards.

---

## Stage 1: Contrastive (CLIP) Pretraining

Before fine-tuning, we trained a CLIP-style image–report model:
* **Image encoder:** DINOv2-small with LoRA.
* **Text encoder:** multilingual E5-base.
* **Readout:** frozen features with MLP heads trained on the LM labels. That scored 0.756 private (0.788 public).

We stopped this line of work for three reasons:
1. **Fine-tuning was much stronger.** Fine-tuning DINOv2-S on the LM labels gave +0.036 LM-test, +0.045 private and +0.038 public over the frozen CLIP model.
2. **Frozen heads fell behind on Gold-58.** On OmniRad-base, a frozen CLIP-adapted encoder with an MLP head matched full fine-tuning on LM-test (+0.0072 [−0.0047, +0.0191]). On Gold-58 it was lower by −0.0433 [−0.0778, −0.0115].
3. **CLIP init added nothing once fine-tuning was strong.** Starting fine-tuning from the CLIP weights changed LM-test by:
   * +0.0026 [−0.0089, +0.0142] with 3 trained blocks;
   * −0.0052 [−0.0144, +0.0046] with 9 trained blocks.

   The CLIP start gave a head start that faded. We compared A30 with A30-CLIP, which uses the same recipe and seed and differs only in its starting weights:

   | Epoch | 1 | 5 | 10 | 20 | 30 |
   |---|---:|---:|---:|---:|---:|
   | Training loss, A30 | 0.502 | 0.430 | 0.371 | 0.322 | 0.315 |
   | Training loss, A30-CLIP | 0.475 | 0.414 | 0.361 | 0.319 | 0.314 |
   | Validation AUROC, A30 | 0.781 | 0.858 | 0.866 | 0.870 | 0.869 |
   | Validation AUROC, A30-CLIP | 0.850 | 0.868 | 0.883 | 0.885 | 0.885 |

   Training loss started lower with the CLIP start, but the gap closed by epoch 30. Validation AUROC stayed a little higher, yet on the test sets the two models score almost the same:

   | Test set | A30 | A30-CLIP | Δ A30-CLIP − A30 [95% CI] |
   |---|---:|---:|---|
   | LM-test (432 studies) | 0.8917 | 0.8865 | −0.0052 [−0.0144, +0.0046] |
   | Gold-58 (58 studies) | 0.8814 | 0.8768 | −0.0046 [−0.0240, +0.0147] |

   Both intervals cross zero.

**Translated vs original reports.** 2,671 of the 4,407 reports are not in English. We translated them with Claude (Opus), fixed a few by hand, and kept the 1,736 English reports as they were. We then retrained the contrastive model with the same text encoder (multilingual E5), seed and settings, so only the report text changed. The score is image–report retrieval Recall@1 on 441 validation studies, averaged over both directions, with a paired bootstrap 95% CI:

| Reports | Studies | Original text | Translated text | Δ Recall@1, pp [95% CI] |
|---|---:|---:|---:|---:|
| All | 441 | 10.66% | 10.20% | −0.45 [−2.95, +2.04] |
| Originally English | 175 | 10.00% | 10.86% | +0.86 [−2.57, +4.29] |
| Translated | 266 | 11.09% | 9.77% | −1.32 [−4.70, +2.07] |

Translation did not help overall. Retrieval went up on reports that were already in English and down on the translated ones. The English text is identical in both runs, so that gain comes from training on an all-English corpus. The drop suggests some detail was lost in translation. Every interval crosses zero and this is one seed, so treat it as a trend, not a result.

**Our own LM labels.** We also labelled the translated reports with several language models and scored each one against the Gold-58 human labels. The submitted models train on **@pilkwang**'s labels. For this labelling, the models read Claude's rewrite of every report, the English ones included. That rewrite put the reports into standard phrasing and sometimes spelled out abbreviations (for example, "ACL" became "anterior cruciate ligament"). The translation experiment above kept the original English reports.

---

## What Did Not Work

Each idea is compared with the model in the "Compared with" column. "DINOv2-S" there means the fine-tuned DINOv2-S step. "Null" means the CI crosses zero. "Vetoed" means Gold-58 dropped by more than 0.024. Rows marked *frozen* used frozen features with MLP heads, so their numbers are not directly comparable with the fine-tuned rows.

### Training and pretraining

| Idea | Compared with | Δ LM-test [95% CI] | Δ Gold-58 | Verdict |
|:---|:---|:---|:---|:---|
| Add an image–report matching loss (label loss + 0.1 × CLIP-style loss) | u9 | −0.0025 [−0.0135, +0.0084] | **−0.0281** | Vetoed. Gold-58 Lateral Meniscus −0.117, Medial Meniscus −0.110 |
| Start from CLIP-trained weights (A30-CLIP) | A30 | −0.0052 [−0.0144, +0.0046] | −0.0046 | Null alone; kept as a Deep Three member |
| Match similar reports, not only exact pairs (*frozen*) | Plain CLIP | +0.0040 [−0.010, +0.017] | — | Null |
| DeCLIP loss instead of the CLIP loss (*frozen*) | Plain CLIP | −0.0021 [−0.0101, +0.0056] | +0.0027 | Null |
| RAD-DINO as the image encoder, DeCLIP loss (*frozen*) | DINOv2-S | **−0.031** [−0.046, −0.016] | −0.047 | Worse |
| RadioDino backbone (3 blocks, 0/1 labels) | DINOv2-S | −0.0187 [−0.0305, −0.0062] | −0.0437 | Worse |
| CoAtNet backbone (CoAtNet-rmlp-2, 384 px) | Soft + 336 px | −0.0583 [−0.0752, −0.0425] | −0.0802 | Worse |

### Data and inputs

| Idea | Compared with | Δ LM-test [95% CI] | Δ Gold-58 | Verdict |
|:---|:---|:---|:---|:---|
| Random rotation, zoom and brightness (rotation ±8°, zoom up to 1.08, shift, intensity ±10%) | A30 | −0.0028 [−0.0111, +0.0055] | −0.0166 | Null |
| 384 px images, not 336 px | Soft + 336 px | +0.0019 [−0.0051, +0.0087] | +0.013 | Null |
| Central 90% of slices, not 60% (positions 0.05–0.95 instead of 0.2–0.8) | A30 | −0.0068 [−0.0156, +0.0025] | −0.0062 | Null |
| Add the non-fat-suppressed scans: 6 series instead of 3 (*frozen* heads on u9 features) | u9 | −0.0044 [−0.0095, +0.0011] | −0.0009 | Null |
| 16 windows per plane, not 8 | DINOv2-S | +0.0032 [+0.0001, +0.0063] | −0.0067 | Small gain; not kept (21% of planes got empty windows) |
| 5 overlapping slice groups, at test time only (*frozen*) | CLIP + LM heads | +0.0154 [+0.004, +0.026] | −0.018 | Failed our rule that Gold-58 must not drop |

### Architecture and loss

| Idea | Compared with | Δ LM-test [95% CI] | Δ Gold-58 | Verdict |
|:---|:---|:---|:---|:---|
| Double loss weight on the 3 lateral findings (Lateral Meniscus, Lateral OA, Synovitis) | u9 | −0.0111 [−0.0243, +0.0015] | **−0.0250** | Vetoed |
| Learned window weights per finding, instead of a plain average | DINOv2-S | −0.0080 [−0.0160, −0.0005] | −0.0071 | Worse |
| Pool the inner and outer halves of the knee separately | DINOv2-S + soft | −0.0150 [−0.0267, −0.0033] | −0.0154 | Worse |
| Train unknown (`UNK`) labels as 0.28, instead of leaving them out of the loss | DINOv2-S + soft | −0.0037 [−0.0153, +0.0072] | −0.0131 | Null |
| 0/1 labels instead of soft targets | Soft + 336 px | −0.0191 [−0.0346, −0.0043] | — | Worse |

### After training (A30, no retraining)

Each row is compared with A30 alone.

| Idea | Δ LM-test [95% CI] | Δ Gold-58 [95% CI] |
|:---|---:|---:|
| Average weights: A30 + augmented A30 | +0.0034 [−0.0028, +0.0096] | −0.0052 [−0.0213, +0.0108] |
| Average weights: A30 + u9 | −0.0030 [−0.0095, +0.0035] | −0.0111 [−0.0305, +0.0079] |
| Average weights: epochs 22 + 30 | +0.0012 [−0.0004, +0.0030] | +0.0014 [−0.0033, +0.0061] |
| Test-time shifts: 1 extra pass | +0.0010 [−0.0014, +0.0037] | +0.0025 [−0.0033, +0.0086] |
| Test-time shifts: 4 extra passes | +0.0019 [−0.0007, +0.0047] | +0.0017 [−0.0046, +0.0085] |
| Best model for each finding (chosen on one half of LM-test, scored on the other) | +0.0020 [−0.0141, +0.0180] | — |

All of these are null.

---

## Kaggle Engineering

**JPEG-Lossless decoding.**
* **Problem:** some hidden-test series are JPEG-Lossless. Kaggle's offline image has no decoder for JPEG-Lossless or JPEG-LS.
* **What breaks:** an affected slice is silently replaced by a neighbouring slice. A fully affected plane crashes the notebook. In simulation this cost about −0.005 LM-test AUROC per 10% of affected planes.
* **Fix:**
  1. Ship the `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg` and `pylibjpeg-rle` wheels in the submission dataset.
  2. Decode a known test file at startup and check the output matches exactly.
  3. Pin the Pillow version used for lossy JPEG.
  4. Mask any plane that still fails to decode.

**Python version.** Kaggle's default image moved to Python 3.13 during the competition. Our Python 3.12 wheels then failed to install, so we pinned the kernels to the Python 3.12 image:
`gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

**Runtime.** Deep Three took 1.98 s per study, summed over the three models, in our local export test. The submission runs on a Kaggle T4, offline.

---

## If We Could Do It Again

1. **Test precision first.** We trained every model with mixed precision (FP16) but never measured its effect. A quick early comparison of FP16, BF16 and FP32 would have shown the best trade-off between speed and score.
2. **More data for CLIP-style training.** Image–report training is weak supervision and usually needs much more data than we had. We would add other image–report data and use stronger augmentation in this stage.
3. **Explore architecture changes with a clear reason,** like the 3D RoPE idea from the [32nd-place team](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/33rd-place-silver-solution-no-external-data), which tells the model where each image sits in 3D. One idea we would try:
   * **Slot-specific experts:** one expert network per slot (plane × scan setting), possibly with attention between the experts. This mimics a radiologist comparing MRI scans taken with different settings.
4. **Download only the cache.** We downloaded the full dataset (500+ GB). Building the preprocessed cache on Kaggle and downloading only that would have saved disk space and time.

---

## Limitations

1. **Single seed.** Every model was trained once. Retraining the Soft + 336 px model with a new seed changed LM-test by −0.0029 and Gold-58 by +0.0066. Several adopted steps are about that size.
2. **A30 is confounded.** It saw about 2.5× more training samples than u9 (30 × 4,946 vs 17 × 3,438). We never ran the control with the same number of samples.
3. **LM-test is optimistic.** We picked models and ensembles on the same 432 studies. A split-half check puts the bias at about 0.004 for the ensemble picks.
4. **LM-test uses the training labeller.** It measures agreement with the same LM that made the training labels.
5. **Gold-58 is small and was reused many times.** Single-model CIs are about ±0.02–0.03.
6. **No patient-level split.** The data has no usable patient ID, so splits are by study.
7. **One setting per null result.** Each negative result tried only one strength or setting.
8. **Small dataset for CLIP-style training.** Contrastive image–report training is usually done with hundreds of thousands to millions of pairs. We had about 3,500 training studies, compared 32 at a time. That is probably too few for the report text to teach the image encoder much beyond what label fine-tuning already does, which fits the small, short-lived benefit of CLIP pretraining we saw.

---

## Code

Repository: [github.com/Qiucha/novo-project](https://github.com/Qiucha/novo-project)


* **Training:** `scripts/training/finetune_supervised.py --config configs/finetune/<run>.json`
* **Export:** `scripts/submission/export_finetune_bundle.py`
* **Inference notebook:** `notebooks/kaggle_submission.ipynb`
* **Configs:**
  * A30: `configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30.json`
  * u9: `configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1.json`
  * A30-CLIP: `configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-from-clip.json`

## Team and Contributions

* **Yun-Hsuan Ku:** Ran the experiments: trained the fine-tuned models, compared them on LM-test and Gold-58 with paired bootstrap CIs, built the ensembles and prepared the Kaggle submission kernels.
* **Yeshwanth Patil:** Data preparation and report review: explored the data, set up the 58-study human-labelled split, and translated and cleaned the non-English radiology reports.
* **Zihan Zhou:** CLIP (image–report) model: proposed ideas we tested, including DeCLIP and RAD-DINO, and raised the concern that the gain from contrastive training on DINOv2 might come from domain adaptation to knee MRI rather than from the report text.
* **Raymond Chou:** Report labelling and evaluation: extracted labels from the translated reports with several language models and scored each one against the Gold-58 human labels.
* **Bowen Liu:** Code checks and model comparison: checked the code, and compared local translation models (NLLB, M2M100, mT5) on the Gold-58 reports.

## Acknowledgements

**Alex Benjamin**, for guidance on which ideas and experiments to try, and for keeping our focus on the parts of the pipeline that mattered most.
