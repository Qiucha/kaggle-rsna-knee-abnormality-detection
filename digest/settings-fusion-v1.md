# Do the non-fat-suppressed series add label information?

2026-10-10. Frozen-feature probe on the u9-b1 rung (`r4a-k8-omnirad-base-soft-336-u9-b1`).

**Answer in one line: no.** Adding the non-fat-suppressed series to what the rung already reads
moves LM-test macro AUROC by −0.0002 under a pretrained encoder and −0.0044 under the rung's own,
both intervals spanning zero. Keep the three-slot input; a six-slot fine-tune is not worth building.

**Artifacts.** [`settings-fusion-v1.json`](settings-fusion-v1.json) (histories stripped; the full
report and the 1.2 GB of features are under `checkpoints/settings-fusion/`). Scripts:
`scripts/extract_setting_features.py`, `scripts/train_setting_heads.py`.

## Why this experiment existed

`pick_study_slots` gives every study one series per plane, the fat-suppressed one when the plane
has any (the CSV's `Fluid_Sensitive` is a byte copy of `Fat_Suppression`, see
[`acquisition-metadata-trust.md`](acquisition-metadata-trust.md)). The other weighting, 42% of all
series, never reached a model. Learned pooling over the series already lost twice, once as the
contrastive tower's plane attention ([`plane-pooling-protocol.md`](plane-pooling-protocol.md)) and
once as R4b's per-label attention. So the open question was information, not fusion.

## Design

- **Slots.** Six per study: the manifest's own slot per plane (*primary*) and the best series of
  that plane with the opposite fat-suppression flag (*secondary*), ranked as
  `rank_plane_candidates` ranks. Secondary coverage across all 4,354 studies: sagittal 90%,
  coronal 73%, axial 19%.
- **Windows.** The rung's evaluation path: 8 stratum-centre windows per slot, 336 px, crop 130 mm,
  band 0.2–0.8, right knees mirrored.
- **Two encoders.** *Finetuned* is u9-b1's adapted backbone. It was trained on the primary slots
  of the same training studies the heads train on, which favours arm A. *Pretrained* is
  OmniRad-base as u9-b1 started, which has seen neither slot. Pretrained is the clean test.
- **Heads.** Same populations, soft targets and selection rule as the rung, with three seeds per arm:
  - **A:** mean of the primary planes.
  - **B:** A concatenated with the mean of the present secondary planes and three presence bits.
  - **C:** mean of all present slots, each with a learned slot embedding added.
  - **D:** 12 label queries attending over the slots.
- **Gate.** The rung's own head on arm A's finetuned features reproduced its saved predictions to
  within 0.0002 (tolerance 0.02), so the extraction matches the rung's evaluation.

## Results

Mean of three seeds. Δ is paired against arm A on the same seed; the interval is a 1,000-resample
study bootstrap of the seed-averaged predictions.

| Encoder | Arm | LM-test | Δ LM-test [95%] | Gold | Δ gold [95%] |
| --- | --- | --- | --- | --- | --- |
| finetuned | A primary mean | 0.8766 | | 0.8662 | |
| finetuned | B + secondary | 0.8722 | −0.0044 [−0.0095, +0.0011] | 0.8653 | −0.0009 [−0.0113, +0.0122] |
| finetuned | C slot mean | 0.8756 | −0.0010 [−0.0090, +0.0031] | 0.8666 | +0.0004 [−0.0134, +0.0129] |
| finetuned | D label attention | 0.8751 | −0.0015 [−0.0086, +0.0045] | 0.8650 | −0.0012 [−0.0124, +0.0144] |
| pretrained | A primary mean | 0.8368 | | 0.7600 | |
| pretrained | B + secondary | 0.8366 | −0.0002 [−0.0076, +0.0120] | 0.7636 | +0.0036 [−0.0239, +0.0176] |
| pretrained | C slot mean | 0.8328 | −0.0040 [−0.0187, +0.0025] | 0.7449 | −0.0150 [−0.0487, +0.0023] |
| pretrained | D label attention | 0.8428 | +0.0060 [−0.0083, +0.0145] | 0.7697 | +0.0098 [−0.0238, +0.0319] |

## Reading

- **B under the pretrained encoder is the answer, and it is zero.** Given the primary features, the
  secondary series add nothing a head can use.
- **Pretrained D is the only positive cell, but it is not evidence.** Both intervals span zero.
  Its largest per-label gains (synovitis, Baker's, lateral OA) are not where the non-fat-suppressed
  series should help (meniscus on sagittal PD, fracture and marrow on T1), and ACL drops 0.035.
- **Under the finetuned encoder every arm sits at or below A.** This is the confound the
  pretrained encoder was there to remove.
- **Limits.** These are frozen features: a backbone fine-tuned on six slots could still learn
  something the heads cannot reach. The result gives no reason to pay the cost of that run (about
  1.8× decode and inference on Kaggle, and more exposure to the JPEG-Lossless gap).

## Reproduce

Run from the repository root. The prerequisite is a trained u9-b1 rung. Its config
`configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1.json` is committed on branch
`feat/ensemble-selection` (local only as of 2026-10-13), not on this one:

```bash
git show "feat/ensemble-selection:configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1.json" \
  > configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1.json
python scripts/finetune_supervised.py \
  --config configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1.json \
  --output checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1
```

Then extract the features and fit the heads:

```bash
python scripts/extract_setting_features.py \
  --checkpoint checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1 \
  --output checkpoints/settings-fusion/u9-b1-features
python scripts/train_setting_heads.py \
  --features checkpoints/settings-fusion/u9-b1-features \
  --output checkpoints/settings-fusion/u9-b1-heads.json --device cuda
```

Inputs not in git:

- `paths.json` with `data_root`: the DICOM tree (`train_series/`) and its `train_series.csv`.
- The cohort manifest `data/processed/contrastive-cohort-gold-holdout-v1.jsonl` and the
  LM labels `data/lm_label/lm_labels_kaggle.csv` (third-party, never commit).
- The trained rung directory: `config.json`, `best.pt` and `predictions.npz` (the sanity
  gate reads the last two).
- The Hugging Face snapshots the rung pins (`Snarcy/OmniRad-base`, `facebook/dinov2-base`)
  in the local cache.
- Optionally the rung's evaluation window cache `checkpoints/finetune/window-cache-k8-336`;
  without it the primary windows are decoded from the DICOM tree.

To vary a run: `--limit N` for a smoke test, `--encoders`, `--arms` and `--seeds` to
select what the heads fit, and `--checkpoint` to probe another windowed rung.
