# RSNA Knee Abnormality Detection: Fine-Tuning on Language-Model Labels + Rank Ensemble of Deep Models

**Private 0.889 / Public 0.902** with the Deep Three ensemble. The private score sets the final ranking.

> **Team:** Ouch! My Knee Hurts!
> **Authors:** Yun-Hsuan Ku, Yeshwanth Patil, Zihan Zhou, Raymond Chou, Bowen Liu
> **Full report:** [kaggle-writeup-refined.md](kaggle-writeup-refined.md) ([HTML version](kaggle-writeup-refined.html))

We predict 12 knee MRI findings by fine-tuning OmniRad-base on labels that a language model (LM) extracted from the radiology reports. The final submission averages the per-label ranks of three fine-tuned models. This README summarizes the report section by section, then explains how to run and reproduce the results.

## Summary

| Model | Private | Public | LM-test | Gold-58 |
|---|---:|---:|---:|---:|
| Deep Three ensemble | **0.889** | **0.902** | 0.9017 | 0.8897 |
| A30 single model | 0.878 | 0.890 | 0.8917 | 0.8814 |

* **No external images.** We trained only on the competition DICOMs and the LM labels from **@pilkwang**'s CC0 dataset [`rsna-knee-llm-labels`](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels). The only external asset is the pretrained backbone `Snarcy/OmniRad-base`.
* **What helped:** fine-tuning instead of frozen features, soft LM targets, 336 px input, training 9 of 12 blocks, lateral oversampling (A30) and a rank ensemble of three deep models.
* **What did not help:** image–report matching losses, image augmentation, learned window weights, weight averaging and test-time shifts.

[Final submission records](kaggle/submissions/README.md) hold the configs, bundle hashes and kernel versions.

## Labels and Validation

* **Targets:** 12 findings scored by macro ROC-AUC. Each LM label is `YES`, `NO` or `UNK`, with a soft value. We train on the soft values and leave `UNK` out of the loss.
* **Splits:** one fixed split of 4,354 studies.

  | Split | Studies | Used for |
  |---|---:|---|
  | Train | 3,438 | training |
  | Validation | 426 | picking the checkpoint |
  | LM-test | 432 | comparing models against the LM labels |
  | Gold-58 | 58 | sanity check against the human labels |

* **Comparisons:** paired bootstrap 95% CIs on LM-test. Gold-58 breaks ties and vetoes any change that drops it by more than 0.024.

## Preprocessing

* Pick the fat-suppressed series in each plane, if there is one.
* Flip right knees, crop 130 mm and resize to 336 × 336.
* Keep the central 60% of slices and split them into 8 windows. Each window is 3 adjacent slices stacked as RGB.
* That gives 24 images per study (3 planes × 8). Missing planes are masked.

## Model

* **Backbone:** OmniRad-base (ViT-B/14), with blocks 4–12 trained.
* **Head:** CLS token and mean patch token per image, averaged over valid images, then a 256-unit MLP to 12 logits.
* **Loss:** binary cross-entropy on the soft targets.
* **A30:** studies positive for Lateral Meniscus, Lateral OA or Synovitis are seen twice per epoch, the lateral targets get small noise, and training runs 30 epochs.

## Step-by-Step Progress

* **Stage 1** trained a CLIP-style image–report model and put small heads on its frozen features: 0.788 public at best.
* **Stage 2** fine-tuned the image encoder on the LM labels. Each step added a little: OmniRad backbone, soft targets + 336 px, 9 trained blocks (u9), then A30 at 0.890 public.
* **Fine-tuning gave the largest gain.** Most later single-model steps were within noise on LM-test. We adopted them on the point estimate plus Gold-58.

## Ensembling

* We average the per-label ranks of each model's predictions.
* Mixing in shallower models raised LM-test but lowered Gold-58 Lateral Meniscus.
* **Deep Three** (A30 + u9 + A30-CLIP) kept only deep models. It beat A30 on LM-test by +0.0100 [+0.0048, +0.0155] and scored best on both leaderboards.

## Stage 1: Contrastive (CLIP) Pretraining

* Image encoder DINOv2-small with LoRA, text encoder multilingual E5-base.
* We stopped this line because fine-tuning was much stronger, and frozen heads fell behind on Gold-58.
* Starting fine-tuning from CLIP weights gave a head start that faded by epoch 30. It still helped as an ensemble member (A30-CLIP).
* Translating the non-English reports did not improve image–report retrieval.

## What Did Not Work

Ideas with a CI crossing zero, a worse score, or a Gold-58 veto:

* **Training:** image–report matching loss, DeCLIP, RAD-DINO, RadioDino and CoAtNet backbones.
* **Data and inputs:** image augmentation, 384 px, a wider slice band, the non-fat-suppressed series, 16 windows per plane.
* **Architecture and loss:** double weight on lateral findings, learned window weights, split knee pooling, training on `UNK` labels, 0/1 labels.
* **After training:** weight averaging, test-time shifts and picking the best model per finding.

## Kaggle Engineering

* **JPEG-Lossless:** Kaggle's offline image cannot decode some hidden-test series. We ship the `pylibjpeg` wheels, run a decode self-test at startup, and mask any plane that still fails.
* **Python version:** Kaggle moved to Python 3.13 during the competition, so we pin the kernels to the Python 3.12 image.
* **Runtime:** Deep Three took 1.98 s per study in our local test. The submission runs offline on a Kaggle T4.

## If We Could Do It Again

1. Measure the speed–score trade-off of FP16, BF16 and FP32 early.
2. Add more image–report data and stronger augmentation for CLIP-style training.
3. Explore architecture changes with a clear reason, like 3D RoPE or slot-specific expert networks.
4. Download only the preprocessed cache instead of the full 500+ GB dataset.

## Limitations

* Every model was trained with one seed. Several adopted steps are within seed noise.
* A30 saw about 2.5× more training samples than u9, and we never ran that control.
* LM-test was reused for model selection and shares the training labeller. Gold-58 is small.
* Splits are by study, as the data has no usable patient ID.

---

## Usage

### Setup

Run commands from the repository root. Training needs Python 3.13 and a CUDA GPU:

```bash
python3.13 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-cuda-lock.txt
cp configs/paths.template.json paths.json
```

Then:

1. **Paths:** set `data_root` in `paths.json` to the competition data directory. Optionally set `train_series_mirror` to a copy on a fast disk.
2. **Files not in git** (ask the team):
   * `data/lm_label/lm_labels_kaggle.csv`: the third-party LM labels. Never commit it.
   * `data/processed/contrastive-cohort-gold-holdout-v1.jsonl`: the fixed split. Every config pins its SHA-256.
3. **Pinned weights:**
   ```bash
   huggingface-cli download Snarcy/OmniRad-base --revision 3e42381bcf0e6981a8b438e335c807c70b857e9a
   huggingface-cli download facebook/dinov2-base --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415
   ```
4. **Check:** `.venv/bin/python -m pytest -q` (CPU, about 10 s).

### Predict with an exported bundle

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-cuda/bin/python checkpoints/kaggle/<bundle-name>/bundle/predict_kaggle.py \
  --bundle checkpoints/kaggle/<bundle-name>/bundle \
  --data-root <competition data dir> \
  --output submission.csv --device cuda
```

On Kaggle, the same bundle runs through [`notebooks/kaggle_submission.ipynb`](notebooks/kaggle_submission.ipynb).

## Reproducing the Results

Each model is one config in `configs/finetune/`:

| Model | Config |
|---|---|
| A30 | `r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30.json` |
| u9 | `r4a-k8-omnirad-base-soft-336-u9-b1.json` |
| A30-CLIP | `r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-from-clip.json` |

**1. Train each model.** A30 takes about 26 min per epoch on one GPU.

```bash
.venv-cuda/bin/python scripts/training/finetune_supervised.py \
  --config configs/finetune/<config>.json \
  --output checkpoints/finetune/<config-name>
```

Each run writes `report.json` with the LM-test and Gold-58 macro AUROC, plus `best.pt` and `predictions.npz`.

A30-CLIP starts from a CLIP checkpoint (its `init_checkpoint`), which is not in git. It was trained with the settings in [`configs/contrastive/omnirad-clip-run5-v1.json`](configs/contrastive/omnirad-clip-run5-v1.json). Ask the team for it, or retrain it first.

**2. Export the Deep Three bundle.** Pass the runs in this order:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-cuda/bin/python scripts/submission/export_finetune_bundle.py \
  --run checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30 \
  --run checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1 \
  --run checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-from-clip \
  --output checkpoints/kaggle/<bundle-name>
```

The export checks each model against its saved predictions and scores the bundle on LM-test. Pass a single `--run` for the A30 single model.

**3. Submit on Kaggle.** Upload only the `bundle/` folder as a private dataset, push the notebook pinned to the Python 3.12 image, and submit its output. The [record files](kaggle/submissions/README.md) hold the exact dataset and kernel metadata we used.

**Expect small differences.** CUDA training is not deterministic here. Retraining one model with a new seed moved LM-test by 0.003 and Gold-58 by 0.007.

[Train and submit](TRAIN_AND_SUBMIT.md) has the full commands, config keys and troubleshooting.

## Code

* [Training script](scripts/training/finetune_supervised.py) and [bundle exporter](scripts/submission/export_finetune_bundle.py)
* [Inference notebook](notebooks/kaggle_submission.ipynb) and [DICOM codecs](kaggle/codecs/README.md)
* [Script index](scripts/README.md), [pipeline guide](src/contrastive_learning/README.md) and [known limitations](KNOWN_LIMITATIONS.md)

## Team and Acknowledgements

Yun-Hsuan Ku, Yeshwanth Patil, Zihan Zhou, Raymond Chou and Bowen Liu. See the [full report](kaggle-writeup-refined.md#team-and-contributions) for each person's contributions.

Thanks to **Alex Benjamin** for guidance, to **@pilkwang** for the LM labels, and to RSNA, the annotators and Kaggle for hosting the competition.
