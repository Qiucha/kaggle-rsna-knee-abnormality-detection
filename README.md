# RSNA Knee Abnormality Detection

We predict 12 knee MRI findings by fine-tuning OmniRad-base on labels extracted
from radiology reports by a language model (LM). Our final model averages the
prediction ranks of three fine-tuned models.

This is a summary of the [refined report](kaggle-writeup-refined.md).

## Results

Scores are macro ROC-AUC. The private leaderboard determines the final ranking.

| Model | Private | Public | LM-test | Gold-58 |
| --- | ---: | ---: | ---: | ---: |
| Deep Three ensemble | **0.889** | **0.902** | 0.9017 | 0.8897 |
| A30 single model | 0.878 | 0.890 | 0.8917 | 0.8814 |

[Final submission records](kaggle/submissions/README.md) include model configs,
bundle hashes and kernel versions.

## Final method

- **Inputs:** competition DICOMs, with one series per plane. Prefer fat-suppressed
  series, align knee orientation, crop to 130 mm and resize to 336 × 336.
- **Slices:** take eight windows from the central 60% of each plane. Stack three
  adjacent slices as RGB channels: 24 images per study. Sample random positions
  during training and window centres at inference. Mask missing or unreadable planes.
- **Backbone:** `Snarcy/OmniRad-base`, a ViT-B/14 pretrained on RadImageNet.
  Fine-tune the last nine of its twelve blocks.
- **Head:** concatenate the CLS token and mean patch features, average across valid
  images, then use a 256-unit MLP to predict 12 findings.
- **Loss:** binary cross-entropy with soft LM targets. Ignore `UNK` labels.
- **A30:** train for 30 epochs. Sample studies positive for Lateral Meniscus,
  Lateral OA or Synovitis twice per epoch. On extra copies, train only those labels.
  Add target noise to Lateral Meniscus and Lateral OA. Keep epoch 22.
- **Ensemble:** average per-label prediction ranks from A30, u9-b1 and CLIP-init u9.
  All three use the backbone and inputs above.

Labels come from **@pilkwang**'s CC0 dataset
[rsna-knee-llm-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels).
We used no external training images beyond the backbone's pretraining.

## Evaluation

The fixed cohort contains 4,354 studies:

| Split | Studies | Purpose |
| --- | ---: | --- |
| Train | 3,438 | Fit model weights |
| Validation | 426 | Select the lowest-loss checkpoint |
| LM-test | 432 | Compare models against LM labels |
| Gold-58 | 58 | Check performance against human labels |

Comparisons use paired bootstrap 95% confidence intervals. Most single-model
changes had LM-test intervals that crossed zero. We adopted them using point
estimates and Gold-58 as a tiebreaker. Deep Three improved LM-test over A30 by
0.0100, with a 95% interval of [0.0048, 0.0155].

## Experiment tracks

We started with image–report contrastive learning and frozen prediction heads.
Fine-tuning on LM labels raised public AUROC from 0.788 to 0.826. Later changes
led to A30 and the final ensemble.

Contrastive pretraining gave no clear single-model gain after strong fine-tuning,
but a CLIP-initialized model helped the ensemble. Image augmentation, attention
pooling, weight averaging and test-time augmentation gave no clear gains in the
tested settings. The joint image–report loss hurt Gold-58.

## Limitations

Most models used one seed. A30 also saw about 2.5 times as many training samples as
u9-b1, so its gains cannot be attributed to oversampling alone.

LM-test shares the training labeller and was reused for model selection. Gold-58
is small and was also reused. Splits are by study because usable patient IDs were
unavailable. Small gains need more seeds and independent validation.

## Setup

Run commands from the repository root. Use Python 3.13 and a CUDA-capable GPU:

```bash
python3.13 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-cuda-lock.txt
cp configs/paths.template.json paths.json
```

Set `data_root` in `paths.json`. Restore the private LM labels, frozen cohort
manifest and pinned model weights before training. See
[Train and submit](TRAIN_AND_SUBMIT.md) for required files and full commands.

## Code and guides

- [Training configs](configs/finetune/): A30, u9-b1 and CLIP-init variants.
- [Training script](scripts/finetune_supervised.py) and [bundle exporter](scripts/export_finetune_bundle.py).
- [Inference notebook](notebooks/kaggle_submission.ipynb): offline Kaggle predictions.
- [DICOM codecs](kaggle/codecs/README.md): bundled decoders and startup checks.
- [Pipeline guide](src/contrastive_learning/README.md) and [known limitations](KNOWN_LIMITATIONS.md).
