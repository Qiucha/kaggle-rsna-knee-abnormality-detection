# Knee MRI image–report contrastive learning pipeline

Date: 2026-09-02

Branch: `contrastive-learning`

Status: implemented and intentionally uncommitted

## Executive summary

The pipeline learns a shared embedding space for one knee MRI study and its radiology report. It reads reports from `data/sample/train.csv`, locates the corresponding DICOM study under `/mnt/d/train_series`, encodes the study with DINOv2, encodes the report with a selectable medical language model, and optimizes symmetric image-to-text/text-to-image InfoNCE.

The implementation keeps the image-reading behavior of `minimal-baseline:train.ipynb`. Its runnable defaults are:

- Image encoder: `facebook/dinov2-small`, matching the baseline notebook's DINOv2-small choice.
- Text encoder: `UCSD-VA-health/RadBERT-RoBERTa-4m`, a radiology-report encoder and the closest available starting domain to MRI reporting.
- Shared embedding size: 256.
- Input: three anatomical planes per study, with three DICOM slices stacked as the channels of each plane image.
- Loss: learned-temperature, bidirectional InfoNCE over the paired studies in a minibatch.
- Validation: bidirectional Recall@1/5/10 and median/mean rank.

The separate [model research note](model-encoder-research.md) gives primary-source evidence for the encoder shortlist and recommends DINOv2-base as the first higher-capacity experiment. Small remains the code default so the initial run is faithful to the baseline and less expensive.

## Data contract and observed inventory

The two modalities are deliberately loaded from different configured roots:

| Modality | Default source | Join key |
|---|---|---|
| English report | `data/sample/train.csv`, column `Report` | `StudyInstanceUID` |
| Series metadata | `/mnt/d/train_series.csv` | `StudyInstanceUID`, `SeriesInstanceUID` |
| DICOM pixels | `/mnt/d/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm` | Directory names from the manifest |

A read-only inventory check found:

- 4,407 report rows, 4,407 unique study UIDs, no duplicate UIDs, and no blank reports.
- Report lengths from 8 to 690 whitespace-separated words, with a median of 129 words.
- 24,371 series rows covering the same 4,407 studies.
- 9,864 sagittal, 8,609 coronal, and 5,898 axial series; 14,010 series are marked fluid-sensitive.

Per the project requirement, the text path assumes every report is English. It does not detect language, translate, or discard reports based on language. The twelve pathology label columns in the CSV are not used for contrastive pretraining.

Only studies satisfying all of these conditions enter training:

1. A non-empty report exists.
2. Sagittal, coronal, and axial series exist in the series manifest.
3. The selected series directories contain DICOM files.
4. By default, each selected plane successfully decodes at least one sampled slice.

The UID split happens after filtering, so a study cannot occur in both train and validation. A patient-level identifier is not available in the supplied tables, so the code cannot guarantee patient-disjointness when one patient has multiple study UIDs.

## End-to-end data flow

```text
data/sample/train.csv                     /mnt/d/train_series.csv + DICOM tree
          |                                             |
          +---------- inner join on Study UID ----------+
                                |
               choose sagittal/coronal/axial series
                                |
          three baseline-compatible [3, 224, 224] tensors
                                |
                    DINOv2, once per plane
                                |
         CLS + mean(patch tokens), learned plane attention
                                |
                     image projection + L2 norm
                                |
                     paired symmetric InfoNCE
                                |
     report tokenizer -> medical LM -> CLS -> projection + L2 norm
```

One dataset item has the following shape:

- `images`: `[3 planes, 3 sampled slices/channels, 224, 224]`, `float32`, in `[0, 1]`.
- `report`: one complete report string.
- `study_uid`: the pairing and audit identifier.

The tokenizer runs in the collator, pads only to the longest report in the minibatch, and truncates to 256 tokens by default.

## Image preprocessing fidelity

The implementation in `src/contrastive_learning/preprocessing.py` reproduces the current notebook path as follows.

### Series selection

For each of `Sagittal`, `Coronal`, and `Axial`:

1. Consider series in that plane.
2. Prefer rows where `Fluid_Sensitive == 1`; fall back to any series in the plane.
3. Choose the series directory containing the most `.dcm` files.
4. Preserve manifest order when slice counts tie.
5. Reject a study if any plane is unavailable or empty.

### Slice ordering and sampling

Slices are ordered by projected DICOM geometry (`ImagePositionPatient` on the normal derived from `ImageOrientationPatient`). The deterministic fallbacks are `SliceLocation`, then `InstanceNumber`, then SOP UID/file name. Three indices are sampled evenly over the central 20–80% of the ordered stack. If a selected slice cannot decode, its nearest successfully decoded neighbor fills it; the plane is rejected only if every selected slice fails.

### Pixel processing

1. Apply `RescaleSlope` and `RescaleIntercept`.
2. Stack the three sampled slices as channels.
3. If valid `PixelSpacing` exists, take a centered 130 mm square crop when that crop is sensible for the source dimensions.
4. Compute the 1st and 99th percentiles over the selected series tensor, clip to that range, and scale to `[0, 1]`.
5. Bilinearly resize to 224×224.

### Laterality and augmentation

Laterality resolves in this order: Series Laterality tag, Image Laterality tag, then center-x patient geometry with a 20 mm dead zone. Right coronal/axial images are horizontally flipped; right sagittal stacks reverse their slice/channel order. Unresolved studies remain unchanged.

Training applies the same augmentation family as the baseline: rotation up to ±8°, zoom-in up to 8%, translation up to 5%, and intensity gain up to ±10%. It intentionally applies no random horizontal or vertical flips because those can corrupt anatomy and laterality.

## Encoders and fusion

### DINOv2 image tower

Each plane is passed independently through one shared DINOv2 backbone. The code concatenates its class token with the mean of its patch tokens, matching the useful `CLS + mean` representation tested in the baseline. A learned plane embedding is added, then a small attention module produces one weighted study representation from the sagittal, coronal, and axial features. A two-layer projection head maps that result into the shared space and L2-normalizes it.

The default DINOv2-small model uses 224×224 inputs. The trainer rejects sizes not divisible by the 14-pixel DINOv2 patch size. The last three image transformer blocks are trainable by default, matching the notebook setting; the other blocks are frozen. For a projection-only warm start, set `--image-unfreeze-last 0`.

### Medical report tower

The report encoder uses `AutoTokenizer` and `AutoModel`. Its first special token is the global feature, followed by a separate two-layer projection and L2 normalization. The final two language-model blocks train by default; `--text-unfreeze-last 0` gives a frozen-encoder warm start.

Supported short names are:

| CLI name | Checkpoint | Intended role |
|---|---|---|
| `radbert-va` | `UCSD-VA-health/RadBERT-RoBERTa-4m` | Default; broad radiology-report pretraining and the closest MRI-domain fit |
| `radbert` | `StanfordAIMI/RadBERT` | Independent radiology-specific comparison |
| `cxrbert` | `microsoft/BiomedVLP-CXR-BERT-specialized` | Prior multimodal contrastive training, but chest-X-ray-biased |
| `biomedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | Broad biomedical-literature control |
| `clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | Broad clinical-note control |

An arbitrary Hugging Face model ID can also be passed to `--text-model`. See [the encoder research](model-encoder-research.md) for citations, caveats, and a controlled comparison plan.

## Objective

For a batch of `B` unique study/report pairs, the model produces unit image embeddings `I` and text embeddings `T`. Similarities are:

```text
logits = exp(logit_scale) * I @ T.transpose()
```

The diagonal contains the true pairs. Training minimizes:

```text
loss = 0.5 * (cross_entropy(logits, diagonal_targets)
            + cross_entropy(logits.transpose(), diagonal_targets))
```

The temperature is learned and its inverse is capped at 100 for stability. Every item contains all three plane views in a single study embedding, rather than treating same-study views as separate examples, so they cannot become false negatives against their own report. Other semantically similar studies can still be false negatives; larger-batch or multi-positive objectives are appropriate follow-up experiments if this becomes limiting.

Contrastive quality depends strongly on the number of in-batch negatives. The default batch size is 16. If memory allows, increase it; gradient accumulation alone does not create additional in-batch negatives.

## Optimization and evaluation

The optimizer is AdamW with independent parameter groups:

| Parameters | Default learning rate |
|---|---:|
| Unfrozen DINOv2 blocks | `1e-5` |
| Unfrozen text blocks | `2e-5` |
| Plane attention, projection heads, and temperature | `1e-3` |

Training uses 20 epochs, weight decay 0.02, 10% linear warmup followed by cosine decay, automatic mixed precision on CUDA, and gradient clipping at norm 1.0. Randomness uses seed 2026, including the UID split and data-loader generator.

At each epoch the full validation embedding matrix is evaluated in both directions. Reported metrics are Recall@1/5/10, median rank, and mean rank for image→text and text→image. `mean_r@1`, the average of both Recall@1 values, chooses the best checkpoint. This avoids using the unused pathology labels as a selection target.

Artifacts under the selected output directory are:

- `config.json`: complete resolved run configuration.
- `metrics.jsonl`: one machine-readable record per completed epoch.
- `last.pt`: latest model, optimizer, scheduler, epoch, metrics, and config.
- `best.pt`: checkpoint with the best validation `mean_r@1`.

Checkpoint writes go to a temporary sibling and are renamed into place, avoiding a partially written final checkpoint if serialization is interrupted.

## Running the pipeline

Create and activate the project environment, then install dependencies:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Validate pairing and DICOM decoding without downloading model weights or training:

```bash
python scripts/train_contrastive.py --prepare-only
```

Run the default baseline-compatible experiment:

```bash
python scripts/train_contrastive.py \
  --data-root /mnt/d \
  --reports-csv data/sample/train.csv \
  --text-model radbert-va \
  --output-dir checkpoints/contrastive/radbert-va-dinov2-small
```

Run a cheap integration check before a full job:

```bash
python scripts/train_contrastive.py \
  --max-studies 64 \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 0 \
  --output-dir checkpoints/contrastive/smoke
```

Run the higher-capacity image configuration recommended by the model research:

```bash
python scripts/train_contrastive.py \
  --image-model facebook/dinov2-base \
  --text-model radbert-va \
  --image-unfreeze-last 0 \
  --text-unfreeze-last 0 \
  --output-dir checkpoints/contrastive/radbert-va-dinov2-base-frozen
```

For text-model comparison, keep every other flag and the seed fixed and change only `--text-model` across the five short names.

## Implementation map

| File | Responsibility |
|---|---|
| `scripts/train_contrastive.py` | Repository-local executable entry point |
| `src/contrastive_learning/cli.py` | CLI parsing, model aliases, input checks |
| `src/contrastive_learning/config.py` | Serializable defaults and model registry |
| `src/contrastive_learning/data.py` | CSV join, series selection, study split, dataset, tokenizing collator |
| `src/contrastive_learning/preprocessing.py` | Baseline-compatible DICOM ordering, decoding, crop, normalization, laterality, augmentation |
| `src/contrastive_learning/model.py` | DINOv2 tower, medical text tower, plane fusion, projections, InfoNCE |
| `src/contrastive_learning/training.py` | loaders, optimizer/schedule, AMP loop, retrieval evaluation, checkpoints |
| `tests/` | Loss, retrieval, laterality, and augmentation invariants |

## Verification performed

- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed.
- `.venv/bin/python -m pip check`: passed with no broken requirements.
- `.venv/bin/python -m pytest -q`: 5 tests passed:
  - a matched similarity matrix has lower symmetric InfoNCE than an ambiguous matrix;
  - aligned embeddings achieve perfect bidirectional Recall@1;
  - non-square logits are rejected;
  - left-knee transformations use the baseline axes;
  - augmentation preserves tensor shape, dtype, and `[0, 1]` range.
- Real DICOM preparation on four studies passed: all four reports paired with all three required planes and no sampled plane failed decoding.
- A one-epoch end-to-end CPU smoke run passed with the default `facebook/dinov2-small` and `UCSD-VA-health/RadBERT-RoBERTa-4m` checkpoints. It exercised model/tokenizer loading, DICOM preprocessing, forward and backward passes, symmetric InfoNCE, retrieval evaluation, and best/last checkpoint serialization. The smoke split contained three training studies and one validation study and had training loss `0.7108`. Its perfect Recall@1 is not a quality estimate because the validation set had only one item. Smoke model downloads are cached at `/tmp/novo-project-hf`, and its checkpoints are at `/tmp/novo-project-contrastive-smoke`.
- Installed test environment versions were Python 3.13, PyTorch 2.13.0 CPU, torchvision 0.28.0 CPU, transformers 5.16.1, pydicom 3.0.2, pandas 3.0.5, and pytest 9.1.1.

## Known limits and recommended next experiments

1. The 256-token default truncates the longest reports. Record tokenizer length percentiles for each candidate model and compare 256 versus 512 tokens or section-aware findings/impression extraction.
2. DINOv2 is pretrained on natural images, not MRI. Compare frozen, last-block, and baseline-style last-three-block adaptation before full fine-tuning.
3. A 16-study batch provides only 15 negatives per anchor. Larger batches, cross-GPU embedding gathering, or a feature queue may materially help.
4. The current split is study-disjoint but cannot be proven patient-disjoint without a patient identifier.
5. Retrieval quality is not clinical validity. Any downstream diagnostic use requires separate labels, calibration, external validation, and clinical review.
6. DICOM compressed-transfer-syntax support depends on the installed pixel decoders. If decode validation reports failures, add the decoder appropriate to the dataset rather than skipping validation for the final run.
