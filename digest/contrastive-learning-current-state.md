# Contrastive learning: current-state audit

Date: 2026-09-04 (Asia/Taipei)

## Bottom line

The repository contains a **runnable image-report contrastive-learning prototype** and a useful 58-study feasibility experiment. It does not yet contain evidence for a reliable or clinically useful model.

Three statements best describe the current state:

1. **The mechanics work on the pilot cohort.** The 58 translated reports pair with 58 studies; every study has decodable sagittal, coronal, and axial inputs; the seven unit tests pass; and completed model checkpoints exist.
2. **The reported retrieval signal is preliminary.** Frozen-backbone five-fold Recall@1 is above the fold-specific random-ranking baseline, but folds contain only 11 or 12 studies, the variance is large, and the same validation fold is used to choose and report the best epoch.
3. **A full 4,407-study run is not ready.** The default full-cohort CSV exposes only the raw `Report` column. A matched 58-study audit shows that 30/58 raw reports are non-English and require the separate `Translated Report` field, which does not exist for the full cohort.

Status by layer:

| Layer | Status | Interpretation |
|---|---|---|
| Data pairing and DICOM decoding | Green for the 58-study pilot | All 58 studies have three decodable planes |
| Forward/backward training path | Green as a smoke/prototype path | Prior one-epoch end-to-end smoke and completed five-epoch runs exist |
| Unit tests | Green but shallow | 7/7 pass; important model, data, checkpoint, and metric edge cases are untested |
| Retrieval evaluator | Red | Collapsed identical embeddings are incorrectly scored as 100% Recall@1 |
| Full-cohort text contract | Red | The code assumes English, but the 4,407-row default file contains raw multilingual reports |
| Experiment validity | Amber/red | Only 58 translated studies, small folds/batches, and no independent test set |
| Reproducibility and operations | Red | All contrastive source is uncommitted; runs cannot resume safely; dependencies/model revisions are not pinned |
| Clinical/downstream evidence | Not started | Retrieval alone does not establish diagnostic usefulness |

## What is implemented

The pipeline is:

```text
DICOM study
  -> select largest fluid-sensitive sagittal/coronal/axial series
  -> sample three slices from each plane and preprocess to [3, 3, 224, 224]
  -> shared DINOv2 image backbone
  -> CLS + mean-patch features and learned three-plane attention
  -> normalized 256-dimensional image embedding

report
  -> medical-language-model tokenizer and backbone
  -> CLS feature
  -> normalized 256-dimensional text embedding

paired image/text embeddings
  -> learned-temperature symmetric InfoNCE
  -> bidirectional Recall@1/5/10 and rank metrics
```

Relevant implementation:

- [`config.py`](../src/contrastive_learning/config.py) defines the model and optimization defaults.
- [`data.py`](../src/contrastive_learning/data.py) joins studies/reports, selects series, creates folds, caches tensors, and tokenizes reports.
- [`preprocessing.py`](../src/contrastive_learning/preprocessing.py) orders and decodes DICOMs, crops/normalizes images, normalizes laterality, and augments training tensors.
- [`model.py`](../src/contrastive_learning/model.py) implements the two towers, projection heads, three-plane fusion, and symmetric InfoNCE.
- [`training.py`](../src/contrastive_learning/training.py) implements loaders, optimization, retrieval evaluation, and best/last checkpoints.
- [`train_contrastive.py`](../scripts/training/train_contrastive.py) is the executable entry point.

Default code settings are DINOv2-small plus UCSD/VA RadBERT, 256-dimensional embeddings, batch size 16, 20 epochs, and partial unfreezing of the last three image blocks and last two text blocks. The saved experiments do **not** use those training defaults: they use 58 translated studies, batch size 8, five epochs, and usually frozen backbones.

## Evidence collected in this audit

### Executable checks

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | 7 passed |
| `.venv/bin/python -m compileall -q src scripts tests` | Passed |
| `git diff --check` | Passed |
| Environment | Python environment imports successfully; PyTorch 2.13.0 CPU; CUDA unavailable |
| Real 58-study `--prepare-only` with pixel validation | 58/58 paired, 0 missing-plane, 0 decode failures |
| Persistent image-cache audit | 58 tensors, all `[3,3,224,224]`, no non-finite/zero/constant channels, range `[0,1]` |

The real preparation check took about two minutes on the current CPU/storage path. A full training run on this host would be impractically slow.

### Data contract audit

- `data/sample/train.csv`: 4,407 rows and 4,407 unique study UIDs; only a raw `Report` column is available.
- `data/gold_translated.csv`: 58 rows and 58 unique study UIDs; both `Report` and `Translated Report` are populated.
- The 58 raw reports exactly match the corresponding reports in the 4,407-row file.
- The gold subset labels 28 reports as English and 30 as non-English; translation changes exactly those 30 non-English reports.
- Therefore the default full-data command cannot satisfy the documented English-report assumption. At least 30 known rows are non-English, and the actual full-cohort language distribution is not recorded.
- No exact normalized report duplicates were found among the 58 translated reports.
- A DICOM-header scan found a populated `PatientID` for all 4,407 studies, but every value was unique. This reveals no observable repeated patients, but it cannot establish whether identifiers were anonymized per patient or per study. The trainer does not group by patient.

### Saved experiments

There are 17 completed run directories and about 18 GB of checkpoints. The main controlled session contains ten frozen-backbone cross-validation runs, three unfreezing runs, and one token-length run.

Five-fold frozen-backbone results reported from the best epoch on each fold:

| Text encoder | Image-to-text R@1 | Text-to-image R@1 | Bidirectional mean R@1 |
|---|---:|---:|---:|
| Stanford RadBERT | 18.79% +/- 10.65% | 15.61% +/- 9.65% | 17.20% +/- 8.92% |
| BiomedBERT | 18.94% +/- 7.31% | 20.61% +/- 7.60% | 19.77% +/- 6.59% |
| Random ranking for these fold sizes | 8.62% | 8.62% | 8.62% |

The folds contain 12, 12, 12, 11, and 11 examples. Consequently, one additional correct retrieval changes a directional fold R@1 by 8.33 or 9.09 percentage points. BiomedBERT's 2.57-point mean advantage over Stanford RadBERT is much smaller than fold-to-fold variation and is not enough to declare it the better encoder.

At fixed epoch 5 instead of selecting the best epoch on each evaluated fold, pooled bidirectional R@1 is about 16.4% for BiomedBERT and 13.8% for Stanford RadBERT. This still suggests feasibility, but also shows that the reported endpoint depends materially on selection policy.

Other pilot findings:

| Ablation | Mean R@1 | What it supports |
|---|---:|---|
| BiomedBERT, 0 blocks unfrozen | 16.67% | Frozen baseline |
| BiomedBERT, 1 block in each tower unfrozen | 20.83% | Candidate, single holdout only |
| BiomedBERT, 2 blocks in each tower unfrozen | 20.83% | Candidate, single holdout only |
| BiomedBERT, 256 tokens | 16.67% | Current pilot default |
| BiomedBERT, 512 tokens | 12.50% | No R@1 improvement on one holdout |

These results support “the model can learn some pairing signal on this sample.” They do not support choosing a final encoder, unfreezing policy, or token length.

## What is missing or incorrect

### P0: fix before trusting another experiment

#### 1. Retrieval ties are scored incorrectly

[`retrieval_metrics`](../src/contrastive_learning/training.py) computes rank as one plus the number of scores strictly greater than the true-pair score. All tied candidates are therefore assigned the best possible rank.

Observed probe:

```text
input: four identical normalized image embeddings and four identical text embeddings
reported: image-to-text R@1 = 100%, text-to-image R@1 = 100%
```

A collapsed representation must not receive perfect retrieval. Implement a documented tie policy, add collapsed/tied regression tests, and re-evaluate saved checkpoints. Exact ties may not have affected the existing runs, but that cannot be established from the saved aggregate metrics alone.

#### 2. The full-cohort reports are not language-consistent

The default code uses `data/sample/train.csv` and `Report`, while the selected language models are English biomedical/radiology encoders. The full file lacks both language labels and translated reports. The 58-study matched subset proves that the raw field includes non-English text.

Choose one explicit contract before scaling:

- translate all reports to English and retain source text, detected language, translation model/version, and QC status; or
- choose and validate a multilingual medical text encoder.

Do not treat the current raw 4,407-report default as an English training set.

#### 3. Validation is also used for model/epoch selection

Each fold's `mean_r@1` selects `best.pt`, and the selected value from that same fold is reported as performance. Encoder choice and unfreezing/token ablations also reuse these small validation data. This makes the estimate optimistic.

Use an inner validation split for epoch/hyperparameter selection and a locked outer test fold, or predeclare a fixed training duration using only training-side evidence and evaluate each outer fold once. Save per-query rankings and study IDs so uncertainty and errors can be audited.

### P1: required for reliable full-scale training

#### 4. The experiment is much too small for the stated architecture

The empirical work uses 46 training studies in the single holdout and 46-47 per cross-validation fold. Even with frozen backbones, the projection and fusion modules contain many trainable parameters. Batch size 8 provides only seven in-batch negatives per anchor, and incomplete training batches are dropped.

The 58-study set should remain an integration/feasibility set. The next meaningful representation experiment needs the language-corrected full cohort, larger effective negative pools, and a GPU.

#### 5. Runs are not safely reproducible or resumable

- The entire contrastive implementation, tests, digest, and translated CSV are currently uncommitted. The branch points at the same commit as `main`/`origin/main`.
- `metrics.jsonl` is opened in append mode, but rerunning an output directory resets `best_score`, overwrites configs/checkpoints, and does not restore model/optimizer/scheduler state.
- Checkpoints contain full backbones plus optimizer state; current artifacts consume about 18 GB.
- Package versions are mostly unpinned, and Hugging Face model revisions are not recorded/pinned.
- Split UIDs, per-query predictions/ranks, source-data hashes, git commit, and preprocessing code version are not saved.

Add output-directory protection, explicit resume semantics, immutable run IDs/manifests, dependency/model revision locking, and a lightweight inference/evaluation artifact.

#### 6. Image caching will not scale cleanly

With `cache_images=True`, [`KneeReportDataset`](../src/contrastive_learning/data.py) eagerly loads every study tensor into RAM. For 4,407 tensors of shape `[3,3,224,224]` in float32, that is about 7.4 GiB before model, optimizer, loader, and Python overhead.

The persistent-cache fingerprint also includes settings and paths but not DICOM content/mtime or an automatic preprocessing-code hash, so stale tensors can survive data or code changes.

Use a disk-backed lazy cache with a bounded RAM LRU, a content/versioned manifest, atomic concurrency-safe writes, and preprocessing QC summaries. Avoid `--cache-images` for the full cohort under the current implementation.

#### 7. Training/evaluation health signals are incomplete

The trainer records only average training loss and retrieval metrics. It does not record validation contrastive loss, temperature/logit scale, learning rates, gradient norms, embedding variance, positive/negative similarity distributions, or collapse/alignment diagnostics. These are necessary to distinguish learning, overfitting, and collapse.

Also address or explicitly test these optimization choices:

- frozen backbones are returned to train mode with the whole model, so dropout remains active during a nominally frozen run;
- AdamW weight decay is applied to biases, normalization parameters, and the learned logit scale;
- the clamped exponential logit scale can receive zero gradient after it crosses the cap;
- CLS pooling is hard-coded by `build_model`, even for masked-language-model encoders that were not trained as sentence encoders.

### P2: representation and clinical-quality follow-ups

- Only nine 2D slices represent an entire 3D study; important findings outside sampled locations can be missed.
- Series selection uses plane, fluid-sensitivity, and slice count, but has no sequence-quality or artifact QC.
- Pixel decode errors are swallowed, differing decoded shapes are silently zero-filled, and laterality conflicts across planes are not surfaced.
- Ordinary paired InfoNCE treats clinically similar studies as negatives; this becomes more important as batch size grows.
- There is no inference/retrieval command, downstream linear probe, external cohort, calibration, subgroup analysis, or clinical review path.

## Recommended next steps

### Now, on the current CPU host

1. **Version the work.** Commit source, tests, configs, and digests; keep raw/private data and 18 GB of checkpoints in an artifact store. Record artifact checksums and the commit used for every run.
2. **Repair the evaluator.** Define tie behavior, add tied/collapsed tests, add an evaluation-only CLI, and re-score the existing 58-study checkpoints. Include random baselines and confidence intervals in the report.
3. **Create a canonical cohort manifest.** One row per study with report language/provenance, patient grouping key, selected series, laterality, preprocessing status/QC, split assignment, and source hashes.
4. **Resolve the text contract.** Translate/QC all 4,407 reports or select a validated multilingual encoder. Preserve both original and normalized text; never overwrite source reports.
5. **Harden run management.** Add fail-if-output-exists, resume, config compatibility checks, fixed split files, metric/prediction export, model revision pins, and bounded disk caching.

### Next experiment session, on a GPU

1. Run a projection-only baseline on the language-consistent full cohort with a patient/group-disjoint train/validation/locked-test protocol.
2. Increase the negative pool (larger batch, cross-device gather, or a validated queue). Gradient accumulation by itself does not add contrastive negatives.
3. Compare text pooling and report inputs: CLS versus masked mean pooling; full report versus findings/impression-aware text.
4. Compare one change at a time: frozen towers, image-only unfreezing, text-only unfreezing, then both. Do not promote the pilot's two-block result without full-cohort confirmation.
5. Evaluate per-query retrieval with bootstrap confidence intervals, random baselines, duplicate/multi-positive handling, and qualitative error review.

### Before claiming usefulness

1. Freeze the representation and run predefined downstream pathology probes against labels that were not used to tune the representation.
2. Evaluate on a genuinely external cohort and clinically meaningful subgroups.
3. Add calibration, failure analysis, and clinical review. Retrieval improvement alone is not evidence of diagnostic performance.

## Decision

The correct immediate decision is **do not launch a full default training run yet**. First fix the metric, establish a language-consistent full dataset, and make evaluation independent of selection. The existing 58-study checkpoints are valuable as integration artifacts and hypothesis generators, not as a final model comparison.
