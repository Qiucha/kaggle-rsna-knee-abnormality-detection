# Kaggle submission and benchmark pipeline

Checked against the working tree and local artifacts on 2026-09-21. The root README's
original multilingual matrix is historical context; newer downstream experiments
include both-tower adaptation, LoRA, DeCLIP, label-similarity targets, and gold holdouts.
Existing user changes were retained.

In this project, a competition benchmark measures the trained MRI image pipeline:
image encoder, fitted pathology heads, macro ROC-AUC, and submission inference.
Kaggle Benchmarks and its Model Proxy are a separate product for evaluating external
LLMs; use that workflow only when an LLM evaluation is explicitly requested.

## Current model choice

The default is the **holdout-trained six-block, rank-64 LoRA encoder** at
`checkpoints/contrastive/both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026/best.pt`.
The existing `digest/downstream-probe-lora-holdout-v1.json` reports **0.75936**
macro AUROC for its uniform mean CLS+patch features. The regularized LoRA variant
scores 0.71261. Both exclude the gold cohort from encoder training and selection.
The older three-seed four-block holdout checkpoints remain an explicit comparison.
Higher exposed-cohort results in other digests are not directly comparable.

The original contrastive checkpoints have no pathology classifier. This pipeline
fits twelve standardized logistic heads on the **56 eligible gold-labeled studies**
in `data/cleaned.csv`. Two additional gold studies remain locked test and are never
used. Blank competition training labels and fractional LM label confidences are
rejected as supervised targets. This is a gold-only first submission, not a classifier
trained on all 4,407 studies or a pseudo-label training experiment.

## Verified local results

The pipeline was executed on the RTX 3070 using the existing CUDA environment:

| Candidate | 56-study OOF macro AUROC | Public example inference | Bundle size |
| --- | ---: | ---: | ---: |
| Unadapted DINOv2 | 0.643992 | benchmark reference only | — |
| Four-block holdout, three-seed probability ensemble | 0.696292 | 3 studies / 18.64 s | 265.6 MB |
| **Six-block LoRA holdout, default** | **0.759362** | **3 studies / 7.17 s** | **99.5 MB** |

LoRA peak allocated GPU memory was 183.6 MB for the three-study inference call.
These timings exclude Python import/startup and are hardware- and cache-dependent.
The maximum export/reload probability difference was 2.94e-7 for LoRA.
All 56 benchmark cases were confirmed `excluded` in each encoder's own cohort.

Local artifacts are under `checkpoints/kaggle/{lora-holdout-v1,gold-holdout-v1}/`:
`benchmark.json`, `bundle/`, `submission.csv`, and `submission.benchmark.json`.
Both CSVs passed schema, ID-order, completeness, and probability validation with
Hugging Face offline mode enabled. Tests: **139 passed, 1 skipped**, including seven
new deployment tests. The seven deployment tests were also rerun after formatting;
Ruff and notebook code syntax checks passed.

The default bundle was also validated in a private Kaggle notebook version on
2026-09-23 using a T4 with Internet disabled. The hosted run completed successfully:
three studies in 30.11 seconds, 184.6 MB peak allocated CUDA memory, and the same
bundle hash (`cb571c7e...bc91cf`) as the local export. Its output had the expected
3-by-13 shape, study order, and columns; all probabilities were finite and within
[0,1]. The maximum absolute prediction difference from the saved local CUDA output
was 4.14e-6 (mean 8.14e-7), consistent with runtime floating-point variation.
The private assets are
[`kyunhsuan/rsna-knee-lora-submission-bundle`](https://www.kaggle.com/datasets/kyunhsuan/rsna-knee-lora-submission-bundle)
and
[`kyunhsuan/rsna-knee-lora-submission`](https://www.kaggle.com/code/kyunhsuan/rsna-knee-lora-submission).

## First leaderboard result

Kaggle code submission **56479080** used private notebook version 1 and completed on
2026-09-23. It received a **0.717 public macro ROC-AUC**, ranking **3,427 of 4,202**
teams in the leaderboard snapshot downloaded immediately after scoring. That places
775 teams behind it (18.4%); the snapshot median was 0.917 and its top score was
0.958. Kaggle exposes no private score while the competition is active.

The public score is 0.042 lower than the 56-study OOF macro AUROC of 0.759, but those
figures use different cohorts and should not be treated as a paired generalization
estimate. The live result confirms that the complete image preprocessing, LoRA image
encoder, fitted pathology heads, and submission schema execute successfully on the
hidden test set. It also establishes that this small gold-only head is not currently
competitive and is the baseline for subsequent leaderboard experiments.

## Build and benchmark

Run from the repository root using the existing GPU environment, which includes
scikit-learn. A fresh environment also needs `scikit-learn==1.9.1` for fitting.
The default uses pinned Hugging Face weights already in the local cache; it makes
no network requests. Full and compact checkpoints and LoRA adapters are supported.

```bash
.venv-cuda/bin/python scripts/build_kaggle_bundle.py \
  --config configs/kaggle/lora-holdout-v1.json \
  --output checkpoints/kaggle/lora-holdout-v1 \
  --device cuda \
  --image-cache-dir checkpoints/contrastive/full-multilingual-v1/image-cache
```

For the older three-seed ensemble, use `configs/kaggle/gold-holdout-v1.json` and a
different output directory. Existing output directories are refused. To benchmark
another checkpoint, copy a config, set its checkpoint list and population manifest,
and keep an explicit record of why it was chosen. Paths in configs are repo-relative;
absolute paths are also accepted. Preprocessing comes from each checkpoint.

The build produces:

- `benchmark.json`: per-label and macro OOF AUROC, average precision, unadapted
  DINOv2 comparison, per-checkpoint encoder exposure, source/input hashes,
  dependencies, wall time, and export/reload errors.
- `member-*-oof.npz` and `ensemble-oof.npz`: study IDs, targets, and held-out
  probabilities; member files also contain the per-label fold assignments.
- `bundle/`: complete image backbone weights, LoRA parameters when applicable,
  final heads fitted on all 56 studies, preprocessing settings, source snapshot,
  checksummed member manifest, inference script, and dependency requirements.
- `kaggle_submission.ipynb`: notebook for the Kaggle code submission.

The eight folds are stratified **separately for each label**, matching the existing
probe. Scaling is fitted only within each training fold. Ensemble probabilities
are averaged across members on identical folds. No hyperparameter tuning occurs.
Export/reload predictions are checked on an actual image before packaging succeeds.

The gold cohort has been examined repeatedly during development. OOF results are
useful development benchmarks, not a fresh external test or leaderboard score.
Patient identity cannot be established from the anonymized study IDs. The final
head uses all eligible gold labels after OOF evaluation; OOF scores describe the
fold-trained heads, not an evaluation of the all-gold final head.

## Validate locally

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-cuda/bin/python checkpoints/kaggle/lora-holdout-v1/bundle/predict_kaggle.py \
  --bundle checkpoints/kaggle/lora-holdout-v1/bundle \
  --data-root /mnt/d \
  --output checkpoints/kaggle/lora-holdout-v1/submission.csv \
  --device cuda
```

Replace `/mnt/d` with the competition directory on another machine. Inference needs
only `test.csv`, `test_series.csv`, `test_series/`, and `sample_submission.csv`.
It never reads reports, training labels, or the local training manifests.
The original series selection, crop, normalization, slice sampling, and laterality
code are included in the bundle. There is no Hugging Face cache dependency after export.

The CSV must have exactly the sample's study order and twelve label columns, unique
IDs, and finite [0,1] probabilities. A missing plane raises an explicit error. A
selected plane that does not decode falls back to the next candidate series (logged),
and the run fails only if no series of that plane decodes; there is no silent row
dropping or constant-prediction fallback. Decode counters per transfer syntax are
printed at the end and stored in the benchmark JSON.

DICOM codecs on Kaggle (vendored wheels, per-syntax Pillow pin, self-test, counters,
`KNEE_STRICT_DECODE`, series fallback): see `kaggle/codecs/README.md`.
The accompanying `submission.benchmark.json` records end-to-end runtime, GPU peak
allocation, study count, and artifact hashes. Runtime on three public example
studies is a smoke benchmark, not an estimate for Kaggle's hidden test runtime.

## Run on Kaggle

1. Upload **only `bundle/`** as a private Kaggle dataset. Do not upload the parent
   benchmark directory, which contains gold targets and study IDs.
2. Import `notebooks/kaggle_submission.ipynb`; attach that private dataset and the
   `rsna-knee-abnormality-detection` competition data. Attach only one bundle.
3. Use a GPU and disable Internet. Ensure the runtime satisfies
   `requirements-kaggle.txt`. For an incompatible Kaggle image, download compatible
   Linux/Python wheels ahead of time, attach them as a private dataset, and install
   using `pip install --no-index --find-links <wheel-directory> -r <bundle>/requirements.txt`.
   The bundle does include one set of wheels: the DICOM codecs in `codecs/` (JPEG
   Lossless, JPEG-LS), which the notebook installs and self-tests before prediction.
   The local environment is Python 3.13; do not copy its native extensions into a
   different Kaggle Python version.
4. Run and save a notebook version. It produces `/kaggle/working/submission.csv` and
   `submission.benchmark.json`. The private version above validates this path. A
   future leaderboard submission should use that version's output and record the
   Kaggle version, score, and runtime alongside the local benchmark.

This is a code competition: the three-row local example CSV is only a smoke output;
the notebook must regenerate predictions when Kaggle supplies the hidden test data.
The private dataset and notebook have been uploaded and tested. No leaderboard
submission should be made without an explicit experiment record; submission
56479080 is the recorded first baseline.

The official [evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
specifies macro ROC-AUC across twelve findings and the study-level CSV schema.
The [data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
describes the public example test inputs. The code-requirements page did not return
readable requirements during the original documentation check; the private hosted
smoke run verifies the current bundle against Kaggle's notebook environment.
