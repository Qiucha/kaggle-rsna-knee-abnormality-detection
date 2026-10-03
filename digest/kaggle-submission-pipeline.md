# Kaggle submission and benchmark pipeline

Checked against the working tree and local artifacts on 2026-09-25. The root README's
original multilingual matrix is historical context; newer downstream experiments
include both-tower adaptation, LoRA, DeCLIP, label-similarity targets, and gold holdouts.
Existing user changes were retained.

In this project, a competition benchmark measures the trained MRI image pipeline:
image encoder, fitted pathology heads, macro ROC-AUC, and submission inference.
Kaggle Benchmarks and its Model Proxy are a separate product for evaluating external
LLMs; use that workflow only when an LLM evaluation is explicitly requested.

## Current model choice

The default submission is the **DINOv2-small + E5-base LM-MLP bundle**. It scored
**0.788** public macro ROC-AUC in submission 56526493, the best score so far. Its
local assets are `checkpoints/kaggle/lm-mlp-tower-size-58-dinov2-small-e5-base/`,
built from `configs/kaggle/lm-mlp-tower-size-58-dinov2-small-e5-base.json`. The
Kaggle assets are `kyunhsuan/rsna-knee-lm-mlp-dinov2s-e5base-{bundle,submission}`,
with notebook version 1.

- **Encoder:** frozen **holdout-trained six-block, rank-64 LoRA encoder** at
  `checkpoints/contrastive/both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026/best.pt`.
  This is the same checkpoint as the first submission. It excludes the gold cohort
  from encoder training and selection.
- **Head:** three seeded 256-unit MLPs averaged by probability. They were trained on
  `data/lm_label/lm_labels_kaggle.csv` `YES`/`NO` verdicts, with `UNK` masked, and
  selected by minimum validation BCE. Gold labels were never used for fitting or
  epoch selection. The local score is 0.7659 AUROC on the 58 gold cases.

Changing only the head, from gold logistic to LM-verdict MLP, raised the public score
from 0.717 to 0.788. The DINOv2-small + E5-small LM-MLP bundle scored 0.774.
Among local metrics, only LM-test AUROC ranks all three submissions correctly. The
leaderboard section below gives the evidence and its label-bias limitations. Neither
gold AUROC nor I→T R@1 should be used alone to select a bundle.

### Previous default (first submission, 0.717 public)

The first submitted bundle used the same LoRA encoder with twelve standardized
logistic heads fitted on the 56 eligible gold-labeled studies in `data/cleaned.csv`.
`digest/downstream-probe-lora-holdout-v1.json` reports **0.75936** OOF macro AUROC for
those features; the regularized LoRA variant scores 0.71261. The older three-seed
four-block holdout checkpoints remain an explicit comparison. Higher exposed-cohort
results in other digests are not directly comparable. The two additional gold studies
were not used by that bundle but were later scored in the 58-case LM-head experiment
and are no longer untouched. Blank competition training labels and fractional LM
label confidences are rejected as supervised targets.

## Verified local results

The pipeline was executed on the RTX 3070 using the existing CUDA environment:

| Candidate | 56-study OOF macro AUROC | Public example inference | Bundle size |
| --- | ---: | ---: | ---: |
| Unadapted DINOv2 | 0.643992 | benchmark reference only | — |
| Four-block holdout, three-seed probability ensemble | 0.696292 | 3 studies / 18.64 s | 265.6 MB |
| **Six-block LoRA holdout, first submission** | **0.759362** | **3 studies / 7.17 s** | **99.5 MB** |

LoRA peak allocated GPU memory was 183.6 MB for the three-study inference call.
These timings exclude Python import/startup and are hardware- and cache-dependent.
The maximum export/reload probability difference was 2.94e-7 for LoRA.
All 56 benchmark cases were confirmed `excluded` in each encoder's own cohort.

## New local LM-head comparison on all 58 gold cases

Four frozen, gold-holdout contrastive image encoders were each fitted with the same
three-seed 12-label MLP head using `data/lm_label/lm_labels_kaggle.csv` (`YES`/`NO`
targets; `UNK` masked). The head training and validation splits contain 3,438 and
426 studies. All 58 gold studies were excluded from both, including the two from
the former test split. Each AUROC below was independently recomputed from the
saved 58-study predictions.

| Image tower | Text tower | LM-linear 58-case AUROC | LM-MLP 58-case AUROC |
| --- | --- | ---: | ---: |
| DINOv2-small | E5-base | 0.7336 | **0.7659** |
| DINOv2-small | E5-small | 0.7312 | 0.7541 |
| DINOv2-base | E5-base | 0.7330 | 0.7504 |
| DINOv2-base | E5-small | 0.7362 | 0.7605 |

The linear heads show much smaller late validation-loss rebound, but the
validation-selected MLP ensembles score higher on gold in every pairing. The
MLP-minus-linear paired 95% intervals exclude zero for small-image/E5-base and
base-image/E5-small, and include zero for the other two pairings. This supports
early stopping of the MLP; it does not justify using its overtrained final epoch.

The 58-case LM-head scores and the earlier 56-case gold-only OOF scores use
different training labels and evaluation populations. They are not paired
estimates of an improvement over the existing Kaggle submission. Every paired
size contrast within the 58-case experiment has a 95% study-bootstrap interval
spanning zero. The full protocol, loss curves, per-label results, and links to
the four local bundles are in the [LM-head report](lm-head-tower-size-58-v1.md)
and [HTML digest](lm-head-tower-size-58-v1.html). The two DINOv2-small bundles were
submitted on 2026-09-24; see the leaderboard section below.

Local artifacts are under `checkpoints/kaggle/{lora-holdout-v1,gold-holdout-v1}/`:
`benchmark.json`, `bundle/`, `submission.csv`, and `submission.benchmark.json`.
Both CSVs passed schema, ID-order, completeness, and probability validation with
Hugging Face offline mode enabled. Tests: **139 passed, 1 skipped**, including seven
new deployment tests. The seven deployment tests were also rerun after formatting;
Ruff and notebook code syntax checks passed.

The first-submission bundle was also validated in a private Kaggle notebook version on
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

## LM-head leaderboard results: which local criterion tracks the public score?

Two DINOv2-small LM-MLP bundles were submitted on 2026-09-24 to test whether
contrastive image-to-text R@1 or 58-case gold macro AUROC better predicts the public
leaderboard. Before submission, the expected outcomes were recorded in each run's
`leaderboard-submission.json`: a higher E5-small score would favor R@1, and a
higher E5-base score would favor gold AUROC. Each bundle was uploaded to its own
private dataset. Each private T4 notebook, version 1, reproduced the local bundle
hash, and its predictions were within 3.5e-6 of local CUDA output.

| Submission | Towers | I→T R@1 | 58-case gold AUROC | Public macro ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| 56479080 (baseline) | DINOv2-small + E5-base, gold logistic head | 13.97% | 0.7594 (56 OOF) | 0.717 |
| **56526493** | DINOv2-small + E5-base, LM-MLP head | 13.97% | **0.7659** | **0.788** |
| 56526496 | DINOv2-small + E5-small, LM-MLP head | **14.55%** | 0.7541 | 0.774 |

**Neither local metric ranks all three submissions correctly.**

| Pair (public gap) | Local gold AUROC | I→T R@1 |
| --- | --- | --- |
| E5-base LM vs E5-small LM (+0.014) | agrees | disagrees |
| E5-base LM vs first submission (+0.071) | agrees | tie (same encoder) |
| E5-small LM vs first submission (+0.057) | **disagrees** (0.7541 < 0.7594) | agrees |

- **Head effect:** both large gaps separate an LM-verdict MLP head from the gold-only
  logistic head. The head type is the one factor that matches the public ordering.
- **I→T R@1:** it cannot explain a gap between two bundles with the same encoder.
- **Local AUROC:** the first-submission AUROC is a 56-case OOF score for heads fitted
  on 56 gold labels. The LM-head scores use 58 held-out cases and heads trained on
  3,438 studies. These protocols are not comparable, and the 56-case OOF score
  overstated that small head's public performance.
- **Within one head protocol:** the only comparison is E5-base versus E5-small. The
  ordering agrees with gold AUROC, but the R@1 gap is below the seed noise floor, and
  the gold-AUROC paired 95% interval includes zero. One pair does not establish
  either metric as a leaderboard proxy.

### Standard local report: LM-test AUROC

**LM-test AUROC** is a standard local performance report for every exported bundle,
alongside gold AUROC. It is the masked macro AUROC against LM `YES`/`NO` verdicts on
the cohort's **432 non-gold locked-test studies**.

- **Clean population:** no contrastive checkpoint selection, LM-head fitting or
  epoch selection, or gold-head fitting used these studies.
- **Exclusions:** gold studies and every study sharing a gold patient group are
  removed. `UNK` cells are ignored.
- **Implementation:** `src/contrastive_learning/lm_proxy.py`. Features come from each
  bundle's frozen encoder and are cached by checkpoint checksum under
  `checkpoints/kaggle/lm-proxy-cache/`. Scoring uses the deployed `predict_head`.
- **Output:** every run writes `<run>/lm-proxy.json`. `scripts/train_lm_head.py` and
  `scripts/build_kaggle_bundle.py` do this automatically.
- **Comparison:** `scripts/eval_lm_proxy_heads.py <run dirs> --output <json>` adds
  paired study-bootstrap intervals and pairwise agreement with recorded public scores.
- **Digest:** the LM-head digest (`lm-head-tower-size-58-v1.md`) shows it next to gold
  AUROC.

| Run | Gold AUROC (local) | LM-test AUROC (432) | Public |
| --- | ---: | ---: | ---: |
| DINOv2-small + E5-base, LM-MLP (56526493) | 0.7659 | 0.8336 | **0.788** |
| DINOv2-small + E5-small, LM-MLP (56526496) | 0.7541 | 0.8275 | 0.774 |
| First submission, gold logistic (56479080) | 0.7594 (56 OOF) | 0.6844 | 0.717 |
| DINOv2-base + E5-base, LM-MLP | 0.7504 | 0.8324 | — |
| DINOv2-base + E5-small, LM-MLP | 0.7605 | **0.8401** | — |

**LM-test AUROC orders all 3 public-score pairs correctly.** Gold AUROC orders 2 of 3,
and I→T R@1 orders 1 of 3. Paired 95% intervals are in `digest/lm-proxy-eval-v2.json`:

- **DINOv2-base + E5-small vs DINOv2-small + E5-small:** +0.0126 [+0.0028, +0.0223].
  This is the only encoder contrast whose interval excludes zero.
- **DINOv2-base + E5-small vs DINOv2-small + E5-base (current best public):** +0.0065
  [-0.0026, +0.0163].
- **Other LM-head pairs:** all intervals include zero.
- **First submission vs LM heads:** about -0.15, with intervals near [-0.17, -0.12].

Unsubmitted DINOv2-base + E5-small has the highest LM-test score. It is the best
candidate for the next leaderboard check of this metric.

Limitations:

- **Label bias:** heads trained on LM verdicts are scored against those same verdicts.
  The metric therefore favors LM-trained heads over heads trained with other labels.
  It overstated the gold-logistic → LM-MLP public gap by about 2×.
- **Best use:** comparing encoders under one LM-head recipe.
- **Evidence:** only 3 public scores have been checked. The only encoder-only public
  pair differs by 0.006 LM-test AUROC, and its interval includes zero.

Record LM-test AUROC with every future submission to continue testing its agreement.
`digest/lm-proxy-eval-v1.json` also contains the first three runs on the
426-study LM validation split. LM heads selected epochs on that split, so it is a
secondary reference only.

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
