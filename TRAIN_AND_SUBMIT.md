# Train a variant and submit it

Two recipes, written for a person at a terminal: no agent, chain script or session
tooling needed. Commands run from the repository root. `python` means the GPU
environment (`.venv-cuda/bin/python`, see [Setup](README.md#setup)) unless a
step says CPU.

The current best single model is **A30**:
`configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30.json`
(LM-test 0.8917, gold-58 0.8814, public 0.890). Start variants from it or from its
base, `r4a-k8-omnirad-base-soft-336-u9-b1.json` (public 0.888).

---

## 0. One-time setup

1. **Machine paths.** `cp configs/paths.template.json paths.json` and set:
   - `data_root`: the competition data directory, holding `train_series.csv` and
     `train_series/<study>/<series>/*.dcm`.
   - `train_series_mirror` (optional): a local byte-identical copy of
     `train_series/`. Fine-tuning re-decodes DICOMs every epoch, so put it on a fast
     disk if `data_root` is a network or `/mnt` mount.
   `paths.json` is git-ignored. Configs never contain host paths.
2. **Inputs not in git** (ask the team):
   - `data/lm_label/lm_labels_kaggle.csv`: third-party LM labels. **Never commit it**,
     or any per-study excerpt of it.
   - `data/processed/contrastive-cohort-gold-holdout-v1.jsonl`: the frozen cohort
     manifest. Every config pins its SHA-256 (`manifest_sha256`).
3. **Pinned model weights.** Each config names `image_model` / `image_model_base`
   with a `*_revision`, and loading is offline. Download those exact revisions once:
   ```bash
   huggingface-cli download Snarcy/OmniRad-base --revision 3e42381bcf0e6981a8b438e335c807c70b857e9a
   huggingface-cli download facebook/dinov2-base --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415
   ```
4. Check the environment: `.venv/bin/python -m pytest -q` (CPU, about 10 s).

---

## 1. Train a variant

### 1.1 Write the config

Copy the closest existing config and change only what the variant tests. Keep the
data contract (`cohort_manifest`, `manifest_sha256`, `lm_labels_csv`,
`gold_labels_csv`, `gold_splits`, `expected_gold_studies`) untouched, or the result
is not comparable with anything recorded.

```bash
cp configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30.json \
   configs/finetune/<new-name>.json
# edit "name" to <new-name>, then the keys below
```

| To vary | Keys | Notes |
|---|---|---|
| Seed (replicate) | `seed` | Do this before believing any gain under about 0.024. |
| Unfrozen depth | `image_unfreeze_last` | u9 = 9 blocks. |
| Resolution | `image_size` | Window caches are keyed by settings, so a new size rebuilds them (slow first epoch). |
| Training length | `epochs`, `patience`, `divergence_stop` | A30 uses 30 epochs with patience off and a divergence stop. |
| Batch | `batch_size` × `grad_accumulation` | Effective batch 16 everywhere; "b1" means `batch_size` 1. |
| Weak-label rebalancing | `oversample`, `label_noise`, `label_weights` | See the A30 config for the shape of each block. |
| Image augmentation | `image_augmentation` | Object with `rotation_degrees`, `zoom`, `shift`, `intensity`; `{}` takes the defaults. A null result on A30 (`digest/finetune-image-augmentation-v1.md`). |
| Targets | `targets` (`soft` / `verdict`), `unk_weight` | |
| Backbone | `image_model`, `image_model_revision`, `image_model_format`, `image_model_base*` | Copy these together from a config that already uses the backbone. |
| Warm start | `init_checkpoint`, or `resume_from` + `save_last` on the parent | `resume_from` continues a parent run from its `last.pt`. The parent must have `save_last: true`. |

`validate_config` in `src/contrastive_learning/finetune.py` rejects unknown values
before any training starts. Commit the new config: it is the record of the variant.

### 1.2 Train

```bash
python scripts/training/finetune_supervised.py \
  --config configs/finetune/<new-name>.json \
  --output checkpoints/finetune/<new-name>
```

A run takes hours (A30: about 26 min per epoch on one GPU). The output directory
must not exist yet. It ends with:

- `config.json`: the config as run, including the resolved mirror path.
- `history.jsonl`: per-epoch loss and validation metrics.
- `best.pt`, plus `last.pt` when `save_last` is set.
- `predictions.npz`: validation, LM-test and gold-58 predictions.
- `report.json`: `lm_test.macro_auroc`, `gold.macro_auroc`, per-label scores and
  `best_epoch`.

### 1.3 Judge it

Read the two headline numbers:

```bash
python -c "import json; r = json.load(open('checkpoints/finetune/<new-name>/report.json')); print(r['lm_test']['macro_auroc'], r['gold']['macro_auroc'], r['best_epoch'])"
```

Compare against the base with **paired** bootstrap intervals over the same LM-test
studies. `scripts/analysis/compare_lateral_rebalance.py` is the current template: add the
run to its `RUNS` registry and the comparison to `PAIRS`, then run:

```bash
python scripts/analysis/compare_lateral_rebalance.py --output digest/<study>-v1.json
```

Decision rule (see [Labels and Validation](README.md#labels-and-validation) in the README):

- **Adopt** only if the LM-test interval against the base excludes zero.
- **Gold-58 is a veto only**: −0.024 or worse rejects. It never selects on its own.
- A single seed cannot resolve a difference under about 0.024. Replicate first.

Record the result as `digest/<study>-v1.md` with its JSON. Add both to
`src/digest_site/catalog.py` (the site build fails on uncatalogued files), then
run `python scripts/digests/build_digest_site.py`.

---

## 2. Submit to Kaggle

The competition is a code competition. You upload a model bundle as a private
dataset, push a notebook that predicts offline, and submit the notebook's output.

### 2.1 Export the bundle

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/submission/export_finetune_bundle.py \
  --run checkpoints/finetune/<new-name> \
  --output checkpoints/kaggle/<bundle-name>
```

- Each `--run` is rebuilt from its `config.json` + `best.pt` and checked against
  its own saved gold predictions before packaging.
- Pass several `--run` flags for an ensemble (per-label rank average).
- To combine bundles already exported on their own, use
  `scripts/submission/assemble_ensemble_bundle.py --member <bundle dir> ...`.
- The export also scores the bundle on LM-test unless `--skip-lm-test` is given.
- Add `--device cpu` if the GPU is busy.

The output holds `bundle/` (weights, inference source, vendored DICOM codecs,
`bundle.json`) and `kaggle_submission.ipynb`.

### 2.2 Check it locally

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python checkpoints/kaggle/<bundle-name>/bundle/predict_kaggle.py \
  --bundle checkpoints/kaggle/<bundle-name>/bundle \
  --data-root <competition data dir> \
  --output checkpoints/kaggle/<bundle-name>/submission.csv --device cuda
```

The CSV must match `sample_submission.csv`: same study order, twelve label
columns, and probabilities in [0, 1].

### 2.3 Upload and run on Kaggle

You need the `kaggle` CLI with an API token, and you must have accepted the
competition rules.

1. **Dataset.** Write `checkpoints/kaggle/<bundle-name>/bundle/dataset-metadata.json`.
   Copy one from an earlier bundle, e.g. A30's, and change `id` and `title`.
   Upload **only `bundle/`**, never its parent: the parent holds gold targets and
   study IDs.
   ```bash
   kaggle datasets create -p checkpoints/kaggle/<bundle-name>/bundle --dir-mode zip
   ```
   Keep the dataset private.
2. **Notebook.** Put `kaggle_submission.ipynb` and a `kernel-metadata.json` in
   `checkpoints/kaggle/<bundle-name>/kernel/`. Copy A30's metadata and change `id`,
   `title` and `dataset_sources`. Keep these fields:
   ```json
   "enable_gpu": "true", "enable_internet": "false", "is_private": "true",
   "competition_sources": ["rsna-knee-abnormality-detection"],
   "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
   "docker_image_pinning_type": "original"
   ```
   **Pin the image.** Kaggle's latest image is Python 3.13, and the bundle's vendored
   codec wheels are built for 3.12. On an unpinned kernel the codec self-test fails
   and nothing is predicted.
   ```bash
   kaggle kernels push -p checkpoints/kaggle/<bundle-name>/kernel
   kaggle kernels status <user>/<kernel-slug>
   ```
   The notebook installs the codecs, runs a decode self-test, and writes
   `submission.csv`. It stops rather than predict with a missing decoder.
3. **Submit** the finished kernel version's output:
   ```bash
   kaggle competitions submit rsna-knee-abnormality-detection \
     -k <user>/<kernel-slug> -v <kernel version> -f submission.csv -m "<bundle-name>"
   ```

### 2.4 Record it

Submissions are limited, so each one is an experiment. Write
`checkpoints/kaggle/<bundle-name>/leaderboard-submission.json` with:

- the run and why it was submitted;
- the predicted score (`prediction_if_LM_test_tracks`) and the baseline;
- `submission_ref`, `kernel`, `kernel_version`, `dataset` and `bundle_sha256`;
- `public_score` and `recorded_at_utc` once scored.

A30's file is the template. Add a title for the new bundle to `TITLES` in
`scripts/digests/build_leaderboard_digest.py`, then regenerate the leaderboard history and
the site:

```bash
python scripts/digests/build_leaderboard_digest.py      # rewrites digest/leaderboard-history-v1.json
python scripts/digests/build_digest_site.py
```

Commit the updated `digest/leaderboard-history-v1.json` together with the title.

The builder reads every `checkpoints/kaggle/*/leaderboard-submission.json`. Those
files are git-ignored, so run it on the machine that holds all recorded
submissions. Elsewhere it refuses to drop rows rather than shrink the history.

---

## Where things go wrong

| Symptom | Cause |
|---|---|
| `Series missing from the local mirror` | `train_series_mirror` in `paths.json` points at an incomplete copy. Fix the copy, or remove the key to read from `data_root`. |
| `Unknown path setting(s) in paths.json` | A misspelled key, or a checkout older than `fix/host-paths` that doesn't know `train_series_mirror`. |
| Manifest SHA mismatch | The cohort file differs from the one the config pins. Restore the team copy; don't edit `manifest_sha256`. |
| Model not found offline | The pinned revision isn't in the Hugging Face cache (setup step 3). |
| Kaggle notebook stops at the codec self-test | The kernel isn't pinned to the Python 3.12 image (§2.3). |
| `Output already exists` | Training and export refuse to overwrite. Choose a new name. |

Background: [Kaggle bundle format and history](digest/kaggle-submission-pipeline.md),
[DICOM codecs on Kaggle](kaggle/codecs/README.md),
[pipeline code guide](src/contrastive_learning/README.md).
