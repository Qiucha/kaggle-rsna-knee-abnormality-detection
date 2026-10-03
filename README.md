# Knee MRI contrastive experiments

This project learns shared representations of knee MRI studies and their radiology
reports, so that a study retrieves its own report and vice versa. Retrieval is the
intermediate check; the working goal in [ROADMAP.md](ROADMAP.md) is whether those
representations improve prediction of the dataset's pathology labels.

Open the [interactive project digest](digest/project-state.html) for current results,
the model architecture, and data limitations. It works offline; regenerate its
artifact snapshot with `.venv/bin/python scripts/build_project_digest.py`.

The current experiment contract is
[full-multilingual-v1](configs/contrastive/full-multilingual-matrix.json); its
selection rules and stated limitations are in the
[frozen protocol](configs/contrastive/full-multilingual-v1-protocol.md).
The [pipeline code guide](src/contrastive_learning/README.md) explains the modules
and what each one hides. Older translated-English pilot notes in `digest/` describe
historical experiments and are not current instructions.

For offline Kaggle inference, supervised label-head fitting, and current checkpoint
selection, see the [submission and benchmark guide](digest/kaggle-submission-pipeline.md).

## Start here: the current best pipeline (2026-10-03)

The leaderboard models are no longer contrastive encoders with a fitted head. They
are **image backbones fine-tuned end to end on LM-extracted labels** (the "R4a"
rung: K=8 windows of 3 slices per plane, mean-pooled). The contrastive code
remains as the source of the initial checkpoint for one member, and as a record of
the experiments that led here. See [Experiment tracks](#experiment-tracks) for how
the two lines of work relate.

| What | Where |
| --- | --- |
| Best public score, 0.879 (run 5, submission 56789799) | `configs/finetune/r4a-k8-omnirad-base-soft-336.json` (OmniRad-base, soft targets, 336 px) |
| Previous best, 0.839 | `configs/finetune/r4a-k8-omnirad-base-pretrained.json` (OmniRad-base, hard verdict targets, 224 px) |
| Three-member candidate, not yet submitted | soft-target R4a (`r4a-soft-224-k8-mean.json`) + OmniRad-base + OmniRad-base soft 336 (`r4a-k8-omnirad-base-soft-336.json`), combined by per-label rank mean |
| Train one member | `.venv-cuda/bin/python scripts/finetune_supervised.py --config configs/finetune/<rung>.json --output checkpoints/finetune/<rung>` |
| Export members as one Kaggle bundle | `scripts/export_finetune_bundle.py --run <run dir> [--run ...] --output checkpoints/kaggle/<name>` |
| Combine already-exported bundles | `scripts/assemble_ensemble_bundle.py` |
| Run a bundle as Kaggle does | `scripts/predict_kaggle.py` and `notebooks/kaggle_submission.ipynb`; DICOM codecs are vendored in `kaggle/codecs/` |
| Model code | `src/contrastive_learning/finetune.py` (training), `submission.py` (inference, ensembling), `timm_dinov2.py` / `timm_backbone.py` (OmniRad, RadioDino, CoAtNet) |
| Results and decisions | `digest/` (open `digest/index.html`): `finetune-ladder-v1.json` for the R-rungs; `kaggle-submission-pipeline.md` covers the bundle format but its leaderboard table stops at the 0.788 LM-MLP bundle (R4a DINOv2-small later scored 0.826, OmniRad-base 0.839) |

How candidates are judged: by the LM-test interval excluding zero. The 58 gold
cases have been reused for prompt tuning, encoder selection and head fitting, so
gold-58 is only a veto (−0.024 or worse) and a tiebreaker, never a selector on its
own. Single-seed differences under about 0.024 are within run-to-run noise.

**Not in git, needed to train** (ask the team for copies):

- `data/lm_label/lm_labels_kaggle.csv`: third-party LM labels, never commit it.
- `data/processed/contrastive-cohort-gold-holdout-v1.jsonl`: the frozen cohort
  manifest. Configs pin its SHA-256, and it has no committed builder yet.
- The DICOM tree (set `data_root` in `paths.json`, see
  `configs/paths.template.json`) and, for the soft R4a member, the contrastive
  init checkpoint named in its config's `init_checkpoint`.
- `train_series_mirror.to` in each fine-tuning config points at one machine's
  local copy of `train_series`. Edit it, or delete the key, to read from `data_root`.

Image caches are keyed by `PREPROCESSING_VERSION`, the image settings and each
series' `<study>/<series>` identity. Commit `084d086` (2026-09-29, exact-extent
crop) changed the pixels of small field-of-view series without bumping
`PREPROCESSING_VERSION`, so a cache written before it can serve the older pixels
under the same key. The R4a window caches were all written after it. Rebuild an
older image cache rather than migrating it with `scripts/migrate_cache_keys.py`.

For the next image and text tower runs, read the prioritized
[model candidate and comparison note](digest/next-model-candidates.md) before deploying an experiment.

## Experiment tracks

Most experiments since 2026-09-29 do **not** train with the contrastive pipeline.
They fine-tune an image backbone directly on LM-extracted labels. From here the
work is split into two tracks, which share one data and evaluation core: the
cohort manifest and its splits, DICOM preprocessing, the K-window input path,
LM-test, and the gold-58 guard.

**Track 1: supervised label training.** This track asks what kind of supervised
label training to do: label form (hard verdicts or soft targets), UNK handling,
resolution, slice sampling, multi-slice pooling and backbone. It has reached a
stage we are satisfied with. The reference recipe is run 5,
`r4a-k8-omnirad-base-soft-336`, at public 0.879. Its ladder and decisions are in
`digest/finetune-ladder-v1.json`. Runs still queued against run 5 (a second seed,
more epochs, deeper unfreezing, 384 px) will be reviewed before the next
direction is set.

**Track 2: report-based representation learning.** The LM labels compress each
report into twelve YES/NO/UNK verdicts, and weak supervision on those verdicts is
limited by the size of the labelled cohort (about 3,400 training studies). Track 2
asks whether image–report contrastive learning, which uses the full report text,
adds to the Track-1 recipe when the two are combined. The combinations to test
are:

- (a) contrastive pretraining, then Track-1 fine-tuning from that checkpoint;
- (b) a contrastive auxiliary loss trained jointly with the label loss.

**What the earlier combined runs do and do not show.** The DINOv2-small R4a rungs
(`r1c-*`, `r4a-hard/soft-224-k8-*`, `r4b-*`, among them the soft R4a ensemble
member) start from the contrastive checkpoint `both-lora-holdout-v1`. That
checkpoint was trained on a single 3-slice stack per plane, while R4a fine-tunes
on K=8 jittered windows per plane. Pretraining and fine-tuning therefore saw
different inputs. The matched-init comparisons below (configs differ only in
`init_checkpoint`; paired study-bootstrap 95% intervals) are confounded by that
mismatch. They should not be read as the size of the contrastive benefit:

| Recipe (DINOv2-small, 224 px, hard targets) | LM-test Δ, contrastive − pretrained init | Gold-58 Δ |
| --- | ---: | ---: |
| R1, one 3-slice stack per plane (input matches pretraining) | +0.014 [+0.002, +0.027] | +0.010 [−0.020, +0.039] |
| R4a K=8, seed 2026 (input does not match) | +0.007 [−0.003, +0.017] | −0.010 [−0.041, +0.020] |
| R4a K=8, seed 2027 (input does not match) | +0.005 [−0.005, +0.015] | −0.011 [−0.038, +0.017] |

The first matched test of (a) is
`r4a-k8-omnirad-base-soft-336-from-clip`, which is in progress. It pretrains
OmniRad-base + E5-base contrastively on the run-5 input path (K=8, 336 px), then
fine-tunes with the run-5 recipe. Its comparator is run 5 together with run 5's
second seed, which is queued directly after it.

Open questions for Track 2:

- **Measurement.** LM-test scores agreement with the LM verdicts, so it cannot
  credit information beyond them. Gold-58 can, but it is small and has been reused.
- **Checkpoint selection.** Contrastive checkpoints are still selected by
  validation retrieval (mean R@1). That may not be the checkpoint that fine-tunes
  best.

---

## Contents

- [What the code is](#what-the-code-is) — modules and scripts
- [Setup](#setup) — CPU and GPU environments, restoring private artifacts
- [Deploy a single training run](#deploy-a-single-training-run)
- [Deploy the full experiment grid](#deploy-the-full-experiment-grid)
- [Evaluating and reporting](#evaluating-and-reporting)
- [Reproducibility rules](#reproducibility-rules)

---

## What the code is

### Library — `src/contrastive_learning/`

One package holds all pipeline logic; scripts are thin wrappers around it. Full
detail is in the [pipeline code guide](src/contrastive_learning/README.md).

| Module | Responsibility |
| --- | --- |
| `config.py` | `TrainConfig`: every setting of a run, its validation, saved-config loading |
| `cli.py` | Training flags and their translation into a configuration |
| `training.py` | Run lifecycle: optimizer groups, schedule, epoch loop, resume, checkpointing |
| `inputs.py` | Cohort loading, split policy, and loaders shared by training and evaluation |
| `data.py` | Study/report pairing, dataset, image caches, batch collation |
| `manifest.py` | Building and reading frozen, study-grouped cohorts (the grouping key is one study per group — see the [integrity audit](digest/integrity-coverage-audit.md)) |
| `preprocessing.py` | DICOM ordering, cropping, normalization, orientation, augmentation |
| `text.py` | Report views: full, impression, findings+impression, impression-first |
| `model.py` | Image and text towers, plane attention pooling, contrastive loss |
| `evaluation.py` | Tie-aware retrieval metrics and checkpoint evaluation reports |
| `checkpoints.py` | Checkpoint format, best-epoch selection, run provenance, resume |
| `experiments.py` | Matrix execution, immutable validation selection, checkpoint rescoring |
| `artifacts.py` | Atomic artifact writes and SHA-256 content hashes |

### Scripts — `scripts/`

Every script is run from the repository root, so relative paths in matrices and
configurations resolve consistently.

| Script | What it does |
| --- | --- |
| `build_contrastive_manifest.py` | Freeze a cohort: pair reports with DICOM series, resolve the text policy, assign study-grouped train/validation/test splits, and write the manifest with checksums. Run once per cohort version. |
| `train_contrastive.py` | Train one run. The single training entry point; every flag is listed by `--help`. |
| `run_measured_training.py` | Wrap `train_contrastive.py` unchanged and additionally record elapsed time and peak CUDA memory to a telemetry file. Requires CUDA. |
| `run_contrastive_matrix.py` | Run a frozen experiment matrix, or (`--dry-run`) print the exact command for each experiment without executing it. |
| `finalize_contrastive_matrix.py` | Verify every run's best epoch against its own history, write the immutable `validation-selection.json`, and optionally spend the locked test split. |
| `evaluate_contrastive.py` | Score one saved checkpoint on a split; write its metrics report and per-query rankings. |
| `rescore_contrastive_checkpoints.py` | Re-evaluate every saved `best.pt` under current metric definitions and summarize which checkpoints actually moved. |
| `build_project_digest.py` | Rebuild `digest/project-state.html` from local experiment artifacts. Takes no arguments. |
| `evaluate_downstream_probe.py` | Score representations on the downstream label-extraction task before and after contrastive learning: cross-validated per-label logistic probes over the gold cohort, pooled out-of-fold AUROC, paired bootstrap intervals. Read-only; never loads the locked test split. |
| `analyze_plane_pooling.py` | Decompose why the image tower's learned plane attention reads labels worse than averaging the same three planes: per-plane probes, the attention weights, and the pooling with and without its plane-embedding term. Reuses the downstream probe's contract by import, not by copy. |
| `translate_reports_claude.py` | Prepare, run, and merge report translations used by the `translated_english` cohorts. Only `run --execute` calls out to Claude; everything else is offline. |
| `templates/` | HTML template used by `build_project_digest.py`. |

One-off audit scripts for completed experiments have been removed; their outputs
remain in `digest/` (for example `digest/full-projection-v1-data-audit.json`,
`digest/english-medical-text-results.json`, `digest/full-manifest-migration.json`).

### Tests

```bash
.venv/bin/python -m pytest -q
```

Covers report views, the image cache, cohort manifest invariants, retrieval metrics
and tie handling, checkpoint selection and the run lifecycle, image tower shapes and
gradients, experiment matrix handling, and the translation runner. Tests requiring
`pydicom` skip when it is unavailable.

---

## Setup

### CPU environment

Use Python 3.13.12 and install `requirements-lock.txt` in a virtual environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
.venv/bin/python -m pytest -q
```

### GPU environment

The isolated GPU environment is `.venv-cuda`, using the same PyTorch 2.13.0 and
torchvision 0.28.0 releases with CUDA 12.6 wheels. The supported pairing is listed in
[PyTorch's version instructions](https://pytorch.org/get-started/previous-versions/).
Restore it from the separately captured `requirements-cuda-lock.txt`:

```bash
python3.13 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-cuda-lock.txt
.venv-cuda/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

Use `.venv-cuda/bin/python` for every GPU command below. On this WSL device CUDA is
exposed through `/dev/dxg`; invoke `/usr/lib/wsl/lib/nvidia-smi` when it is absent
from PATH. GPU commands must run with host device access — the agent sandbox can
hide `/dev/dxg`. The installed Windows NVIDIA driver already supplies CUDA support
to WSL; see [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

### Machine directories

No directory that differs between machines is written into the code. One JSON
file assigns them:

```bash
cp configs/paths.template.json paths.json   # then edit data_root
```

`paths.json` is git-ignored and read from the repository root; set
`CONTRASTIVE_PATHS_CONFIG` or pass `--paths-config` to keep it elsewhere. It may
assign `data_root` (the DICOM mount, required by anything that reads DICOM data),
`reports_csv`, `cohort_manifest`, `output_dir`, and `image_cache_dir`. Relative
values resolve against the file's own directory, and a command-line flag always
overrides the file, so a frozen experiment matrix runs unchanged on a new host.
`run_contrastive_matrix.py` trains in subprocesses that inherit the environment, so
`CONTRASTIVE_PATHS_CONFIG` reaches them without a flag of its own.

To resume a run recorded before this change, `data_root` must resolve to the value
in that run's `config.json`. A resume compares every immutable setting, so a
different mount — including no mount where one was recorded — is rejected.

### Private and local artifacts

Restore the artifacts listed with SHA-256 hashes in
[the artifact inventory](configs/contrastive/full-multilingual-v1-artifacts.json):

- The cohort manifest and its `.meta.json` belong in `data/processed/`.
- DICOM series live under `<data_root>/train_series`, with the series CSV at
  `<data_root>/train_series.csv`, where `data_root` is whatever you assigned in
  `paths.json` (see [Machine directories](#machine-directories); this project was
  developed with `/mnt/d`). The manifest stores absolute series paths, so a cohort
  built on one mount must be rebuilt if that mount moves.
- Copy the two pinned Hugging Face snapshots, including referenced blob contents,
  into the training host's Hugging Face cache. Revisions are pinned in the matrix,
  and compact checkpoints require those exact frozen backbones at reload time.
- `data/translation/gold_translated.csv` is an external input recorded in the inventory.

Building a new cohort instead of restoring one:

```bash
.venv/bin/python scripts/build_contrastive_manifest.py \
  --output data/processed/contrastive-cohort.jsonl \
  --text-policy multilingual_raw \
  --validate-pixels
```

The reports CSV and the series paths come from `paths.json`; pass `--reports-csv`,
`--series-csv`, or `--series-root` to override any of them for one build.

A new cohort has a new checksum, so it is a new experiment version: update the
matrix's `manifest_sha256` rather than reusing an existing one.

---

## Deploy a single training run

**1. Confirm the environment and inputs.** This indexes the cohort, prints its
summary, and stops before building a model:

```bash
.venv-cuda/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv-cuda/bin/python scripts/train_contrastive.py \
  --cohort-manifest data/processed/contrastive-cohort.jsonl --prepare-only
```

**2. Train.** A run refuses to write into an existing `--output-dir`, so each run
gets its own:

```bash
.venv-cuda/bin/python scripts/train_contrastive.py \
  --cohort-manifest data/processed/contrastive-cohort.jsonl \
  --text-model multilingual-e5 --text-pooling mean --text-prefix 'query: ' \
  --text-model-revision d128750597153bb5987e10b1c3493a34e5a4502a \
  --image-model facebook/dinov2-small \
  --image-model-revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --max-text-length 512 --image-unfreeze-last 0 --text-unfreeze-last 0 \
  --epochs 20 --batch-size 32 --num-workers 4 \
  --image-cache-dir checkpoints/contrastive/image-cache \
  --image-memory-cache-size 16 --trainable-checkpoints --local-files-only \
  --output-dir checkpoints/contrastive/my-run
```

Add `--local-files-only` whenever the host should not reach the network. Prefix with
`scripts/run_measured_training.py --telemetry PATH --` to also record runtime and
peak CUDA memory.

**3. Read the output directory.** A completed run contains:

| File | Contents |
| --- | --- |
| `config.json` | The resolved configuration, exactly as the run used it |
| `run_manifest.json` | Run id, Git revision, source-tree hash, input hashes, dependency versions, resolved backbone revisions, resume events |
| `metrics.jsonl` | One line per epoch: training loss and gradient norm, per-group learning rates, retrieval metrics, embedding diagnostics |
| `last.pt` | The most recent epoch, fully resumable |
| `best.pt` | The epoch selected by highest `mean_r@1`, then lower validation loss, then earlier epoch |

**4. Resume an interrupted run.** Keep every flag identical except the epoch target,
and point at the run's own `last.pt`:

```bash
.venv-cuda/bin/python scripts/train_contrastive.py <the same flags> --epochs 20 \
  --resume checkpoints/contrastive/my-run/last.pt
```

Resume restores the optimizer, schedule, loss scaler, loader shuffle, and random
states, and refuses to continue if the saved configuration differs on any field
other than the epoch target, if the metrics history does not end at the checkpoint's
epoch, or if `best.pt` disagrees with that history. Runs saved under a superseded
selection rule must be audited separately, not silently finalized under the new one.

---

## Deploy the full experiment grid

A grid is one JSON matrix — [`full-multilingual-matrix.json`](configs/contrastive/full-multilingual-matrix.json) —
that fixes the cohort by checksum, the shared arguments, the per-experiment
overrides, the output root, and the selection policy before anything runs. The
current grid is eight experiments comparing text truncation, pooling, report view,
and which final blocks are unfrozen.

**1. Validate the grid without running it.** This performs every check — cohort
checksum, selection policy, experiment names — and prints the exact command for each
experiment. It works on a CPU-only host:

```bash
.venv/bin/python scripts/run_contrastive_matrix.py --dry-run
```

**2. GPU smoke and resume check.** Same pinned models and preprocessing, a separate
output directory, at most 64 cohort rows, two epochs total. It never touches the
test split:

```bash
.venv-cuda/bin/python - <<'PY'
import json, subprocess, sys
from pathlib import Path
import torch
assert torch.cuda.is_available()
m = json.loads(Path('configs/contrastive/full-multilingual-matrix.json').read_text())
out = 'checkpoints/contrastive/full-multilingual-v1-gpu-smoke'
command = [sys.executable, 'scripts/train_contrastive.py', *m['base_args'],
           '--output-dir', out, '--max-studies', '64', '--batch-size', '32',
           '--num-workers', '0', '--local-files-only']
subprocess.run([*command, '--epochs', '1'], check=True)
subprocess.run([*command, '--epochs', '2', '--resume', out + '/last.pt'], check=True)
subprocess.run([sys.executable, 'scripts/evaluate_contrastive.py', out + '/best.pt',
                '--split', 'validation', '--output', out + '/reload-validation.json'], check=True)
PY
```

**3. Run the first experiment and inspect its cost.** Record runtime, peak GPU
memory, and the validation diagnostics before committing to the rest of the grid:

```bash
.venv-cuda/bin/python scripts/run_contrastive_matrix.py --experiment 00-projection-full-mean-512
```

**4. Run the remaining experiments.** Omit `--experiment` to run the whole matrix in
order; each experiment trains in a fresh process, so GPU memory is released between
them. Repeat `--experiment` to select a subset:

```bash
.venv-cuda/bin/python scripts/run_contrastive_matrix.py
```

If one experiment is interrupted, resume it with the exact command printed by
`--dry-run` plus `--resume <that experiment's output dir>/last.pt`, keeping the
original 20-epoch target.

**5. Select on validation.** This verifies every run's `best.pt` against its own
history and writes `validation-selection.json` under the matrix output root:

```bash
.venv-cuda/bin/python scripts/finalize_contrastive_matrix.py
```

The selection file is immutable: re-running with different results is an error, not
an overwrite. Add `--allow-incomplete` only to inspect a partially finished grid.

**6. The locked test split — once.** Only after the selection above exists, and only
when the experiment is genuinely finished:

```bash
.venv-cuda/bin/python scripts/finalize_contrastive_matrix.py --evaluate-locked-test
```

Do not invoke `--evaluate-locked-test` during baseline development. Selection reads
validation only, which is what `locked_test_was_used_for_selection: false` in the
selection file records.

---

## Evaluating and reporting

Score a saved checkpoint on any split:

```bash
.venv-cuda/bin/python scripts/evaluate_contrastive.py \
  checkpoints/contrastive/my-run/best.pt --split validation \
  --image-cache-dir checkpoints/contrastive/image-cache
```

The checkpoint's own saved configuration defines the model and cohort, so a report
cannot describe a different setup than the one trained. It writes
`evaluation-<split>-<checkpoint>.json` and `rankings-<split>-<checkpoint>.jsonl`
beside the checkpoint, containing recall at 1/5/10 in both directions, mean and
median rank, expected credit for tied candidates, embedding diagnostics, per-query
rankings, bootstrap 95% intervals, the random-ranking baseline for the split size,
and any difference from the metrics stored at training time. Existing reports are
never replaced without `--overwrite`.

Re-score every saved checkpoint after a metric definition changes:

```bash
.venv/bin/python scripts/rescore_contrastive_checkpoints.py \
  --checkpoint-root checkpoints/contrastive --summary digest/contrastive-checkpoint-rescore.json
```

Rebuild the browsable project snapshot:

```bash
.venv/bin/python scripts/build_project_digest.py
```

---

## Reproducibility rules

- **A cohort's checksum is its identity.** The manifest reader rejects missing or
  mismatched checksums. Never recompute a checksum to accept changed data; changed
  data means a new experiment version. The full manifest's checksum was added by an
  audited migration recorded in [digest/full-manifest-migration.json](digest/full-manifest-migration.json).
- **Selection is one-way.** Validation selection is written before any test
  evaluation and cannot be rewritten with different content.
- **Runs record their own source.** Runs are routinely launched from an uncommitted
  working tree, so `run_manifest.json` stores both the Git revision and a hash of the
  actual source tree and configuration files.
- **Frozen protocols are historical records.** Files under `configs/` and `digest/`
  that pin `source_sha256` describe the code as it was when a completed experiment
  ran. Later refactors do not invalidate those results, and those hashes are not
  updated to match current code.
- **Compact checkpoints need their backbones.** `--trainable-checkpoints` stores only
  trained weights and therefore requires pinned image and text model revisions, which
  must be present in the Hugging Face cache at reload time.
