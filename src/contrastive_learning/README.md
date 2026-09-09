# Contrastive learning pipeline

This experiment learns to match knee MRI studies with their reports. DINOv2 encodes
three MRI planes; a text encoder reads the report. Both produce normalized vectors.
Training brings matching image/report vectors together with a symmetric contrastive
loss, and validation measures retrieval in both directions.

For installing, running, and reporting, see the [root README](../../README.md).
This file documents the code itself.

## Where the code lives

| Module | Responsibility |
| --- | --- |
| `paths.py` | The paths configuration: which directories a machine may reassign, and how they resolve |
| `config.py` | `TrainConfig`: every setting of a run, its validation, and saved-config loading |
| `cli.py` | Training flags, argument-to-config translation, and input-path checks |
| `training.py` | Run lifecycle: optimizer groups, schedule, epoch loop, resume, checkpointing |
| `inputs.py` | Cohort loading, split policy, and loader construction shared by training and evaluation |
| `data.py` | Study/report pairing, the dataset, the image caches, and batch collation |
| `manifest.py` | Building and reading frozen, patient-grouped train/validation/test cohorts |
| `preprocessing.py` | DICOM slice ordering, cropping, normalization, orientation, and augmentation |
| `text.py` | Report views: full, impression, findings+impression, impression-first |
| `model.py` | Image and text towers, plane attention pooling, projections, contrastive loss |
| `evaluation.py` | Embedding collection, tie-aware retrieval metrics, and checkpoint reports |
| `checkpoints.py` | Checkpoint format, best-epoch selection, run provenance, and resume |
| `experiments.py` | Frozen matrix execution, immutable validation selection, checkpoint rescoring |
| `artifacts.py` | Atomic artifact writes and SHA-256 content hashes used by every module above |

Dependencies run one way. `training.py` calls `evaluation.py` and `checkpoints.py`;
`evaluation.py` never imports training, so a saved checkpoint can be scored without
the training stack. Both reach data through `inputs.py`, so they cannot disagree
about what a split contains. `checkpoints.py` depends only on settings and
`artifacts.py`. `paths.py` and `artifacts.py` are leaves: they import nothing from
this package, so any module may use them.

## What each module hides

Each of these owns a decision completely, so no caller has to restate it.

- **`checkpoints.py` owns the checkpoint file format.** Callers hand over live
  objects — `save_checkpoint(...)`, `load_model_weights(model, checkpoint)`,
  `restore_training_state(checkpoint, model=..., optimizer=..., ...)` — and never
  name a key inside the payload. Both weight formats (`full_v1` and the compact
  `trainable_delta_v1`) and the four random-number-generator states that make a
  resume bit-reproducible are therefore changeable in one place. It also owns
  `SELECTION_POLICY`: the single authoritative statement of how the best epoch is
  chosen, which frozen matrices copy and refuse to run without.
- **`inputs.py` owns "which studies, in what order".** `load_cohort(config)` returns
  a cohort with `split(name)`; splitting is deferred until requested, so
  `--prepare-only` can index without committing to a split, and evaluation can ask
  for one split alone. `make_loader(records, tokenizer, config)` owns cache
  settings, report preparation, and worker seeding. Only two named arguments differ
  between callers: `for_training` (enforces text policy and pixel checks) and
  `training` (shuffles and drops the final partial batch).
- **`manifest.py` owns cohort identity.** Patient-grouped splitting, text-policy
  resolution, pixel eligibility, and checksums are decided once when a manifest is
  built; afterwards `read_cohort_manifest` verifies every invariant and only reads.
  A missing or mismatched checksum is an error, never a warning.
- **`evaluation.py` owns the tie policy.** Reports in this cohort are sometimes
  identical, producing genuinely tied similarity scores. Every metric awards the
  expected credit of a uniformly random tie-break, and each report records
  `metric_tie_policy` so numbers computed under different rules are never compared.
- **`experiments.py` owns the one-way door.** `run_matrix` verifies the cohort
  checksum and selection policy before training; `finalize_matrix` re-derives each
  run's best epoch from its own history, writes an immutable
  `validation-selection.json`, and only then can be asked to touch the test split.
- **`paths.py` owns machine-specific locations.** No module names a directory that
  differs between hosts; the DICOM mount and the other reassignable paths come from
  one git-ignored JSON file, resolved by a command-line entry point *before* a
  `TrainConfig` exists. Everything downstream therefore still receives concrete
  paths, and run artifacts stay as reproducible as they were. `data_root` has no
  default and is required only where a mount is actually read, so a run driven by a
  frozen cohort manifest needs none.
- **`artifacts.py` owns durability.** Every artifact is written to a temporary
  sibling and renamed, so no reader sees a partial file, and content hashes are
  computed one way everywhere.

Keep command scripts atomic: parse arguments, call one workflow, print its result.
Put reusable policy in the module that owns the workflow.

## Follow a training batch

1. Load the frozen cohort manifest, or pair reports and DICOM series from CSVs.
2. Select one series per plane, preferring fluid-sensitive series and then the
   largest slice count. Read three slices from the central 20–80% of each series.
3. Apply the 130 mm physical center crop, percentile normalization, resize, and
   left-knee orientation. These pixels are what the caches store.
4. Stack studies, select the configured report view, and tokenize the reports.
5. Augment training images on device. Encode each plane with DINOv2, concatenate
   its CLS token with its mean patch features, and pool the three planes with
   learned attention.
6. Project image and text features to unit vectors, score every image/report pair
   in the batch, and optimize the symmetric loss in both directions.
7. Encode the full validation split, record metrics, and checkpoint the epoch.

| Value | Shape |
| --- | --- |
| One study | `[3 planes, 3 slice channels, H, W]` |
| Image batch | `[batch, 3, 3, H, W]` |
| Image or report vectors | `[batch, embedding_dim]` |
| Training similarity scores | `[batch, batch]` |

## Results and checkpoint selection

Training writes `metrics.jsonl`, `last.pt`, `best.pt`, `config.json`, and
`run_manifest.json` into the run directory. The best epoch has the highest
validation `mean_r@1`; ties break by lower validation loss, then the earlier epoch.
Resume re-derives that from history and rejects a `best.pt` that disagrees.

`evaluate_contrastive.py` writes, per checkpoint and split:

- Image-to-text and text-to-image recall at 1, 5, and 10, plus mean and median rank.
- Expected recall and rank when candidates have equal similarity scores.
- Validation loss, positive/negative similarities, embedding spread, temperature.
- Per-query rankings with study identifiers, top matches, and tie counts.
- Bootstrap 95% intervals, the random-ranking baseline for the split size,
  checkpoint hashes, and differences from the metrics stored at training time.

## Tests

`pytest` covers the parts that can run without a GPU or the DICOM mount: report
views, the image cache, cohort manifest invariants, retrieval metrics and ties,
checkpoint selection and the run lifecycle, image tower shapes and gradients,
experiment matrix handling, path resolution, and the translation runner.
Preprocessing tests that need `pydicom` skip when it is unavailable.
