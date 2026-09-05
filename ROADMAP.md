# Project roadmap and milestones

Last reviewed: 2026-09-05 (Asia/Taipei)

## Working goal

Build a reproducible knee MRI research pipeline that learns study representations
from paired MRI images and multilingual radiology reports, then measures whether
those representations improve prediction of the dataset's 12 pathology labels.
Bidirectional image/report retrieval is an intermediate representation check.

This goal is inferred from the [classification EDA notebook](eda_exploration.ipynb)
and the [contrastive pipeline](digest/contrastive-learning-pipeline.md). No separate
product specification was found. The proposed research endpoint is a reproducible
model, an inference/evaluation path, and an honest comparison against baselines on
held-out data. Clinical deployment is a separate future scope requiring external
validation and clinical review.

## Current state

The project is between pilot feasibility and full-cohort experimentation.

| Area | Evidence as of this review | Remaining gap |
| --- | --- | --- |
| Exploration | Notebook covers class imbalance, MRI series, multilingual reports, and LM-label comparisons | Establish label provenance and an explicit downstream evaluation protocol |
| Contrastive pipeline | Two towers, plane fusion, preprocessing, multilingual text handling, evaluation CLI, bounded cache, resume support, and compact checkpoints exist | Full-scale validation and stronger integration coverage |
| Pilot experiments | Saved 58-study experiments and a multilingual CPU smoke run exist; the implementation note records 18 checkpoint re-scores with no metric changes | Pilot results do not establish a winning architecture or downstream utility |
| Full cohort | Manifest reader accepts 4,407 eligible rows; all record pixel QC `passed`; splits are 3,526 train / 441 validation / 440 test | Checksum audited and enforced; see `digest/full-manifest-migration.json` |
| Experiment plan | Eight experiments in [the matrix](configs/contrastive/full-multilingual-matrix.json); all eight commands dry-run successfully | No `full-multilingual-v1` output directory exists |
| Hardware | Installed PyTorch is `2.13.0+cpu`; CUDA is unavailable | A working CUDA training environment |
| Verification | `pytest`: 25 passed; `git diff --check`: passed | Passing unit tests do not cover all orchestration behavior |
| Versioning | Experiment setup frozen as `full-multilingual-v1`; see README, protocol, and artifact inventory | GPU environment and completed baseline artifacts |

Pixel QC above is the saved manifest result, not a fresh DICOM decode sweep.
Training and checkpoint re-scoring were not repeated during this review.

The [September 4 audit](digest/contrastive-learning-current-state.md) is historical:
its retrieval-tie, text-contract, caching, and run-management findings have largely
been addressed by the [implementation follow-up](digest/contrastive-learning-implementation-status.md).
That follow-up still says full pixel QC is being built; the artifact is now present.
The [experiment session log](digest/experiment-session-log.md) tracks the pilot
session, rather than the whole project. This file is the project-wide tracker.

## Milestones

Check a task only when its acceptance evidence exists. A milestone is complete
when every task in it is checked; add the completion date and artifact or commit.

### M0 — Exploration and pilot feasibility: complete

- [x] Explore pathology prevalence, series structure, reports, and label comparisons.
- [x] Implement study/report pairing, MRI preprocessing, dual encoders, and retrieval.
- [x] Run the translated 58-study pilot and record controlled encoder/length/unfreezing experiments.
- [x] Correct tied retrieval scores and record saved-checkpoint re-scoring.

Evidence: EDA notebook, [session log](digest/experiment-session-log.md),
[re-score summary](digest/contrastive-checkpoint-rescore.json), and implementation
follow-up. Completed locally by 2026-09-04. This is a feasibility milestone only.

### M1 — Freeze the data and experiment contract: in progress

- [x] Implement explicit multilingual and translated-English text policies.
- [x] Materialize full and pilot manifests with grouping, fixed splits, and pixel QC.
- [x] Add run identity, source/model/dependency provenance, resume, and bounded caching.
- [x] Resolve the full manifest's missing checksum through an audited rebuild or migration; require a checksum for the frozen experiment artifact and test missing/mismatched values.
- [x] Use one checkpoint-selection rule in training and finalization, including resume; test tied Recall@1 with different validation losses and verify the selected metrics describe the actual checkpoint.
- [x] Document patient-ID anonymization semantics if obtainable; otherwise retain the explicit limitation that grouping cannot prove patient independence.
- [x] Review and version source, tests, configs, dependency locks, and documentation; document locations/checksums for data and model artifacts needed to reproduce runs.
- [x] Add concise setup/run instructions reflecting the current multilingual matrix and distinguish historical defaults in older docs.
- [ ] Freeze retrieval selection metrics, downstream label provenance, leakage checks, and intended downstream metrics before further tuning.

Acceptance: a recorded source revision, verified immutable cohort/splits, consistent
selection behavior, and reproducible setup instructions. Patient-identity uncertainty
must remain explicit if it cannot be resolved.

Selection gap fixed on 2026-09-05: training, resume, and finalization use highest
Recall@1, lowest loss, then earliest epoch. Resume/finalization verify `best.pt`
against the selected history row; mismatched legacy checkpoints fail explicitly.
The protocol records label limitations and intended metrics; independent label
provenance and duplicate/leakage auditing remain open before further tuning.

### M2 — Full-cohort projection baseline: blocked on M1 and CUDA

- [ ] Provision/verify CUDA, pinned model access, dataset paths, storage, and the dependency environment on the training host.
- [ ] Run a short GPU smoke check including compact-checkpoint reload and resume.
- [ ] Run `00-projection-full-mean-512` on the frozen full-cohort manifest.
- [ ] Review learning curves, validation loss, temperature, embedding variance, similarity distributions, and per-query errors against random retrieval baselines.
- [ ] Record runtime, memory, batch size/negative count, source revision, manifest hash, and artifact paths.

Acceptance: a complete, reloadable baseline with validation diagnostics and a written
continue/revise decision. A completed run is not by itself evidence of useful learning.

### M3 — Controlled representation selection: planned; depends on M2

- [ ] Run the declared 256/512-token, CLS/mean, and report-section comparisons.
- [ ] Compare image-only, text-only, and both-tower last-block adaptation.
- [ ] Audit multilingual and truncation errors; tokenizer coverage alone does not validate report semantics. Existing QC records 625/4,407 full reports over 512 tokens.
- [ ] Review duplicate/clinically similar pairs and false negatives before changing the contrastive objective or negative pool.
- [ ] Assess stability of promising configurations with predeclared repeat seeds using training/validation data only.
- [ ] Freeze the validation-based selection and record all attempted experiments and deviations from the matrix.

Acceptance: a reproducible selection with uncertainty and error analysis, using no
locked-test feedback. Treat larger encoders, more slices/3D inputs, and larger
negative pools as conditional follow-ups driven by observed failures.

### M4 — Locked retrieval evaluation: planned; depends on M3

- [ ] Verify the chosen checkpoint hash and its agreement with the selection record.
- [ ] Evaluate the selected model on the locked test split after selection is final.
- [ ] Save bidirectional Recall@1/5/10, rank metrics, confidence intervals, random baselines, and per-query rankings.
- [ ] Record language/report-length and image-QC failure patterns where metadata permits, including small subgroup limitations.

Acceptance: an immutable test report with the complete selection history and stated
limitations. Further tuning after seeing test results needs a new evaluation protocol.

### M5 — Downstream pathology utility: planned; protocol starts in M1

- [ ] Audit the 12 labels, missingness, class imbalance, and whether labels are human or report/LM-derived; separate independent reference labels from weak supervision.
- [ ] Compare frozen contrastive image features with unadapted DINOv2 features under the same splits and classifier budget; add a supervised baseline if label volume supports it.
- [ ] Train pathology probes on training data, select thresholds/hyperparameters on validation data, and evaluate once on the reserved downstream test data.
- [ ] Report per-class and macro/micro AUROC/AUPRC, validation-selected threshold metrics, uncertainty, and calibration where support permits; account for classes lacking positives/negatives.
- [ ] Package the selected encoder/preprocessing configuration with an inference command and a reproducible evaluation report.

Acceptance: a defensible answer to whether contrastive pretraining improves pathology
prediction over the image-only baseline. A negative result is a valid completed
research outcome. Report-derived labels cannot establish independent diagnostic validity.

### M6 — External validation and use decision: future, conditional

- [ ] Define the intended use and acquire an appropriate independent cohort and reference labels.
- [ ] Evaluate generalization, relevant subgroups, calibration, and failure modes with clinical collaborators.
- [ ] Record whether the evidence supports further research, a bounded application, or stopping.

Acceptance: an external evidence report and explicit use decision. This is a separate
gate beyond the core research deliverable; no deployment commitment is assumed.

## Immediate next steps, in order

1. Close the M1 manifest-integrity and checkpoint-selection gaps, with focused regression tests.
2. Freeze/version the experiment inputs and protocol; prepare the CUDA handoff.
3. Run and review the full projection baseline before launching the rest of the matrix.

From the repository root, preview the planned commands:

```bash
.venv/bin/python scripts/run_contrastive_matrix.py --dry-run
```

After M1 and the GPU checks are complete, start the baseline:

```bash
.venv/bin/python scripts/run_contrastive_matrix.py \
  --experiment 00-projection-full-mean-512
```

## Progress log and maintenance

Update this file when a milestone, blocker, or experiment decision changes. Record
the date, task, result, and evidence link/hash; keep detailed run histories in the
experiment log and link them here. Revise the working goal if a project specification
is adopted. Do not infer completion from code presence alone.

| Date | Update | Evidence |
| --- | --- | --- |
| 2026-09-05 | Created project-wide roadmap; verified full manifest inventory and reader, 17 passing tests, CPU-only hardware, and eight matrix dry-run commands. Identified missing full-manifest checksum and selection-policy mismatch. | Current source, local manifest metadata, and checks described above |

| 2026-09-05 | Migrated missing full-manifest checksum without changing cohort bytes/splits; unified selection and resume; froze protocol, resolved baseline config, model/data hashes, and CPU dependency closure. 25 tests and eight dry-run commands pass. Baseline launch attempted and stopped by CUDA guard; GPU smoke and full run remain pending. | `digest/full-manifest-migration.json`, `configs/contrastive/full-multilingual-v1-*`, `README.md` |
