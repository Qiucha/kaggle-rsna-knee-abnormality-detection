# Project roadmap and milestones

Last reviewed: 2026-09-07 (Asia/Taipei)

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
| Experiment plan | Eight experiments in [the matrix](configs/contrastive/full-multilingual-matrix.json); all eight commands dry-run successfully | Full baseline completed and compact validation reload verified |
| Hardware | Host check found RTX 3070 8 GB through WSL `/dev/dxg`; driver CUDA initialization passes. The sandbox hid the device. | CUDA environment and smoke/baseline validated; see results |
| Verification | `pytest`: 27 passed with CUDA; `git diff --check`: passed | Passing unit tests do not cover all orchestration behavior |
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

### M2 — Full-cohort projection baseline: complete (2026-09-05)

- [x] Provision/verify CUDA, pinned model access, dataset paths, storage, and the dependency environment on the training host.
- [x] Run a short GPU smoke check including compact-checkpoint reload and resume.
- [x] Run `00-projection-full-mean-512` on the frozen full-cohort manifest.
- [x] Review learning curves, validation loss, temperature, embedding variance, similarity distributions, and per-query errors against random retrieval baselines.
- [x] Record runtime, memory, batch size/negative count, source revision, manifest hash, and artifact paths.

Evidence: [completed baseline results](digest/full-projection-v1-results.md),
[learning curves](digest/full-projection-v1-curves.html), verified epoch-20 checkpoint
and local execution artifacts. Continue to controlled comparisons after leakage/error review.

Acceptance: a complete, reloadable baseline with validation diagnostics and a written
continue/revise decision. A completed run is not by itself evidence of useful learning.

### M3 — Controlled representation selection: in progress

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

1. Retain full text after the completed [impression-first comparison](digest/claude-translated-dfd07db-comparison-results.md): 10.6576% versus 9.6372% mean validation R@1. The negative seed-2026 difference did not trigger repeats. Both reloads and provenance passed; candidate errors and long-report behavior were reviewed.
2. Review the completed English/medical text screens on the Claude-translated quarantined cohort. English E5 reached 8.8435% and NeuML PubMedBERT Embeddings 9.4104%, versus the matched frozen multilingual baseline at 10.6576%. Both completed 20 epochs and exact reloads; fresh artifact verification passed. Neither triggered repeat seeds, and no follow-up training is queued. Read [verified results](digest/english-medical-text-results.md) and all twenty pairs per candidate in the [error-review packet](digest/english-medical-text-error-review-packet.md); write the pending qualitative review before selecting a new bounded experiment. The [image-adaptation error review](digest/claude-translated-dfd07db-image-block-error-review.md) is already complete. Both single-seed paired intervals include zero; keep final selection open. See the [handoff notes](digest/handoff/) on the machine that ran it.
3. Preserve exact-pair retrieval and original source reports. Two confirmed conflicting duplicate-image training entries are quarantined; broader duplication and patient independence remain unresolved. The competition has no reliable cross-study patient ID. The user identifies 58 ground-truth cases; other labels remain weak supervision, with only eight labeled validation cases and zero positive MCL. Keep test locked and downstream fitting separate.

The original eight-run matrix is historical, not an unattended launch queue.
Current evidence supersedes older pending status in the progress log below.

## Progress log and maintenance

Update this file when a milestone, blocker, or experiment decision changes. Record
the date, task, result, and evidence link/hash; keep detailed run histories in the
experiment log and link them here. Revise the working goal if a project specification
is adopted. Do not infer completion from code presence alone.

| Date | Update | Evidence |
| --- | --- | --- |
| 2026-09-05 | Created project-wide roadmap; verified full manifest inventory and reader, 17 passing tests, CPU-only hardware, and eight matrix dry-run commands. Identified missing full-manifest checksum and selection-policy mismatch. | Current source, local manifest metadata, and checks described above |

| 2026-09-05 | Migrated missing full-manifest checksum without changing cohort bytes/splits; unified selection and resume; froze protocol, resolved baseline config, model/data hashes, and CPU dependency closure. 25 tests and eight dry-run commands pass. Baseline launch attempted and stopped by CUDA guard; GPU smoke and full run remain pending. | `digest/full-manifest-migration.json`, `configs/contrastive/full-multilingual-v1-*`, `README.md` |

| 2026-09-05 | Corrected hardware diagnosis: WSL host exposes an RTX 3070 8 GB and working driver; sandbox/PATH checks had missed it. User authorized local CUDA setup and explicitly prohibited commits/pushes without permission. Installing isolated `.venv-cuda`; CPU environment retained. | Host `/usr/lib/wsl/lib/nvidia-smi`, successful `cuInit(0)` |

| 2026-09-05 | CUDA 12.6 environment installed with torch 2.13.0 / torchvision 0.28.0; 27 GPU-enabled tests pass. Fixed CPU/CUDA validation scalar mismatch and smoke reload subset handling. Batch-32 smoke, resume, and compact reload agree; full baseline launched. No commits or pushes. | `checkpoints/contrastive/cuda-execution-20260905/`, `requirements-cuda-lock.txt` |

| 2026-09-05 | M2 complete: 20 epochs, selected epoch 20, mean validation R@1 10.66% vs 0.227% chance; compact reload has zero metric deltas. Runtime 116.71 minutes, peak allocated GPU 1.808 GiB. Quantitative error/curve review complete; semantic/leakage review remains before tuning. No locked-test evaluation or new commits/pushes. | `digest/full-projection-v1-results.md`, `digest/handoff/` (local) |

| 2026-09-05 | Continued pre-tuning audit: 22 exact-report groups span splits (118 studies); read 20 worst validation pairs, found report disagreements and one internal inconsistency; 70/441 validation reports exceed 512 tokens. Label support is 48 train / 8 validation, with no positive validation MCL. Gate remains open for pixel/near-duplicate checks and provenance. | `digest/full-projection-v1-data-review.md`, `digest/full-projection-v1-data-audit.json` |

| 2026-09-07 | Updated baseline 10.2041%; quarantined full-text baseline 10.6576%; impression-first 9.6372%. Exact reloads, snapshot hashes, paired differences and candidate report review verified. Retain full text; repeat rule not triggered. Image-only final-block adaptation selected, not launched. | `digest/claude-translated-dfd07db-comparison-results.md`, `digest/claude-translated-dfd07db-comparison-audit.json` |

| 2026-09-07 | Image-only adaptation completed at three matched seeds; mean gain +1.2094 pp (sample SD 0.6643), all reloads/provenance verified. Interactive HTML digest added. Candidate error/subgroup review remains before text-only adaptation. | `digest/project-state.html`, `digest/claude-translated-dfd07db-image-block-audit.json` |

| 2026-09-07 | User redirected next research toward English and/or medical English text-encoder swaps on Claude translations. Handoff rewritten with model-contract research, matched baseline design and operational context; no new training launched. | `digest/handoff/` (local) |

| 2026-09-07 | Completed primary-source English/medical sentence-encoder shortlist, pinned metadata/tokenizers, frozen matched configs and bounded repeat protocol. Reverified previous artifacts and source compatibility. No candidate weights or training launched; image-adaptation error review and GPU preflight remain. | `digest/english-medical-text-encoder-research.md`, `digest/english-medical-text-protocol.md`, `digest/english-medical-text-preflight.json` |

| 2026-09-07 | English/medical queue completed: E5 8.8435%, PubMedBERT 9.4104%, matched baseline 10.6576%. Both screens verified; neither triggered repeats. Text qualitative error review remains. Refreshed digest and next-session handoff; no new training or test evaluation. | `digest/english-medical-text-results.md`, `digest/handoff/` (local) |
