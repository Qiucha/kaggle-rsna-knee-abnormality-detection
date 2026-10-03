# Project roadmap and milestones

Last reviewed: 2026-09-11 (Asia/Taipei)

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
| Verification | `pytest` on 2026-09-11 (CPU `.venv`): 63 passed, 1 skipped (CUDA-only). The historical "27 passed with CUDA" figure predates several suite additions and was stale. | Passing unit tests do not cover all orchestration behavior |
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
- [x] Compare image-only, text-only, and both-tower last-block adaptation. All three arms are measured at three matched seeds against the same gold labels. Image-only is the only one that moves the image representation (+0.024 to +0.027 macro AUROC); text-only leaves DINOv2 bit-identical by construction, verified at 0.0 deviation; both-tower reproduces the image-only gain but adds nothing to it, with all three seed-matched paired intervals spanning zero. See [`downstream-both-block-v1.md`](digest/downstream-both-block-v1.md).
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

First measurement taken on 2026-09-11, against the unadapted-DINOv2 comparison in the
second task above: see [the downstream probe](digest/downstream-probe-v1.md). No task is
checked, because none of them is finished by it. The probe scores 56 of the 58 gold
studies, all of which the representation has already seen, so it is a development
readout statistic and cannot satisfy an acceptance that requires reserved downstream
test data. What it does establish is directional: the image-block arm gains +0.026 macro
AUROC at three seeds with intervals excluding zero, and every frozen-tower run is a
provable zero because it never updates a DINOv2 weight.

### M6 — External validation and use decision: future, conditional

- [ ] Define the intended use and acquire an appropriate independent cohort and reference labels.
- [ ] Evaluate generalization, relevant subgroups, calibration, and failure modes with clinical collaborators.
- [ ] Record whether the evidence supports further research, a bounded application, or stopping.

Acceptance: an external evidence report and explicit use decision. This is a separate
gate beyond the core research deliverable; no deployment commitment is assumed.

## Immediate next steps, in order

1. Retain full text after the completed [impression-first comparison](digest/claude-translated-dfd07db-comparison-results.md): 10.6576% versus 9.6372% mean validation R@1. The negative seed-2026 difference did not trigger repeats. Both reloads and provenance passed; candidate errors and long-report behavior were reviewed.
2. Review the completed English/medical text screens on the Claude-translated quarantined cohort. English E5 reached 8.8435% and NeuML PubMedBERT Embeddings 9.4104%, versus the matched frozen multilingual baseline at 10.6576%. Both completed 20 epochs and exact reloads; fresh artifact verification passed. Neither triggered repeat seeds, and no follow-up training is queued. The qualitative review is now complete: all twenty pairs per candidate in the [error-review packet](digest/english-medical-text-error-review-packet.md) were read and recorded in [the error review](digest/english-medical-text-error-review.md), alongside [verified results](digest/english-medical-text-results.md). It recommends, for the user's decision, text-only final-block adaptation on the baseline encoder at three matched seeds; no matrix is frozen and no run is launched. The [image-adaptation error review](digest/claude-translated-dfd07db-image-block-error-review.md) is already complete. Both single-seed paired intervals include zero; keep final selection open. See the [handoff notes](digest/handoff/) on the machine that ran it.
3. Preserve exact-pair retrieval and original source reports. Two confirmed conflicting duplicate-image training entries are quarantined. Duplication is now screened cohort-wide and the known pair remains the only pixel-identical group in train and validation; one train/test candidate is gated on a locked-test decision. Patient independence is **unverifiable, not merely unresolved** — anonymization is per study, so every reported number is a study-level estimate. See the [integrity audit](digest/integrity-coverage-audit.md) and [gold-case inventory](digest/gold-case-inventory.md). The user identifies 58 ground-truth cases; other labels remain weak supervision, with only eight labeled validation cases and zero positive MCL. Keep test locked and downstream fitting separate.

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

| 2026-09-10 | Corrected in place a documentation error: `digest/image-shape-audit.md` and the local handoff both advertised `src/contrastive_learning/shape_trace.py`, `scripts/audit_image_shapes.py`, `tests/test_image_shape_trace.py`, and `CONTRASTIVE_SHAPE_TRACE=1`. None exist in the working tree or anywhere in git history, so the stated reproduction commands could never have run. The trace output `digest/image-shape-trace.jsonl` (1,176 records; 585 paired operations) was committed in `62a2a8f` and remains valid, but cannot be regenerated. Both documents now say so and point to the real coverage in `tests/test_image_tower.py`. The audit’s substantive conclusions are unaffected. No training, no source changes. | `digest/image-shape-audit.md` |

| 2026-09-11 | Completed the pending qualitative error review of the two frozen English/medical text screens: all forty worst-pair records (twenty per candidate) read in full and recorded in `digest/english-medical-text-error-review.md`. Neither candidate is promotable and that is unchanged. The two fail by different routes — e5 regresses on queries the baseline handled, PubMedBERT mostly fails to rescue already-bad ones. Dominant readable error in both is template and register matching over finding content. Tokenizer-defined subgroups are not comparable across candidates: PubMedBERT over-512 is 0 of 19 and its lost-heading row is 4 studies. Laterality is never contradicted in these forty pairs. Recommends, for the user’s decision only, text-only final-block adaptation at three matched seeds; no matrix frozen, nothing launched, test still locked. | `digest/english-medical-text-error-review.md`, `digest/english-medical-text-results.md` |
| 2026-09-11 | Closed the two optional follow-ups the text error review left open, from saved artifacts only — no model run, no evaluation, test untouched. (1) The `horn<mangled>root` mojibake is in the supplied source reports (`data/sample/train.csv`), not in this pipeline: 14 of 4,407 cohort records, all `source_english`, no translated report affected. It is inside `report_sha256`, so repairing it would change the pinned cohort manifest hash; left as an observation. (2) Retrieval-hub concentration is general, not a PubMedBERT tail artifact — all three runs including the baseline sit far outside a uniform Monte-Carlo null. e5 is the most concentrated on every statistic, while PubMedBERT is indistinguishable from the baseline on target diversity (179 vs 181 distinct rank-1 targets) and milder on single-hub share (11 vs 21), so it is not uniquely hub-prone. Hub identity is encoder-specific (0/10 top-10 overlap across all three). Two hub claims in the review corrected. Analysis kept as a reusable script. | `scripts/analyze_text_screen_followups.py`, `digest/english-medical-text-error-review.md` |
| 2026-09-11 | Acted on three user decisions. (1) Froze, but did **not** launch, the text-only final-block adaptation matrix: `configs/contrastive/claude-translated-dfd07db-text-block-v1.json`, three seeds (2026/2027/2028) budgeted up front, `--text-unfreeze-last 1` (final block + final layer norm), DINOv2 frozen, full report, 512 cap; validated by a total `--dry-run` that verifies the selection policy and cohort SHA-256 and emits the exact commands without training. The output root does not exist and no selection file was written, so the one-way door is untouched. All three matched frozen baselines already exist and are reused, so the arm costs three new runs rather than six. (2) Shape-trace harness: decided not to rebuild; `digest/image-shape-audit.md` stands as corrected and `tests/test_image_tower.py` is the real coverage. (3) Fixed `tests/test_inputs.py` to match the current `DataSummary` — the dirty source *removed* the redundant `reports_with_text` field rather than adding one, contrary to the earlier handoff note; both construction sites now use keyword arguments. Suite: 63 passed, 1 skipped, from 3 failed / 60 passed / 1 skipped. No `src/` file touched; nothing committed. | `configs/contrastive/claude-translated-dfd07db-text-block-v1.json`, `digest/claude-translated-dfd07db-text-block-protocol.md`, `tests/test_inputs.py` |
| 2026-09-11 | Executed the two read-only items the feedback response gated on. **§7:** cohort-wide duplicate screen over all 4,407 studies (header-based, positive-control validated) found 7 acquisition-duplicate groups; the six without a test member were decoded and **only the known quarantined pair is pixel-identical**. Joins are one-to-one on both sides. **Patient independence is unverifiable, not merely unresolved** — anonymization is per study, proven by that pair carrying two different `PatientID` values on identical pixels — so every reported number is a study-level estimate. One train/test candidate group is gated on a locked-test decision, **which the user then took the same day: decoded, pixels differ on all three planes, CLEARED — no group remains gated and the quarantined pair is the only pixel-identical duplicate in the cohort**. **§9:** 48 of the 58 gold studies are training pairs and 8 are in the checkpoint-selection split; **the only 2 clean of all exposure are locked test**, so any probe on them is a development statistic by construction. `drop.csv` is a blank template, not a second label set. All 12 labels are estimable pooled over 56; the "zero positive MCL" claim reproduces for the 8-case validation slice. Estimand drafted, **not signed off**; nothing fitted. Also reviewed the seven dirty `src/contrastive_learning/` files (a dedup pass plus a CSV header-validation bug fix; metrics bit-exact against HEAD) and backed up all 21 modified tracked files outside git. No commit, no training, test split untouched. | `digest/integrity-coverage-audit.{json,md}`, `digest/gold-case-inventory.{json,md}`, `digest/acquisition-duplicate-screen.json`, `digest/duplicate-group-pixel-verification.json`, `digest/gated-duplicate-resolution.json`, four `scripts/audit_*`/`verify_*` scripts, `/home/q/novo-project-backups/2026-09-11-dirty-tracked/` |
| 2026-09-11 | First downstream measurement of whether contrastive learning helps the actual task. Verified the two pixel-identical duplicate studies are already excluded from training — the reader selects 4,405 of 4,407 rows, 3,524 train — so no pipeline change was needed. Then probed representations against the 12 gold labels with pooled out-of-fold AUROC, protocol frozen in advance per the gold-case inventory (per-label L2 logistic, 8 stratified folds, scaler inside each fold, no tuning). **Two results.** (1) The image-block arm improves macro AUROC from 0.6438 to 0.6687/0.6706/0.6709 at three seeds, **+0.026, every paired interval excluding zero** — a more stable signal than its own +1.2094 pp retrieval gain. Ten of twelve labels rise; the gains concentrate on labels the unadapted features read near chance (Synovitis +0.072, PF OA +0.050, Fracture +0.048). (2) Every `--image-unfreeze-last 0` run leaves DINOv2 **byte-identical** — asserted at 0.0 max deviation — so the retained baseline, both text screens and the text-only arm could not have changed the image representation at all. The learned plane attention, which retrieval selects on, gives back most of the backbone gain on label readout (+0.006). LM report-reading extraction reaches 0.82–0.89 on the same studies, bounding label noise. Estimand is readout on an exposed cohort, not generalisation: 48 of 56 were training pairs, 8 selected checkpoints, and the 2 unexposed gold studies are locked test and were never loaded. No training, no test evaluation, no commit. | `digest/downstream-probe-v1.{json,md}`, `configs/downstream/gold-probe-v1.json`, `scripts/evaluate_downstream_probe.py` |
| 2026-09-11 | Took apart the one anomaly in the downstream probe and folded both results into the HTML digest. **Mechanism.** The image tower's learned plane attention reads labels worse than a plain mean of the same three planes: −0.0185/−0.0214/−0.0205 macro AUROC at the three seeds, paired against the uniform mean of the *same* checkpoint. Direction agrees across seeds to within 0.003 and 90–95% of replicates favour the mean, but **every interval spans zero** at 56 studies, so this is a direction rather than a demonstrated deficit. Two candidate causes eliminated: dropping the additive plane-embedding term moves the score 0.0007, and concatenating all three planes (2304-d) scores 0.6689 against the mean's 0.6701, so no recoverable information is lost. What remains is the re-weighting — all three seeds independently put ≈0.5 mass on Axial (argmax Axial for 46–48 of 56 studies). Per-plane probes explain the cost: PF OA reads best on Axial (0.732) and gains +0.082, while MCL is coronal (0.596) and Lateral Meniscus sagittal (0.583) and they lose −0.082 and −0.091. Axial advantage per label tracks attention benefit at r = 0.62 (Spearman 0.53, p = 0.075, n = 12). One plane preference, twelve labels that disagree about planes; the attention fits its retrieval objective, which is one question, not twelve. Exporting pre-attention plane features costs no training and recovers the difference. **Digest.** Added two cards to `digest/project-state.html` and corrected three claims already on that page: patient independence is unverifiable rather than unresolved, downstream fitting has now occurred (on exposed data, test untouched), and every run in the evidence table left DINOv2 byte-identical. No training, no test evaluation, no commit. | `digest/plane-pooling-analysis.json`, `configs/downstream/gold-probe-planes-v1.json`, `scripts/analyze_plane_pooling.py`, `digest/downstream-probe-v1.md`, `digest/project-state.html` |
| 2026-09-12 | Closed the last cell of the M3 adaptation comparison: **both-tower** final-block adaptation, at the user's request, to test whether aligning the image tower harder toward the report teaches it more medical semantics. Froze `configs/contrastive/claude-translated-dfd07db-both-block-v1.json` (three seeds 2026/2027/2028 budgeted up front, `--image-unfreeze-last 1 --text-unfreeze-last 1`, per-tower learning rates copied by value from the two single-factor arms so each tower matches the arm it is compared against), dry-run validated, then trained all three on the RTX 3070. **Result: a null.** The primary contrast is seed-matched both-block versus image-block — not versus unadapted DINOv2, which would confound two simultaneous changes — and it gives +0.0005/−0.0001/−0.0030 macro AUROC with every paired 95% interval spanning zero and replicate fractions straddling 50%. Retrieval agreed first: +0.57/+0.11/−1.02 pp mean validation R@1, sign flipping, mean −0.11 pp. Both arms *do* clear the unadapted anchor (both-block +0.0253/+0.0268/+0.0240, intervals excluding zero), so the whole contrastive gain comes from unfreezing the image block and the text block adds nothing. **Two reproduction checks passed to four decimals**: the image-block arm reproduces `downstream-probe-v1.md` (0.6687/0.6706/0.6709) and its `plane_attention_768` reproduces `plane-pooling-analysis.json` (0.6503/0.6491/0.6504), both under a config written independently. **Text-only settled directly**: all three text-block runs assert 0.0 max deviation from unadapted DINOv2, so that arm provably never changed what the image tower encodes. Caveats recorded: estimand is readout on an exposed cohort (56 of 58 gold cases exposed) and the both-block arm trains *more* parameters on it, which strengthens the null; every selected checkpoint sits well past its validation-loss minimum (all three bottomed at epoch 6, selected at 14/20/15) and seed 2027 selected the final epoch, so its R@1 never turned over inside the frozen budget, which was not extended. Also found and documented — **not patched** — that `scripts/evaluate_downstream_probe.py` only works on CPU (`extract_features` never moves the batch to the device; `--device` defaults to `cuda`; `.venv-cuda` has no `sklearn`); left unfixed so both arms come from one code path. No test evaluation, no selection file finalized for this matrix, no commit. | `digest/downstream-both-block-v1.md`, `digest/downstream-probe-both-block-v1.json`, six `digest/downstream-both-vs-image-block-*.json`, `configs/contrastive/claude-translated-dfd07db-both-block-v1.json`, `configs/downstream/gold-probe-both-block-v1.json`, `scripts/compare_downstream_arms.py` |
