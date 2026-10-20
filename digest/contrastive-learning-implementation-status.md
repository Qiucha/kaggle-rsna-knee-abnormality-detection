# Contrastive learning implementation status

Date: 2026-09-04 (Asia/Taipei)

This is the implementation follow-up to `contrastive-learning-current-state.md`. It covers requested items 2–7. Everything described here is local and uncommitted; no branch was committed or pushed.

## Outcome

The prototype has been converted into a reproducible experiment path with a corrected evaluator, explicit multilingual text contract, versioned cohort manifests, safe run/resume behavior, compact checkpoints, and a predeclared GPU experiment matrix. The complete path has passed an end-to-end CPU smoke run on the translated pilot.

The full projection baseline and follow-up matrix cannot be executed on this host because its installed PyTorch build is CPU-only. The runner checks this before creating experiment artifacts. The locked test split has not been evaluated.

## Requested items

### 2. Fix retrieval metrics and re-score checkpoints

Status: implemented and verified.

- Retrieval ties now receive uniform expected credit rather than an artificially best rank. Four fully collapsed embeddings correctly produce Recall@1 = 0.25 and mean rank = 2.5.
- Evaluation exports per-query image-to-text and text-to-image rankings, candidate ties, expected ranks, and study identifiers.
- Evaluation-only checkpoint loading supports both the historical full checkpoints and new trainable-delta checkpoints.
- Results include random-ranking baselines, deterministic 95% bootstrap confidence intervals, embedding health metrics, checkpoint hashes, and differences from stored metrics.
- Re-scoring all 18 current best checkpoints (17 historical plus the new smoke checkpoint) found zero tied queries and zero changes to stored retrieval metrics. The metric bug was real but did not alter these saved results. The aggregate is written to `digest/contrastive-checkpoint-rescore.json`; individual evaluation JSON and JSONL ranking files live beside each checkpoint.

### 3. Establish a multilingual text contract

Status: implemented for the next representation experiment; semantic/clinical validation remains open.

- The selected default for the full-data matrix is `intfloat/multilingual-e5-base`, pinned to revision `d128750597153bb5987e10b1c3493a34e5a4502a`.
- The implementation uses masked mean pooling, the model's required `query: ` prefix, and a 512-token limit.
- Tokenizer QC covers all 4,407 raw reports without exporting report text. It found zero unknown tokens, but 625 full reports exceed 512 tokens. The detailed result is in `multilingual-text-qc.json`.
- Report inputs can be full text, impression/conclusion, or findings plus impression. Missing sections fall back deterministically to the full report.
- A separate translated-English policy preserves source text and records translation provenance. The 58-study translated pilot manifest validates this path.

Tokenizer coverage is not evidence that the encoder understands specialized multilingual radiology language. This remains a model-risk item for error review and downstream evaluation.

### 4. Create a canonical cohort manifest

Status: schema and pilot are complete; full 4,407-study pixel QC is being materialized.

Each JSONL row records the study UID, pseudonymized grouping key, fixed train/validation/test assignment, eligibility, training and source reports, report hashes, language and provenance, selected plane paths and fingerprints, laterality, preprocessing status and QC, and labels. Sidecar metadata records the schema/preprocessing versions, source hashes, split seed/fractions, text policy, build configuration, and manifest hash.

Reader validation rejects duplicate studies, invalid splits, patient groups spanning splits, incomplete eligible rows, and changed manifest content. DICOM cache keys now include source file metadata and a preprocessing version.

The translated pilot artifact contains 58 eligible studies with fixed splits of 46 train, 6 validation, and 6 test; all 58 passed pixel validation. The full cohort uses the multilingual-raw contract because full translations do not exist.

One limitation remains, and as of 2026-09-11 it is settled in the unhelpful direction: all observable DICOM `PatientID` values are unique, and the split is grouped by a field whose every group holds one study. Anonymization is per study, not per patient — the two studies of the known pixel-identical duplicate pair carry different `PatientID` values. Patient independence is therefore unverifiable rather than merely unproven; see the [integrity audit](integrity-coverage-audit.md).

### 5. Harden run management

Status: implemented and tested.

- A new run fails if its output directory exists.
- Resume requires an in-directory checkpoint, compatible immutable configuration, consistent metric history, and a larger target epoch.
- Checkpoints restore the model, optimizer, scheduler, scaler, sampler generator, and CPU/CUDA random-number state.
- Run manifests include an immutable run ID, UTC timestamp, git commit/branch/dirty status, source-tree and input-artifact hashes, dependency versions, pinned resolved model revisions, and resume history.
- New checkpoints can store only trainable parameters. In the CPU smoke run, `best.pt` and `last.pt` are each 9.6 MB.
- Image caching is lazy, disk-backed, atomically written, validated on load, and bounded by an in-memory LRU instead of eagerly retaining the full cohort.
- Training records validation loss, temperature, positive/negative similarity, embedding variance, gradient norm, and learning rates. Frozen backbones remain in evaluation mode, AdamW excludes biases/norms/logit scale from decay, and logit scale is clamped after the optimizer step.

### 6. Run the projection-only full-cohort baseline

Status: execution blocked by hardware; configuration and end-to-end path are verified.

The primary experiment is `00-projection-full-mean-512`: frozen DINOv2-small and multilingual E5 towers, trainable fusion/projections, masked mean text pooling, 512 tokens, batch size 32, 20 epochs, fixed manifest splits, and bounded persistent image caching.

The CPU smoke run used the same pinned backbones and text contract on 46/6 pilot train/validation studies. It completed one epoch, saved compact checkpoints, and reloaded them for evaluation with zero metric drift. Its validation mean Recall@1 was 0.1667 on six studies, exactly the random Recall@1 baseline; this is only an integration check, not model evidence.

The matrix runner exits with `This matrix requires CUDA` on this host. It did not create the full-run output directory, and the locked test split remains untouched.

### 7. Prepare the next experiments

Status: eight experiments are declared and dry-run successfully.

The matrix contains:

1. projection-only, full report, mean pooling, 512 tokens;
2. projection-only with 256 tokens;
3. CLS instead of masked mean pooling;
4. impression/conclusion report view;
5. findings plus impression report view;
6. final image block unfrozen;
7. final text block unfrozen; and
8. final block in both towers unfrozen.

The finalizer selects only from validation mean Recall@1 (validation loss breaks ties), writes an immutable selection record, and evaluates the selected checkpoint on the locked test split only when explicitly requested. Test output protection prevents accidental replacement.

## Verification

- Unit tests: 17 passed.
- Python compilation: passed for `src` and `scripts`.
- Patch whitespace validation: passed.
- Matrix dry-run: all eight commands generated.
- Real matrix launch on this host: rejected before training because CUDA is unavailable.
- CPU multilingual end-to-end smoke: passed, including saved-checkpoint reload.
- Saved-checkpoint re-score: 18/18 completed; zero metric changes and zero tied queries.

## Immediate handoff

Once the full manifest build has completed and a CUDA environment is available:

```bash
.venv/bin/python scripts/training/run_contrastive_matrix.py \
  --experiment 00-projection-full-mean-512
```

After reviewing the baseline's learning curves and per-query validation failures, run the remaining experiments one change at a time. Do not evaluate the locked test set until the validation-based selection is final. Then run:

```bash
.venv/bin/python scripts/evaluation/finalize_contrastive_matrix.py \
  --evaluate-locked-test
```

Before interpreting retrieval as useful, add predefined downstream pathology probes, subgroup/failure analysis, and an external cohort. The multilingual encoder choice and improved retrieval mechanics do not themselves establish clinical validity.
