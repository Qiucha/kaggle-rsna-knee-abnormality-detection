# Full projection baseline results — 2026-09-05

The frozen full-cohort projection baseline completed all 20 epochs on the local
RTX 3070. Epoch 20 wins the predeclared validation checkpoint rule. Reloading the
compact checkpoint reproduces every saved validation metric exactly. The locked
test split was not evaluated.

## Retrieval on 441 validation studies

| Metric | Image → report | Report → image | Random ranking |
| --- | ---: | ---: | ---: |
| Recall@1 | 10.20% | 11.11% | 0.227% |
| Recall@5 | 31.52% | 34.01% | 1.134% |
| Recall@10 | 56.01% | 56.24% | 2.268% |
| Mean rank | 16.05 | 15.67 | 221 |
| Median rank | 9 | 9 | 221 |

Mean bidirectional Recall@1 is **10.66%**. The existing per-study bootstrap gives
a descriptive 95% interval of 8.28–13.15%. This is validation data used to select
an epoch; it is not an unbiased locked-test estimate and does not account for
unknown repeated patients. Expected top-1 match counts are 45/441 and 49/441.
Five image→report queries have tied positive candidates; retrieval uses expected
credit for ties. No report→image positive ties were recorded.

## Curves and errors

[Learning curves](full-projection-v1-curves.html) show train loss falling from
2.659 to 0.602 and full-validation loss from 4.531 to 3.699. The losses use different
candidate counts (32 training, 441 validation), so their absolute gap is not a
direct generalization-gap measurement. Validation loss reached its minimum of
3.671 at epoch 17, while the primary Recall@1 metric continued improving to epoch 20.
No epochs beyond the planned 20 were added.

Inverse temperature increased from 14.46 to 18.27, well below its clamp of 100.
Mean embedding-dimension standard deviations at the selected epoch are 0.0601
(image) and 0.0593 (text); these aggregate diagnostics show no obvious embedding
collapse. Mean paired similarity is 0.6068 versus mean unpaired similarity 0.0514.
Paired similarity spans -0.0738 to 0.8266, with median 0.6308 and 10th percentile
0.4468. Among incorrect top-1 results, median top-candidate similarity is 0.7040
(image→report) and 0.7211 (report→image): many errors still have strong competing
similarities. This alone cannot distinguish semantic equivalents from true errors.

Worst positive ranks are 303 and 325 respectively. The review JSON saves the ten
worst queries in each direction, with query/candidate IDs, positive and top scores,
and expected ranks. Full per-query rankings are also saved. These are a quantitative
error review; clinical/semantic review of the competing reports remains pending.
The similarity summaries cover positive and top-candidate distributions plus the
aggregate unpaired mean, not the entire unpaired-score distribution.

**Decision: continue to the controlled-comparison milestone after its leakage and
report-error review gate.** The result supports measurable pair-retrieval learning
and retaining this baseline as the comparison anchor. It does not establish a
winning architecture, patient independence, or pathology prediction utility.
Do not launch the remaining matrix or use the locked test automatically.

## Runtime and reproducibility

- Training: 2026-09-05 10:53:39–12:50:21 UTC, 7,002.55 seconds / 116.71 minutes.
  This includes initial file fingerprinting/preprocessing; it excludes the later
  checkpoint reload. The initial scan over the Windows-mounted dataset was slow.
- Peak PyTorch allocated memory: 1,941,709,312 bytes / 1.808 GiB.
  Peak reserved memory: 2,774,532,096 bytes / 2.584 GiB. These are allocator peaks,
  not total device memory across all applications.
- RTX 3070 8 GB; Python 3.13.12; torch 2.13.0+cu126; torchvision 0.28.0+cu126.
- 3,526 training studies, 441 validation, 440 locked test; batch 32, 31 in-batch
  negatives per query, 110 updates/epoch, final six training samples dropped per
  shuffled epoch, 2,200 scheduled updates. Both backbones frozen; seed 2026.
- Baseline checkpoint SHA-256:
  `dca40bb0f6b71b291f1562d6e51d2fcb25c8a0b65441e7c586ddd80ee12fdd5e`.
- Cohort SHA-256:
  `54bd803e23acf3299daeb815d651a34249c736689e20d790f862fc71c0520b63`.
- Git base: `baeb1525dfe14e0d1ae6d540398d6903f20b492f`; follow-up changes are
  intentionally uncommitted per user instruction. Actual training source-tree hash:
  `693748e682423ef0a40f7e0755785ac69e54157952663e6c8d3c3b0f972832f1`.
  A hashed source snapshot preserves the actual uncommitted code.
- 27 tests passed with CUDA before launching the full baseline. Dependencies passed
  `pip check`; all 13 inventoried data/model artifacts matched their hashes.

Run artifacts: `checkpoints/contrastive/full-multilingual-v1/00-projection-full-mean-512/`
contains config, run manifest, 20-row metrics history, best/last compact checkpoints,
`reload-validation.json`, `rankings-validation-best.jsonl`, and `baseline-review.json`.
Execution artifacts: `checkpoints/contrastive/cuda-execution-20260905/` contains
runtime/memory telemetry, logs, driver/dependency records, source snapshot, and
preserved launch/review scripts. These local artifacts are ignored by Git.

## Continuation data review

The [data/report audit](full-projection-v1-data-review.md) found 22 exact-report
groups spanning splits (118 studies). All 20 worst pairs received a provisional
text comparison; image/clinical adjudication remains pending. Seventy validation
reports exceed 512 tokens. Keep the pre-tuning gate open for image duplicate
checks, report-duplicate handling and provenance; the baseline metrics are unchanged.
