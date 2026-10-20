# Updated translation baseline and input audit

The separately versioned `dfd07db857fceb55ac20269a7dc693fd5a390e53` baseline uses
all 4,407 studies with unchanged image metadata, splits, seed and hyperparameters.
It retains 1,736 source-English reports and uses 2,671 Claude translations. The
manifest hash is `7479e2194d1ab3f3c219dc8137ce5e0a9cb40d8658f20b27e56d57866749a21c`.
The full 20-epoch updated baseline completed: best epoch 18, validation mean
Recall@1 10.2041%, exact checkpoint reload verified. The quarantined full-text
baseline also completed: best epoch 19, mean Recall@1 10.6576%. The impression-first
comparison completed at 2026-09-07 06:01 Asia/Taipei: best epoch 19, mean R@1
9.6372%, exact reload verified. Retain full text; the -1.0204 percentage-point
difference did not trigger repeat seeds. See the
[completed comparison and candidate error review](claude-translated-dfd07db-comparison-results.md). Consult the
[local handoff notes](handoff/) and execution status files for live state.
Test remains locked.

## Audit findings

[Input audit](claude-translated-dfd07db-input-audit.json) covers all 3,967 training
and validation studies, with no test pixels, report examples or labels inspected.

- 464/3,526 training reports and 67/441 validation reports exceed 512 tokens.
- 300 training and 43 validation reports have their final explicit English
  impression/conclusion heading entirely beyond the model's text cutoff.
- Exact cached tensors identify one duplicate pair, both in training. Their
  reports differ; one says LEFT while its manifest says R. Source verification confirms identical ordered decoded pixels in all 87 slices
  across the three selected series; fresh preprocessing matches both caches.
  Both DICOM studies are tagged R. See the [verification](claude-translated-dfd07db-duplicate-verification.json).
- Pooled 16x16 image cosine screening at >=0.98 finds only that same pair. This is
  a limited screening heuristic, not proof that other images/patients are independent.
- Existing translated-corpus audit finds 20 exact report groups spanning splits
  (78 studies), versus 22 groups/118 studies for the original source reports.
  Translation wording variation can conceal repeated source reports.

## Previous translated checkpoint error review

Read all 20 worst direction/query pairs from the `9bec1bd` checkpoint packet.
These remain its original reports/rankings; they are not relabeled as results of
training on the updated corpus. None has an identical normalized competing report.
Packet indices below refer to the saved JSON in that run's parent directory.

| Pairs | Provisional report-level observation |
| --- | --- |
| 1, 12 | Medial meniscal/ACL/MCL findings compete with lateral meniscal disease or differing ACL status. |
| 2, 11 | Leading tibial/pes projection is not reproduced by the competitor; this outlier is only 418 tokens. |
| 3 | Meniscal tearing/chondrosis competes with a report denying those findings. |
| 4 | Medial meniscal tear competes with normal menisci and different ligament findings. |
| 5, 13 | Source-English report states medial meniscal tear in findings but a boilerplate no-other-significant-abnormality line in conclusion; competitor denies meniscal tear. |
| 6 | Patellofemoral chondropathy with no meniscal tear competes with medial meniscal tearing. |
| 7 | Shared normal structures, but cartilage, effusion and fat-pad findings differ. |
| 8, 19 | Reciprocal competitors share injury vocabulary but differ in fracture site, patellar/MPFL and meniscal findings. |
| 9 | Patellar dislocation with normal menisci competes with meniscal root tearing/degeneration. |
| 10, 14, 15 | Partial meniscal/degenerative overlap; extent and morphology differ. Not established as interchangeable positives. |
| 16 | Same long template and medial degeneration, but competitor adds stress fracture. |
| 17 | Acute ACL/pivot-shift pattern competes with degeneration and suspected meniscal tearing. |
| 18 | Similar template, but advanced cartilage damage competes with explicitly intact cartilage. |
| 20 | Long templates with opposing meniscal findings and different bone findings. |

Four packet entries (three unique queries) exceed 512 tokens. The longest
reports demonstrably lose their final impression, but most worst queries are
short. Report-to-image assessment uses the candidate image's paired report as
a proxy, not an image interpretation. No clinical adjudication or target relabeling
was performed. The existing section extractor also misses inline headings and
can include later sections in an impression-first source report; do not blindly
launch its impression-only setting as a validated clinical-content extraction.

## Decision and controlled comparison protocol

Keep the old and updated full-cohort baselines as immutable exploratory anchors.
Retain the exact-pair, uniform-expected-tie-credit evaluation policy. Do not merge
report duplicates into positives: equal/template reports do not establish image
or clinical equivalence. No negative-pool or objective changes are justified yet.

Both conflicting entries are quarantined in `claude-translated-dfd07db-q2-v1`
by setting training eligibility false. All reports and split assignments remain
unchanged, with 3,524 eligible training studies, 441 validation and 440 test.
A fresh matched baseline and ordering comparison use exactly this cohort.
This resolves the confirmed conflict for exploratory comparisons; it does not
resolve all possible duplication or establish patient independence.
The user confirms that the competition provides no reliable cross-study patient
identity. Existing local `patient_group_id` values are implementation metadata,
not evidence of patient-disjoint splits. The user identifies the 58 ground-truth
labels as the exception; labels outside those 58 are not independently annotated
from images. Those other labels remain weak supervision and cannot serve as
independent diagnostic evaluation targets. The 58-case set still has only eight
validation cases under this split, with zero positive validation MCL cases.

The first bounded exploratory comparison uses a deterministic impression-first
full-report ordering at the same 512-token limit against the full-report baseline.
Preserve source text and all words; move a confidently identified late impression
forward, retaining full text otherwise. Validate extraction on inline, multiline,
impression-first and heading-free examples before training. This tests visibility
of the summary without simultaneously changing encoders, pooling or negative count.
It does not correct inconsistent source reports or guarantee all findings fit.

Use seed 2026 for the first matched comparison, 20 epochs and the existing
checkpoint rule (mean R@1 descending, validation loss ascending, earliest epoch).
If the candidate improves mean validation R@1, repeat both configurations with
seeds 2027 and 2028; report every run, the paired query difference and seed spread.
Do not promote on one selected validation score. After report-input comparison,
compare last-image-block-only and last-text-block-only adaptation separately,
with both-block adaptation conditional on those results. Locked-test evaluation
requires a finalized selection and is not part of this continuation.

## Execution and validation

`checkpoints/contrastive/claude-translated-dfd07db-comparison-execution/run.py`
waits for the full-cohort anchor and verifies its 20-epoch history, selection and
exact reload. It then runs an impression-first smoke, the quarantined full-report
baseline, and the matched impression-first experiment. Only a positive seed-2026
R@1 difference triggers repeats of both configurations at seeds 2027 and 2028.
Maximum: six full comparison runs plus the smoke. No automatic unfreezing, test
evaluation, commit or push. Failures stop the queue and are recorded in status.json.

The code/config snapshot is fixed before queue launch. Each run has separate
telemetry, logs, compact checkpoints and validation reload. Results are written
to `digest/claude-translated-dfd07db-results.md` after the reference run and
`digest/claude-translated-dfd07db-comparison-results.md` during the comparison.

The new ordering has unit coverage for inline/plural headings, heading-free and
impression-first input, ambiguous repeated headings, and preservation of negation
and inconsistent findings. Full CPU suite: 37 passed, one CUDA-only test skipped.
The cohort-level audit checks exact word-multiset preservation for all 3,965
eligible train/validation reports. 2,046 training and 254 validation inputs change.
292/300 late training summaries and 42/43 late validation summaries move forward;
ambiguous/unrecognized headings retain their full original text. This only proves
heading placement, not full summary fit or clinical completeness after truncation.

Report duplicates retain the existing single-positive objective and exact-pair
metric in both arms. This is a declared exploratory policy, not proof that
duplicate reports are independent. No downstream label fitting is included.
Unfreezing stays conditional on review of these results rather than running the
entire architecture matrix unattended.

## User-confirmed provenance

The user clarified during this continuation that the competition has no
PatientID column/tag supplied for tracking patients across scans. Earlier notes
about populated local DICOM tags must not be interpreted as reliable competition
patient identifiers. The two identical-image studies with different observed
group identifiers reinforce why these groups do not prove independence.

The user identifies 58 ground-truth cases; all other pathology labels are not
independently image-annotated. No downstream fitting is launched. A future probe
must distinguish those 58 cases from weak labels, account for tiny per-class
support, and document the ground-truth annotation protocol when available.

## Completed comparison review — 2026-09-07

Verified both histories, selected checkpoint hashes/reloads, 46 snapshot hashes,
matched configs, telemetry and paired rankings. All 20 worst candidate pairs
received report-level review. Long-report R@1 fell from 9.7015% to 7.4627%;
summary-heading recovery did not yield an observed gain. Retain full text for
image-only final-block adaptation, selected as the next isolated experiment but
not launched. Descriptive paired interval [-3.0612, 1.0204] pp includes zero;
no repeat-seed stability or downstream diagnostic claim follows. Both runs log
an infinite gradient norm at epoch 2 under AMP; skipped-step counts were not
recorded. Later norms and exact reload metrics are finite. Full details and
reproduction are in the linked comparison review.
