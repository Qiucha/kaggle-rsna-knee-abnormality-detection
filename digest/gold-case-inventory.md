# Gold-case inventory: provenance, exposure, and what 58 studies can support

Read-only audit, 2026-09-11 (Asia/Taipei). Produces the three pieces of evidence
[the response §9](knee-clip-feedback-response.md) requires before any downstream number
is interpretable — provenance, exposure, per-label feasibility — and proposes the
fourth, an estimand and fold plan, **for sign-off rather than as a decision**.

Nothing is fitted. No probe was trained, no representation was scored, no label was
selected on. Evidence: [`gold-case-inventory.json`](gold-case-inventory.json). Script:
`scripts/audit_gold_case_inventory.py`.

**Test gate.** Gold studies in the locked test are counted and nothing more: their
labels are not tabulated, their text is not compared, and they are excluded from every
feasibility count below.

## The finding that governs everything else

| Gold studies | 58 |
| --- | --- |
| In the train split, `training_eligible` | **48** |
| In the validation split | **8** |
| In the locked test split | **2** |
| Quarantined or otherwise ineligible | 0 |
| Absent from the cohort | 0 |

All 58 are in the cohort, and **56 of them have already been used.** The 48 train cases
were image–report training pairs in every completed contrastive run on this cohort. The
8 validation cases are inside the split that selected every checkpoint and decided every
matrix. Holding a case out of a later probe does not undo either exposure.

That leaves **two gold studies untouched by representation learning, and both are in
the locked test.** There is no unexposed, usable gold data. Any probe fitted on these
cases measures how well a representation that has already seen them can be read out —
a development statistic. It is not, and cannot be made into, an estimate of
generalisation to new patients. The response says this; the inventory now shows the
exact size of the problem, which is that the exposure is essentially total.

## Provenance: three files, one label set, one trap

| File | sha256 (first 16) | Studies | Populated label cells |
| --- | --- | --- | --- |
| `data/cleaned.csv` | `f4c2798f3207f27a` | 58 | 696 |
| `data/translation/gold_translated.csv` | `f111d544a9afea18` | 58 | 696 |
| `data/drop.csv` | `a7ea513ece555746` | 58 | **0** |

All three cover the same 58 studies. `cleaned.csv` and `gold_translated.csv` carry
**identical labels in all 696 cells**, so there is one gold label set, not two competing
ones. `drop.csv` carries the same 58 studies and the same reports with every label
column empty — it is the blank annotation template.

That last file is a trap worth naming. It has the expected shape, the expected columns
and the expected study list; read as labels it silently yields 58 studies with no
positives on any of the 12 labels. Any future probe code must load `cleaned.csv` (or
the gold translation) and assert a non-zero positive count before fitting.

Labels are binary 0/1 with **no missing cells**. There is no "unknown" or "uncertain"
code, so the uncertainty-aware label contract the feedback asks for has nothing to read
in this file. Absence of an unknown code is not evidence that every case was
adjudicable; it means the encoding cannot express that it was not.

## Text exposure: same documents, different translations

Is the text the model trained on the text the labels were adjudicated against? For the
56 non-test gold studies:

| | Studies |
| --- | --- |
| Byte-identical after whitespace normalisation | 27 |
| Differ, but the **source report is identical** | **29** |
| Differ because the source report differs | **0** |

Every divergence is translation variance, not a provenance failure. The 27 matches are
the reports that were English at source. The 29 differences are non-English reports —
Spanish 9, Turkish 6, Croatian 4, Greek 3, Bulgarian 3, Dutch 2, German 2 — where the
gold file's translation and the cohort's `dfd07db` Claude translation are two renderings
of the same clinical document. For example, both renderings of one report describe
marginal amputation of the lateral meniscal body with no rupture, in slightly different
word order and wording.

So a label adjudicated from the gold translation describes the same findings the model
read, worded differently. This is a caveat on any text-derived target built from the
gold file, not a reason to distrust the pairing.

## Per-label feasibility

Train + validation gold studies only: **56 cases**. Thresholds declared before reading
the counts — fewer than 2 positives is not estimable at all; fewer than 5 cannot support
grouped folds.

| Label | Positives | Negatives | Prevalence | Train / validation positives | Verdict |
| --- | --- | --- | --- | --- | --- |
| Effusion | 33 | 23 | 0.59 | 27 / 6 | estimable |
| Medial Meniscus | 26 | 30 | 0.46 | 24 / 2 | estimable |
| Synovitis | 26 | 30 | 0.46 | 23 / 3 | estimable |
| ACL | 23 | 33 | 0.41 | 20 / 3 | estimable |
| Lateral Meniscus | 22 | 34 | 0.39 | 18 / 4 | estimable |
| PF OA | 20 | 36 | 0.36 | 17 / 3 | estimable |
| Contusion | 19 | 37 | 0.34 | 16 / 3 | estimable |
| Fracture | 17 | 39 | 0.30 | 15 / 2 | estimable |
| Medial OA | 14 | 42 | 0.25 | 12 / 2 | estimable |
| Baker's | 12 | 44 | 0.21 | 9 / 3 | estimable |
| Lateral OA | 11 | 45 | 0.20 | 7 / 4 | estimable |
| MCL | 8 | 48 | 0.14 | **8 / 0** | estimable |

Two checks on claims already in circulation:

- **"Only eight labelled validation cases, with zero positive MCL" reproduces exactly.**
  Eight gold studies are in validation, and MCL has zero positives among them.
- **Every one of the 12 labels is estimable over the pooled 56**, which the earlier
  framing did not make clear. The infeasibility is specific to the *existing retrieval
  validation split* — eight cases is too few for any per-label estimate — not to the
  gold set pooled. That distinction changes which protocols are available: a pooled
  out-of-fold design over 56 is arithmetically viable where scoring the 8-case
  validation slice is not.

Smallest cell is MCL at 8 positives. Under 8-fold cross-validation that is one positive
per held-out fold: AUROC is *defined* there but computed against a single positive, so
it is uninformative rather than undefined. Under any fold count above 8 it becomes
genuinely undefined for some folds. Either way the pooled out-of-fold construction is
the only usable option for MCL.

## Proposed contract — pending sign-off, nothing is authorised

This is a draft of the response's probe-contract table filled in with the numbers above.
**It is not a decision and licenses no fitting.** §9's closure criterion requires a
*signed-off* estimand, and §8's exploratory protocol must not be executed until this
inventory is accepted.

| Element | Proposal | Why, given the inventory |
| --- | --- | --- |
| Estimand | Readout quality of a frozen representation on a cohort it was trained on. Explicitly **not** generalisation. | 56 of 58 gold cases were exposed; the estimand has to say so. |
| Population | The 56 train+validation gold studies. Test membership stays a count. | Using the 2 test cases is a separate protocol decision. |
| Model | Per-label L2-regularised logistic probe, penalty fixed in advance, no threshold search, no MLP. | 3,072 weights against 56 cases; anything tuned is fitted to noise. |
| Features | One prespecified frozen representation set, identical folds across methods. | Comparability, and no feature selection on gold labels. |
| Folds | 8-fold stratified per label, stratification computed inside the fold loop. Grouped folds are not available: there is no patient grouping to preserve. | See [the integrity audit](integrity-coverage-audit.md) — every group is one study, and the 19 studies in acquisition-duplicate candidate groups are the only leakage candidates the cohort can express. |
| Preprocessing | Standardisation and any reduction fitted inside each training fold. | Leakage through the scaler is the usual silent failure. |
| Rare labels | MCL (8 positives) reported pooled-OOF only; fold-level AUROC declared uninformative for it in advance, and undefined above 8 folds. | One positive per held-out fold at 8 folds. |
| Predictions | Exactly one held-out score per study per label; repeats kept as a study-by-repeat table, never counted as extra studies. | |
| Metrics | Positive/negative counts beside every estimate; pooled out-of-fold AUROC; macro-average eligibility named explicitly. | A macro over 12 labels of wildly different support hides MCL. |
| Uncertainty | Bootstrap over studies, with the independence limitation stated: resampling studies is not resampling patients, because patients are not identifiable. | |
| Tuning | Every hyperparameter fixed before the first fit, recorded in the config, and the config frozen the way experiment matrices are. | |

**Open decisions that are the user's, not this document's:**

1. Whether the 2 test-split gold studies may be used, and under what separation.
2. Whether a development-only readout statistic is worth producing at all, given that no
   unexposed gold data exists to contrast it against.
3. Whether to seek additional independently adjudicated studies, which is the only path
   to a credible final estimate and is outside this repository.
