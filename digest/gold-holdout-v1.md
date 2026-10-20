# Does the 0.7105 gold-probe AUROC survive contact with unexposed data?

2026-09-14. Every element pre-registered before the run that tested it — including the control's
reading, fixed before the control landed. The locked test split was never loaded.

**Answer in one line: yes, mostly.** The gold probe was exposure-inflated by **≈0.0165** of macro
AUROC — about a quarter of the measured gain — and the remaining **+0.0500** over unadapted DINOv2
is real on 56 studies the encoder has never seen.

**Artifacts.** Authoritative (four arms, one pass):
[`downstream-probe-unexposed-v2.json`](downstream-probe-unexposed-v2.json) +
[`gold-holdout-comparisons-v2.json`](gold-holdout-comparisons-v2.json), under
`configs/downstream/gold-probe-unexposed-v2.json`. The v1 pair is retained as the record of the
primary result before the control existed; v2 reproduces it exactly. Runs:
`checkpoints/contrastive/{gold-holdout-v1/holdout,datasize-control-v1/control}-unfreeze4-seed{2026,2027,2028}`.
Six runs, ~22 min each. Protocol: [`gold-holdout-probe-protocol.md`](gold-holdout-probe-protocol.md).

## Why this experiment existed

Every gold-probe figure this project has published was a **development statistic by construction**:
56 of the 58 gold studies were training pairs (48) or in the checkpoint-selection split (8), and the
[matched-N audit](matched-n-loss-audit.md) then showed the encoder retrieves its own training pairs
**3.0×** better than unseen ones. Together those said 0.7105 *might* be substantially exposure
inflation — but no unexposed gold data existed to say by how much.

This experiment made some: all 56 usable gold studies were held out of **both** representation
training and checkpoint selection (cohort `contrastive-cohort-gold-holdout-v1.jsonl`, sha
`d1043376…`, 3,526/441 → 3,478/433, test split byte-identical), the 4-block matrix was retrained at
three matched seeds, and both arms were scored **in one probe pass** so they share the same 56
studies, the same folds and the same bootstrap resamples.

## Integrity

The exposed arm re-scored here reproduces the published headline **exactly** — 0.7105 / 0.7039 /
0.7173, mean **0.7106** against the published **0.7105**. Same folds, same features, same resamples.
The probe population resolves to exactly 56 studies with 231 positives (the `drop.csv` blank-template
trap avoided), and both arm keys were asserted present in the output before any number was read.

## Result 1 — the primary estimand

Pooled paired difference in macro out-of-fold AUROC, unexposed minus exposed, bootstrapped over
studies on the **per-study difference** with shared resample indices:

| Seed | exposed | unexposed | paired difference | 95% CI | P(unexposed higher) |
| --- | --- | --- | --- | --- | --- |
| 2026 | 0.7105 | 0.6970 | **−0.0135** | [−0.0295, +0.0033] | 0.052 |
| 2027 | 0.7039 | 0.6825 | **−0.0212** | [−0.0384, −0.0046] | 0.006 |
| 2028 | 0.7173 | 0.7018 | **−0.0157** | [−0.0338, +0.0017] | 0.040 |
| **pooled** | **0.7106** | **0.6938** | **−0.0168** | per-seed sd 0.0039 | — |

**Same sign in all three seeds, tightly clustered, and the bootstrap favours the exposed arm in
95–99% of resamples in every seed.** One of three per-seed CIs excludes zero — which is why the
pooled paired difference, not a per-seed rule, was pre-registered as the primary estimand: per-seed
intervals on a 56-study probe are ~±0.02 wide and could not have resolved an effect this size,
exactly the failure the plane-pooling session recorded.

## Result 2 — contrastive adaptation still works on data it never saw

| Contrast | pooled | per-seed CIs excluding zero |
| --- | --- | --- |
| **unexposed** vs unadapted DINOv2 (0.6438) | **+0.0503** | **3 of 3** |
| exposed vs unadapted DINOv2 (published) | +0.0668 | 3 of 3 |

Per seed: **+0.0535** [+0.0180, +0.0914], **+0.0391** [+0.0001, +0.0823], **+0.0583**
[+0.0183, +0.1008].

**This is the most important number on the page.** The adaptation gain is not an artifact of
exposure. On 56 studies the encoder has never seen in any capacity, 4-block contrastive adaptation
still beats unadapted DINOv2 by **+0.0503**, and every seed's interval excludes zero.

## Two kinds of number on this page, and they differ in the third decimal

**Point-estimate levels are the arithmetic spine:** unadapted **0.6438**, unexposed **0.6938**,
exposed **0.7106**. Those subtract exactly: +0.0500 unexposed gain, −0.0168 exposed-minus-unexposed,
+0.0668 exposed gain.

**Bootstrap means are the interval-bearing estimates** and are computed over 5,000 resamples, so they
differ from the level differences by ~0.0003 by construction — the +0.0503 in Result 2 is the
bootstrap mean of the same quantity the levels put at +0.0500. Neither is wrong; do not mix them
inside one subtraction.

## What this does and does not establish

1. **The headline was inflated, but modestly.** On the point-estimate levels, exposure-plus-data-loss
   accounts for **−0.0168 of the +0.0668** measured gain — about **25%**. Three quarters of it
   survives on unexposed studies.
2. **The data-size confound is measured and negligible — so −0.0168 is essentially all exposure.**
   The holdout arm changed two things at once: it lost the gold studies' exposure **and** trained on
   56 fewer studies. The protocol required a control if the arm dropped, and it dropped, so the
   control ran: 48 train + 8 validation *non-gold* studies dropped at matched counts, gold studies
   left in place (`contrastive-cohort-datasize-control-v1.jsonl`, sha `8798fccb…`). All four arms
   were then scored in **one** probe pass. See Result 3.

3. **The project now has a gold-probe number that is an estimate of generalisation** —
   **0.6938** — rather than a readout on studies the encoder trained on. That is a first.
4. **It is still study-level, not patient-level.** De-identification re-randomised every identity
   tag, so patient-disjointness is unverifiable. And it is still 56 studies: the levels carry wide
   intervals; the paired construction is what makes the *difference* estimable.
5. **One residual exposure channel, measured: 1 study.** Exactly one study remaining in training
   carries a `report_sha256` identical to a held-out gold report, so that report text was still seen
   paired with a different patient's images. One of 3,478, detected by exact hash.

## Result 3 — the decomposition, with the control

All contrasts paired on the same 56 studies, folds and resamples. **Integrity re-asserted after
re-extraction:** the exposed arm again gives 0.7105 / 0.7039 / 0.7173 and the unexposed arm again
gives 0.6970 / 0.6825 / 0.7018 — identical to the v1 pass.

| Arm | seed 2026 | 2027 | 2028 | mean |
| --- | --- | --- | --- | --- |
| unadapted DINOv2 | — | — | — | 0.6438 |
| **exposed** (published) | 0.7105 | 0.7039 | 0.7173 | **0.7106** |
| **control** (56 fewer studies, exposure kept) | 0.7180 | 0.7049 | 0.7071 | **0.7100** |
| **unexposed** (gold held out) | 0.6970 | 0.6825 | 0.7018 | **0.6938** |

| Paired contrast | pooled | per-seed sd | CIs excluding 0 |
| --- | --- | --- | --- |
| control − exposed — *cost of 56 fewer studies* | **−0.0005** | 0.0092 | **0 of 3** (signs flip: +0.0077, +0.0012, −0.0104) |
| unexposed − exposed — *exposure + data loss* | **−0.0168** | 0.0039 | 1 of 3 |
| **unexposed − control — *exposure alone, data size matched*** | **−0.0163** | 0.0096 | **2 of 3** |

**The pre-registered reading was "control − exposed ≈ 0 ⇒ the deficit is essentially all exposure",
and that is what landed.** Dropping 56 training studies costs nothing measurable (−0.0005, signs
flipping across seeds). Two independent routes to the exposure estimate — against the exposed arm
(−0.0168) and against the data-size-matched control (−0.0163) — **agree to 0.0005**.

**Exposure inflation is therefore ≈ −0.0165 of macro AUROC**, and the −1.4% training-set reduction
explains none of it.

## Consequences for the rest of the repository

- **Every existing probe figure remains a development statistic** and its caveat stands. What changes
  is that the caveat now has a measured size rather than an unknown one.
- **0.7105 should not be quoted as the project's headline representation-quality number without
  0.6938 beside it.** The first is a readout on trained-on studies; the second is an estimate of
  generalisation.
- **Selection remains retrieval R@1 and nothing was selected on label AUROC**, here or anywhere.

## Scope

Validation R@1 for these runs (0.1409 / 0.1270 / 0.1132) is over **433** candidates, not 441. It is
**not** comparable to the published 0.1383 / 0.1434 / 0.1202 and must never share a table with them —
see [the matched-N audit](matched-n-loss-audit.md). AUROC figures here are the 56-study 12-label
pooled out-of-fold probe and are not comparable to any Recall@1 `pp` figure in `digest/`.
