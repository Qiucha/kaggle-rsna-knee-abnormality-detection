# The 150mm crop does not help the twelve labels either: −0.0201 macro AUROC

Measured 2026-09-15 (Asia/Taipei). Pre-registered in
[`configs/downstream/gold-probe-crop150-v1.json`](../configs/downstream/gold-probe-crop150-v1.json),
whose `PRE_REGISTERED_READING` block fixed the reading of **every** outcome before the run.
Evidence: [`downstream-probe-crop150-v1.json`](downstream-probe-crop150-v1.json) ·
[`crop150-vs-crop130-comparisons-v1.json`](crop150-vs-crop130-comparisons-v1.json).

## The question

The 150mm crop was chosen on **measured tissue retention** — 95.7% against 86.1% at 130mm — not on a
predicted score, and retrieval fell (0.1340 → 0.1190). Retrieval and label readout have dissociated
twice in this project, so the retrieval drop did not settle the label question.
[`crop150-rebaseline-v1.md`](crop150-rebaseline-v1.md) left it explicitly open. This answers it.

## Result

Pooled paired difference in macro out-of-fold AUROC, crop150 minus crop130, bootstrapped over studies
on the **per-study difference** with shared resample indices — the same construction as
[`gold-holdout-v1.md`](gold-holdout-v1.md), so the two are directly comparable:

| Seed | crop130 | crop150 | paired difference | 95% CI | P(crop150 higher) |
| --- | --- | --- | --- | --- | --- |
| 2026 | 0.7105 | 0.6828 | **−0.0277** | [−0.0549, −0.0011] | 0.020 |
| 2027 | 0.7039 | 0.6840 | **−0.0198** | [−0.0523, +0.0110] | 0.108 |
| 2028 | 0.7173 | 0.7047 | **−0.0127** | [−0.0395, +0.0138] | 0.177 |
| **pooled** | **0.7106** | **0.6905** | **−0.0201** | per-seed sd 0.0075 | — |

**Same sign in all three seeds; the bootstrap favours crop130 in 82–98% of resamples in every seed.**
**One of three per-seed CIs excludes zero** — which is exactly why the pooled paired difference was
pre-registered as primary. Per-seed intervals on a 56-study probe run ~±0.03 here and cannot resolve
an effect this size; the gold-holdout result had the identical shape (1 of 3, pooled −0.0168).

**By the pre-registered reading this is the third branch: "crop150 below crop130".** The wider crop
costs **both** retrieval **and** label readout. Of the two mechanisms
[`crop150-rebaseline-v1.md`](crop150-rebaseline-v1.md) offered, **dilution** — the extra field being
disproportionately muscle and subcutaneous fat, adding patch tokens without discriminative content —
is now the better-supported one. Losing resolution on anatomy already in frame would be expected to
cost fine-grained retrieval; costing the *label* readout as well fits "the added field is not
carrying signal".

## The unadapted floors move too, and in the opposite direction

| unadapted DINOv2, no training at all | macro AUROC |
| --- | --- |
| at 130mm | 0.6438 |
| **at 150mm** | **0.6529** |

**The wider crop helps the *unadapted* backbone (+0.0091) while hurting the *adapted* one.** So the
150mm field does carry information a generic encoder can use — the deficit is not "there is nothing
out there". It is the adaptation that fails to exploit it, gaining **+0.0376** over its own floor
against crop130's **+0.0668**.

> **Do not difference those two gains.** They are measured against **different feature sets** —
> unadapted DINOv2 at 130mm and at 150mm are not a common reference — so `+0.0376` versus `+0.0668`
> is not a difference-in-differences and the 0.0292 gap between them is not an estimate of anything.
> The paired crop150-minus-crop130 row above is the estimand; these floors are context.

## Exposure — every level here is a development statistic

**Both arms trained on `contrastive-cohort.jsonl`**, which holds 48 gold studies in train and 8 in
validation. Every **level** on this page is therefore exposure-inflated by ≈0.0168
([`gold-holdout-v1.md`](gold-holdout-v1.md)) and **must never be quoted against the 0.6938
generalisation estimate**. The paired **difference** is a contrast between arms sharing the same
exposure and is **not** corrected by that figure.

## Integrity

**The crop130 arm reproduced the published levels exactly — 0.7105 / 0.7039 / 0.7173, pooling to
0.7106.** That is the check that matters here: it confirms running a two-preprocessing config did not
disturb the arm whose numbers are already on record.

## What this settles, and what it does not

1. **The 150mm crop is not adopted.** It costs retrieval (0.1340 → 0.1190) *and* label readout
   (−0.0201), and the convergence audit already established it was not simply under-trained
   ([`convergence-curves.html`](convergence-curves.html)): it reached its validation minimum at
   epoch 9.0 and then overfit. Three independent readouts agree.
2. **"Retain more tissue" is refuted as stated, not merely unsupported.** The retention argument
   assumed retained tissue is useful tissue; Otsu foreground counts thigh and calf muscle, which the
   audit flagged as an upper bound rather than an estimate. This is what that caveat looked like when
   it bound.
3. **It does not refute reading more of the study.** This changes the *field of view of one slice*.
   The untried direction remains the one that **adds** information — the 89.9% of slices and ~2.7
   series per study the pipeline never opens
   ([`acquisition-metadata-trust.md`](acquisition-metadata-trust.md)). The unadapted floor rising at
   150mm is weak evidence that more field does carry something.
4. **Selection stays retrieval R@1.** Nothing here was selected on label AUROC.

## Provenance

56 gold studies (48 train + 8 validation); the 2 locked-test gold studies are counted and dropped
before any pixel is read. Probe hyperparameters copied unchanged from `gold-probe-unexposed-v2`, so
construction is comparable to every earlier probe; nothing was tuned on these studies.

**This is the project's first probe whose arms do not share one preprocessing.** It required
`extract_features_by_preprocessing` in `evaluate_downstream_probe.py`: each arm is extracted at its
own `crop_mm`, from its own populated cache, against the unadapted floor at *its* crop. That change
was verified inert on single-preprocessing configs by reproducing `gold-probe-unexposed-v2`'s stored
integrity assertion exactly (exposed 0.7105/0.7039/0.7173, unexposed 0.6970/0.6825/0.7018,
floor 0.6438) — **re-verified after every edit to the scoring path**, not only the first.

Nothing committed; the test split was never loaded.
