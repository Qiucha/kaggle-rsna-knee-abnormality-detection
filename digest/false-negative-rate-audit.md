# Are there false negatives to cancel? A gate on loss-function redesign

**Read-only. No training was run, nothing was launched, the test split was not touched.**
Evidence: [`digest/false-negative-rate-audit.json`](false-negative-rate-audit.json).

## The question

The training objective is CLIP-style symmetric InfoNCE over a batch of 32
(`model.symmetric_contrastive_loss`). Every off-diagonal entry is treated as a negative. Knee MRI
reports are templated, so if a batch contains two studies whose reports say the same thing, the
loss actively pushes them apart — a *false negative*. Two published remedies were on the table:
cancelling such negatives (drop them from the denominator) and replacing the one-hot target with a
graded similarity target.

Both need one thing first: **a similarity source that identifies genuinely equivalent reports, and
a rate high enough that a batch of 32 actually contains them.** This note measures that, on the
3,526 training reports of `data/processed/contrastive-cohort.jsonl` (6,214,575 pairs), before any
implementation.

## Result 1 — the frozen text tower's own similarity does not work

Reports embedded exactly as the arms would see them (frozen `multilingual-e5-base`, `query: `
prefix, mean pooling, 512 tokens). The similarity is **dominated by the shared structured
template, not by the findings**:

| Percentile of all pairs | 50 | 90 | 99 | 99.9 |
| --- | --- | --- | --- | --- |
| cosine | 0.889 | 0.937 | 0.975 | 0.993 |

The distribution is compressed so high that a threshold must sit near the ceiling, and even there
it is not selective. Read at the cut, pairs at **0.991** are clinically unrelated:

> **A** — `[report excerpt removed]`
> **B** — `[report excerpt removed]`

Only above ~0.995 are pairs genuinely equivalent, and that is **0.19 expected collisions per batch
of 32**. The rate that *would* fire often enough (0.99 → 1.14 per batch) fires on unrelated
findings. **This source is unusable as a false-negative detector**, which also matters beyond this
note: it is the same embedding the retrieval R@1 selection metric ranks with.

## Result 2 — TF-IDF works, and says the duplicates are rare

Document frequency discounts the template by construction. TF-IDF cosine (sublinear, 1–2 grams,
`min_df=3`, `max_df=0.5`) behaves correctly on inspection: pairs above 0.90 are genuinely
equivalent, pairs at 0.80 are not.

> **0.905** — two reports that are byte-identical through `Conclusion: Normal study.`
> **0.837** — `[report excerpt removed]` vs `[report excerpt removed]` — *not* equivalent.

At the cut where pairs are real, they are rare:

| TF-IDF cut | 0.80 | 0.85 | **0.90** | 0.95 |
| --- | --- | --- | --- | --- |
| expected collisions per batch of 32 | 0.24 | 0.14 | **0.10** | 0.08 |
| batches containing at least one | 22% | 13% | **9%** | 7% |
| training studies with any partner | 15.5% | 11.1% | **8.1%** | 5.9% |

## Result 3 — graded soft targets fail for the same reason

The obvious fallback is to drop the hard threshold and use a graded target: replace the one-hot
diagonal with a distribution built from the similarity matrix. That only helps if the off-diagonal
target mass lands **on the genuine duplicates**. It does not. Over 600 random batches of 32, the
share of all off-diagonal target mass that reaches a TF-IDF > 0.90 pair:

| Target construction | off-diagonal mass | share of it on genuine duplicates |
| --- | --- | --- |
| normalised TF-IDF, power 1 | 46.5% | **0.4%** |
| normalised TF-IDF, power 2 | 18.6% | **1.4%** |
| normalised TF-IDF, power 3 | 8.5% | **2.9%** |
| softmax, τ=0.2 | 23.1% | **0.8%** |
| softmax, τ=0.15 | 7.6% | **3.1%** |
| softmax, τ=0.1 | 1.2% | 21.7% |

Sharpening does not fix this, it trades one failure for another. Loose settings put a quarter to a
half of the target mass on true negatives. The one setting that aims well, τ=0.1, moves
0.0122 × 0.217 = **0.26% of total target mass** onto real duplicates — an intervention too small to
change a three-seed result. In between, the mass is smeared across all 31 in-batch negatives
regardless of content: that is **uniform label smoothing wearing a similarity matrix**, and the
9%-of-batches duplicate rate from Result 2 is why. In ~91% of batches there is no duplicate to
credit, so every unit of off-diagonal mass necessarily lands on a true negative.

## Amended 2026-09-13

[`digest/duplicate-confusion-audit.md`](duplicate-confusion-audit.md) revisits this note and
**amends one of its two objections**:

- The *rate* finding below **stands** and is firmer — counting byte-identical reports rather than a
  TF-IDF cut gives 0.057 exact-duplicate pairs per batch of 32, in 4.9% of batches.
- The *detector* finding below is **void**. This note searched for a similarity detector and never
  tried exact report identity. `report_sha256` is already in the manifest, needs no threshold, and has
  no false positives. It also shows the duplicates are **concentrated, not diffuse** — 77.9% of the
  720 exact training pairs lie inside one group of 34 normal knees — and that the rate reaches 0.94
  pairs per batch in 46% of batches at batch 128.

Read the two together before re-proposing or re-closing this lever.

## Verdict

**The loss-redesign lever fails at batch size 32, in both forms, and neither was implemented.**

- *Cancellation* needs a detector. The only source that identifies genuine duplicates (TF-IDF)
  finds roughly **one per ten batches — one pair in 496**. The source dense enough to fire often
  (the text tower's own embedding) fires on unrelated findings.
- *Graded targets* need the mass to land on duplicates. **At most 3% of it does** at any setting
  that is large enough to matter.

Both failures trace to the same fact: **at batch 32 there is almost nothing to fix.** Six GPU runs
would have measured noise.

This is a gate outcome, not a failed experiment: it was pre-registered as the condition for
launching, it cost no GPU time, and it rules the lever out on evidence rather than on taste.

## What this does and does not rule out

- **Ruled out:** false-negative handling, in either form, **at the current batch size**.
- **Not ruled out — batch size is the binding constraint, not target shape.** Collisions scale with
  the pair count, so batch 64 quadruples the duplicate rate and batch 128 multiplies it by
  sixteen. Nothing here was measured at any batch but 32. If the batch ever grows for another
  reason, this gate should be re-run rather than assumed.
- **Independent and untouched:** the truncation/positional-limit lever. Nothing here bears on it.

## Caveats

1. The equivalence judgements are **the author's reading of sampled pairs**, not an adjudicated
   label. They are reproducible from the saved embeddings but are not a gold standard.
2. TF-IDF is lexical. Two reports saying the same thing in different languages — this cohort is
   multilingual — score near zero. The true duplicate rate is therefore a **lower bound**, though
   the e5 result shows the multilingual detector available here does not recover the difference.
3. Measured on the **training split only**, which is the split the loss sees.
