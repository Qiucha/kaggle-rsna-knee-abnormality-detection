# Do duplicate reports explain the retrieval misses?

A gate on the selection metric, and an amendment to the false-negative audit.

**Read-only. No training was run, nothing was launched, the test split was not loaded.**
Evidence: `digest/duplicate-confusion-audit.json` (not public: contains report text), produced by
[`digest/handoff/analyze_duplicate_confusion.py`](handoff/analyze_duplicate_confusion.py).

## The question, and why it is not the one already answered

[`digest/false-negative-rate-audit.md`](false-negative-rate-audit.md) closed false-negative
handling because near-duplicate reports almost never co-occur **in a batch of 32** — one genuine
pair in 496. That verdict was about the *loss*.

The selection metric is a different object. Validation retrieval ranks each query against **all 441
studies in the split**, not against 31 batchmates, so duplicates that a batch never sees are
candidates here on every query. If the rank-1 winner is routinely a study whose report says the
same thing as the query's, then Recall@1 is penalising correct behaviour, and every lever in
[`digest/image-block-variants-v1.md`](image-block-variants-v1.md) was selected on a metric that
misreads its own successes. That is worth knowing before designing any objective around duplicates.

Measured on the three matched 4-block baseline runs
(`checkpoints/contrastive/image-block-depth-v1/original-unfreeze4-seed{2026,2027,2028}`), at the
selected checkpoint, on `data/processed/contrastive-cohort.jsonl`. All three were re-scored from
`best.pt` first and **reproduced every saved metric exactly** (`metric_deltas` all `0.0`), so the
rankings analysed here belong to the runs the headline numbers come from. TF-IDF settings are
copied from the earlier audit, which established that pairs above 0.90 are genuinely equivalent and
pairs at 0.80 are not; vocabulary is fit on the training split only.

## Result 1 — duplicate confusion is real, and enriched up to 130× over chance

Among misses, the TF-IDF similarity between the query's report and the rank-1 winner's report is far
above the background pair rate. Background: 0.095% of the 97,020 validation pairs exceed 0.80,
0.019% exceed 0.90.

| Run | direction | misses > 0.90 | share of misses | enrichment |
| --- | --- | --- | --- | --- |
| seed 2026 | image→text | 2 / 381 | 0.52% | 28× |
| seed 2026 | text→image | 8 / 379 | 2.11% | **114×** |
| seed 2027 | image→text | 0 / 381 | 0.00% | — |
| seed 2027 | text→image | 9 / 374 | 2.41% | **130×** |
| seed 2028 | image→text | 1 / 392 | 0.26% | 14× |
| seed 2028 | text→image | 8 / 384 | 2.08% | **112×** |

The effect is not a threshold artifact. At the top of the range the confused pairs are
**byte-identical reports** — one cluster of Hoffa fat-pad impingement studies recurs, three
separate studies whose reports match to the character:

> **Q** — `[report text removed]`
> **W** — *(identical)*

For a text query drawn from that cluster, no objective can ever put the right study first. The
signal the metric demands does not exist in the input.

## Result 2 — it is confined to one direction, in all three seeds

text→image is **4.0× and 8.2×** more duplicate-affected than image→text in seeds 2026 and 2028, and
in seed 2027 the comparison is 9 misses against 0. The sign is the same in every seed. The mild hubness in the rank-1 winners runs the same way: 177–212 distinct winners out
of 441 for text→image against 221–254 for image→text, with the top ten winners taking 16–23% of all
queries versus 11–15%, and a single image winning as many as 20 queries.

A mechanism accounts for the **duplicate** asymmetry. Two studies with identical reports produce
**identical text embeddings**, so a report query scores their two images almost equally and which one
wins is arbitrary. Reverse the direction and the two *images* are different patients, so an image
query still separates them even though both reports read alike. **Duplicate confusion in this cohort
is a text-side degeneracy, not a symmetric one.**

The hubness is **also** concentrated on text→image, but that is an observation and not explained by
the same mechanism: 5 duplicate studies cannot produce one image winning 20 of 441 queries. Hubness
is a property of the image embedding distribution and its cause is **not established here**. Both
findings share one practical consequence: `mean_r@1` averages the two directions and hides them.

## Result 3 — and it is far too small to matter

Crediting every such miss gives a duplicate-tolerant Recall@1. At the 0.90 cut it buys
**+0.000 to +0.020**, against a seed spread in the metric itself of 0.1202 → 0.1434 = **0.023**.
The whole correction fits inside the noise it would have to beat.

The clean version of the bound needs no threshold at all. Grouping by `report_sha256`, a text query
whose report is shared by *k* studies can score at most 1/*k*:

| Split | studies | sharing a report | groups | group sizes |
| --- | --- | --- | --- | --- |
| train | 3,526 | 136 (3.86%) | 36 | 34, 10, 9, 5, 5, 5, 4, 4, 3×4, 2×24 |
| **validation** | **441** | **5 (1.13%)** | **2** | **3, 2** |
| test | 440 | 8 (1.82%) | 4 | 2, 2, 2, 2 (not loaded; counted from the manifest) |

On validation that is **5 queries of 441** with a ceiling below 1.0 — the 3-group capped at 1/3, the
2-group at 1/2 — so exact duplicates cost **0.0068 of text→image Recall@1 in expectation**, against
a measured 0.14 and a gap to perfect of 0.86. **0.8% of the gap.** Stated the other way: even a model
that solved every resolvable query would score 0.9932, so the ceiling is not what is binding.

Note that this per-query averaging is the right bound for the *metric* but says nothing about how
duplicates are distributed. Result 4 measures that, and it does not follow the same pattern.

## Result 4 — the duplicates are one clinical class, and an exact detector already existed

Grouping the training split by `report_sha256` contradicts the picture of duplicates as diffusely
rare. **They are concentrated, and they are the normal studies.** The four largest groups:

| size | report |
| --- | --- |
| **34** | `[report text removed]` — a fully normal knee, Turkish |
| 10 | `[report text removed]` |
| 9 | `[report text removed]` |
| 5 | `[report text removed]` — normal, English structured |

The 34-study group is **not a de-identification artifact**: 34 distinct `patient_group_id`, 34
distinct `series_fingerprints`, 17 left and 17 right knees. It is 34 genuinely different normal knees
whose reports are the same template sentence. Concentration follows:

- **720 exact-duplicate pairs** in the training split, of which the single largest group supplies
  **561 — 77.9%.** The top three supply **89.2%.**
- So collisions are **bursty, not diffuse**: a batch that touches the normal cluster tends to draw
  several of its members at once.

Two consequences that change what is open.

**The detector problem is solved, and was solved all along.** The earlier audit's cancellation arm
died partly because no usable *similarity* detector existed — the text tower fired on unrelated
findings, TF-IDF was accurate but lexical. It never tried **exact report identity**, which is a
field already in the manifest, has no threshold to tune, and has **zero false positives** by
construction. That removes the audit's second blocking objection, though not its first.

**The first objection — rate — still holds at batch 32, and scales sharply:**

| batch | expected exact-duplicate pairs per batch | batches containing ≥ 1 |
| --- | --- | --- |
| **32** | **0.057** | **4.9%** |
| 64 | 0.234 | 16.8% |
| 128 | 0.942 | 46.3% |
| 256 | 3.782 | 86.1% |

Byte-identical text thus accounts for roughly **half** of the audit's measured TF-IDF > 0.90 rate
(0.10 per batch), and nearly all of that half comes from one cluster.

## Result 5 — what the misses actually are

The same measurement names the real bottleneck. The **median** miss has TF-IDF **0.275–0.327** to its
winner — clinically unrelated — and the median query has only **5 to 7** of 440 candidates above its
positive (mean 12–14). The positive is already in the **top ~1.5%** and is being edged out by a
handful of unrelated reports:

> **Q** — `[report text removed]`
> **W** — `[report text removed]`

Both describe meniscal tears; only the query has an ACL tear. This is a **fine-grained
discrimination failure at the very top of the ranking**, not a false-negative problem. Any further
loss work should be aimed there.

## Verdict

**Closed for the metric; reopened, with a sharper instrument, for the loss.**

- **For the selection metric, duplicates are settled.** The enrichment is real — 130× is not a
  rounding error — and the directional asymmetry is new, but the magnitude rules it out: 0.8% of the
  gap. A duplicate-tolerant Recall@1 should be logged as a diagnostic and **must not** be selected
  on, because it moves the number by less than a seed.
- **For the loss, the earlier verdict needs amending.** It closed the lever at batch 32 on rate and
  on the absence of a detector. The rate finding stands at batch 32 and is if anything firmer
  (4.9% of batches). But the detector objection does not: `report_sha256` is exact and free. What is
  now well-founded — and was not before — is a **multi-positive objective over exact report-identity
  groups at a larger batch**, where the groups are a real clinical equivalence class ("normal
  study") rather than a threshold artifact. At batch 128 that is 0.94 pairs per batch in 46% of
  batches, concentrated enough to produce gradient rather than noise.
- **This is a proposal, not a result.** Nothing was implemented and nothing was launched. It should
  be gated the same way, on the same pre-registration discipline, and it competes with Result 5
  below on merit — not because duplicates are the more interesting story.

## Caveats

1. **One depth, one cohort.** Measured at 4 unfrozen image blocks on the original-language cohort.
   The directional asymmetry is a property of the text tower and should hold, but was not measured
   at other depths.
2. **TF-IDF is lexical and this cohort is multilingual.** Two reports saying the same thing in
   different languages score near zero, so the duplicate rate is a **lower bound** — the same
   caveat the earlier audit carries. The `report_sha256` bound in Result 3 is exact but strictly
   narrower, catching only byte-identical text.
3. **Equivalence above the cut is the author's reading of sampled pairs**, not an adjudicated label.
   The byte-identical cases are not a judgement call; the 0.80–0.95 band is.
4. **Validation only.** This bears on the selection metric, which is a development statistic by
   construction. Nothing here touches the label probe or the locked test split.
5. **Rank-1 winners only.** Duplicates sitting at rank 2–10 are not counted, so Result 3's
   duplicate-tolerant figure understates a duplicate-tolerant R@5 or R@10. Selection is on R@1.
6. **Result 4's batch table is arithmetic, not a measurement.** The expected pair count per batch is
   exact under sampling without replacement, but the share of batches containing at least one pair
   treats the 36 groups as independent. At these sizes the error is small and in the conservative
   direction; it is not a substitute for re-running the earlier audit's gate at the new batch size,
   which remains required before anything is launched.
7. **Result 4 licenses a proposal, not a conclusion.** That a clean detector exists and that
   collisions become frequent at batch 128 says the intervention is now *measurable*. It does not say
   it helps. The earlier audit's warning still applies with full force: cancelling these negatives
   removes the hardest ones, and the shuffled-pairing control showed per-study instance discrimination
   is doing real work in this cohort, so a negative result is a live possibility.
8. **The normal-knee cluster may be a labelling story rather than a loss story.** 34 studies sharing
   one fully-normal report means those images have almost no discriminative text signal. Whether the
   right response is a multi-positive loss, or treating "normal" as the free weak label it evidently
   is, was not investigated.
