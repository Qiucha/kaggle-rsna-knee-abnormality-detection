# Pre-registration: uniform plane pooling instead of learned plane attention

**Written before the first run of this arm, and before any result was seen.** Fixed in advance:
the arms, the seeds, the metric, the selection rule, the decision criterion, and the prediction.

## The hypothesis

The image tower pools its three plane vectors with learned attention
(`ImageTower.forward`, `model.py`). The label probe reads the same checkpoints out with a
**uniform** plane mean (`uniform_mean_768` = CLS ⊕ patch-mean, 2 × 384). So training shapes the
backbone for features that will be combined with the learned weights, while the probe combines them
equally. **The hypothesis is that removing that train/readout mismatch improves the probe.**

## Why this lever, on evidence already in hand

1. **Reading the same checkpoints out uniformly beats reading them out with the learned weights, in
   8 of 8 arms measured** — `digest/downstream-probe-extended-v3.json`, `uniform_mean_768` versus
   `plane_attention_768` macro AUROC:

   | arm | uniform | attention | delta |
   | --- | --- | --- | --- |
   | original-unfreeze4-seed2026 | 0.7105 | 0.6978 | +0.0127 |
   | original-unfreeze4-seed2027 | 0.7039 | 0.6941 | +0.0098 |
   | original-unfreeze4-seed2028 | 0.7173 | 0.6988 | +0.0185 |
   | original-unfreeze2-seed2026 | 0.6838 | 0.6779 | +0.0059 |
   | original-unfreeze8-seed2026 | 0.7164 | 0.6965 | +0.0199 |
   | original-multilingual-seed2026 | 0.6695 | 0.6604 | +0.0091 |
   | original-multilingual-seed2027 | 0.6684 | 0.6457 | +0.0227 |
   | original-multilingual-seed2028 | 0.6722 | 0.6494 | +0.0228 |

   Mean over the three 4-block baseline seeds: **+0.0137**. Note that
   `digest/handoff/handoff-20260912-image-block-variants.md` quoted **0.0185** for this; that is
   seed 2028 alone, the largest of the three. The three-seed figure is the one to use.

2. **The attention's stated rationale does not hold on this cohort.** Measured on
   `original-unfreeze4-seed2026/best.pt` over all 441 validation studies, the learned weights are
   near-constant at **[0.2615, 0.2835, 0.4550]**: normalised entropy **0.947** (1.0 is uniform), mean
   max weight 0.475, per-plane standard deviation only **0.064–0.087** against a 0.19 spread in the
   means, no study above 0.90 on any plane, and plane 3 is the argmax for 365 of 441 studies. It has
   learned approximately a **fixed reweighting**, not the content-dependent routing the module
   docstring claimed. A fixed reweighting is exactly the thing a uniform mean can replace.

## Design

- **Baseline:** the existing `image-block-depth-v1/original-unfreeze4-seed{2026,2027,2028}` runs.
  Nothing is re-trained for the baseline.
- **Arm:** identical in every respect except `--image-plane-pooling uniform`. Matched:
  `contrastive-cohort.jsonl`, `full` view, frozen multilingual-e5 (`--text-unfreeze-last 0`, mean
  pooling, `query: ` prefix, 512 tokens), dinov2-small at `ed25f3a`, `--image-unfreeze-last 4`,
  `--lr-image 1e-5`, 20 epochs, batch 32, same image cache, same three seeds.
- **New output root** `checkpoints/contrastive/plane-pooling-v1/`. No existing matrix or
  `validation-selection.json` is touched.
- **Primary metric:** gold-probe macro AUROC at `uniform_mean_768`, with the **probe configuration
  byte-identical to `configs/downstream/gold-probe-extended-v3.json`**. Per handoff concern #6,
  nothing in the probe is tuned — not `C`, not `class_weight`, not the fold count.
- **Selection stays plain validation retrieval R@1**, unchanged, for comparability with the three
  baseline runs.

## Decision criterion, fixed now

Following the convention the depth result was judged by: **the arm is called a real effect only if
all three seed deltas share a sign and each seed's descriptive 95% bootstrap interval excludes
zero.** Two of three, or a sign flip, is a null. The comparison is run once, at three seeds; there
is no "add a seed and re-look".

## Pre-registered so that a null reads as an answer

1. **Predicted effect: smaller than +0.0137, and possibly zero.** That figure is how much worse the
   attention *readout* is on a fixed checkpoint. This arm changes *training*, and the probe already
   reads uniformly, so what is being removed is only the train/readout mismatch — 0.07 of pooling
   weight moved off two planes. **A null here does not contradict the readout table.**
2. **This lever is not isolated.** `plane_pooling="uniform"` builds no `plane_embeddings` and no
   `plane_attention` MLP, so it removes trainable parameters as well as changing the pooling. The
   arm tests "uniform pooling *and* a smaller head" as one thing. It cannot be decomposed in this
   tower and no claim will be made that it was.
3. **The selection metric is less sensitive to this change than the probe is.** Validation R@1 reads
   the projected 256-d space, which still passes through the projection head; the probe reads the
   backbone features the change is aimed at. Selection stays on R@1 anyway, for comparability — but
   **R@1 is not the metric this hypothesis is about**, and a null on R@1 with a probe gain is a
   coherent outcome rather than a contradiction.
4. **It may hurt.** If the learned weighting is mildly useful during training even though it is a
   poor readout, uniform pooling loses it. A negative result is a finding.

## Cost

Three runs, ~21 minutes each, plus one CPU probe pass. No new cohort, no new preprocessing.

## The locked test split is not involved and will not be loaded.


---

# Addendum, written after the runs — corrections to this document's own provenance claim

**The pre-registered content above is unedited.** These are corrections to it, added after the
result, and they are marked as such rather than folded in.

1. **This document mis-attributed the handoff's "0.0185".** It said that figure was seed 2028 of the
   `downstream-probe-extended-v3.json` readout table. It is not. It comes from
   [`digest/plane-pooling-analysis.json`](plane-pooling-analysis.json) —
   `image-block-seed2026/attention_pooled_768`, delta **−0.018455** against the uniform mean on the
   same checkpoint. The near-identical magnitude to v3's seed-2028 delta (+0.0185) is a coincidence.
2. **That prior figure's own interval spans zero**, which the handoff's "free win, already measured"
   did not convey: −0.0185 with CI [−0.0479, +0.0099]. Across its three seeds the prior artifact
   gives −0.0185 / −0.0214 / −0.0205, all the same sign, **none excluding zero**. Its
   "plane concatenation is +0.0010" is likewise seed 2026 only; seeds 2027 and 2028 give −0.0007 and
   −0.0035.
3. **Prior work had already measured the attention weights**, on the translated-cohort 1-block runs
   over the 56 gold studies. This session's measurement is a **replication** at 4 blocks over all 441
   validation studies, not a discovery, and §2 of this protocol should have cited it. The two agree:
   prior mean weights [0.256, 0.236, 0.508] with normalised entropy 0.908 versus
   [0.2615, 0.2835, 0.4550] at 0.947 here.
4. **The prior artifact names the planes**, which this protocol did not: the order is
   **Sagittal, Coronal, Axial**, so the up-weighted plane is the **axial** one.
5. **The 8-of-8 readout table in §1 is unaffected** — it was computed in this session directly from
   `downstream-probe-extended-v3.json` and stands as written.
