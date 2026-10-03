# Uniform plane pooling versus learned plane attention

**Null by the pre-registered criterion. Consistent in direction across three seeds, and
individually underpowered in all three.** Pre-registration:
[`digest/plane-pooling-protocol.md`](plane-pooling-protocol.md), written before the first run.
Evidence: [`digest/downstream-probe-plane-pooling-v1.json`](downstream-probe-plane-pooling-v1.json),
[`digest/plane-pooling-comparisons-v1.json`](plane-pooling-comparisons-v1.json). Probe config
[`configs/downstream/gold-probe-plane-pooling-v1.json`](../configs/downstream/gold-probe-plane-pooling-v1.json).

## What was changed

One lever: `--image-plane-pooling uniform` replaces the image tower's learned plane attention with a
plain mean over the three planes. Everything else matched the three existing 4-block baselines
exactly — same cohort, view, frozen text tower, dinov2-small at `ed25f3a`, `--image-unfreeze-last 4`,
`--lr-image 1e-5`, 20 epochs, batch 32, same image cache, same three seeds.

## The result

Gold-probe macro AUROC on 56 of 58 gold studies × 12 labels, `uniform_mean_768`, probe configuration
byte-identical to `gold-probe-extended-v3` with nothing tuned. Unadapted DINOv2 = **0.6438**.

| seed | attention | uniform | delta | descriptive 95% CI | excludes 0 | P(replicates favour uniform) |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | 0.7105 | 0.7159 | +0.0055 | [−0.0144, +0.0263] | no | 0.69 |
| 2027 | 0.7039 | 0.7219 | +0.0183 | [−0.0044, +0.0405] | no | 0.94 |
| 2028 | 0.7173 | 0.7270 | +0.0096 | [−0.0063, +0.0271] | no | 0.88 |

Mean delta **+0.0110**; arm mean 0.7216 against baseline mean 0.7106.

**All levels in this table are exposed** — 56 of the 58 gold studies were training pairs. The
4-block baseline level is **0.6938** on unexposed studies, not 0.7106
([`gold-holdout-v1.md`](gold-holdout-v1.md)); the uniform arm was never re-run held out. The deltas
are paired within the same exposure and are unaffected.

**The pre-registered criterion was: all three deltas share a sign *and* each seed's interval excludes
zero. Three of three share a sign; zero of three exclude zero. That is a null, and it is recorded as
a null.**

Validation retrieval R@1, the selection metric, agrees that nothing decisive happened: 0.1349 /
0.1412 / 0.1395 (mean 0.1385) against 0.1383 / 0.1434 / 0.1202 (mean 0.1340). The protocol
pre-registered that R@1 is *not* the metric this hypothesis is about, and it is reported only for
comparability.

## The pre-registration was right about the effect size and wrong about the test

The protocol predicted the effect would be **"smaller than +0.0137, and possibly zero"**. Observed:
**+0.0110**. The prediction held.

But the criterion it paired with that prediction cannot detect an effect of that size. The per-seed
descriptive intervals here are roughly ±0.02 wide on a 56-study probe, so **no effect near +0.011
could ever have passed "each seed's interval excludes zero"**, whatever its truth. The criterion was
borrowed from the depth result, where the effect was +0.04 — nearly four times larger. **Copying a
decision rule across a fourfold change in expected effect size was a design error, and it was
visible at pre-registration time, in the protocol's own prediction.** It is recorded here rather
than quietly replaced.

What follows from that is a limit on what this session can claim, not a licence to re-call the
result. A one-sided reading of three independent same-sign results is suggestive; the honest summary
is **"consistent direction, individually underpowered, not established."** No pooled or
across-seed test is reported, because none was pre-registered and choosing one after seeing the
numbers is exactly the move the protocol existed to prevent.

## What is established

Two things, neither of which required the runs:

1. **The learned attention is a fixed reweighting, not content-dependent routing — replicating
   prior work.** On `original-unfreeze4-seed2026/best.pt` over all 441 validation studies: weights
   **[0.2615, 0.2835, 0.4550]**, normalised entropy **0.947**, per-plane sd **0.064–0.087** against a
   0.19 spread in the means, no study above 0.90 on any plane, and the third plane the argmax for 365
   of 441. [`digest/plane-pooling-analysis.json`](plane-pooling-analysis.json) had **already measured
   this** on the translated-cohort 1-block runs over the 56 gold studies, and agrees closely: mean
   weights [0.256, 0.236, 0.508], normalised entropy 0.908. This session's figures are a replication
   at a different depth and a larger population, not a discovery.

   **That artifact also names the planes, which matters clinically: the order is Sagittal, Coronal,
   Axial, so the plane the attention up-weights to ~0.46–0.51 is the axial one** — the least
   informative of the three for meniscal and cruciate pathology, which read sagittal and coronal.
   The module docstring's own rationale ("a meniscal tear reads from sagittal, a collateral ligament
   from coronal") is therefore not merely unrealised but inverted in practice. `model.py`'s docstring
   now records the measurement instead of the claim.
2. **Reading a fixed checkpoint out uniformly beats reading it out through the attention, in 8 of 8
   arms** (+0.0059 to +0.0228 macro AUROC, computed in this session from
   `downstream-probe-extended-v3.json`). This was already true before this session and is already
   banked in the 0.7105 headline, because the probe has always read `uniform_mean_768`.

   **The handoff's "0.0185 below a uniform mean" is not from that table, and it is weaker than it
   sounds.** It is `plane-pooling-analysis.json`'s `image-block-seed2026/attention_pooled_768`,
   **−0.0185 with CI [−0.0479, +0.0099] — spanning zero.** Its three seeds give −0.0185 / −0.0214 /
   −0.0205: same sign, **none excluding zero**. Its companion claim "plane concatenation is +0.0010"
   is also seed 2026 only (2027: −0.0007, 2028: −0.0035). Calling that a "free win, already measured"
   overstated it. The near-identical magnitude to v3's seed-2028 delta (+0.0185) is coincidence.

## Per-label movement

Only three of twelve labels move consistently across all three seeds:

| label | 2026 | 2027 | 2028 | mean | consistent |
| --- | --- | --- | --- | --- | --- |
| Baker's | +0.1023 | +0.0644 | +0.0398 | **+0.0688** | yes |
| Lateral OA | +0.0788 | +0.0222 | +0.0283 | **+0.0431** | yes |
| Synovitis | −0.0385 | −0.0192 | −0.0333 | **−0.0303** | yes |
| ACL | −0.0935 | +0.0053 | −0.0356 | −0.0413 | no |
| *(eight others)* | | | | −0.002 to +0.026 | no |

The macro movement is carried by **Baker's cyst and lateral OA**, with **synovitis consistently
worse**. With 8–33 positives per label (concern #4 in the handoff) individual labels are fragile, so
the consistency of sign across seeds is the only part of this worth weighting, and even that rests
on three runs.

## Integrity checks that passed

- **The three re-extracted baselines reproduce `gold-probe-extended-v3` exactly** — macro AUROC
  differences of `0.00e+00` on all three seeds. The new arm therefore sits on the same folds,
  resamples and features as the published table.
- `plane_attention_768` is **absent** for the uniform arm, as it must be: those towers have no
  attention parameters. `uniform_mean_768`, the primary metric, is identically defined for both arms.
- Every uniform-pooling checkpoint re-scores with `metric_deltas` all `0.0`.
- 66 tests pass, 1 skipped (was 63 + 1).
- **The locked test split was never loaded.**

## Two defects found and fixed in the probe, which matter beyond this arm

1. **`build_towers` iterated a hardcoded `("after_cl", "frozen_tower_null")`.** Any arm added to a
   probe config under a new key was **silently skipped** — it would have produced a complete-looking
   artifact with the new arm simply missing. Now derived from the config's own arm keys.
2. **The feature loop dereferenced `tower.plane_embeddings` unconditionally**, so it crashed on a
   uniform-pooling tower. It now branches, and computes `projection_256` from the pooling the tower
   actually uses.

Also: the text tower is now released after each checkpoint's provenance is recorded. Building every
tower up front, each with a 278M-parameter text backbone that is never read, exhausted memory and
killed the first probe run.

## The probe cannot resolve effects of this size at three seeds — twice now

The prior investigation and this one have produced the **same shape of result by different routes**:

| | what was compared | three-seed deltas | same sign | any CI excludes zero |
| --- | --- | --- | --- | --- |
| `plane-pooling-analysis.json` | attention *readout* vs uniform readout, fixed checkpoints | −0.0185 / −0.0214 / −0.0205 | yes | **no** |
| this arm | attention *training* vs uniform training | +0.0055 / +0.0183 / +0.0096 | yes | **no** |

Both point the same way; neither is established. The common cause is the instrument, not the
hypotheses: on a 56-study, 12-label probe the per-seed descriptive interval is roughly ±0.02–0.03
wide, so **no effect of order 0.01–0.02 can clear a per-seed "interval excludes zero" bar at three
seeds, whatever its truth.** Two independent attempts have now been spent discovering this.

**The operational consequence for the next arm:** if the predicted effect is ~0.01–0.02 macro AUROC,
a per-seed criterion is the wrong test and will return a null by construction. Fix a **pooled
across-seed estimand in advance**, and choose the seed count from the width that estimand needs —
before running anything. Checking that the decision rule can detect the protocol's own predicted
effect size should be a step in writing any future pre-registration.

## Caveats

1. **Not isolated.** `uniform` builds no `plane_embeddings` and no `plane_attention` MLP, so this arm
   changes the pooling *and* removes trainable parameters. Pre-registered as inseparable in this
   tower; no claim is made that it was decomposed.
2. **Not generalisation.** 56 of 58 gold studies were contrastive training pairs or in the selection
   split. Every number here is a readout statistic on a cohort the representation has seen — handoff
   concern #3, unchanged by this arm.
3. **One depth, one cohort, three seeds.** 4 unfrozen blocks on the original-language cohort.
4. **`projection_256` sign-flips** (+0.0301 / −0.0155 / +0.0233) and is confounded by construction;
   it is reported in the artifact and is not evidence either way.
5. **Same population as v3, verified.** The probe config inherits v3's *translated* q2 cohort
   manifest while both arms trained on `contrastive-cohort.jsonl`. That is intentional and matches
   v3: the manifest fixes the study population and pixels, not the reports. Verified rather than
   assumed — `population` string identical to v3's and the same **56** `study_uids` in the same
   order, so the contrast is over one population.
6. **The probe was not tuned** — no `C`, `class_weight` or fold-count sweep, per concern #6. If this
   arm is ever revisited, the temptation to raise its number that way must stay refused.
