# What actually moves downstream label readout: depth, not language or schedule

Measured 2026-09-12 (Asia/Taipei). Nine full-cohort contrastive runs across three levers — report
language policy, depth of image-tower adaptation, and schedule length — all scored against the gold
pathology labels in **one** probe extraction pass, so every arm shares identical folds and resamples.

Training configs: [`image-block-cohort-compare-v1.json`](../configs/contrastive/image-block-cohort-compare-v1.json).
Probe config: [`gold-probe-extended-v1.json`](../configs/downstream/gold-probe-extended-v1.json).
Evidence: [`downstream-probe-extended-v1.json`](downstream-probe-extended-v1.json),
[`downstream-extended-comparisons-v1.json`](downstream-extended-comparisons-v1.json).
Curves: [`image-block-variants-curves.html`](image-block-variants-curves.html).

## The answer

> **Unfreezing 4 of DINOv2's 12 blocks instead of 1 raises macro AUROC by +0.036 to +0.046,
> replicated at three seeds with every interval excluding zero. Depth plateaus at 4 of 12 blocks.
> Translating the reports is null at three seeds. Doubling the schedule to 40 epochs is null.
> Adaptation depth is where the gain is.**

> **Exposure caveat, measured after the fact (2026-09-14).** Every level on this page — including
> the 0.7105 headline — is a **development statistic** read on studies the encoder trained on: 56 of
> the 58 gold studies were training pairs. A held-out replication ([`gold-holdout-v1.md`](gold-holdout-v1.md))
> measured the inflation at **−0.0168** of macro AUROC (−0.0163 by the data-size-matched route), so
> the generalisation estimate beside the pooled exposed level of **0.7106** is **0.6938**, and the
> **+0.0668** gain over unadapted DINOv2 is **+0.0500** on unexposed studies. The levels subtract
> exactly: 0.6438 unadapted, 0.6938 unexposed, 0.7106 exposed. **0.7105 is seed 2026**, not the pool. The *paired differences* on this page — depth, language, schedule, shuffled
> control — are contrasts between arms that share the same exposure and are not corrected by that
> figure; only the **levels** are. Quote 0.6938 beside 0.7105, never instead of it.

> **And the image/report objective does contribute.** A shuffled-pairing control at the same depth,
> learning rate, schedule and seed loses +0.0311 to +0.0538 macro AUROC against the true pairing
> across three permutation seeds, all agreeing in sign, two of three intervals excluding zero. But
> the control still recovers about a third of the total gain with no correct pairing at all, so the
> effect is not purely semantic alignment. See [the control section](#is-the-contrastive-objective-what-helps).

Superseded numbers: an earlier single-seed pass reported the depth gain as +0.0413 and the control as
recovering ~53% of the gain. Three seeds give +0.036 to +0.046 and ~33% respectively. The machine-readable
evidence for the final figures is `downstream-probe-extended-v3.json` and
`downstream-extended-comparisons-v3.json`; the v1 and v2 artifacts are retained as the record of what
the earlier passes actually produced.

## All nine runs, against unadapted DINOv2

Pooled out-of-fold AUROC over 56 gold studies × 12 labels, `uniform_mean_768` features (matched
pooling and dimensionality, so the only difference is the adapted backbone). Unadapted
DINOv2-small reads these labels at **0.6438**.

| Run | Blocks unfrozen | Epochs | Seed | macro AUROC | macro AUPRC | vs unadapted | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original-multilingual | 1 | 20 | 2026 | 0.6695 | 0.5650 | +0.0257 | +0.0112 to +0.0409 |
| original-multilingual | 1 | 20 | 2027 | 0.6684 | 0.5599 | +0.0247 | +0.0084 to +0.0417 |
| original-multilingual | 1 | 20 | 2028 | 0.6722 | 0.5644 | +0.0285 | +0.0127 to +0.0455 |
| translated-english | 1 | 20 | 2026 | 0.6685 | 0.5622 | +0.0247 | +0.0095 to +0.0406 |
| translated-english | 1 | 20 | 2027 | 0.6713 | 0.5609 | +0.0275 | +0.0120 to +0.0432 |
| translated-english | 1 | 20 | 2028 | 0.6705 | 0.5660 | +0.0266 | +0.0106 to +0.0434 |
| original-40ep | 1 | **40** | 2026 | 0.6726 | 0.5641 | +0.0288 | +0.0106 to +0.0476 |
| original-unfreeze2 | **2** | 20 | 2026 | 0.6838 | 0.5693 | +0.0400 | +0.0143 to +0.0684 |
| original-unfreeze4 | **4** | 20 | 2026 | **0.7105** | 0.5793 | **+0.0670** | +0.0299 to +0.1060 |

Every arm improves on unadapted DINOv2 with an interval excluding zero. The six `unfreeze 1` runs
span 0.6684–0.6722 — a seed spread of 0.0038, which is the noise floor any claimed effect must clear.

## Paired contrasts

Same studies, folds and 5,000 study resamples; paired within seed wherever a seed match exists.

| Contrast | Baseline | Candidate | Difference | 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 → 4 blocks (s2026) | 0.6695 | 0.7105 | **+0.0413** | +0.0095 to +0.0739 | **excludes zero** |
| 1 → 4 blocks (s2027) | 0.6684 | 0.7039 | **+0.0356** | +0.0027 to +0.0695 | **excludes zero** |
| 1 → 4 blocks (s2028) | 0.6722 | 0.7173 | **+0.0455** | +0.0125 to +0.0809 | **excludes zero** |
| 4 → 8 blocks (s2026) | 0.7105 | 0.7164 | +0.0062 | −0.0227 to +0.0344 | spans zero — **plateau** |
| 2 → 4 blocks (s2026) | 0.6838 | 0.7105 | **+0.0270** | +0.0012 to +0.0527 | **excludes zero** |
| 1 → 2 blocks (s2026) | 0.6695 | 0.6838 | +0.0143 | −0.0030 to +0.0334 | spans zero (93.9% favour) |
| 20 → 40 epochs (s2026) | 0.6695 | 0.6726 | +0.0031 | −0.0050 to +0.0109 | spans zero |
| original → translated (s2026) | 0.6695 | 0.6685 | −0.0010 | −0.0061 to +0.0043 | spans zero |
| original → translated (s2027) | 0.6684 | 0.6713 | +0.0028 | −0.0037 to +0.0093 | spans zero |
| original → translated (s2028) | 0.6722 | 0.6705 | −0.0019 | −0.0069 to +0.0027 | spans zero |

### Report language: a null result at three seeds

The three seed-matched differences are −0.0010, +0.0028 and −0.0019. Every interval spans zero, the
sign flips across seeds, and the replicate fractions straddle 50% (34.1%, 80.0%, 21.8%). All three
differences are smaller than the 0.0038 seed spread. Translating the reports to English does not
change what the image tower learns. This replicates, at three seeds, the single-seed finding in
[`downstream-original-vs-translated-v1.md`](downstream-original-vs-translated-v1.md).

### Schedule length: a null result, as the curves predicted

Doubling to 40 epochs moves macro AUROC by +0.0031 — inside the seed spread, interval spanning zero.
The curves said this in advance: validation loss reaches its minimum at epoch 8 and drifts upward
thereafter while training loss keeps falling to 0.38, and the late-epoch R@1 climb tracks the cosine
schedule annealing the learning rate to zero rather than continued learning. More schedule buys
nothing measurable on the goal metric.

### Adaptation depth: the one lever that moves

| Label | Positives | unadapted | 1 block | 2 blocks | 4 blocks |
| --- | --- | --- | --- | --- | --- |
| ACL | 23 | 0.5244 | 0.5547 | 0.5863 | 0.6285 |
| MCL | 8 | 0.7135 | 0.7396 | 0.7214 | 0.7083 |
| Medial Meniscus | 26 | 0.7923 | 0.7821 | 0.7692 | 0.7167 |
| Lateral Meniscus | 22 | 0.5709 | 0.6016 | 0.6243 | 0.6511 |
| Medial OA | 14 | 0.8044 | 0.8231 | 0.8197 | 0.9048 |
| Lateral OA | 11 | 0.6040 | 0.5818 | 0.5818 | 0.5818 |
| PF OA | 20 | 0.5194 | 0.5681 | 0.5986 | 0.6389 |
| Effusion | 33 | 0.7681 | 0.7984 | 0.7945 | 0.8538 |
| Synovitis | 26 | 0.4679 | 0.5333 | 0.5141 | 0.6256 |
| Baker's | 12 | 0.6799 | 0.6970 | 0.7765 | 0.7311 |
| Contusion | 19 | 0.7952 | 0.8236 | 0.8592 | 0.8193 |
| Fracture | 17 | 0.4857 | 0.5309 | 0.5596 | 0.6667 |
| **macro AUROC** | | **0.6438** | **0.6695** | **0.6838** | **0.7105** |
| *macro AUROC, unexposed (4 blocks only)* | | *—* | *—* | *—* | ***0.6938*** |
| **macro AUPRC** | | **0.5281** | **0.5650** | **0.5693** | **0.5793** |

The gain is broad rather than one column: 8 of 12 labels read better at 4 blocks than at 1, and 9 of
12 beat unadapted. The largest gains are Synovitis (+0.0923 over 1 block), Medial OA (+0.0817),
Fracture (+0.1358) and ACL (+0.0738) — labels the 1-block arm barely lifted off chance. Two labels
move the other way: Medial Meniscus falls 0.7821 → 0.7167 and MCL 0.7396 → 0.7083, the two labels
unadapted DINOv2 already read best, which is where more adaptation has the most to lose.

Depth costs memory, not time: peak VRAM rises 3,186 → 3,618 → 4,480 MiB from 1 to 2 to 4 blocks,
while elapsed time stays flat at 1,235–1,292 s.

## Is the contrastive objective what helps?

Every contrast above measures an adapted backbone against an unadapted one, which confounds *the
image/report objective* with *domain adaptation of DINOv2 to knee MRI at all*. The control separates
them: identical cohort, depth, learning rate, schedule and seed, with only the image/report
correspondence destroyed — each training study paired with a different study's report, as a
derangement, validation and test pairings left intact.

**The control is valid.** Validation Recall@1 collapsed to 0.0011 against a 0.0023 chance baseline
while validation loss rose monotonically 6.09 → 8.40. It learned no alignment, which is the
precondition for the control to mean anything.

| Arm | macro AUROC | vs unadapted |
| --- | --- | --- |
| Unadapted DINOv2 | 0.6438 | — |
| Shuffled pairings, 4 blocks (mean of 3 seeds) | 0.6659 | +0.0221 |
| True pairings, 4 blocks (mean of 3 seeds) | 0.7106 | +0.0668 |

Both arms here are exposed. On unexposed studies the true-pairing level is **0.6938** and the gain
over unadapted DINOv2 is **+0.0500** ([`gold-holdout-v1.md`](gold-holdout-v1.md)); the shuffled
control was never re-run held out, so the +0.0450 objective decomposition below is an exposed-arm
contrast and carries no generalisation estimate.

Seed-matched, depth-matched, the objective is worth **+0.0450** on average:

| Permutation seed | Shuffled | True | Difference | 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- |
| 2026 | 0.6795 | 0.7105 | +0.0311 | −0.0044 to +0.0706 | spans zero (95.7% favour) |
| 2027 | 0.6504 | 0.7039 | +0.0538 | +0.0170 to +0.0932 | **excludes zero** |
| 2028 | 0.6677 | 0.7173 | +0.0502 | +0.0161 to +0.0852 | **excludes zero** |

**Both mechanisms are real; the objective is the larger half.** All three seeds favour the true
pairing, two intervals exclude zero, and the objective accounts for roughly two thirds of the total
+0.0668 gain. The remaining third is reproduced by adapting the same blocks against a target carrying
no image/report correspondence, so "contrastive learning improves label readout" still overstates the
mechanism.

What the control actually did is worth stating precisely. Its training loss fell 3.47 → 1.41 while
validation loss rose to 8.40 — it **memorised** the random pairings. Doing that requires features that
separate individual studies, which is closer to instance discrimination than to domain adaptation.
These runs cannot distinguish those two explanations; an images-only self-supervised arm would.

Note the control at 4 blocks (0.6659 mean) is not distinguishable from the true pairing at 1 block
(0.6700 mean). Depth with a meaningless target buys about as much as shallow adaptation with a real
one.

## Correctness checks that passed

- **Frozen-tower identity.** Both `--image-unfreeze-last 0` controls produce `uniform_mean_768`
  features bit-identical to unadapted DINOv2 (max absolute deviation 0.0), declared in advance.
- **Population.** 48 train / 8 validation / 2 test / 0 absent, 56 scored, under the pinned manifest.
- **No GPU contention.** All nine runs are within 1,207–1,292 s except the 40-epoch run at 1,824 s,
  which is the expected cost of twice the schedule. Runs executed strictly one at a time.
- **Locked test never loaded.** Test-split gold studies are dropped before any pixel is read.

## Limitations

- **Depth is replicated at three seeds and is no longer provisional.** All three 1 → 4 block
  contrasts exclude zero. The 8-block screen shows a plateau, so 4 of 12 is near the useful optimum;
  the plateau itself is single-seed.
- **The control's own gain over unadapted is not individually established.** Point estimate +0.0221,
  and the seed-2026 arm's interval spans zero. The decomposition into "objective" and "everything
  else" is supported in direction and ordering, not as a precise split.
- **Depth and learning rate are not separated by this design.** All arms use `--lr-image 1e-5`, a
  value chosen when only one block was trainable. Four blocks at that same rate is four times the
  trainable surface at an LR tuned for one, so "more depth helps" and "this LR suits more blocks
  better" are confounded here. The gain is real at this configuration; attributing it to depth alone
  would require an LR sweep at each depth.
- **Readout, not generalisation.** 56 of the 58 gold studies were contrastive training pairs or
  members of the selection split. Every number here is a readout statistic on seen data.
- **Retrieval and the goal metric disagree about depth.** Selection uses validation R@1; the labels
  are the working goal. Nothing here was selected on label AUROC, and nothing should be.
- **Two labels regress at depth**, and with 8–33 positives per label on 56 studies, per-label
  movements of this size are individually fragile.
- **Cohort asymmetry.** The translated manifest drops two studies quarantined for identical decoded
  pixels with incompatible paired reports; the base manifest retains them.
- **Environment.** The probe runs on CPU: `evaluate_downstream_probe.py` never moves the batch to
  the selected device, so `--device cuda` raises a device mismatch in `_plane_features`. A latent
  bug, not a result. Scored under scikit-learn 1.9.0, which emits the `penalty='l2'` deprecation the
  probe config anticipates; the fitted model is unchanged. Exposure has since been measured: **−0.0168** of macro AUROC, generalisation estimate **0.6938** ([`gold-holdout-v1.md`](gold-holdout-v1.md)).
