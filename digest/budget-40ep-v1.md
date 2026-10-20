# The budget arm: doubling the schedule recovers a third of the drop-path deficit, and the rest is ambiguous

Measured 2026-09-15 (Asia/Taipei). Pre-registered in the budget-arm block of
[`handoff/handoff-20260913-overfitting-confirmed.md`](handoff/handoff-20260913-overfitting-confirmed.md)
**before any run**, including the decision rule and the power check that forced the rule to be
rewritten. Evidence: [`budget-40ep-v1.json`](budget-40ep-v1.json).
Runner: [`handoff/queue11-budget-arm.sh`](handoff/queue11-budget-arm.sh) ·
Analysis: [`handoff/analyze_budget_arm.py`](handoff/analyze_budget_arm.py).

## Why this arm exists

The [dose-response sweep](drop-path-sweep-v1.md) trained every drop-path rate for the same 20 epochs
and read the result as "stochastic depth is closed at 4 blocks". A
[convergence audit](convergence-curves.html) then found the high rates were **still improving when
the budget ended** — the validation-loss rise from each arm's own minimum collapsed +0.259 → +0.010
across the rate ladder, and rate 0.1 had barely begun to overfit at epoch 20.

That made rate confounded with budget. This arm removes the confound.

## The design point that makes it work

**Both arms were retrained at 40 epochs, not just the treatment.** The cosine schedule anneals over
`epochs`, so a 40-epoch run has a different learning-rate trajectory *throughout* — it is not the
20-epoch run continued. Comparing rate 0.1 at 40 epochs against the recorded rate 0.1 at 20 epochs
would confound budget with schedule. Retraining both arms makes the schedule change **common to
both, so it cancels in the paired difference** — the same logic that makes pairing on seed work.

The learning rate at epoch 1 is exactly half the 20-epoch run's (2.52e-06 vs 5.05e-06), because a
0.1 warmup fraction over 40 epochs is a 4-epoch ramp instead of 2. That is the schedule change,
visible in the first row of data.

## Result

| seed | rate 0.0 @40ep | rate 0.1 @40ep | paired difference |
| --- | --- | --- | --- |
| 2026 | 0.1361 (ep 35) | 0.1224 (ep 25) | −0.0136 |
| 2027 | 0.1434 (ep 23) | 0.1217 (ep 32) | −0.0217 |
| 2028 | 0.1412 (ep 33) | 0.1264 (ep 27) | −0.0147 |
| **pooled** | **0.1402** | **0.1235** | **D40 = −0.0167** |

**Per-query paired bootstrap: D40 = −0.0167, 95% CI [−0.0337, +0.0005]**, same sign 3/3, 97.1% of
resamples favouring the baseline. The interval **grazes zero without excluding it**.

> **The difference column is not the subtraction of the two level columns, and should not be read as
> one.** The levels come from `metrics.jsonl` at each selected epoch; the differences come from
> per-query R@1 credit on eval-mode embeddings, which is what the bootstrap resamples. Subtracting
> the levels gives −0.0137 / −0.0217 / −0.0148 against the bootstrap's −0.0136 / −0.0217 / −0.0147 —
> **same quantity, two estimators, agreeing to a ten-thousandth. Not a disagreement.** The pooled
> D40 quoted everywhere on this page is the per-query one, because that is the estimator the interval
> belongs to.

**On the learned temperature, which this page otherwise does not use.** τ⁻¹ rises monotonically
through training, so a checkpoint selected earlier carries a lower one. Seed 2027's rate-0.0 arm
shows **23.76 at its selected epoch 23** against 28.92 (ep 35) and 28.29 (ep 33) for its siblings —
which looks like an outlier and is not: its **final** τ⁻¹ is **29.05**, in family with 29.16 and
28.91. The spread is selection epoch, not a divergent run. None of this touches D40: **R@1 is
τ-invariant** ([the temperature result](handoff/handoff-20260913-overfitting-confirmed.md)), so
retrieval cannot be moved by τ in either direction.

## The pre-registered rule returns AMBIGUOUS, and that is the finding

| rule, fixed before the numbers existed | fires? |
| --- | --- |
| CI excludes zero **and** D40 ≤ −0.015 → harm persists | **No** — CI includes zero by 0.0005 |
| CI spans zero **and** \|D40\| ≤ 0.010 → harm did not reproduce | **No** — \|D40\| = 0.0167 |
| otherwise → **partial or ambiguous** | **Yes** |

**Do not round this to either neighbour.** The third band exists precisely so that a result landing
between them gets recorded rather than argued into the more convenient one.

## What did move, and is not ambiguous

- **Both arms converged inside the new budget.** Best epochs are 23–35 of 40; none is at the
  boundary (≥39), against 0 of 3 for the baseline and a validation minimum at epoch 16.0 for rate 0.1
  at the old budget. The pre-registered "still truncated" trap did **not** fire. This is what makes
  D40 a comparison at convergence rather than another truncated one.
- **Budget helped both arms, and helped the regularised one more than twice as much.**

  | arm | 20 epochs | 40 epochs | gain |
  | --- | --- | --- | --- |
  | rate 0.0 | 0.1340 | 0.1402 | **+0.0062** |
  | rate 0.1 | 0.1085 | 0.1235 | **+0.0150** |

  That asymmetry is exactly what the truncation hypothesis predicted: the arm that had not finished
  improving gained more from being allowed to finish. **Read as direction only** — these level
  comparisons are unpaired across schedules.
- **The deficit shrank and its interval stopped excluding zero.** At 20 epochs rate 0.1 was the
  *only* rate in the sweep whose CI excluded zero, at [−0.0421, −0.0094]. At 40 epochs the same
  contrast is [−0.0337, +0.0005]. The sweep's single solid negative is no longer solid.

## Secondary, and under-powered — quote the number with the caveat attached

D20 −0.0255 → D40 −0.0167 is a shift of **+0.0088**, about **35%** of the deficit recovered.

**This is not a test.** At three seeds the half-width of `D40 − D20` is roughly **0.023** — √2 times
the 0.0163 half-width of a single paired difference — so it can only detect a near-total reversal.
The 35% is a point estimate with an interval far wider than itself. This was written down
**before the run**, after a power check killed the first version of the decision rule, which had made
this difference-of-differences primary with bands of ±0.0064 that it could never have resolved.

## What a successor should take from this

1. **"Stochastic depth is closed at 4 blocks" remains too strong, and "it was all truncation" is
   also wrong.** Truncation was real and accounts for roughly a third of the measured harm; a
   directional deficit survives at convergence but no longer excludes zero. Both clean stories fail.
2. **A fixed epoch budget is not a neutral choice when the lever is a regulariser.** Any future
   regularisation arm — projection-head dropout and random erasing are both still untested — must
   either train to a convergence criterion or report budget as a co-varying factor.
3. **The baseline itself gained +0.0062 from 40 epochs.** The recorded 0.1340 is a 20-epoch number,
   not a ceiling. Anything that compares against it should say which budget it means.
4. **Selection stays retrieval R@1.** No label AUROC was computed for these checkpoints and nothing
   here was selected on it.

## Provenance

Six runs, zero FAILED, `QUEUE11 COMPLETE`. Every training argument byte-identical to
`queue9-drop-path-sweep.sh` apart from `--epochs 40` and the output root; each run's identity
(`image_drop_path_rate`, `seed`, `epochs`) read from its **own** `config.json`, never from a
filename. Validation R@1 is over **441** candidates for all six runs, asserted in the analysis script
— the 433-candidate gold-holdout figures must never appear beside these. The bootstrap contract
(per-query R@1 credit, pooled across-seed resampling, 2000 resamples, seed 20260914) is copied
verbatim from `analyze_drop_path_arm.py` so D40 is constructed identically to D20.

Nothing committed; the test split was never loaded.
