# Stochastic depth: the memorisation/retrieval trade is parallel, with no knee

Measured 2026-09-15 (Asia/Taipei). Pre-registered as a **characterisation, not a test**, in the
dose-response block of [`handoff/handoff-20260913-overfitting-confirmed.md`](handoff/handoff-20260913-overfitting-confirmed.md).
Evidence: [`drop-path-sweep-v1.json`](drop-path-sweep-v1.json).
Runner: [`handoff/queue9-drop-path-sweep.sh`](handoff/queue9-drop-path-sweep.sh).
Analysis: [`handoff/analyze_drop_path_sweep.py`](handoff/analyze_drop_path_sweep.py).

> ## ⚠ AMENDMENT 2026-09-15 — every rate was trained to a fixed 20-epoch budget, and the higher
> rates had not finished improving when it ended.
>
> This page's conclusion is **narrower than it was written**. A convergence audit of the epoch
> histories ([`convergence-curves.html`](convergence-curves.html),
> [`convergence-audit.json`](convergence-audit.json)) shows the validation-loss rise from each arm's
> own minimum collapsing monotonically with rate — **+0.259 → +0.184 → +0.125 → +0.102 → +0.057 →
> +0.010** — and at **rate 0.1 validation loss is still falling at epoch 20 in 3 of 3 seeds**. The
> validation minimum moves from epoch 11.3 at the baseline to 16.0 at rate 0.1.
>
> So the sweep did not compare six rates **at convergence**. It compared them at progressively
> earlier points along their own trajectories, because a regulariser that delays overfitting also
> slows optimisation. **Rate is confounded with training budget**, and these runs cannot say how much
> of the deficit at high rates is the rate and how much is truncation.
>
> **What survives unchanged:** the numbers in the table, the memorisation-gap measurements (eval-mode,
> drop-path identity at every rate), the τ-matched loss handling, and the pre-registered refusal to
> nominate a rate. **What must be narrowed:** "stochastic depth is **closed** at 4 blocks" becomes
> **"closed at a 20-epoch budget"**. The cheapest decisive follow-up is a *budget* arm — one rate at
> a longer schedule against the same rate at 20 epochs — not another rate.
>
> ### ✅ RESOLVED IN PART 2026-09-15 — the budget arm ran: [`budget-40ep-v1.md`](budget-40ep-v1.md)
>
> Both arms retrained at 40 epochs, 3 seeds. **Both converged inside the new budget** (best epochs
> 23–35 of 40), so this is a comparison at convergence. **D40 = −0.0167, 95% CI [−0.0337, +0.0005]**
> against this page's **D20 = −0.0255, CI [−0.0421, −0.0094]**.
>
> **Neither clean story survives.** Truncation was real — rate 0.1 gained **+0.0150** from the longer
> budget against the baseline's +0.0062, and the interval that was this sweep's only CI excluding
> zero now grazes zero. But a directional deficit of −0.0167 remains at convergence, same sign 3/3.
> The pre-registered rule returns **AMBIGUOUS**, and it is recorded as such rather than rounded.
>
> Read this page's rate-0.1 row as **overstating the harm by roughly a third** — a point estimate
> whose own interval is ≈0.023 wide, so do not treat "a third" as measured precisely.

> The 40-epoch evidence cuts *toward* this reading rather than against it: the **unregularised** arm
> at 40 epochs got worse (R@1 0.1190 vs 0.1383, validation loss 3.687 → 4.752), which is what an arm
> that is already overfitting at epoch 11 should do. Whether a *regularised* arm can use the extra
> budget is the untested case.

Six rates × three matched seeds (2026/2027/2028), exposed cohort, 441 validation candidates,
4 adapted blocks, 20 epochs. The 0.0 and 0.05 anchors are reused from earlier work; **{0.005, 0.01,
0.02, 0.10} are new**, chosen with three points below the known-harmful 0.05 specifically to look for
a knee.

## The curve

| rate | val R@1 | vs baseline | 95% CI | same sign | memorisation gap | train R@1 | learned τ⁻¹ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 0.1340 | — | — | — | **+0.2689** | 0.4029 | 20.08 |
| 0.005 | 0.1202 | −0.0138 | [−0.0285, +0.0015] | 3/3 | +0.2354 | 0.3556 | 19.49 |
| 0.01 | 0.1264 | −0.0076 | [−0.0227, +0.0074] | 1/3 | +0.2412 | 0.3676 | 19.46 |
| 0.02 | 0.1238 | −0.0102 | [−0.0257, +0.0053] | 2/3 | +0.2319 | 0.3556 | 19.37 |
| 0.05 | 0.1187 | −0.0153 | [−0.0302, +0.0002] | 3/3 | +0.1588 | 0.2775 | 18.12 |
| **0.1** | 0.1085 | **−0.0255** | **[−0.0421, −0.0094]** | 3/3 | **+0.1150** | 0.2234 | 17.27 |

## The answer to the question this sweep was designed to ask

> **There is no knee. The two curves are parallel.** Every rate that reduces memorisation also costs
> retrieval, across a **20× range** of rates. No setting buys a smaller train/validation gap for free.

The memorisation gap falls monotonically with rate — **+0.2689 → +0.1150**, a 57% reduction at 0.1 —
and validation R@1 falls with it, **0.1340 → 0.1085**. The regulariser does exactly what a regulariser
should do, at every dose, and at no dose does the model generalise better for it.

**The only interval excluding zero is at rate 0.1**, the largest and worst: −0.0255, [−0.0421,
−0.0094], 3/3 seeds, 99.9% of resamples favouring the baseline. Harm is *established* only at the top
of the range; below it the direction is consistent but each interval grazes or spans zero.

**The low-rate region is flat and noise-dominated.** 0.005, 0.01 and 0.02 land at −0.0138, −0.0076 and
−0.0102 with heavily overlapping intervals and inconsistent per-seed signs (3/3, 1/3, 2/3). Do not
read 0.01 as "better than 0.005" — that ordering is noise. The honest summary of the bottom three
rates is *a small consistent cost of roughly −0.01*, not a ranking.

## What this closes, and what it does not

- **Stochastic depth is closed for this project at 4 blocks.** Six rates spanning 20×, three matched
  seeds each, one clean monotone trade and no knee. Do not re-propose it without changing something
  structural — the adapted depth, the batch size, or the data.
- **Lever D is not closed.** Projection-head dropout and random erasing remain untested, and they are
  different interventions: erasing *removes information from the input*, where stochastic depth
  *perturbs the forward pass*.
- **The mechanism conclusion strengthens.** [`drop-path-5e-2-arm.md`](drop-path-5e-2-arm.md) found that
  cutting memorisation 41% bought nothing. The sweep shows that was not a property of one rate: the
  memorisation gap can be cut **57%** and retrieval still only falls. **Whatever limits retrieval here
  is not memorisation capacity.** A successor should stop treating the 3.0× gap as the thing standing
  between this project and better numbers.
- **The information reading is the one left standing.** Every lever that *removes or perturbs*
  information has now failed. The untried direction is the one that *adds* it — the 89.9% of slices
  and 2.7 series per study the pipeline never opens
  ([`acquisition-metadata-trust.md`](acquisition-metadata-trust.md)).

## Reading notes

- **No rate is nominated, by pre-registration.** A maximum taken over six noisy points is biased
  upward; any rate that looked good here would need a fresh confirmatory 3-seed run. None does.
- **τ⁻¹ falls monotonically with rate** (20.08 → 17.27). Losses are therefore compared only at a
  matched τ; R@1 needs no such correction, being τ-invariant, which is why it carries the finding.
- **The memorisation gap is measured on eval-mode embeddings with augmentation off**, where drop-path
  is identity at every rate. `train_loss` from the metrics log is **not** usable across these arms —
  the forward pass differs by construction.
- Validation R@1 is over **441** candidates. Never place these beside the 433-candidate gold-holdout
  figures.
