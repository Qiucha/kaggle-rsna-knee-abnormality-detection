# Matched-N train versus validation loss — the overfitting question, measured directly

2026-09-13. Three 4-block baselines `image-block-depth-v1/original-unfreeze4-seed{2026,2027,2028}`
at `best.pt` (epochs 19 / 18 / 15). Compute spent: one forward pass per checkpoint to encode the
**train (3,526)** and **validation (441)** splits with `make_loader(..., training=False)` —
**augmentation off, cohort order, every study**. No training, no `src/` edit, locked test split never
loaded. Reproduce with `digest/handoff/encode_splits.py` then
`digest/handoff/analyze_matched_n_loss.py` → [`.json`](matched-n-loss-audit.json).

## Why this was necessary

The published pair "train loss 0.298 vs validation loss 3.709" **compares two different tasks**:
`training.py:158` averages the **in-batch InfoNCE over batches of 32, with augmentation on**, while
`evaluation.py:200` computes the **full-matrix InfoNCE over all 441 validation studies with
augmentation off**. Chance level alone differs by log(441) ≈ 6.09 versus log(32) ≈ 3.47.

A previous audit ([`val-loss-vs-retrieval-audit.md`](val-loss-vs-retrieval-audit.md)) *inferred* from
that structure that the gap was "mostly a candidate-count artifact" and concluded the memorisation
premise did not survive. **That inference was wrong**, and it is withdrawn there. This page replaces
it with the measurement.

## Pipeline verification

The reimplementation reproduces the stored epoch-19 metrics for seed 2026 **exactly**:
`mean_r@1` 0.13832199, `validation_loss` 3.702520, `inverse_temperature` 20.385719. The unaugmented
batch-32 train loss computed here is **0.2811** against the logged augmented **0.28241** — so
augmentation accounts for ~0.001 and is not what any gap below is made of.

## Result 1 — the gap is real and large, on every route

| Route | seed 2026 | seed 2027 | seed 2028 | mean |
| --- | --- | --- | --- | --- |
| **Matched N=441, train** (random subsets, 200 resamples) | 1.7697 | 1.7798 | 2.0354 | **1.862** |
| **Matched N=441, validation** | 3.7025 | 3.7686 | 3.6582 | **3.710** |
| **→ loss gap** | −1.933 | −1.989 | −1.623 | **−1.848** |
| Chance-normalised train (N=3,526, −log N) | −4.475 | −4.460 | −4.168 | −4.368 |
| Chance-normalised validation (N=441, −log N) | −2.387 | −2.320 | −2.431 | −2.379 |
| **→ gap, second route** | −2.089 | −2.140 | −1.737 | **−1.989** |
| **Batch-32 both splits, augmentation off** | 0.281 / 1.217 | 0.283 / 1.284 | 0.366 / 1.169 | **0.310 / 1.224** |

**The two independent routes agree** (−1.85 and −1.99). The batch-32 route, which matches the logged
number's protocol on both splits, shows the same thing at **4× the loss on unseen data**.

## Result 2 — retrieval says it more plainly than the loss does

| At N=441 | seed 2026 | seed 2027 | seed 2028 | mean |
| --- | --- | --- | --- | --- |
| **train R@1** | 0.4336 | 0.4247 | 0.3538 | **0.404** |
| **validation R@1** | 0.1383 | 0.1434 | 0.1202 | **0.134** |
| **gap** | +0.295 | +0.281 | +0.234 | **+0.270** |

**The model retrieves its own training pairs 3.0× better than unseen ones, at an identical candidate
count.** Seed 2028, which stopped earliest (epoch 15), has both the smallest train R@1 and the
smallest gap — the dose-response you would expect.

## Result 3 — duplicates are not causing it

Random 441-subsets of train carry **5.3** duplicate studies on average versus **3** in validation.
Redrawing the subsets with at most one study per `report_sha256` moves the gap by **≤0.005** in all
three seeds (−1.938 / −1.990 / −1.620). The duplicate-group structure is not the explanation.

## Result 4 — temperature: two separate statements, and only one is a lever

**(a) Tuning τ cannot improve any number this project reports.** τ is a positive scalar multiplying
every logit, so it cannot reorder a similarity matrix. Measured across τ ∈ {1, 5, 10, 20, 30, 50,
100} and the learned value, validation R@1 is **bit-identical** — 0.138322 / 0.143424 / 0.120181 at
every τ. R@10, median rank and the gold-probe AUROC are invariant for the same reason. **"Adjust the
temperature" is a no-op on retrieval and on the probe.** It changes only the loss value and, during
training, the gradients.

**(b) But the learned τ is itself a clean overfitting diagnostic.**

| | seed 2026 | seed 2027 | seed 2028 |
| --- | --- | --- | --- |
| τ⁻¹ minimising **validation** loss | 11.50 | 11.50 | 11.50 |
| τ⁻¹ **learned** | 20.39 | 20.36 | 19.49 |
| τ⁻¹ minimising matched-N **train** loss | 32.00 | 30.00 | 25.50 |

The learned temperature sits **between** the two, in all three seeds: sharper than validation wants,
softer than the training distribution wants. That is what a model fit to a distribution it has
memorised looks like. Forcing τ⁻¹ to 11.5 would lower validation loss from 3.70 to 3.34 and **change
retrieval by exactly zero**.

## Conclusions

1. **The model overfits substantially, and item D's premise is confirmed.** Train R@1 0.404 versus
   validation 0.134 at matched N, with a matched-N loss gap of −1.85 to −1.99 on two independent
   routes and no duplicate-structure explanation.
2. **Temperature is not the fix.** It cannot reorder anything; every retrieval and probe number is
   exactly invariant to it. Do not spend an arm on it. Its only use here is as the diagnostic in
   Result 4(b).
3. **The fix, if one is wanted, is capacity or regularisation** — the prescription item D always
   carried, now with evidence. The tower has **no dropout, no random erasing and no stochastic
   depth**, and `weight_decay` applies only to matrix-shaped parameters. Augmentation exists
   (±8° rotation, 8% zoom, 5% shift, 10% intensity, per plane) and is evidently not enough.
   *This is a proposal, not an authorized arm.*
4. **This makes concern #3 quantitative, and merges two worries into one.** An image tower that
   retrieves seen pairs 3× better than unseen ones, evaluated on 56 gold studies that were
   themselves training pairs, is precisely the mechanism by which the 0.7105 probe AUROC would fail
   to transfer. Representation exposure and memorisation were being tracked as separate concerns;
   they are the same concern with a number on it now.
5. **Fix the reporting.** `train_loss` and `validation_loss` must never be quoted as a pair again
   without saying that one is batch-32-augmented and the other full-441-clean. The docstring in
   `embedding_health_metrics` asserts they are "comparable to train loss" and is **wrong**. No source
   edit was made here; correcting it is the user's call.

## Scope

R@1 figures on this page are Recall@1 at N=441 under the repository tie policy. They are **not**
comparable to the gold-probe **AUROC** deltas quoted elsewhere in `digest/`, and must not share a
table with them.

Hard-negative gate data for lever A was computed while these embeddings were in memory and is kept
separately in [`hard-negative-gate-data.json`](hard-negative-gate-data.json). **The gate fails at
batch 32**: on real training batches the top-1 negative already carries **0.73–0.75** of the
off-diagonal softmax mass and the top-3 carries **0.94–0.95** over 31 negatives, so the objective is
already almost entirely a hard-negative loss and explicit reweighting would measure noise. The
matched-N=441 shares (top-1 0.269) are the same quantity at a different candidate count and are
**not** the gate — spreading one softmax over 440 negatives instead of 31 necessarily lowers the
top-1 share. Lever A remains unauthorized either way; this is evidence for the decision, not the
decision.
