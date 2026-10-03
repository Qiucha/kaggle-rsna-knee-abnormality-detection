# Validation loss versus retrieval quality — the late-window disagreement

> ## ⚠️ AMENDED 2026-09-13 — conclusion 1 of this page was WRONG and is withdrawn
>
> This page concluded that the memorisation premise "does not survive". **It does survive. The model
> overfits, substantially.** A direct matched-N measurement —
> [`digest/matched-n-loss-audit.md`](matched-n-loss-audit.md) — encoded the training split and found
> **train loss 1.77 versus validation 3.70 at the same N=441**, and **train R@1 0.434 versus
> validation 0.138**.
>
> **The method on this page could not have detected that.** It measured *deltas over a late window*
> and inferred the *level* from log(N) at chance. That inference was not a measurement and it was
> wrong. What this page does establish is the answer to a **different** question — why validation
> loss and validation R@1 *disagree over the late window* — and that answer stands. Conclusions 2–5
> below are unaffected. Conclusion 1 is replaced in place.

2026-09-13. Read-only audit over `metrics.jsonl` in **18** run directories already on disk.
No training was launched, no checkpoint was loaded, no model was run, the locked test split was
never touched, and `src/` was not edited. Reproduce with
`digest/handoff/analyze_val_loss_vs_r1.py` (writes [`.json`](val-loss-vs-retrieval-audit.json))
and `digest/handoff/analyze_val_loss_rank_depth.py` (writes
[`val-loss-rank-depth.json`](val-loss-rank-depth.json)).

## The hypothesis this tested

The standing handoff item D reads the pattern "train loss 0.298 vs validation loss 3.709, validation
loss flat from epoch 10 while validation R@1 keeps rising to epoch 20" as evidence that the image
tower **memorises its 3,526 training studies** — which would make the 0.71 gold-probe AUROC
untrustworthy — and uses it to motivate adding dropout, random erasing or stochastic depth.

The competing explanation, pre-registered and tested first: **validation loss and validation R@1 do
not measure the same object.** The loss is in-batch InfoNCE over batches of 32 at a **learned**
temperature; R@1 is retrieval over all 441 candidates and is scale-free. A sharpening temperature
inflates the loss on an unchanged ranking.

**Discriminator:** scale-free full-split ranking measures (median rank, R@1/R@5/R@10) over the window
from each run's validation-loss minimum to its final epoch. If ranking keeps improving there, the
loss rise is calibration, not degradation.

## Integrity check first

All 18 `metrics.jsonl` files have **unique, monotonic epochs starting at 1** — 0 rejected. This was
checked rather than assumed, because this repository has already produced one run directory with
duplicate epochs from interleaved concurrent writes.

## Results

**The divergence is systematic, not a seed artifact.** The validation-loss minimum falls at epochs
**7–13** while the retrieval optimum falls at **15–20**, and validation R@1 peaks *after* the
validation-loss minimum in **16 of 18** runs. (The companion statement "validation loss is worse at
the final epoch than at its minimum in 18/18" is **near-tautological** — the window start was
selected as that minimum — and carries almost no evidence; it is recorded only for completeness. The
window start being selected on *loss* is what makes the retrieval improvements below informative:
nothing about retrieval entered the choice of window.)

**Temperature is rising in the same window in 18 of 18 runs**, mean `inverse_temperature` change
**+2.62** over the window (+9.53 in the 40-epoch run). The loss and the ranking are not on a common
scale across epochs.

**Every scale-free retrieval measure improves over the window, in the true-pairing runs.** Means over
the 15 true-pairing runs, from validation-loss minimum to final epoch:

| Measure | True pairing (15 runs) | Shuffled-pairing control (3 runs) |
| --- | --- | --- |
| Validation loss | worse in 15/15 | worse in 3/3, by **+2.0 to +2.3** |
| mean R@1 | **+0.0248** | −0.0011 |
| mean R@5 | **+0.0462** | −0.0068 |
| mean R@10 | **+0.0368** | −0.0091 |
| text→image **median** rank | **−1.33** (better), improves in 14/15 | **+22.33** (far worse) |
| `inverse_temperature` | +2.51 | +0.37 |
| positive−negative similarity margin | +0.0182 | +0.0058 |

**The control is the clean positive control for real memorisation, and it looks nothing like the
true-pairing runs.** The shuffled-pairing arm, which is *known* to memorise random pairings, shows
validation loss rising an order of magnitude more (+2.0 to +2.3 versus +0.06 to +0.31), retrieval
quality **falling** on every measure, and median rank collapsing by 7–41 places — all while its
temperature barely moves (+0.29 to +0.44). Memorisation in this setup has a distinctive signature,
and the true-pairing runs do not carry it.

**A docstring in `src/` is wrong about the two losses being comparable.** (This section originally
also claimed the level gap was "largely a candidate-count artifact" — **withdrawn**, see the
amendment at the top; the structural mismatch below is real, the inference drawn from it was not.) `embedding_health_metrics` in `src/contrastive_learning/evaluation.py:200` computes
`validation_loss` as the symmetric InfoNCE over the **full 441 × 441** validation similarity matrix,
while `train_loss` is the in-batch loss over **32**. Chance level alone differs by
log(441) ≈ **6.09** versus log(32) ≈ **3.47**, so the headline "train 0.298 vs validation 3.709" is
comparing two different tasks and most of that gap is the candidate count. The function's docstring
asserts the opposite — *"the reported validation loss is comparable to train loss"* — which is how
this premise entered the handoff. **No source edit was made here** (that is the user's call), but the
docstring is wrong and should be corrected.

**How far the temperature mechanism is actually established.** It is *argued plus correlated*, not
directly measured: positive/negative similarity **means** cannot reconstruct an InfoNCE value, so
this is the ceiling available from disk. The argument is that at R@1 ≈ 0.14 roughly **86%** of
validation queries have their positive off top-1, and sharpening τ inflates the loss on exactly those
without bound. The strongest single datum is the 40-epoch run, which has both the largest temperature
move (**+9.53**) and the largest loss move (**+1.066**).

**One real caveat, and it is a different effect.** `mean` rank improves in only 7 of 18 runs while
`median` rank improves in 14 of 15 true-pairing runs. Late training therefore improves the bulk of
the ranking while pushing a **small tail of queries much further down** — enough to move the mean
against the median. That is consistent with the
[duplicate-confusion audit](duplicate-confusion-audit.md)'s finding that the bottleneck is
fine-grained discrimination at the very top of the ranking, and it is a tail effect, not
memorisation.

## Conclusions

1. **Item D's premise DOES survive — see the amendment above.** What this page settles is narrower,
   and three claims have to be kept apart.
   - *The loss/R@1 disagreement is explained* — learned-temperature scale plus a ranking tail. Over
     precisely the window where validation loss worsens, R@1, R@5, R@10 and median rank all improve.
   - *The inference from that observation to memorisation is unsupported.*
   - *The magnitude of the train-vs-validation level gap* is **not** addressed by anything on this
     page. It was measured directly afterwards and it is large: see
     [`matched-n-loss-audit.md`](matched-n-loss-audit.md).

   The control does less work than a first reading suggests. The shuffled arm has no learnable
   signal, so it is a **pure**-memorisation control: its signature being absent rules out pure
   memorisation, not a mixture — and a mixture is exactly what the matched-N measurement found.
2. **Do not motivate regularisation work (dropout, random erasing, stochastic depth) from this
   observation.** Its stated evidence is a scale artifact plus a tail effect. Regularisation may
   still be worth trying, but it needs a different justification.
3. **Validation loss is not a usable model-selection signal here** and selection on retrieval R@1 was
   the right call. Selecting on validation loss would have stopped at epoch 7–13 and given up
   +0.025 R@1.
4. **The exposure objection is untouched and remains the real threat to the probe number.** 56 of 58
   gold studies were training pairs or in the selection split. That is a separate argument from
   memorisation dynamics and this audit says nothing about it either way.
5. **New, cheap monitoring suggestion:** log **median** rank alongside mean rank when judging late
   training. Mean rank and median rank disagree systematically here, and mean rank alone would have
   supported the wrong conclusion.

## Scope

Retrieval R@1 figures on this page are **validation Recall@1 over 441 candidates**. They are not
comparable to the gold-probe **AUROC** deltas quoted elsewhere in `digest/`, and the two must not
appear in a common table.

Reading the three `plane-pooling-v1` runs' `metrics.jsonl` here is not a re-score of those runs; the
standing instruction not to re-read them concerns banking the plane-pooling result, which is not done
here.
