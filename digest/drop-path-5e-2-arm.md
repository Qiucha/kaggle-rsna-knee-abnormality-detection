# Stochastic depth at 5e-2 cuts memorisation by 41% and does not buy generalisation

Measured 2026-09-14 (Asia/Taipei). Pre-registered in the lever-D block of
[`handoff/handoff-20260913-overfitting-confirmed.md`](handoff/handoff-20260913-overfitting-confirmed.md)
before any code was written, including the predicted effect size and the null threshold.
Evidence: [`drop-path-5e-2-arm.json`](drop-path-5e-2-arm.json).
Runner: [`handoff/queue8-drop-path.sh`](handoff/queue8-drop-path.sh).
Analysis: [`handoff/analyze_drop_path_arm.py`](handoff/analyze_drop_path_arm.py).

One lever: `--image-drop-path-rate 5e-2`, masked to the 4 adapted DINOv2 blocks. Exposed cohort, 441
validation candidates, three matched seeds, 20 epochs.

**One-lever discipline verified at the strongest available level**, not by reading the run scripts:
the *resolved config recorded inside each checkpoint* was diffed, baseline against arm, for all three
seed pairs. Ignoring `seed`/`output_dir`/`resume`/`run_id`, the **only** differing field is
`image_drop_path_rate` (0.0 → 0.05) in every pair. The `BASE` argument array is also identical to
`queue2.sh`, the queue that actually produced the baselines — checked against queue2 directly rather
than through queue5's assertion that *its* args came from queue2.

> **Superseded in scope, confirmed in substance (2026-09-15).** A six-rate sweep over 20× of rate —
> [`drop-path-sweep-v1.md`](drop-path-sweep-v1.md) — found this is not a property of 5e-2: there is
> **no knee** anywhere. The memorisation gap can be cut **57%** and retrieval still only falls. Read
> the sweep for the general result; this page remains the detailed single-rate record.

## The answer

> **The regulariser worked, and it did not help.** The matched-N train/validation R@1 gap — the
> direct measure of memorisation, and the reason this lever was chosen — fell from **+0.2679 to
> +0.1579**, a **41% reduction**, same direction at three seeds. Validation loss improved even after
> controlling for temperature. But **validation retrieval R@1 got slightly worse**: pooled paired
> difference **−0.0153**, same sign at 3 of 3 seeds.
>
> **Less memorisation did not become more generalisation.** That is the finding, and it is a
> different one from "the lever did nothing".

## Retrieval — the primary estimand, pre-registered

| seed | baseline | drop-path 5e-2 | paired difference | paired-query 95% CI |
| --- | --- | --- | --- | --- |
| 2026 | 0.1383 | 0.1224 | −0.0159 | [−0.0420, +0.0102] |
| 2027 | 0.1434 | 0.1224 | −0.0210 | [−0.0476, +0.0074] |
| 2028 | 0.1202 | 0.1111 | −0.0091 | [−0.0374, +0.0193] |
| **pooled** | **0.1340** | **0.1187** | **−0.0153** | **[−0.0308, +0.0004]** |

**State the strength honestly.** Three of three seeds share the sign and **97.1%** of pooled
resamples favour the baseline, but the pooled 95% interval **grazes zero at +0.0004** and every
per-seed interval spans zero. This is a **consistent directional harm that falls just short of
exclusion at 95%** — not a demonstrated harm, and not a null.

**The pre-registration did not anticipate the sign.** Its "below +0.005 ⇒ null" clause was written
for a small positive effect. A consistent negative of three times that magnitude must not be filed
under it, and is not.

## Memorisation — the pre-registered diagnostic, and the reason this run matters

Read at matched N=441 on eval-mode embeddings with augmentation off, so drop-path is identity in
**both** arms and the two are measured on the same footing.

| | train R@1 | validation R@1 | gap | ratio |
| --- | --- | --- | --- | --- |
| baseline | 0.4019 | 0.1340 | **+0.2679** | **3.00×** |
| drop-path 5e-2 | 0.2766 | 0.1187 | **+0.1579** | **2.33×** |

**The gap closed from the train side, not the validation side.** Train R@1 fell 0.4019 → 0.2766
(−31%) while validation fell only 0.1340 → 0.1187. The model memorises substantially less, exactly as
§1's mechanism predicted it would, and reads unseen pairs no better for it.

The matched-N loss gap agrees — **computed at a matched τ for both arms**, which it must be: a
softer τ compresses both splits toward log(N) and shrinks the gap independently of memorisation, so a
gap read at each arm's own learned τ is not comparable across arms. At matched τ the gap is
**−1.83 / −1.90 / −1.56 → −0.98 / −1.31 / −1.08**, shrinking at all three seeds. (The R@1 gap above
needs no such correction: τ cannot reorder a similarity matrix, so it is τ-invariant by §5. That
invariance is why R@1 carries this finding and the loss gap only corroborates it.)

## Validation loss is a real improvement, and only partly temperature

Learned τ⁻¹ moved down, 20.39 / 20.36 / 19.49 → **18.13 / 18.43 / 17.80**. Since τ scales all logits
and cannot reorder a similarity matrix (§5), a τ move changes loss while changing retrieval by
exactly zero — so the raw loss improvement could not be read as representation quality.

Recomputed at a **matched** τ per seed, the improvement survives:

| seed | baseline | drop-path | difference |
| --- | --- | --- | --- |
| 2026 | 3.6319 | 3.3616 | −0.2703 |
| 2027 | 3.7036 | 3.4792 | −0.2244 |
| 2028 | 3.6091 | 3.4880 | −0.1211 |

So the arm genuinely lowers validation InfoNCE loss. **It still does not improve validation
retrieval** — a further instance of the loss/retrieval dissociation already on file in
`val-loss-vs-retrieval-audit.md`.

## What is NOT evidence here

**The `train_loss` rise (0.28 → 0.61) is inadmissible for this lever and must not be quoted as
evidence that memorisation fell.** `train_loss` is computed in train mode (`training.py:158`), so the
arm's logged forward pass has drop-path **active** and the baseline's does not. Any forward-pass
noise raises it whether or not memorisation changed. This is a **third variant** of the trap
reporting rule 1 exists for. The matched-N table above is the admissible route, and it is why the
encode pass was run rather than reading the metrics log.

## What this does and does not settle

- **Does not close lever D.** This is **one rate, one sub-lever, at 4 blocks**. Projection-head
  dropout, random erasing, and lower rates are all untouched. Lever D remains open.
- **Does close one hypothesis**, and this is the transferable part: the project's working assumption
  was that the 3.0× memorisation gap was *the* thing standing between it and better numbers. **It
  was cut by 41% — 3.00× down to 2.33× — and retrieval did not improve.** Whatever limits retrieval here is not only
  memorisation capacity, so a successor should not expect a second regulariser to pay off merely
  because it reduces the gap further.
- **Untested at a lower rate.** 5e-2 may simply be too strong for 4 adapted blocks over 20 epochs;
  the train-side collapse is large. A rate of 1e-2 or 2e-2 is the obvious follow-up **if** the user
  wants one, but the result above is a reason to doubt that direction rather than to expect it to
  work.
- **Selection was retrieval R@1 throughout.** No probe number was used to select anything.
- Validation R@1 here is over **441** candidates. Never place these beside the 433-candidate
  gold-holdout figures.
