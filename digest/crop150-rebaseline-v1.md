# Re-baseline at a 150mm crop: more anatomy, coarser pixels, lower retrieval

Measured 2026-09-15 (Asia/Taipei). Pre-registered in the overnight block of
[`handoff/handoff-20260913-overfitting-confirmed.md`](handoff/handoff-20260913-overfitting-confirmed.md).
Runner: [`handoff/queue10-overnight.sh`](handoff/queue10-overnight.sh) stage D.

**This is a re-baseline, not an arm.** `crop_mm` 130 → 150 changes the pixels, so **no recorded run is
comparable to it**; it becomes the new comparison point for anything built on the new preprocessing.
The constant was chosen on measured tissue retention
([`acquisition-metadata-trust.md`](acquisition-metadata-trust.md)), **not** on a predicted score.

> ## ✓ CONVERGENCE CHECKED 2026-09-15 — this reading stands.
>
> The same audit that narrowed the stochastic-depth conclusion
> ([`convergence-curves.html`](convergence-curves.html)) was applied here, and **the crop150 runs are
> not truncated**. Their validation minimum arrives at epoch **9.0** against the baseline's 11.3 —
> *earlier*, not later — and validation loss then drifts up **+0.280**, slightly more than the
> baseline's +0.259. The best epoch sits at the budget boundary in **0 of 3** seeds.
>
> Every signature says crop150 converged and then overfit, so its lower retrieval is **not** a
> training-budget artifact and the two mechanisms proposed below remain the live explanations.
> This distinguishes it from the drop-path arms, where the same audit found real truncation.

## What changed

| | 130mm | 150mm |
| --- | --- | --- |
| tissue retained (mean / p10) | 0.861 / 0.706 | **0.957 / 0.884** |
| millimetres per pixel at 224² | 0.58 | **0.67** (15% coarser) |
| crop behaviour | crop, silently skipped when FOV ≤ 130mm | crop **or pad**, exact for every series |

## Result

| seed | crop130 (recorded) | crop150 (re-baseline) |
| --- | --- | --- |
| 2026 | 0.1383 | 0.1156 |
| 2027 | 0.1434 | 0.1202 |
| 2028 | 0.1202 | 0.1213 |
| **mean** | **0.1340** | **0.1190** |
| **seed spread** | **0.0232** | **0.0057** |

**Validation retrieval is lower: 0.1340 → 0.1190.** Selected epochs also came earlier (14/14/17
against 19/18/15).

**Do not read this as the crop change being wrong.** It was pre-registered that either direction is
informative and neither is a failure, because the crop was chosen on anatomy retained rather than on
retrieval. Two mechanisms are consistent with the drop and this run does not separate them:

1. **Coarser pixels.** 0.58 → 0.67 mm/px is a 15% resolution loss on anatomy that was already in
   frame, which is the cost side of the trade the retention table made explicit.
2. **Dilution.** The extra 96mm² of retained field is disproportionately muscle and subcutaneous fat,
   which adds patch tokens without adding discriminative content.

## Memorisation: the same parallel trade, on a worse exchange rate

Matched-N gap at a matched τ, on eval-mode embeddings ([`crop150-memorisation.json`](crop150-memorisation.json)):

| | train R@1 | validation R@1 | gap |
| --- | --- | --- | --- |
| crop130 | 0.4013 | 0.1340 | **+0.2674** |
| crop150 | 0.3511 | 0.1190 | **+0.2321** |

The wider crop **also reduces memorisation** — and costs retrieval doing it, which is the same
parallel trade the stochastic-depth sweep found ([`drop-path-sweep-v1.md`](drop-path-sweep-v1.md)).
Two seeds move that way; **seed 2028 moves the other way** (gap +0.2297 → +0.2695), so the direction
is 2 of 3, not 3 of 3.

**The exchange rate is worse here than for drop-path.** Retrieval lost per unit of gap closed is
≈0.42 for the crop against ≈0.14–0.17 for drop-path at 0.05–0.1. **Read that comparison loosely**:
drop-path holds the pixels fixed and perturbs the forward pass, whereas the crop changes the input
itself, so the two "gaps" are measured over different representations of the data. It is suggestive,
not a like-for-like ratio.

## The observation worth following up

**The seed spread collapsed roughly fourfold, 0.0232 → 0.0057.** If that holds, it matters more than
the level: the plane-pooling session recorded a false null purely because a decision rule was applied
against a seed spread wider than the effect, and every arm since has had to pool across seeds to see
anything. A tighter noise floor raises the power of every future experiment.

**Treat it as provisional.** Three seeds give a very noisy estimate of a spread, and this is exactly
the kind of number that regresses. It deserves a direct check before anyone plans around it.

## ✅ ANSWERED 2026-09-15 — the probe ran: [`crop150-probe-v1.md`](crop150-probe-v1.md)

The question this section left open — does retaining 96% of tissue instead of 86% help the twelve
labels? — now has a measured answer, and it is **no**. Pooled paired difference
**−0.0201** macro AUROC, crop150 minus crop130, same sign in all three seeds. The crop costs
**both** retrieval and label readout, so the two readouts did **not** dissociate here.

Of the two mechanisms proposed below, **dilution is the better-supported one**. A notable detail:
the wider crop *helps* the **unadapted** backbone (0.6438 → 0.6529) while hurting the adapted one, so
the added field is not empty — the adaptation simply fails to exploit it.

**The crop is not adopted.** The `config.py` default stays at 130.0.

<details><summary>The original framing, kept as the record</summary>

**No downstream probe has been run on these checkpoints**, so the question the crop was actually
chosen to serve — does retaining 96% of tissue instead of 86% help the twelve labels? — is still open.
Retrieval and label readout have dissociated before in this project
([`val-loss-vs-retrieval-audit.md`](val-loss-vs-retrieval-audit.md),
[`drop-path-sweep-v1.md`](drop-path-sweep-v1.md)), so the retrieval drop does not settle it.

</details>

**Selection remains retrieval R@1.** Nothing here was or should be selected on label AUROC.

## Provenance

Cohort manifest unchanged (its read-time invariants do not involve `crop_mm`). Every training argument
byte-identical to `queue2.sh` apart from the cache directory and `--crop-mm 150`. The 130mm cache was
not touched; the re-baseline built its own at `crop150-v1/image-cache`. `--crop-mm` did not previously
exist on the CLI and was added for this run.
