# The cheap probes: a lower weight really does help the contrastive objective, and none of it reaches AUROC

Measured 2026-09-17 (Asia/Taipei). Follow-up to [`label-similarity-v1.md`](label-similarity-v1.md),
running the two probes that page named. Evidence: [`label-similarity-v2.json`](label-similarity-v2.json) ·
[`downstream-probe-label-similarity-v2.json`](downstream-probe-label-similarity-v2.json) ·
[`label-similarity-paired-v2.json`](label-similarity-paired-v2.json).
Frozen config: [`../configs/contrastive/label-similarity-v2.json`](../configs/contrastive/label-similarity-v2.json) ·
Probe config: [`../configs/downstream/gold-probe-label-similarity-v2.json`](../configs/downstream/gold-probe-label-similarity-v2.json).

## What was tested and why

v1 found a null at λ=0.3 with all 12 findings, and offered two explanations, each testable in one run:

1. **λ=0.3 was too strong.** The validation loss got consistently worse, which suggested the soft
   target was distorting the embedding geometry more than the label signal was worth. → **λ=0.1**.
2. **Mostly-UNK findings diluted the signal.** Synovitis is 84% UNK, Fracture 56%, so most of the
   12-finding target is masking. → **λ=0.3 restricted to findings with UNK ≤ 10%**.

The baseline is v1's `lambda00-seed2026`, not retrained: every argument except the two label settings
is identical and it was trained hours earlier under the same code state.

> **The finding set came from a stated rule, not from v1's prose.** v1 named "Effusion, Medial
> Meniscus, ACL and PF OA" off-hand. Applying *UNK ≤ 10%* to the measured rates gives **Medial
> Meniscus (5.7%), ACL (8.1%), MCL (9.8%), Effusion (9.9%)**. PF OA is 18.5% and does not qualify;
> Lateral Meniscus misses by 0.03pp. The rule selected the set.

## Probe 2 was predicted to fail before it ran, and the prediction was recorded

Restricting to 4 binary findings admits at most **16 distinct clinical profiles**, so 3,478 training
studies collapse onto very few vectors. Measured at batch 32 *before launch* and written into the
frozen config:

| target | distinct off-diagonal values | pairs exactly 1.0 | rows with a perfect rival |
| --- | --- | --- | --- |
| all 12 findings | 46 | 6.0% | 68.6% |
| low-UNK 4 findings | **7** | **16.4%** | **92.6%** |

Fewer findings co-observe more often but describe *less*. The restricted target is **coarser**, not
sharper — closer to label smoothing than to a semantic similarity matrix. It was run anyway because
confirming a pre-registered prediction costs one run and is worth more than assuming it.

## Result 1: λ=0.1 genuinely improves the contrastive objective

> **Reminder on `train_loss`.** Each arm has its own irreducible floor, equal to the mean entropy of
> its target: **0.617** nats at λ=0.1, **1.513** at λ=0.3 all-12, **1.483** at λ=0.3 low-UNK-4, and
> **0** at λ=0. Comparing raw `train_loss` across arms is meaningless; the analysis script refuses to.
> `validation_loss` and `mean_r@1` are scored on the unmodified hard diagonal for every arm.

Per-epoch `validation_loss` against the shared baseline, negative meaning better:

| arm | median Δ | mean Δ | epochs better | plateau (mean, last 10 ep) |
| --- | --- | --- | --- | --- |
| **control rerun** (same config, same seed — noise only) | +0.0006 | −0.0055 | 8/20 | 3.6798 |
| λ=0.3, all 12 | +0.1374 | +0.1187 | 3/20 | 3.7968 |
| λ=0.3, low-UNK 4 | +0.1609 | +0.1331 | 2/20 | 3.8004 |
| **λ=0.1, all 12** | **−0.1227** | **−0.1203** | **17/20** | **3.5294** |

The control row is what makes this readable: pure nondeterminism produces a **balanced** 8/20 split
and a median of +0.0006. The treatments do not. λ=0.1 is better in 17 of 20 epochs, and its plateau
validation loss sits **0.205 below the baseline against a 0.055 noise spread** — roughly four times
the run-to-run variation on the same statistic.

This is the first quantity in either experiment to break cleanly out of noise. **Mixing a little
label agreement into the target is a real regularizer on the contrastive objective.**

There is also a clean dose-response, which is why it reads as mechanism rather than luck: λ=0.1
improves validation loss, λ=0.3 harms it, and λ=0.3 on the coarser 4-finding target harms it slightly
more (+0.1609 vs +0.1374, 2/20 vs 3/20). Probe 2 behaved exactly as pre-registered.

## Result 2: none of it reaches the gold probe

Integrity check: unadapted DINOv2 reproduces **0.6438** again, matching v1 and the project record.

| arm | macro AUROC | macro AUPRC | Δ vs λ=0 | 95% CI | frac favouring |
| --- | --- | --- | --- | --- | --- |
| unadapted DINOv2 (floor) | 0.6438 | 0.5281 | — | — | — |
| λ=0 (baseline) | 0.7056 | 0.5709 | — | — | — |
| **λ=0.1, all 12** | **0.7030** | 0.5758 | **−0.0025** | [−0.0190, +0.0140] | 0.379 |
| λ=0.3, all 12 | 0.7096 | 0.5746 | +0.0040 | [−0.0100, +0.0170] | 0.726 |
| λ=0.3, low-UNK 4 | 0.7093 | 0.5748 | +0.0037 | [−0.0109, +0.0185] | 0.694 |

Every difference is inside the pre-registered ±0.01 null band, every interval spans zero, and the
whole spread across four arms is 0.0066 — smaller than the 0.0086 that the *same* λ=0 configuration
moved between two runs in v1.

**The arm with the best validation loss has the worst AUROC.** λ=0.1 wins the contrastive objective
decisively and lands 0.0025 *below* baseline on the probe. That inversion is the result.

## What this means

v1 concluded that the label target "reshapes the embedding geometry without adding discriminative
information". v2 sharpens that into something more specific, because the regularization effect is now
measurable rather than inferred:

**At λ=0.1 the soft target acts as a regularizer on the contrastive objective — and regularizing the
contrastive objective does not make the features more informative about the findings.** A better
InfoNCE validation loss means the model is better at matching held-out studies to their reports. That
is not the same thing as encoding whether the knee has a torn ACL, and here the two come apart
completely.

This is coherent with why the whole idea was always borderline: the labels are extracted **from the
reports**. Agreement between two studies' labels is a lossy compression of the same text the model
already trains against. Feeding the compression back as a target can smooth the optimization — which
is what λ=0.1 does — but it cannot add information that was not in the reports to begin with.

The 4-finding restriction failed for the reason predicted before it ran: **denser co-observation is
not the same as a better target**, and past a point fewer findings simply means fewer distinguishable
studies.

## Conclusion

**Still not worth a three-seed replication for the purpose it was proposed for.** Neither cheap probe
rescued the idea:

- λ=0.1 buys a real improvement on the contrastive objective and **zero** on the gold probe (−0.0025).
- λ=0.3 restricted to low-UNK findings is the worst arm on validation loss and null on AUROC, as
  pre-registered.
- All four arms sit within 0.0066 macro AUROC of each other, inside single-seed noise.

**One narrow thing is worth remembering.** If a future experiment wants a regularizer for the
contrastive objective specifically — not better findings features — λ=0.1 label mixing is a cheap one
that measurably works, and it is already implemented behind a flag. That is a different goal from the
one this investigation started with, and it should be justified on its own terms rather than inherited
from here.

**What would actually be needed** to test the original hypothesis properly is labels that are *not*
derived from the reports — radiologist adjudication, or structured fields from the acquisition
metadata. The `data/cleaned.csv` gold labels are exactly that, but there are only 58 of them, which
is far too few to shape a 3,478-study training target. That is the real blocker, and no amount of
tuning λ addresses it.

## Reproduce

```bash
# arms (needs .venv-cuda)
PYTHONPATH=src .venv-cuda/bin/python scripts/training/train_contrastive.py \
  --cohort-manifest data/processed/contrastive-cohort-gold-holdout-v1.jsonl \
  --lm-label-csv data/lm_label/lm_labels_kaggle.csv --label-similarity-weight 0.1 \
  ...  # remaining flags in configs/contrastive/label-similarity-v2.json

# probe and pairing (needs .venv — sklearn lives there, and --device cpu is required)
PYTHONPATH=src .venv/bin/python scripts/evaluation/evaluate_downstream_probe.py \
  --config configs/downstream/gold-probe-label-similarity-v2.json \
  --features checkpoints/contrastive/label-similarity-v2/probe-features.npz \
  --output digest/downstream-probe-label-similarity-v2.json --device cpu

PYTHONPATH=src .venv-cuda/bin/python scripts/analysis/analyze_label_similarity.py \
  --experiment-version label-similarity-v2 \
  --experiment-config configs/contrastive/label-similarity-v2.json \
  --baseline checkpoints/contrastive/label-similarity-v1/lambda00-seed2026 \
  --treatment checkpoints/contrastive/label-similarity-v2/lambda01-seed2026 \
              checkpoints/contrastive/label-similarity-v2/lambda03-lowunk-seed2026 \
              checkpoints/contrastive/label-similarity-v1/lambda03-seed2026 \
  --control checkpoints/contrastive/gold-holdout-v1/holdout-unfreeze4-seed2026 \
  --probe digest/downstream-probe-label-similarity-v2.json \
  --paired digest/label-similarity-paired-v2.json \
  --output digest/label-similarity-v2.json
```

The v1 operational notes still apply: `.venv-cuda` trains and has no `sklearn`; `.venv` probes and
must run `--device cpu`, because `evaluate_downstream_probe.py` never moves `images` to the device.
`configs/contrastive/full-projection-v1-resolved.json` gained a third key this round,
`label_findings`, at its default — required by `test_frozen_baseline_matches_resolved_config`.
