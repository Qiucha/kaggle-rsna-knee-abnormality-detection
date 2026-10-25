# LM label agreement as a contrastive target: the mechanism works, the effect is indistinguishable from rerunning the same config

Measured 2026-09-16 (Asia/Taipei). Evidence: [`label-similarity-v1.json`](label-similarity-v1.json) ·
[`downstream-probe-label-similarity-v1.json`](downstream-probe-label-similarity-v1.json) ·
[`label-similarity-paired-v1.json`](label-similarity-paired-v1.json).
Frozen config: [`../configs/contrastive/label-similarity-v1.json`](../configs/contrastive/label-similarity-v1.json) ·
Probe config: [`../configs/downstream/gold-probe-label-similarity-v1.json`](../configs/downstream/gold-probe-label-similarity-v1.json) ·
Analysis: [`../scripts/analysis/analyze_label_similarity.py`](../scripts/analysis/analyze_label_similarity.py).

## The idea

CLIP-style training treats every off-diagonal pair in a batch as a negative. That is wrong whenever
two studies describe the same pathology: a torn ACL with an effusion is not a negative for another
torn ACL with an effusion. `data/lm_label/lm_labels_kaggle.csv` carries LM-extracted verdicts for 12
findings across all 4,406 studies, which is a cheap proxy for "how alike are these two knees
clinically". This experiment mixes that agreement into the contrastive target and asks whether it
buys anything.

Target rule, per batch:

```
sim[i,j] = mean agreement over findings BOTH studies gave a YES/NO verdict for,   rescaled to [0,1]
         = 0                                                                       if they share none
target   = (1 - λ)·I + λ·rownorm(sim with diagonal forced to 1.0)
```

## Three design points, each of which would have produced a fake result if got wrong

**UNK is unobserved, not 50%.** Synovitis is 84% UNK, Fracture 56%. The bare value columns hold only
five distinct numbers per finding — they are a lookup from verdict and confidence, not calibrated
probabilities. Cosine over them yields a near-constant matrix (sd 0.147, p5–p95 0.37–0.88) that
trains as noise and reads as "the direction doesn't work". Masking to co-observed findings gives real
structure: **sd 0.236, p5–p95 0.13–1.00**, with only 2.8% of pairs sharing no observed finding.

**The diagonal has to be forced.** In a batch of 16, **47.6% of rows have some other study agreeing
perfectly on every co-observed finding.** Row-normalized agreement alone would hand those rows a
target that outranks the study's own report. Forcing the diagonal to 1.0 before normalizing, then
mixing with `(1-λ)·I`, keeps the true pair strictly the row max — asserted in the test suite.

**The cohort decides whether the number means anything.** Eight of the nine cohorts in
`data/processed/` put 48 of the 56 gold-probe studies in *train*. Since these labels cover exactly
the 12 findings the probe scores, training on them there would feed the probe's own label space into
the target for the studies it evaluates. `contrastive-cohort-gold-holdout-v1` is the only cohort that
parks all 56 outside train and validation; both arms use it.

## How to read the two loss columns

> **`train_loss` is NOT comparable between the arms, and subtracting the columns is meaningless.**
> The λ=0.3 arm's cross entropy is taken against a softened target whose mean entropy at batch 32 is
> **1.512 nats**. That is its irreducible floor. λ=0 has a floor of 0. Measured against their own
> floors the two arms fit the training set *equally well*: λ=0 ends at 0.272, λ=0.3 ends at 1.834,
> which is **0.322 above its floor** against λ=0's 0.272. The analysis script therefore refuses to
> report a `train_loss` difference at all.

> **`validation_loss` and `mean_r@1` ARE comparable.** `evaluation.py` was deliberately left
> untouched: both are scored against the unmodified hard diagonal over the full 433-study validation
> split, for both arms. `validation_loss` is not comparable to `train_loss` in either arm — different
> candidate counts — which is a pre-existing property of this pipeline, not something this run changed.

## Training curves

| epoch | λ=0 train | λ=0.3 train | λ=0 val | λ=0.3 val | λ=0 R@1 | λ=0.3 R@1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.629 | 3.054 | 4.437 | 4.583 | 0.0335 | 0.0404 |
| 4 | 1.214 | 2.362 | 3.618 | 3.923 | 0.0693 | 0.0635 |
| 8 | 0.843 | 2.140 | 3.561 | 3.709 | 0.0935 | 0.0837 |
| 9 | 0.794 | 2.113 | **3.484** | 3.683 | 0.0935 | 0.1051 |
| 11 | 0.660 | 2.037 | 3.644 | **3.643** | 0.0970 | 0.1201 |
| 16 | 0.353 | 1.874 | 3.765 | 3.826 | **0.1172** | 0.1247 |
| 17 | 0.324 | 1.853 | 3.800 | 3.860 | 0.1097 | **0.1316** |
| 20 | 0.272 | 1.834 | 3.824 | 3.879 | 0.1039 | 0.1259 |

Both arms overfit on the same schedule: validation loss bottoms around epoch 9–11 and rises for the
rest of the run while training loss keeps falling. Adding the label target did not change that.

## The result on retrieval is a null, and the control is what proves it

The λ=0 arm was **retrained** rather than reused, so both arms share one code state. That turned out
to matter for a reason the experiment did not anticipate: an identical run of the same config already
existed at `gold-holdout-v1/holdout-unfreeze4-seed2026`, and **nothing in this repository pins CUDA
determinism**. The two λ=0 runs agree at epoch 1 to 1e-4, then diverge. That divergence is the
yardstick.

| run | best `mean_r@1` | epoch | min `validation_loss` |
| --- | --- | --- | --- |
| λ=0 (retrained here) | 0.1172 | 16 | 3.4838 |
| λ=0.3 | **0.1316** | 17 | 3.6426 |
| λ=0 (pre-existing, same config **and** same seed) | 0.1409 | 15 | 3.5270 |

**λ=0.3 lands between the two λ=0 runs.** Two framings of the noise floor, because the first alone
would not reproduce:

- **Selected checkpoint: floor 0.0237, effect +0.0144.** The effect is inside it. But taking the best
  of 20 noisy epochs inflates both numbers, so this comparison flatters the floor.
- **Per-epoch, the like-for-like comparison:** the distribution of per-epoch `|Δ mean_r@1|` is
  **identical** between the two comparisons — median **0.0087** for pure nondeterminism, median
  **0.0087** for treatment-versus-baseline (means 0.0108 and 0.0111). The λ=0.3 arm deviates from the
  baseline by exactly as much as a rerun of the baseline does.

λ=0.3 leads on `mean_r@1` in 16 of 20 epochs, which looks like signal until you note that the
pre-existing λ=0 run leads the retrained one in 16 of 20 epochs by the same margin. Sign consistency
across epochs is not independent evidence here: the epochs within a run are correlated, so a single
lucky trajectory produces a long run of agreeing signs.

## The dissociation is the most informative thing in the run

**`mean_r@1` up in 16/20 epochs, `validation_loss` worse in 17/20 epochs** (median +0.1374, best-epoch
+0.1588). Both are scored on the same hard diagonal, so this is a genuine, consistent split rather
than a metric artefact.

This is mechanistically what softening predicts. The model spends capacity pulling label-alike studies
together, which blunts the sharp diagonal that InfoNCE scores, while leaving rank-1 retrieval intact.
The target is **reshaping the embedding geometry without adding discriminative information** — and
that reading is exactly what the gold probe independently confirms.

## AUROC on the 56 held-out gold studies

Integrity check first: the unadapted DINOv2 reference reproduces **0.6438**, the documented value, so
this probe run sits on the same footing as the project's existing numbers.

| arm | macro AUROC | macro AUPRC |
| --- | --- | --- |
| unadapted DINOv2 (floor) | 0.6438 | 0.5281 |
| λ=0 | 0.7056 | 0.5709 |
| λ=0.3 | **0.7096** | 0.5746 |

**Paired difference λ=0.3 − λ=0: +0.0040, 95% CI [−0.0100, +0.0170], 72.6% of resamples favouring
λ=0.3.** The config's pre-registered reading — *"a difference inside ±0.01 macro AUROC is a null at
this sample size"* — returns **null**. For scale, the same λ=0 configuration scored 0.6970 in the
earlier `gold-holdout-v1` run against 0.7056 here, a 0.0086 swing from nondeterminism alone, which is
twice the treatment effect.

No per-label difference clears its own interval either; the largest, Lateral OA at +0.0485, has
CI [−0.0090, +0.1111] on 11 positives.

## Conclusion

**Not worth a three-seed replication as specified.** All three reads agree, and the pre-registered
rule in the frozen config says to call this a null:

1. Retrieval: +0.0144 at the selected checkpoint, inside a 0.0237 selection-noise floor, and with a
   per-epoch deviation distribution *identical* to pure nondeterminism.
2. Validation loss: consistently worse, 17/20 epochs.
3. Gold probe: +0.0040, CI spanning zero, inside the pre-registered null band.

The honest summary is that **LM label agreement changes the geometry of the embedding space without
making it more informative about the findings** — which is coherent, because these labels are
extracted *from the reports*. Label agreement is a lossy compression of the text side, not new
supervision. Compressing the text signal into 12 axes and training on the compression does not beat
training on the text.

What was built is reusable and costs nothing when off: at λ=0 the loss is a bit-exact reproduction of
the unmodified one (verified to 1e-6 in `tests/test_label_similarity.py`), and both new config fields
default to the previous behaviour.

**Options if this is revisited** — offered as options, not as a recommendation to spend more GPU time:

- **λ=0.1.** The geometry distortion scales with λ; the validation-loss cost suggests 0.3 is too
  strong. A lower weight may keep the diagonal sharp.
- **Restrict to low-UNK findings.** Effusion, Medial Meniscus, ACL and PF OA are observed in most
  reports; Synovitis (84% UNK) and Fracture (56%) contribute mostly absence. A four-finding target
  would be denser and less dominated by masking.
- **Any replication needs ≥3 seeds.** A single seed cannot see an effect smaller than ~0.024 in
  selected `mean_r@1`, and the effects in play here are smaller than that.

## Operational notes for whoever runs this next

- **`evaluate_downstream_probe.py` is broken under `--device cuda`.** `extract_features` never moves
  `images` to the device, so any tower whose buffers are on the GPU raises a device mismatch. Not
  patched here, deliberately: that script stands behind published numbers. Run it on CPU.
- **The two virtualenvs are not interchangeable.** `.venv-cuda` has CUDA torch but no `sklearn`;
  `.venv` has `sklearn` 1.9.0 and CPU torch. Training needs the first, probing the second.
- **`configs/contrastive/full-projection-v1-resolved.json` gained two keys** (`lm_label_csv`,
  `label_similarity_weight`) at their defaults. That snapshot is asserted field-for-field by
  `test_frozen_baseline_matches_resolved_config`, so appending new fields is required maintenance,
  not drift.
