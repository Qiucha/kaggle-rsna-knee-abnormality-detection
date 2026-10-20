# Does contrastive learning help the downstream task? A first measured answer

Read-only probe evaluation, 2026-09-11 (Asia/Taipei). Config:
[`configs/downstream/gold-probe-v1.json`](../configs/downstream/gold-probe-v1.json).
Evidence: [`downstream-probe-v1.json`](downstream-probe-v1.json). Script:
`scripts/evaluate_downstream_probe.py`. No training was run, no locked-test study was
loaded, and nothing was written into an existing run directory.

## The question, and the two answers

The project selects checkpoints on retrieval Recall@1, but the working goal in
[ROADMAP.md](../ROADMAP.md) is pathology prediction. This is the first time the two have
been measured against each other. The answer splits cleanly in two, and the split is the
finding:

> **Contrastive learning improves downstream label readout by +0.026 macro AUROC
> (0.644 → 0.670), consistently at all three seeds — but only in the arm that actually
> unfroze a DINOv2 block. The frozen-tower runs, which are most of the completed
> experiment history, cannot improve these features at all, and that is provable rather
> than merely observed.**

## Why the frozen-tower runs are a structural zero

Every run with `--image-unfreeze-last 0` trains only the plane attention and the 256-d
projection head. **It never updates a DINOv2 weight.** Its pre-projection image features
are therefore the same numbers the unadapted backbone produces, and the probe confirms
it exactly:

| Frozen-tower run | Max absolute deviation from unadapted DINOv2 | Identical |
| --- | --- | --- |
| `claude-translated-dfd07db-q2-v1/00-full-seed2026` | 0.0 | yes |
| `claude-translated-dfd07db-image-block-v1/full-seed2027` | 0.0 | yes |
| `claude-translated-dfd07db-image-block-v1/full-seed2028` | 0.0 | yes |

This was declared in the config before the run as a correctness check, and it doubles as
the pipeline's own validation: the feature extractor, the fold assignment, and the
scoring path all reproduce a known-zero difference to the last bit.

The consequence is worth stating plainly. The retained baseline at 10.6576% mean R@1,
the impression-first comparison, the two English/medical text screens, and the text-only
adaptation arm all left the image representation byte-identical to off-the-shelf
DINOv2-small. Whatever those experiments measured, it was not a change in what the image
tower knows. Only `--image-unfreeze-last 1` moved it.

## The primary comparison

Matched pooling, matched dimensionality: uniform mean over the three planes, 768-d, so
the only difference between the arms is the adapted final DINOv2 block.

| Arm | Macro AUROC | Difference | Paired 95% interval | Replicates favouring after |
| --- | --- | --- | --- | --- |
| Before CL — unadapted DINOv2-small | 0.6438 | — | — | — |
| After CL — image-block seed 2026 | 0.6687 | **+0.0249** | +0.0090 to +0.0415 | 99.9% |
| After CL — image-block seed 2027 | 0.6706 | **+0.0268** | +0.0112 to +0.0427 | 99.9% |
| After CL — image-block seed 2028 | 0.6709 | **+0.0271** | +0.0108 to +0.0442 | 99.9% |

Three independently seeded runs agree in sign and magnitude, and each interval excludes
zero. This is a more stable signal than the retrieval arm it came from, whose +1.2094 pp
gain had a sample SD of 0.6643 across the same three seeds. Retrieval and downstream
readout point the same way here, which is evidence that R@1 was a usable proxy *for this
arm* — not that it is one in general, as the next section shows.

## Per label

Positives and negatives are over the 56-study population. **After** and **Difference**
are the mean over the three seeds. The **interval** is seed 2026 alone: a paired
bootstrap resamples one arm's actual per-study scores, so it cannot be formed from an
average of three runs. It is therefore a guide to the width of the per-label uncertainty,
not a test of the three-seed difference beside it. Per-seed values are in the JSON
artifact.

| Label | Pos / Neg | Before | After | Difference | Paired 95% interval |
| --- | --- | --- | --- | --- | --- |
| Synovitis | 26 / 30 | 0.468 | 0.540 | **+0.072** | +0.014 to +0.133 |
| PF OA | 20 / 36 | 0.519 | 0.569 | **+0.050** | +0.020 to +0.099 |
| Fracture | 17 / 39 | 0.486 | 0.534 | +0.048 | −0.023 to +0.104 |
| ACL | 23 / 33 | 0.524 | 0.558 | +0.034 | −0.032 to +0.087 |
| MCL | 8 / 48 | 0.714 | 0.745 | +0.031 | −0.020 to +0.093 |
| Lateral Meniscus | 22 / 34 | 0.571 | 0.602 | +0.031 | −0.010 to +0.071 |
| Effusion | 33 / 23 | 0.768 | 0.799 | +0.031 | −0.008 to +0.068 |
| Contusion | 19 / 37 | 0.795 | 0.821 | +0.026 | −0.030 to +0.081 |
| Medial OA | 14 / 42 | 0.804 | 0.820 | +0.015 | −0.018 to +0.055 |
| Baker's | 12 / 44 | 0.680 | 0.693 | +0.013 | −0.057 to +0.078 |
| Medial Meniscus | 26 / 30 | 0.792 | 0.782 | −0.011 | −0.054 to +0.032 |
| Lateral OA | 11 / 45 | 0.604 | 0.579 | −0.025 | −0.063 to +0.015 |
| **Macro** | | **0.644** | **0.670** | **+0.026** | +0.009 to +0.042 |

Ten of twelve labels move up, and only two per-label intervals exclude zero — the macro
gain is broad and small rather than driven by one label. The largest gains are on the
labels the unadapted features read worst: Synovitis, PF OA and Fracture all start within
noise of chance (0.47–0.52) and end merely poor. The labels that already read well
(Medial OA, Contusion, Medial Meniscus, Effusion, all ≥ 0.77 before) gain little or
nothing. MCL is reported pooled-out-of-fold only, as declared in advance: at 8 positives
its fold-level AUROC is uninformative.

## Where the gain is lost again

The primary comparison uses uniform plane averaging for both arms. Substituting the
representation the contrastive model *actually uses* changes the picture:

Each column is the mean of three seeds. The frozen-tower column is the structural
zero — those runs never updated a DINOv2 weight — so it is what each feature set scores
when the backbone provably did not learn anything.

| Feature set | After CL | Frozen-tower null | After CL vs before CL |
| --- | --- | --- | --- |
| Uniform plane mean, 768-d | **0.670** | 0.644 | **+0.026** (+0.009 to +0.042) |
| Learned plane attention, 768-d | 0.650 | 0.644 | +0.006 (−0.027 to +0.041) |
| Projected embedding, 256-d | 0.673 | **0.676** | +0.030 (−0.043 to +0.117) |

Read the rows against the null column rather than down the "After CL" column.

**Uniform mean** is the only row where the adapted arm separates from a backbone that
cannot have learned: 0.670 against a null of exactly 0.644.

**Learned plane attention** — trained to serve retrieval, and carried in every
checkpoint — gives back roughly three quarters of the backbone's gain when the task is
label readout. The next section takes that apart; it has its own evidence file,
[`plane-pooling-analysis.json`](plane-pooling-analysis.json), and its own config,
[`gold-probe-planes-v1.json`](../configs/downstream/gold-probe-planes-v1.json), which
copies this probe's folds, seeds and resamples by value so the numbers are comparable.

**The projected embedding is noise, and its own null proves it.** Its 0.673 looks like
the best number in the table, but the structural-zero arm scores *higher* at 0.676, and
the single highest macro AUROC anywhere in this run — 0.6886 — belongs to
`frozen_tower_null/full-seed2027/projection_256`, a provably unchanged backbone. A
linear probe cannot profit from a learned 768→256 compression, so nothing else was
available for it to find; the spread across seeds (0.656 to 0.689) is variance. Do not
read +0.030 as a result in either direction.

## What this is not

**It is not generalisation evidence.** *(Amended 2026-09-14: this line originally read "and
cannot be made into any", which is now false — the gold-holdout replication
([`gold-holdout-v1.md`](gold-holdout-v1.md)) did exactly that by retraining with the gold studies
excluded, putting the generalisation estimate at **0.6938** against 0.7106 exposed. The narrower
claim below still stands for **this** artifact's own numbers.)* 48 of the 56 studies
scored here were image/report training pairs in every completed run on this cohort, and
the other 8 are in the split that selected every checkpoint. The two gold studies never
touched by representation learning are both in the locked test and were not loaded. The
estimand is readout quality of a frozen representation on a cohort it has already seen;
the cross-validation controls probe overfitting, not representation exposure. See the
[gold-case inventory](gold-case-inventory.md).

**Studies are not patients.** Anonymisation is per study, so the bootstrap resamples
studies and the independence its interval assumes is unverifiable. See the
[integrity audit](integrity-coverage-audit.md).

**The population is 56 studies.** Every number carries the uncertainty that implies, and
the probe was fixed in advance — penalty, folds, seeds, no threshold search, no model
selection — precisely because nothing tuned on 56 cases would mean anything.

## Reference: the text side of the same task

The LM report-reading labels scored against the same gold on the same 56 studies, for
scale:

| Label source | Macro AUROC |
| --- | --- |
| `gpt5.6_sol_medium.csv` | 0.889 |
| `gemini3.7_flash_high.csv` | 0.888 |
| `gpt5.6_sol_high.csv` | 0.843 |
| `gemini3.1_pro_high.csv` | 0.821 |
| Best image representation measured here | 0.673 |

This is not a like-for-like contest: those models read the radiologist's report, which
states the findings, while the probe reads three MRI slices per plane. It bounds the
label noise rather than the images' difficulty — an extractor reading the answer key
still only reaches 0.89, so the gold labels and the reports do not agree perfectly, and
no image model should be expected to exceed that ceiling.

### The notebook's AUROC inflates these numbers

Scored above with midrank AUROC (`sklearn.metrics.roc_auc_score`). Section 6 of the EDA
notebook computes AUROC from a plain `argsort`, which hands every tied score a distinct
consecutive rank in sort order instead of the average rank the tied block shares. LM
label outputs are heavily tied — MCL carries 41 identical `0.0` values over 56 studies —
so this is not a hypothetical. Both were computed on the same columns:

| Label source | Notebook `argsort` macro | Midrank macro | Inflation |
| --- | --- | --- | --- |
| `gemini3.7_flash_high.csv` | 0.9012 | 0.8875 | +0.0136 |
| `gpt5.6_sol_high.csv` | 0.8514 | 0.8426 | +0.0088 |
| `gemini3.1_pro_high.csv` | 0.8252 | 0.8207 | +0.0045 |
| `gpt5.6_sol_medium.csv` | 0.8926 | 0.8894 | +0.0033 |

**All 48 per-label values differ**, in every file and on every label, by a mean of
+0.0075 and up to +0.0576 (`gemini3.7_flash_high` Lateral OA: 0.9091 against 0.8515).
The effect is not uniformly optimistic — `gemini3.1_pro_high` Lateral Meniscus goes the
other way, 0.7807 against 0.8275 — so it moves individual comparisons in both
directions, not just the headline. The macro inflation is small; the per-label
distortion is large enough to reorder models on a given label.

This is the same failure the project already corrected in its retrieval metrics, where
tied similarity scores are given expected credit under `metric_tie_policy`. The
notebook's section 6 predates that correction and was not updated. Both numbers are in
[`downstream-probe-v1.json`](downstream-probe-v1.json) under
`notebook_argsort_minus_midrank` so the claim can be checked rather than taken.

## Why the learned plane pooling reads labels worse than averaging

This is the odd result in the table above, so it is worth being precise about. The image
tower does not average its three plane vectors; it scores them with a small learned
network and pools them by softmax attention. Attention is strictly more expressive than
a mean — it *can* represent the mean — yet on label readout it lands below one.

**First, the size of the claim.** Paired against the uniform mean *of the same
checkpoint*, on the same studies and folds:

| Checkpoint | Uniform mean | Attention pooled | Difference | Paired 95% interval | Replicates favouring uniform |
| --- | --- | --- | --- | --- | --- |
| image-block seed 2026 | 0.6687 | 0.6503 | −0.0185 | −0.048 to +0.010 | 90% |
| image-block seed 2027 | 0.6706 | 0.6491 | −0.0214 | −0.049 to +0.006 | 94% |
| image-block seed 2028 | 0.6709 | 0.6504 | −0.0205 | −0.044 to +0.006 | 95% |

**Every interval spans zero.** Three independent seeds agree in sign and to within
0.003 in magnitude, and 90–95% of bootstrap replicates favour the mean, but at 56
studies this is a direction-consistent observation, not a demonstrated deficit. Read
what follows as a mechanism for an effect of that strength.

### Three candidate causes, two eliminated

| Feature set (mean of 3 seeds) | Macro AUROC | vs uniform mean |
| --- | --- | --- |
| Uniform plane mean, 768-d | 0.6701 | — |
| All three planes concatenated, 2304-d | 0.6689 | −0.001 |
| Attention pooled, 768-d | 0.6499 | −0.020 |
| Attention weights without the plane-embedding term, 768-d | 0.6492 | −0.021 |

**It is not the plane-embedding term.** `ImageTower` adds a learned per-plane vector
before pooling, so the pooled feature carries `Σ wₚeₚ` — a study-dependent nuisance in
the span of three vectors that has nothing to do with the pixels. Removing it changes
the score by 0.0007. Not the cause.

**It is not information loss from pooling 3→1** — and this is the most useful single
number in the decomposition. Concatenating all three planes and letting the probe weight
them itself scores 0.6689, against the plain mean's 0.6701: indistinguishable. Keeping
everything buys nothing over averaging, so **there is nothing in the three planes that
the mean throws away and a better pooling could recover.** That is what makes "export the
pre-attention features and average them" a real recommendation rather than a guess — the
mean is not a lossy compromise to be improved on later, it is already at the ceiling
these three planes support. (At 2304 features over 56 cases this arm could have collapsed
from dimensionality alone; it did not, so it is a usable bound.)

**It is the re-weighting itself** — the only remaining term, and the one the two
eliminations leave.

### What the attention actually does

The weights are not collapsed onto one plane, but they are clearly tilted:

| Checkpoint | Sagittal | Coronal | **Axial** | Mean entropy (uniform = 1.099) | Argmax is Axial |
| --- | --- | --- | --- | --- | --- |
| seed 2026 | 0.256 | 0.236 | **0.508** | 0.998 | 48 / 56 |
| seed 2027 | 0.264 | 0.270 | **0.466** | 1.028 | 46 / 56 |
| seed 2028 | 0.256 | 0.261 | **0.484** | 1.017 | 47 / 56 |

All three seeds independently learned the same thing: put about half the mass on Axial
and split the rest, for nearly every study. Study-to-study variation is real but small
(SD 0.07–0.12), so this is close to a *fixed* plane preference that happens to be
computed per study.

### Why a fixed plane preference cannot serve twelve labels

Probing each plane on its own shows where each label's signal actually lives. Mean of
three seeds; the best plane per label is in bold.

| Label | Sagittal | Coronal | Axial | Best plane | Attention effect |
| --- | --- | --- | --- | --- | --- |
| PF OA | 0.612 | 0.535 | **0.732** | Axial | **+0.082** |
| Lateral OA | 0.357 | 0.609 | **0.685** | Axial | +0.034 |
| Medial OA | 0.728 | 0.788 | **0.851** | Axial | +0.009 |
| ACL | 0.497 | **0.539** | 0.445 | Coronal | −0.004 |
| Medial Meniscus | 0.765 | 0.678 | **0.801** | Axial | −0.008 |
| Baker's | **0.862** | 0.576 | 0.473 | Sagittal | −0.023 |
| Fracture | 0.510 | 0.574 | **0.608** | Axial | −0.027 |
| Contusion | 0.667 | **0.809** | 0.650 | Coronal | −0.039 |
| Synovitis | **0.585** | 0.538 | 0.516 | Sagittal | −0.045 |
| Effusion | **0.802** | 0.710 | 0.696 | Sagittal | −0.048 |
| MCL | 0.447 | **0.596** | 0.484 | Coronal | −0.082 |
| Lateral Meniscus | **0.583** | 0.539 | 0.509 | Sagittal | −0.091 |

The table sorts itself. **The three labels the attention helps most are the three whose
signal is strongest on Axial**, and the two it hurts most — Lateral Meniscus and MCL —
are labels where Axial is the *worst* of the three planes. Across the twelve labels, how
much Axial beats the average of the other two planes correlates with how much the
attention helps that label at r = 0.62 (Spearman 0.53, p = 0.075, n = 12).

Treat that correlation as decoration, not evidence. Both quantities are computed from
the same per-plane probe scores on the same 56 studies, so a label whose axial probe
happened to score high will tend to show a high axial advantage *and* a high attention
benefit for that reason alone. They are not independently estimated, and at n = 12 the
coefficient would be fragile even if they were. The table carries the argument; the
correlation only summarises it.

The anatomy is the ordinary anatomy, which is the reassuring part. Baker's cyst is
posterior and reads off sagittal (0.862, the highest single-plane number in the table).
The medial collateral ligament is a coronal structure (0.596, against 0.447 sagittal).
Patellofemoral osteoarthritis is what the axial view is *for* (0.732). The probe
recovered all three from pixels without being told.

So the mechanism is not that the attention picked a bad plane. It is that **it picks one
plane preference and applies it to every label, while the labels disagree about which
plane shows them.** Averaging is not a weaker version of that choice — it is a refusal to
make it, and refusing is the better policy when twelve different questions need twelve
different answers. Each plane alone scores 0.618–0.624, all well below the mean's 0.670:
the planes are complementary, so anything that concentrates mass gives up part of the
ensemble.

### Why the contrastive objective produced this

Nothing here says the attention is broken. It was trained for one task — match this
study to its report — and that is a single question, not twelve. A report describes a
whole knee, so whichever plane is most globally distinctive is a reasonable thing to
up-weight for retrieval, and the objective never saw a reason to preserve the
coronal-only or sagittal-only detail that MCL and Lateral Meniscus turn out to need.
The pooling is well fitted to its objective and the objective is narrower than the
downstream task.

### What to do about it, if anything

Stated as options, not a recommendation — nothing here is authorised and none of it is
launched:

1. **Cheapest, and available now:** when a representation is exported for downstream
   use, take the pre-attention plane features and pool them uniformly, or keep all three.
   Both score ≈0.670 against the attention's ≈0.650, and it costs no training.
2. Give the attention a reason to preserve per-plane detail — an auxiliary per-plane
   objective, or attention computed per output rather than once per study.
3. Leave it. The effect's interval spans zero at 56 studies, and every number here is a
   readout statistic on a cohort the representation has already seen.

Evidence: [`plane-pooling-analysis.json`](plane-pooling-analysis.json). Reproduce with
`.venv/bin/python scripts/analyze_plane_pooling.py`.

## How this was computed

Protocol frozen in the config before the first fit, following the contract proposed in
the [gold-case inventory](gold-case-inventory.md): per-label L2 logistic probe at fixed
penalty, 8 stratified folds with stratification computed inside the per-label fold loop,
`StandardScaler` fitted inside each training fold, exactly one held-out score per study
per label, no threshold search and no tuning of any kind. Intervals are 5,000 study
bootstrap resamples at a fixed seed, paired so the same resampled studies score both
arms in every replicate.

Two implementation notes, since this repository's numbers are usually checkable:

- Every reported AUROC point estimate comes from `sklearn.metrics.roc_auc_score`. The
  bootstrap replicates use an equivalent rank-sum formulation for speed, and the script
  asserts the two agree to 1e-9 on every label's real scores before building any
  interval.
- The installed scikit-learn deprecates the explicit `penalty='l2'` argument the config
  pins and removes it in 1.10. The fitted model is unaffected — L2 is the default — but a
  rerun on 1.10 must translate it rather than drop it.

Reproduce with:

```bash
.venv/bin/python scripts/evaluate_downstream_probe.py --device cpu
```

## What follows

1. **Unfreezing image blocks is the lever that matters.** It is the only change in the
   completed history that moved the image representation at all, and it is the one that
   helped downstream. Deeper unfreezing (2, 4 blocks) is the obvious next screen.
2. **Retrieval R@1 is not a safe proxy on its own.** The learned plane pooling it
   selected reads labels worse than a plain average of the same three planes, at all
   three seeds, for a mechanism that is now understood — one plane preference applied to
   twelve labels that disagree about planes. The interval spans zero, so this is a
   direction, not a verdict; but exporting the pre-attention plane features costs nothing
   and does not carry the risk.
3. **This gain is small and measured on seen data.** +0.026 macro AUROC from 0.644 is
   real within this cohort and does not establish clinical or generalisation value.
   Obtaining independently adjudicated studies remains the only route to that, and is
   outside this repository.
