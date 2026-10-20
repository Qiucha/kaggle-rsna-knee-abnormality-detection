# Does adapting both towers teach the image tower more medical semantics than adapting the image tower alone?

Measured answer, 2026-09-12 (Asia/Taipei). Training config:
[`configs/contrastive/claude-translated-dfd07db-both-block-v1.json`](../configs/contrastive/claude-translated-dfd07db-both-block-v1.json).
Probe config:
[`configs/downstream/gold-probe-both-block-v1.json`](../configs/downstream/gold-probe-both-block-v1.json).
Evidence: [`downstream-probe-both-block-v1.json`](downstream-probe-both-block-v1.json) and the six
`downstream-both-vs-image-block-*.json` files. Scripts: `scripts/evaluate_downstream_probe.py`
(unmodified) and `scripts/compare_downstream_arms.py` (new).

The paired comparison is split across six per-seed files rather than one combined artifact because
the single six-pair invocation was killed by the host's low-memory guard before writing. The guard
fired on system conditions, not on this workload — each pair scores two arms and bootstraps in about
24 seconds with a few megabytes of peak allocation — so the run was simply split one pair per
invocation. No combined `downstream-both-vs-image-block-v1.json` exists.

## The question

Retrieval selection and the working pathology goal had already been measured against each other in
[downstream-probe-v1.md](downstream-probe-v1.md), which established that contrastive learning
improves downstream label readout by roughly +0.025 macro AUROC **only in the arm that unfroze a
DINOv2 block**. The open item in [ROADMAP.md](../ROADMAP.md) M3 was the remaining cell of that
comparison: image-only, text-only, and **both-tower** last-block adaptation.

The working hypothesis was that aligning the image tower harder toward the medical report — by
letting the text tower move toward the images at the same time — should make the image tower encode
medical semantics better. The measured answer is that it does not.

> **Adapting the final text block on top of the final image block changes downstream label readout
> by between −0.0030 and +0.0005 macro AUROC. All three seed-matched paired intervals include zero
> and the sign flips across seeds. The entire +0.024 to +0.027 contrastive gain comes from
> unfreezing the image block; adding the text block is measurably inert.**

## The primary comparison

Matched pooling, matched dimensionality, matched seed: the only difference between the two arms is
whether `--text-unfreeze-last` was 1 or 0. Unadapted DINOv2 is the anchor, not the contrast —
both-block versus unadapted would confound two simultaneous changes. Paired over the same studies,
folds and 5,000 study resamples as the frozen probe config.

| Seed | image-block | both-block | Difference | Paired 95% interval | Replicates favouring both-block |
| --- | --- | --- | --- | --- | --- |
| 2026 | 0.6687 | 0.6691 | +0.0005 | −0.0039 to +0.0045 | 59.0% |
| 2027 | 0.6706 | 0.6706 | −0.0001 | −0.0045 to +0.0045 | 48.4% |
| 2028 | 0.6709 | 0.6678 | −0.0030 | −0.0090 to +0.0025 | 14.3% |

Every interval spans zero, and the replicate fractions straddle 50%. This is a null result at three
independently seeded replicates, not an underpowered hint of a gain.

The learned plane pooling tells the same story, and also reproduces the known pooling deficit — both
arms read *worse* through `plane_attention_768` (0.646–0.654) than through a trivial uniform mean
(0.668–0.671). The image-block column below reproduces the `attention_pooled_768` values recorded in
[plane-pooling-analysis.json](plane-pooling-analysis.json) — 0.6503, 0.6491 and 0.6504 — to four
decimal places, a second independent reproduction check on this path.

| Seed | image-block | both-block | Difference | Paired 95% interval |
| --- | --- | --- | --- | --- |
| 2026 | 0.6503 | 0.6461 | −0.0041 | −0.0155 to +0.0070 |
| 2027 | 0.6491 | 0.6540 | +0.0049 | −0.0043 to +0.0140 |
| 2028 | 0.6504 | 0.6495 | −0.0009 | −0.0112 to +0.0094 |

## Both arms do clear the unadapted anchor

Placed on the same scale as the published table, the both-block arm reproduces the contrastive gain.
It simply does not exceed the image-only arm.

| Arm | Macro AUROC | Difference vs unadapted | Paired 95% interval | Replicates favouring after |
| --- | --- | --- | --- | --- |
| Unadapted DINOv2-small | 0.6438 | — | — | — |
| image-block seed 2026 | 0.6687 | +0.0249 | +0.0090 to +0.0415 | 99.86% |
| image-block seed 2027 | 0.6706 | +0.0268 | +0.0112 to +0.0427 | 99.90% |
| image-block seed 2028 | 0.6709 | +0.0271 | +0.0108 to +0.0442 | 99.90% |
| both-block seed 2026 | 0.6691 | +0.0253 | +0.0096 to +0.0420 | 99.94% |
| both-block seed 2027 | 0.6706 | +0.0268 | +0.0108 to +0.0434 | 99.90% |
| both-block seed 2028 | 0.6678 | +0.0240 | +0.0090 to +0.0401 | 99.88% |

The three image-block rows reproduce [downstream-probe-v1.md](downstream-probe-v1.md) to four
decimal places. That is a reproduction check on the whole path — probe config, feature extraction,
fold assignment, bootstrap — under a config written independently of it.

## Two structural zeros, both verified rather than assumed

**Text-only adaptation cannot change the image tower, and did not.** The three text-block runs used
`--image-unfreeze-last 0`, so their pre-projection image features must be bit-identical to the
unadapted backbone. Declared in the probe config in advance, then confirmed exactly:

| Text-block run | Max absolute deviation from unadapted DINOv2 | Identical |
| --- | --- | --- |
| `text-block-seed2026` | 0.0 | yes |
| `text-block-seed2027` | 0.0 | yes |
| `text-block-seed2028` | 0.0 | yes |

This settles the text-only arm directly. Whatever text-only adaptation achieved for retrieval, it
provably did not change what the image tower encodes — its downstream macro AUROC is 0.6438, the
unadapted value, to the last bit. It also validates the extraction path on an arm never probed
before.

Combined with the earlier frozen-tower result, the pattern across the whole experiment history is
now: **only `--image-unfreeze-last 1` ever moved the image representation, and moving the text tower
alongside it added nothing.**

## Retrieval said the same thing first

Selection metric, mean validation R@1, against the matched image-block seed:

| Seed | image-block | both-block | Difference |
| --- | --- | --- | --- |
| 2026 | 11.1111 pp | 11.6780 pp | +0.5669 |
| 2027 | 10.4308 pp | 10.5442 pp | +0.1134 |
| 2028 | 10.5442 pp | 9.5238 pp | −1.0204 |

Mean difference −0.11 pp with the sign flipping across seeds. Retrieval and downstream readout agree
that the two arms are indistinguishable, which is mild evidence that R@1 remained a usable proxy
here — as it was in the image-block comparison.

## Caveats that bound this result

**The estimand is readout, not generalisation** (exposure since measured at **−0.0168**; generalisation estimate **0.6938**, [`gold-holdout-v1.md`](gold-holdout-v1.md)). 56 of 58 gold cases were exposed to contrastive
training or checkpoint selection. This measures how well a frozen representation supports label
readout on a cohort it trained on. The both-block arm trains strictly more parameters on that same
cohort than the image-block arm, so if anything it had *more* room to look good here and still did
not. That asymmetry strengthens the null rather than weakening it.

**Every selected checkpoint sits well past its validation-loss minimum.** All three both-block runs
bottomed out at **epoch 6** (3.6961–3.7280) but were selected at epochs 14, 20 and 15 with
validation loss 4.00–4.18. The frozen selection policy is `mean_r@1` descending, so this is the
policy working as written, not a deviation — but these are checkpoints chosen deep into the
overfitting region, and the same pattern held in the text-block arm. Seed 2027 selected the
**final** epoch, meaning its R@1 curve never turned over inside the frozen 20-epoch budget. The
budget was not extended.

**Patient independence remains unverifiable.** Anonymisation is per study, so every number here is a
study-level estimate and the bootstrap resamples studies, not patients.

**Rare labels stay uninformative.** MCL has 8 positives over 56 cases; exactly one label-replicate
was dropped as single-class in each paired bootstrap.

## One tooling defect found, not patched

`scripts/evaluate_downstream_probe.py` only works on CPU. `extract_features` never moves the image
batch onto the target device, so `_plane_features` mixes `cuda:0` and `cpu` tensors and raises; the
closing `.numpy()` also lacks a `.cpu()`. Since `--device` defaults to `cuda` when available, the
default path is broken, and `.venv-cuda` additionally has no `sklearn` installed. Every run of this
script — the published one and this one — therefore used `.venv` on CPU.

The script was deliberately **not** patched for this experiment, so that the both-block numbers come
from the identical code path that produced the image-block numbers they are compared against. Fixing
it is a two-line change plus installing `sklearn` into `.venv-cuda`, and belongs in a separate change
with its own reproduction check.

## What this does not answer

This compares *final-block* adaptation of both towers against final-block adaptation of the image
tower. It does not test deeper unfreezing, a different text encoder under both-tower adaptation, a
longer epoch budget, or any change to the contrastive objective or negative pool. It also does not
touch the locked test split, and no selection record was finalized for this matrix.
