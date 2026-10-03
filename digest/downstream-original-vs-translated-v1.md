# Does translating the reports change what the image tower learns?

Measured 2026-09-12 (Asia/Taipei). Two full-cohort contrastive runs that differ **only** in the
report text policy, both unfreezing the final DINOv2 block with the text tower frozen.

Training config: [`image-block-cohort-compare-v1.json`](../configs/contrastive/image-block-cohort-compare-v1.json).
Probe config: [`gold-probe-cohort-compare-v1.json`](../configs/downstream/gold-probe-cohort-compare-v1.json).
Evidence: [`downstream-probe-cohort-compare-v1.json`](downstream-probe-cohort-compare-v1.json) and
[`downstream-original-vs-translated-v1.json`](downstream-original-vs-translated-v1.json).

## The answer

> **Translating the reports to English changes downstream label readout by −0.0010 macro AUROC
> (95% CI −0.0061 to +0.0043). The interval spans zero. Both report policies produce the same
> +0.025 gain over unadapted DINOv2, and that gain is real in both.**

> **Update, later the same day — replicated at three seeds.** Seeds 2027 and 2028 give −0.0019 and
> +0.0028; all three intervals span zero, the sign flips, and every difference is smaller than the
> 0.0038 seed spread. See [`image-block-variants-v1.md`](image-block-variants-v1.md), which also
> reports the lever that *does* move readout: unfreezing 4 DINOv2 blocks rather than 1
> (+0.0413, 95% CI +0.0095 to +0.0739). The single-seed caveat below is now resolved for the
> report-policy question.

## What was run

| Arm | Cohort manifest | Text policy | Rows | Elapsed | Peak VRAM | Best epoch | val mean R@1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `original-multilingual-seed2026` | `contrastive-cohort.jsonl` (`54bd803e…`) | `multilingual_raw` | 4,407 | 1,250 s | 3,186 MiB | 19 | 0.1111 |
| `translated-english-seed2026` | `…claude-dfd07db-q2-v1.jsonl` (`8f6f0219…`) | `translated_english` | 4,405 | 1,211 s | 3,278 MiB | 18 | 0.1032 |

Both arms: `--image-unfreeze-last 1 --text-unfreeze-last 0 --lr-image 1e-5`, 20 epochs, batch 32,
seed 2026, multilingual-e5 text tower, `facebook/dinov2-small` image tower, both revisions pinned.
The two arms share the same 441 validation studies, so the R@1 figures are directly comparable.
Matched elapsed time and peak memory confirm neither arm was competing for the GPU.

Retrieval R@1 is the selection metric, not the deliverable. Each arm is a **single seed**, so the
0.1111 vs 0.1032 gap has no interval attached and is not a claim.

## The label evaluation

The gold label set is **58 studies × 12 labels**, not 58 labels. Two of those studies sit in the
locked test split and are counted but never loaded, so the probe scores **56 studies**. The probe
resolved 48 train / 8 validation / 2 test / 0 absent under both manifests.

Pooled out-of-fold AUROC, macro-averaged over the 12 labels, 8 stratified folds, 5,000 study
bootstrap resamples — all fixed in the probe config before the first fit.

### Both arms beat unadapted DINOv2; neither beats the other

Matched pooling and dimensionality (`uniform_mean_768`), so the only difference is the adapted block.

| Contrast | Before | After | Difference | 95% CI | Replicates favouring right |
| --- | --- | --- | --- | --- | --- |
| unadapted → original | 0.6438 | 0.6695 | **+0.0257** | +0.0112 to +0.0409 | 100.0% |
| unadapted → translated | 0.6438 | 0.6685 | **+0.0247** | +0.0095 to +0.0406 | 99.9% |
| original → translated | 0.6695 | 0.6685 | −0.0010 | −0.0061 to +0.0043 | 34.1% |

Both contrastive gains exclude zero and reproduce the +0.024 to +0.027 effect already recorded in
[`downstream-probe-v1.md`](downstream-probe-v1.md) and [`downstream-both-block-v1.md`](downstream-both-block-v1.md).
The original-versus-translated interval spans zero and is an order of magnitude tighter than either
gain — this is a null result, not an underpowered one.

Through the learned plane pooling the same contrast is −0.0133 (95% CI −0.0321 to +0.0053), also
spanning zero. That path again reads *worse* than a trivial uniform mean (0.6472 and 0.6604 versus
0.6685 and 0.6695), reproducing the known pooling deficit.

### Per-label AUROC (`uniform_mean_768`, 56 studies)

| Label | Positives | Unadapted | Original | Translated |
| --- | --- | --- | --- | --- |
| ACL | 23 | 0.5244 | 0.5547 | 0.5534 |
| MCL | 8 | 0.7135 | 0.7396 | 0.7422 |
| Medial Meniscus | 26 | 0.7923 | 0.7821 | 0.7808 |
| Lateral Meniscus | 22 | 0.5709 | 0.6016 | 0.5936 |
| Medial OA | 14 | 0.8044 | 0.8231 | 0.8231 |
| Lateral OA | 11 | 0.6040 | 0.5818 | 0.5778 |
| PF OA | 20 | 0.5194 | 0.5681 | 0.5806 |
| Effusion | 33 | 0.7681 | 0.7984 | 0.7958 |
| Synovitis | 26 | 0.4679 | 0.5333 | 0.5410 |
| Baker's | 12 | 0.6799 | 0.6970 | 0.6970 |
| Contusion | 19 | 0.7952 | 0.8236 | 0.8122 |
| Fracture | 17 | 0.4857 | 0.5309 | 0.5249 |
| **macro AUROC** | | **0.6438** | **0.6695** | **0.6685** |
| **macro AUPRC** | | **0.5281** | **0.5650** | **0.5622** |

Per-label differences between the two arms are within ±0.013 everywhere except Contusion (−0.0114)
and PF OA (+0.0125), and MCL has only 8 positives — pooled-OOF only, fold-level AUROC was declared
uninformative in advance.

## Correctness checks that passed

- **Frozen-tower identity.** Both `--image-unfreeze-last 0` controls — one per report policy —
  produce `uniform_mean_768` features **bit-identical** to unadapted DINOv2 (max absolute deviation
  0.0). Declared in advance; a non-zero value would have meant the arm wiring was wrong.
- **Both new arms actually moved the backbone**, at max absolute deviation 6.23 and 6.52.
- The two frozen-tower checkpoints serve as **identity controls only**. One of them,
  `full-multilingual-v1/00-projection-full-mean-512`, was trained under a different matrix with
  different arguments and is not a compact checkpoint, so its `plane_attention_768` and
  `projection_256` figures describe a differently-trained projection head and are not a control for
  anything compared here. Only its `uniform_mean_768` identity result is used.
- **Population identity.** The 56-study gold population resolves identically under both manifests,
  verified before training and again by the probe itself.
- **Locked test never loaded.** Test-split gold studies are dropped before any pixel is read.

## Limitations

- **Single seed per arm.** Prior arms in this project were measured at three matched seeds. These
  are n=1. The null is consistent with the both-block null but has not been replicated across seeds.
- **The cohorts are not row-identical.** The translated manifest drops two studies quarantined for
  identical decoded pixels with incompatible paired reports; the base manifest retains them. The
  original arm therefore trains on two studies the translated arm never sees. Removing this confound
  requires a new manifest and a new checksum.
- **Readout, not generalisation.** 56 of the 58 gold studies were contrastive training pairs or
  members of the selection split. Every number here is a readout statistic on seen data.
- **Environment.** The probe ran on CPU: `evaluate_downstream_probe.py` never moves the batch to the
  selected device, so `--device cuda` raises a device mismatch in `_plane_features`. This is a latent
  bug, not a result — the existing probe artifacts were evidently produced on CPU too. Scored under
  scikit-learn 1.9.0, which emits the `penalty='l2'` deprecation the probe config anticipates; the
  fitted model is unchanged. Exposure has since been measured: **−0.0168** of macro AUROC, generalisation estimate **0.6938** ([`gold-holdout-v1.md`](gold-holdout-v1.md)).
