# Slice range and image augmentation investigation

Audited 2026-10-12. **The training slice band is [0.2, 0.8]. Image augmentation is active in contrastive training, but absent from supervised fine-tuning.**

> **Note (2026-10-13):** this audit predates `feat/label-image-augmentation`, which adds an optional `image_augmentation` object to supervised fine-tuning (one coherent draw per study/plane shared across the K windows). The first trial on the A30 recipe was a null: see [the image-augmentation result](finetune-image-augmentation-v1.md). Statements below that supervised fine-tuning has no image augmentation describe the branches as audited on 2026-10-12. Wider-band training and a supervised image-augmentation ablation remain unmeasured in the inspected configurations.

## Scope and branch coverage

`git fetch --all` completed successfully. The audit inspected all **19 local branches, 9 remote branches, and 14 attached worktrees**, including their working-tree source, configuration files, and notebook source. It also inspected **169 distinct saved run configurations**, deduplicating checkpoint directories shared through symlinks. All **161 saved configurations containing an explicit slice band** specify [0.2, 0.8]; the other eight have no explicit band. Every explicit band in branch and worktree configurations is also [0.2, 0.8].

Exact branch names, commits, file identities, matching source lines, and configuration values were kept in a host-local audit snapshot (an inventory of one machine's worktrees, not committed). This is an audit of branch tips and current worktree files, with targeted inspection of older experiment notes and scratch code; it is not an exhaustive search of every historical commit or checkpoint tensor.

| Branch family | Relevant behavior |
|---|---|
| `contrastive-learning`, `submission`, and their remote counterparts | Same geometric/intensity augmentation helper; central three-slice inputs. |
| `exp/finetune-supervised` and its remote counterpart | Committed fine-tuning implementation has no image augmentation. The main checkout also held newer K8 experiments not yet committed at the time. |
| `pipeline/best-2026-10`, `sync/main-into-pipeline`, `exp/settings-fusion`, `feat/ensemble-selection`, `exp/archive-null-results`, `t3code/combine-training-losses` | K8/random-window fine-tuning without spatial or intensity augmentation; contrastive training uses the helper. |
| `feat/omnirad-clip-pipeline` | K8 contrastive training retains the image transforms, including gradient replay; subsequent supervised training has none. |
| `joint/text-bank`, `backup/joint-text-bank-pre-rebuild` | Adds an `augment_images` switch, default true. The supplied joint experiment explicitly sets it to false to match supervised training. |
| `t3code/fine-tune-weak-labels` | Adds target noise, label weighting, and oversampling, not pixel augmentation. |
| `main`, translation/label-extraction branches, and investigation/review branches | No separate training slice-range or augmentation implementation was found. |

The augmentation function body is identical across all four distinct committed versions of `preprocessing.py` found by the audit.

## What currently happens

The older input selects three geometrically ordered slices spread across the band. The K8 input selects **eight local windows per plane, each containing three adjacent slices as channels**. Training draws a new centre uniformly within each of eight depth strata on every read; evaluation uses deterministic stratum centres. Windows can overlap, so K8 does not necessarily mean 24 distinct slices. The band constrains window centres; neighbouring slices can extend outside it. Even [0, 1] with K8 does not consume every slice.

Sources: `_slice_indices` and `window_indices` in `src/contrastive_learning/preprocessing.py`; `WindowDataset` in `src/contrastive_learning/finetune.py`.

| Image transform | Existing contrastive recipe |
|---|---|
| Rotation | Uniform −8° to +8°. |
| Zoom | Scale 1.00 to 1.08; zoom-in only. |
| Translation | Each axis uniform −0.05 to +0.05 in normalized grid coordinates: about ±2.5% of image extent before zoom, not ±5% of its width. |
| Intensity | Multiplicative gain 0.90 to 1.10, then clamp to [0, 1]. |
| Resampling | Bilinear interpolation with border padding. |
| Flips | None in stochastic augmentation. Right-to-left knee mirroring is deterministic preprocessing. |

All three slice channels share a transform. Each flattened image draws independently: a conventional input draws per study/plane, while a K-window input draws per study/plane/window. Thus calling the helper on K8 does **not** automatically make transforms coherent across all windows in a plane.

Sources: `augment_study_images` in `src/contrastive_learning/preprocessing.py`; its calls in `src/contrastive_learning/training.py`; the supervised training loop in `src/contrastive_learning/finetune.py`; the `augment_images` switch in `src/contrastive_learning/config.py` on branch `joint/text-bank`.

Cropping to 130 mm, percentile normalization, resizing, and laterality normalization are preprocessing. Fresh depth draws are sampling augmentation. Neither establishes that supervised training applies geometric or intensity augmentation. No additional pixel-noise, gamma, blur, elastic deformation, MixUp, or CutMix training recipe was found in the inspected pipeline.

## What changing the range does

I executed the actual deterministic `window_indices` function against the **450 historical series from 150 sampled training studies** in the existing depth audit. These are sampler calculations, not new DICOM reads, cohort-wide measurements, or AUROC results.

| Band, fixed K8 | Median eligible centre slices | Median distinct selected slices | Median centre spacing |
|---|---:|---:|---:|
| [0.2, 0.8] | 19 | 20 | 8.0 mm |
| [0.1, 0.9] | 25 | 24 | 10.5 mm |
| [0.05, 0.95] | 27 | 24 | 12.0 mm |
| [0.0, 1.0] | 30 | 24 | 13.5 mm |

Widening [0.2, 0.8] to [0.1, 0.9] increases the nominal depth span by one third while keeping the same maximum 24 image encodings per study. It adds peripheral coverage and reduces window overlap, but also lowers sampling density. Peripheral signal and the dilution of useful windows under mean pooling are competing hypotheses; their accuracy effects cannot be inferred from these calculations.

The percentile normalization is computed jointly over selected slices, so changing their selection can also change the normalization bounds. This remains part of the deployed recipe unless a separate mechanism experiment fixes those bounds. Evaluation cache keys already include the band, K, crop, size, and preprocessing version, preventing a different band from reusing old tensors. New bands still require new cache entries and decoder QC.

Sources: [sampler analysis](slice-range-sampler-analysis.json), [historical depth audit](fov-and-depth-audit.json); percentile normalization in `src/contrastive_learning/preprocessing.py`; the window-cache key in `src/contrastive_learning/finetune.py`.

## Existing evidence and its limits

Random slice sampling has previously helped supervised training: the R1c random-slice run recorded LM-test AUROC 0.8455, approximately +0.012 over R0, with paired interval [0.003, 0.021]. That result supports testing sampling regularization, but does not isolate wider bands or image transforms. [R4 design record](finetune-r4-multislice-design.md)

The older B4 dense-slice experiment **still decoded only the 20–80% band**. Its results are not evidence of training on [0.1, 0.9] or the full range. There is also full-series inference code in the input-preprocessing scratch harness, but that is a different frozen-feature/inference question. Historical multi-view results disagreed across LM-test and gold: a full-train B3 head refit improved LM-test by +0.0154 while gold-58 fell by −0.018. (Sources: the input/preprocessing hypotheses study's R2 coverage-factorial record, its full-train gate and its full-series harness; these are uncommitted working notes.)

MRNet provides precedent for stronger spatial augmentation, but neither its paper nor the local records establish optimal settings here. See [the primary-source check](knee-mri-augmentation-primary-sources.md) for the literature and the differences from our orientation convention.

## Proposed experiment

Start with one unchanged supervised K8/336/soft-label recipe, preferably the plain OmniRad u9 recipe, and a **2 × 2 comparison**:

| Arm | Train and evaluation band | Training image augmentation |
|---|---|---|
| A: baseline | [0.2, 0.8] | Off |
| B: wider coverage | [0.1, 0.9] | Off |
| C: augmentation | [0.2, 0.8] | Existing conservative recipe |
| D: both | [0.1, 0.9] | Same recipe as C |

Keep K8, random training-window sampling, pooling, model initialization, label targets, training population, resolution, crop, optimizer, effective batch, epoch budget, and checkpoint selection fixed. Train/evaluate each coverage arm with its own band; an inference-only change would answer a different question.

Use one spatial/intensity draw per study/plane shared across its K windows, with the three channels also sharing it, as an explicit initial design choice. Different planes can draw independently. Keep masks intact and invalid windows zero. This deliberately tests a coherent supervised augmentation recipe rather than silently inheriting the helper's per-window draws. Keep stochastic flips disabled to retain the current laterality convention.

Compare validation-selected models using paired seeds (2026, 2027, 2028), report validation macro AUROC and masked BCE, and record per-label changes, runtime, valid-window counts, unique-slice coverage, and decode/substitution counts. Augmentation consumes extra RNG draws, so seed pairing alone does not guarantee identical depth draws; use an independent augmentation generator or explicitly acknowledge that limitation.

Predefine the comparison and validation selection rule before running. Freeze epoch and recipe choices before the final paired LM-test readout. Report `(B−A)`, `(C−A)`, and `(D−B)−(C−A)` to distinguish coverage, augmentation, and their interaction. Use paired study bootstrap intervals; seed variation measures a different source of uncertainty. LM-test is agreement with LM verdicts, and repeatedly inspected gold-58 is exploratory. Do not choose augmentation strengths or bands based on those test scores.

If wider coverage looks promising, a later K10/K12 comparison can test whether sampling density limits it. That changes compute and slice count and needs its own baseline. Likewise, gamma/noise or [0.05, 0.95] should be later independent questions.

**Implementation gap (as of 2026-10-12; closed on `feat/label-image-augmentation`):** supervised `run_finetune` needs a real training-only augmentation hook and recorded parameters. Adding an augmentation field to its JSON alone will not activate image transforms. The joint branch's `augment_images` switch belongs to its separate training loop. Preview transformed windows and verify validation/inference remain deterministic before training.

## Work completed

Completed the branch/worktree/configuration census, sampler calculation, and primary-source check. A synthetic K8 tensor passed the existing augmentation helper's shape, dtype, and [0, 1] range checks. No model training or new accuracy measurement was performed for this audit.
