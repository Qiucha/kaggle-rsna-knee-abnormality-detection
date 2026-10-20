# R4 multi-slice input: design decisions

Decided 2026-09-30 with the user. Source ticket: `.scratch/leaderboard-next/issues/08-r4-multislice-input.md` (map: *Close the leaderboard gap*). This page is the design only; nothing here has been built or run yet.

## Why R4

Random slice sampling on R1c (`r1c-hard-224-from-r0-slice-random`) passed the ticket-03 bar: LM-test 0.8455, +0.012 over R0, CI [0.003, 0.021]. It did so by damping overfitting. So R4 is justified as **regularization**, not only as coverage. **R4's baseline is that run, not R0.**

## Starting point (code facts)

- Each of the 3 planes goes in as one 224 px RGB image. Its 3 channels are slices spread across the 0.2–0.8 band, a median 31 mm apart (`preprocessing._slice_indices`).
- `FinetuneModel.features` pools CLS + mean-patch per image and **averages the planes** (`finetune.py:91`). `validate_config` rejects any `slices_per_plane` other than 3.
- Short series are padded by **repeating** slice indices.
- One step is batch 16 × 3 planes = 48 backbone passes. An epoch takes about 55 s from the local series mirror.

## Decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Unit of depth | **K windows per plane.** Each window is a 3-slice RGB image, so the pretrained RGB stem stays unchanged. |
| 2 | Where attention sits | **Across all planes.** 12 label queries attend over all 3K windows, so each label can learn which plane to rely on. A learned plane embedding is added to each window token. |
| 3 | Training sampling | **Stratified jitter:** the band is split into K strata, one random window is drawn per stratum, and order is kept. This generalizes what passed in ticket 07. |
| 4 | Inference sampling | **Deterministic:** K windows at the stratum centres, for evaluation, LM-test, gold and the exported member. |
| 5 | K and memory (8 GB GPU) | **K = 8 first**, about 24 slices per plane. The 9 frozen blocks run under `no_grad`. Batch 4 × gradient accumulation 4 keeps the effective batch at 16. Peak memory is measured in a 20-step smoke test before the overnight run. **K = 16 is a follow-up if there is headroom.** Expect roughly 8× the compute, about 2 h per run. |
| 6 | Fallback for missing input | **Validity mask**, not repetition. A short stack (fewer than K distinct windows in the band) and a missing or undecodable plane both become masked zero windows. Attention and mean pooling give them zero weight. A study with every plane missing stays excluded, as now. **The report counts masked windows per split**, so we know how often each fallback fires. |
| 7 | Window layout | **Contiguous:** 3 adjacent slices per window, tiled across the band via `slice_group_indices`, with per-stratum jitter in training. The input-preprocessing study's null on contiguous groups was inference-only on frozen features, so it does not carry over. |
| 8 | Attention head | **Single-head attention with 12 learned label queries** over window tokens (768-d CLS + mean-patch plus the plane embedding). It is masked by validity, and each label gets its own linear logit. Parameter count stays near R1c's 256-hidden MLP. **Attention weights are logged on LM-test.** |
| 9 | Attribution | **Two arms**, both at K = 8, 224 px, hard labels, R0 init and random sampling: **R4a** with masked mean pooling, and **R4b** with per-label attention. R4a is judged against the slice-random baseline (does depth help?), and R4b against R4a (does attention help?). Both use the ticket-03 bar: the LM-test CI must exclude zero, with a gold-58 guard of ≥ −0.024. |
| 10 | Kaggle runtime | Accepted **only if** a timing dry-run of the offline bundle on a local test sample projects to **under half the notebook limit**. The current per-study inference time and the limit still have to be measured. |

## Open items

- Measure the current per-study Kaggle inference time and the notebook limit (decision 10).
- Measure peak memory at K = 8 in the smoke test (decision 5).
- The spread from random slice draws is still unmeasured (single seed). A gap below 0.05 gets a 3-seed rerun, per ticket 05.
