# Wide slice band in supervised fine-tuning, and A30's inference-only probes

**Status (2026-10-15): not adopted.** Neither a wider slice band nor any inference-only change to A30 helps.
A30 (0.8917 LM-test, 0.890 public) remains the best single model.

## Trial: A30 with slice band [0.05, 0.95]

Every run so far keeps only the middle of each stack, slice band [0.2, 0.8]. On sagittal stacks the outer 40% holds
the meniscal bodies and the collateral ligaments (leaderboard-next ticket 17, item 1). Run
`checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-band05` copies A30 and changes only
`slice_band`, in training and in every evaluation. It ran 2026-10-14 09:35–22:52 (Taipei), 13.3 h, all 30 epochs,
and selected epoch 25.

Config commit `71209ce` on `t3code/fine-tune-weak-labels` (worktree t3code-d7aef637). Comparison:
`scripts/analysis/compare_slice_band.py` in the same worktree (a copy of the image-augmentation comparison with the
edge-structure labels as targets), run with `CONTRASTIVE_PATHS_CONFIG` pointing at
`checkpoints/finetune/a30-inference-probes/paths-d7aef637.json`, because main's `paths.json` has a
`train_series_mirror` key that this worktree's `paths.py` rejects.

| | LM-test | Δ vs A30 [95% CI] | Gold-58 | Δ vs A30 [95% CI] |
|---|---|---|---|---|
| A30, band [0.05, 0.95] | 0.8849 | −0.0068 [−0.0156, +0.0025] | 0.8752 | −0.0062 [−0.0233, +0.0105] |
| A30 | 0.8917 | — | 0.8814 | — |
| u9-b1 | 0.8834 | | 0.8775 | |
| Noise floor: run 5 seed 2027 − 2026 | | −0.0029 [−0.0110, +0.0061] | | +0.0066 [−0.0153, +0.0291] |

- Both intervals cross zero and both point estimates lean negative. Against u9-b1 it is +0.0015 LM-test and −0.002
  gold-58, so most of A30's edge is gone.
- The edge-structure labels do not gain. On LM-test, Lateral Meniscus moves +0.003, Medial Meniscus +0.007 and
  MCL −0.022. On gold-58 the same labels move −0.047, −0.023 and −0.029.
- The LM-test losses fall on Medial OA (−0.024), PF OA (−0.022) and Contusion (−0.021). With K = 8 windows over a
  wider band, each window covers a wider stretch of slices, which thins the central coverage.
- Caveats: one seed, and only one band width was tried. A wider band with more windows (K = 12 or 16), so that the
  central density is kept, is untested.

## A30 inference-only probes (2026-10-14)

Scripts: `.scratch/leaderboard-next/scratch/a30_inference_probes.py` and `a30_inference_compare.py` (main checkout).
Outputs are in `checkpoints/finetune/a30-inference-probes/`, and the digest copy is `a30-inference-probes-v1.json`.
Re-scoring A30's best checkpoint reproduced its saved scores exactly.

| Probe | Δ LM-test vs A30 [95% CI] | Δ gold-58 |
|---|---|---|
| Average of best (ep 22) and last (ep 30) weights | +0.0012 [−0.0004, +0.0030] | +0.001 |
| Weight soup A30 + A30-with-augmentation | +0.0034 [−0.0028, +0.0096] | −0.005 |
| Weight soup A30 + u9-b1 | −0.0030 [−0.0095, +0.0035] | −0.011 |
| Centre windows + 4 jittered passes | +0.0019 [−0.0007, +0.0047] | +0.002 |

None of these is worth adopting. Each jitter pass also costs a full decode, about 5–7 min per 490 studies.

Sources: `finetune-slice-band-v1.json`, `a30-inference-probes-v1.json`.
