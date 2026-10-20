# Weak-label rebalancing on u9-b1 (Lateral Meniscus, Lateral OA, Synovitis)

**Status (2026-10-12):** the 20-epoch arms lose to u9-b1. Oversampling with a 30-epoch budget gives the best single LM-test score so far (0.8917) and the best public single model (0.890 vs u9-b1 0.888), pending a matched 30-epoch u9-b1 control.

2026-10-11. Two single-seed arms on the u9-b1 recipe (OmniRad-base, 9 unfrozen blocks, soft LM targets,
K8 windows at 336 px, seed 2026). Each changes one thing. Both select their checkpoint the same way u9-b1
does: on validation masked BCE against hard verdicts, with no weighting and no noise.

- **A · oversample + label noise.** Each study with an observed positive on any of the three labels gets
  one extra draw per epoch (1,508 extras; 4,946 draws per epoch instead of 3,438). On the extra draws the
  other nine labels are masked. Every draw adds fresh N(0, 0.1) noise to the two lateral soft targets,
  clipped to [0, 1]. Confound: A takes about 1.44× u9-b1's optimiser steps.
- **B · loss weight.** 2× BCE weight on the three labels, renormalised to mean 1 (1.6 / 0.8).

## Update 2026-10-12: arm A with a 30-epoch budget

A was retrained from scratch: cosine schedule over 30 epochs, patience off, and a divergence stop
(validation BCE 0.02 above its best for 3 epochs). The divergence stop never fired. Validation BCE fell until
epoch 22 (0.4199, against u9-b1's 0.4308), then flattened, so the run used all 30 epochs.

| | LM-test | Δ [95% CI] | Gold-58 | Δ [95% CI] |
|---|---|---|---|---|
| A30 vs u9-b1 | 0.8917 | +0.0083 [−0.0025, +0.0202] | 0.8814 | +0.0040 [−0.0149, +0.0218] |
| A30 vs A (20-epoch) | | +0.0149 [+0.0045, +0.0255] | | +0.0191 [−0.0018, +0.0412] |

- This is the highest single-run LM-test score so far, but the interval against u9-b1 still crosses zero.
- Lateral Meniscus keeps its LM-test gain (+0.0311 [+0.0006, +0.0632]). Lateral OA and Synovitis stay within
  noise.
- The 20-epoch arm's losses on MCL, Effusion and Fracture are gone. No label falls beyond seed noise on LM-test.
  The largest gold-58 dips are ACL (−0.033) and Baker's (−0.016).
- **Confound:** A30 saw about 2.5× u9-b1's training draws (30 × 4,946 vs 17 × 3,438). The matched control
  is u9-b1 with the same 30-epoch budget and stopping rule. Without it, the gain cannot be credited to
  oversampling rather than to longer training.
- **Public leaderboard (2026-10-12): 0.890** (submission 57094902; kernel
  `kyunhsuan/rsna-knee-omnirad-base-u9-a30-submission`), against u9-b1's 0.888. That makes it the best public single
  model, but only by +0.002. It falls below both predictions: about 0.896 if LM-test tracked and about 0.892 if gold-58
  did. Public picked up about a quarter of the LM-test gain, which is within single-seed noise. Record:
  `checkpoints/kaggle/r4a-k8-omb-soft-336-u9-a30/leaderboard-submission.json`.

## Follow-up 2026-10-13: A30 + image augmentation

A30 retrained with training-time image augmentation (rotation, zoom, shift, intensity): 0.8889 LM-test (−0.0028 [−0.0111, +0.0055] vs A30), 0.8648 gold-58 (−0.0166 [−0.0380, +0.0050]). No gain; A30 stays the best single run. Write-up: [finetune-image-augmentation-v1.md](finetune-image-augmentation-v1.md).

## Original 20-epoch result

| Arm | LM-test | Δ vs u9-b1 [95% CI] | Gold-58 | Δ vs u9-b1 [95% CI] | Best / last epoch |
|---|---|---|---|---|---|
| u9-b1 | 0.8834 | — | 0.8775 | — | 12 / 17 |
| A · oversample + noise | 0.8768 | −0.0066 [−0.0172, +0.0041] | 0.8624 | −0.0151 [−0.0391, +0.0065] | 12 / 17 |
| B · 2× weight | 0.8723 | −0.0111 [−0.0243, +0.0015] | 0.8525 | −0.0250 [−0.0470, −0.0007] | 7 / 12 |
| Noise floor: run 5 seed 2027 − 2026 | | −0.0029 [−0.0110, +0.0061] | | +0.0066 [−0.0153, +0.0291] | |

B crosses the gold-58 veto (−0.024). Neither arm has an LM-test interval above zero.

## The targeted labels (paired bootstrap on each label alone, Δ vs u9-b1)

| Label | A LM-test | A gold-58 | B LM-test | B gold-58 |
|---|---|---|---|---|
| Lateral Meniscus | **+0.0375 [+0.0118, +0.0657]** | +0.0516 [−0.0215, +0.1299] | **+0.0437 [+0.0168, +0.0708]** | −0.0112 [−0.0921, +0.0727] |
| Lateral OA | −0.0166 [−0.0425, +0.0080] | +0.0348 [−0.0521, +0.1271] | −0.0084 [−0.0303, +0.0140] | +0.0077 [−0.0476, +0.0658] |
| Synovitis | −0.0052 [−0.0625, +0.0619] | +0.0042 [−0.0727, +0.0713] | +0.0279 [−0.0757, +0.1384] | −0.0108 [−0.1201, +0.0909] |

- Only Lateral Meniscus on LM-test clearly moves, in both arms. LM labels agree with gold on this label only
  about 0.84, and the gold-58 rise (A only) has an interval that crosses zero.
- Lateral OA and Synovitis stay within noise. Synovitis has only 537 observed training studies, and 454 of
  them are positive, so oversampling repeats the same ~450 studies. Its gap is coverage, not imbalance.
- The cost lands mostly on MCL (gold −0.104 for A, −0.138 for B; LM-test −0.088 for B), with smaller losses
  on Medial Meniscus, ACL, Effusion (A, gold −0.099) and Fracture (A, LM-test −0.048).

## Reproduce

Each arm is one committed config under `configs/finetune/`, trained one at a time from the repository root with the
GPU environment. With `X` the config name (for example `r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30`):

```bash
python scripts/finetune_supervised.py --config configs/finetune/X.json --output checkpoints/finetune/X
python scripts/compare_lateral_rebalance.py --output digest/lateral-rebalance-v1.json
python scripts/build_digest_site.py
```

- Arms: `...-u9-b1-lat-os-noise` (A), `...-u9-b1-lat-w2` (B), `...-u9-b1-lat-os-noise-ep30` (A30), and the planned but never run
  `...-u9-b1-ep30` / `...-u9-b1-ep43` (matched-epoch and matched-draw u9-b1 controls) and `...-u9-b1-lat-os-noise-ep50`
  (A50); that queue was abandoned at the 2026-10-20 wrap-up. Rerun the comparison and the site after each run; arms without `predictions.npz` are listed as missing.
- A30 and the controls set `"save_last": true`, so the run keeps `last.pt`. A50 is a warm restart:
  `"resume_from": {"run": "checkpoints/finetune/...-lat-os-noise-ep30", "lr_scale": 0.5}` loads A30's `last.pt`,
  `best.pt`, `history.jsonl` and `report.json`, so A30 must finish first and keep those files.
- The rebalancing keys are `oversample` (labels, copies), `label_noise` (labels, sigma), `label_weights`,
  `divergence_stop` (margin, epochs) and `resume_from` (run, lr_scale); change those and `name` for a new variant.
- Inputs not in git: `data_root` (and optionally `train_series_mirror`) in `paths.json`, the cohort manifest and LM
  labels CSV named in each config, and for the comparison the finished run 5 (both seeds) and u9-b1 run directories.

## If revisited

The Lateral Meniscus LM-test gain is real but bought with other labels. Options worth trying: a smaller
weight (1.25–1.5×), oversampling without the extra steps (fixed epoch length), or a lateral-only
member in an ensemble rather than a single model. Per-label specialists did not replicate before
(ensemble-selection-v1), so the last option needs an out-of-sample check.

Sources: `lateral-rebalance-v1.json` (A30 run dir `...-lat-os-noise-ep30`, which also keeps `last.pt`) (built by `scripts/compare_lateral_rebalance.py` on branch
`t3code/fine-tune-weak-labels`), run dirs `checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-{os-noise,w2}`.
