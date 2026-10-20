# OmniRad CLIP frozen-tower probe

Updated 2026-10-07 17:00:18 +0800 · analysis run 2026-10-05

Question: does plain CLIP pre-training (Phase 1) make OmniRad-base a better *frozen* encoder, and does a head on that frozen tower reach a full label fine-tune (Phase 2)?

Encoders: unadapted OmniRad-base vs the epoch-18 image tower of `omnirad-clip-run5-v1` (plain CLIP, last 6 blocks unfrozen, K8 windows at 336 px, batch 32, seed 2026). Same K8 336 inputs; heads use the run-5 LM-head recipe, three head seeds.

| Readout | Split | Unadapted | CLIP | Δ CLIP − unadapted [95% CI] |
|---|---|---:|---:|---|
| MLP head | LM-test (432) | 0.8413 | 0.8847 | +0.0434 [+0.0327, +0.0545] |
| MLP head | Gold-58 | 0.7566 | 0.7834 | +0.0268 [-0.0053, +0.0589] |
| Linear head | LM-test (432) | 0.8399 | 0.8708 | +0.0309 [+0.0192, +0.0433] |
| Linear head | Gold-58 | 0.7513 | 0.7670 | +0.0157 [-0.0206, +0.0506] |
| kNN (k=15) | LM-test (432) | 0.6992 | 0.7939 | +0.0947 [+0.0731, +0.1160] |

## Frozen head vs full fine-tune

| Comparison | Δ LM-test [95% CI] | Δ gold-58 [95% CI] |
|---|---|---|
| CLIP frozen head − run 5 fine-tune | +0.0072 [-0.0047, +0.0191] | -0.0433 [-0.0778, -0.0115] |
| CLIP frozen head − CLIP then fine-tune | +0.0047 [-0.0060, +0.0155] | -0.0414 [-0.0744, -0.0090] |
| Unadapted frozen head − run 5 fine-tune | -0.0362 [-0.0508, -0.0213] | -0.0701 [-0.1086, -0.0323] |
| Run 5 fine-tune − CLIP then fine-tune | -0.0026 [-0.0142, +0.0089] | +0.0019 [-0.0260, +0.0302] |
| Run 5 − its seed-2027 replicate (noise) | +0.0029 [-0.0061, +0.0110] | -0.0066 [-0.0291, +0.0153] |

## Acquisition control

CLIP also made scanner identity easier to read: manufacturer kNN accuracy 0.6319 → 0.8125 (majority 0.2477); field strength 0.8136 → 0.8910. Effective rank 14.1 → 30.2.

Reading: LM labels come from the reports CLIP aligns to, so the LM-test gain may overstate what CLIP adds; gold-58 is the independent check and its interval includes zero. On gold, every frozen head trails every fine-tune. One CLIP training seed; head-seed noise only; gold-58 is exploratory.

[Probes page](site/pages/probes.html) · [Machine-readable snapshot](omnirad-clip-frozen-probe-v1.json)

## Reproduce

From the repository root on a CUDA host, with `paths.json` naming `data_root`:

1. `python scripts/evaluation/extract_omnirad_probe_features.py --output <features>`: frozen K8 336 features from both towers, after identity checks.
2. `python scripts/analysis/analyze_omnirad_probe.py --features <features> --output <analysis>`: heads, grouping and acquisition control (`results.json`, `head-predictions.npz`).
3. `python scripts/analysis/compare_headonly_vs_finetune.py --heads <analysis>/head-predictions.npz --features <features> --output <analysis>/headonly-vs-finetune.json`.
4. `python scripts/digests/build_omnirad_probe_digest.py --source <analysis>`. The default source is the archived `.scratch/archive/omnirad-probe/full-analysis`.

To probe another checkpoint, change `CLIP_CHECKPOINT` and `CLIP_SHA256` together at the top of the extract script.
Not in git: the CLIP checkpoint (`omnirad-clip-run5-v1`, epoch 18), the run-5 fine-tune predictions under `checkpoints/finetune/`, the cohort manifest, `data/lm_label/lm_labels_kaggle.csv` and the pinned Hugging Face snapshots.
