# Factor isolation: completed reports and paired comparisons

One contrastive factor per arm; selection uses mean retrieval R@1. Frozen LM heads use three seeds.
Gold-58 is exploratory. LM-test is agreement with LM verdicts on 432 non-gold held-out studies.
Paired intervals use 2,000 study resamples, seed 2030. Missing head scores mean evaluation pending.

| Arm | Mean R@1 | Gold-58 | LM-test | Δ gold vs reference [95% CI] | Δ LM-test vs reference [95% CI] |
|---|---:|---:|---:|---|---|
| reference | 0.1357 | 0.7670 | 0.8303 | baseline | baseline |
| dinov2-small-uniform-224 | 0.1374 | 0.7630 | 0.8286 | -0.0040 [-0.0237, +0.0162] | -0.0017 [-0.0089, +0.0052] |
| dinov2-small-k8-224 | 0.1817 | 0.7773 | 0.8658 | +0.0103 [-0.0188, +0.0368] | +0.0355 [+0.0234, +0.0478] |
| dinov2-small-336 | 0.1391 | 0.7619 | 0.8370 | -0.0051 [-0.0275, +0.0166] | +0.0066 [-0.0028, +0.0166] |
| omnirad-base-lora-224 | 0.1368 | 0.7647 | 0.8464 | -0.0023 [-0.0294, +0.0253] | +0.0161 [+0.0051, +0.0274] |

The codec reference is the common anchor; the K8 matched comparator is the uniform-224 arm.
A difference against the common reference is not the isolated K8 effect: it also changes plane pooling.
The OmniRad swap changes model size and pretraining. No unadapted LM-test score is assumed.

Sources: saved summary, metrics.jsonl, config.json, benchmark.json, gold-predictions.npz,
bundle checksums and existing LM-test feature caches. Predictions are checked against reported AUROCs.

[Phase 1 page](site/pages/phase1.html#factor-isolation)
[Verified data](factor-isolation-v1.json)

## Reproduce

From the repository root on a CUDA host, with `paths.json` naming `data_root` (and `train_series_mirror` for the K8 arm):

1. Train one arm: `python scripts/run_contrastive_matrix.py --matrix configs/contrastive/factor-isolation-v1.json --experiment <arm>`.
2. Fit its heads, once per config the arm lists under `lm_head_configs`: `python scripts/train_lm_head.py --config configs/kaggle/<head>.json --features checkpoints/kaggle/<head>-features.npz --output checkpoints/kaggle/<head> --device cuda`.
3. Rebuild this digest: `python scripts/build_factor_isolation_digest.py`. It also reads the chain's off-repo `summary.json` (see its docstring).

A new arm is a new matrix entry. Its `args` override the base recipe. Copy a head config and point `checkpoint` at the new run's `best.pt`.
Not in git: the cohort manifest (checked against `manifest_sha256`), `data/lm_label/lm_labels_kaggle.csv`, and the pinned Hugging Face snapshots.
