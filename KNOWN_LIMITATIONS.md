# Known limitations

Behaviour the code has on purpose or tolerates for now, which is easy to trip over.
Each entry says when it bites and what to do instead. Fixed issues do not belong here.

## Training

### Gradient checkpointing on DINOv2 needs transformers 5.x

`--gradient-checkpointing` patches `Dinov2Encoder.forward` to checkpoint each block
(`model._enable_gradient_checkpointing_for_dinov2`), because Hugging Face's own
`gradient_checkpointing_enable()` is a no-op for that encoder. The patch calls each
`Dinov2Layer` and uses the result as the next hidden state, which assumes the layer
returns a bare tensor. That holds in transformers 5.x (the lock files and the Kaggle
image pin 5.16.1). In 4.x the layer returns a tuple, and the first training step
would fail with a type error.

- **When it bites:** a fresh environment installed from `requirements.txt`, which
  still allows `transformers>=4.40,<6`, then training with `--gradient-checkpointing`.
- **What to do:** install from `requirements-lock.txt` or `requirements-cuda-lock.txt`.
  Inference never uses the patch; it is active only while the backbone is in
  training mode.

## Fine-tuning export and Kaggle bundles

### CoAtNet (timm) runs cannot be exported yet

`scripts/submission/export_finetune_bundle.py` refuses runs with `image_model_format: "timm"`,
because `submission.load_member` rebuilds only Hugging Face `AutoModel` backbones and
has no path for a timm model behind `TimmBackbone`.

- **When it bites:** a CoAtNet run (`configs/finetune/r4a-k8-coatnet-384-*`) scores
  well on LM-test and is meant to join a submission.
- **What to do:** add a timm-aware `load_member`, and make sure timm loads offline on
  Kaggle, before relying on a CoAtNet member. Until then, treat CoAtNet results as
  research only.

### `rank_mean` bundles output ranks, not probabilities

A bundle with `"ensemble": "rank_mean"` averages per-label percentile ranks over
the studies being scored (`submission.combine_members`). The values lie in (0, 1]
but are not calibrated probabilities, and they depend on which studies are scored
together.

- **When it bites:** any metric or downstream use that needs calibration, such as
  log loss, thresholds or a probability average with another model.
- **What to do:** nothing for the leaderboard: its metric is AUROC, which reads
  only order. Otherwise use `"mean_probability"`.

### Assembling a bundle checks less than exporting one

`scripts/submission/assemble_ensemble_bundle.py` copies already-exported members byte for byte
and re-checks their SHA-256. It then copies the current `src/` into the bundle and
compares only the overall LM-test macro AUROC with the saved predictions (tolerance
1e-3). Unlike `scripts/submission/export_finetune_bundle.py`, it does not compare predictions
study by study, and it does not check that the members share the cohort manifest
and label files.

- **When it bites:** members exported at different commits, or from runs on
  different cohorts or label files. A member could then run under code it was not
  validated with, and a small per-study drift could pass an aggregate check.
- **What to do:** in that case, export all the runs together with
  `export_finetune_bundle.py --run … --run …`. It checks each member against its
  run's predictions (|Δp| ≤ 0.02) and the combined ranks (at most one position),
  and refuses members that disagree on cohort or labels.

### Grouped pooling cannot tell a missing run from a zero one

With `pooling: "grouped"`, each plane's K windows split into `window_groups`
contiguous runs, and each run is mean-pooled over its valid windows
(`submission.pool_windows`). A run with no valid window becomes a zero vector, and
the head cannot tell that apart from real features near zero.

- **When it bites:** a plane is masked because none of its series decode, so all
  of its runs are zero. This is rare in training and at inference.
- **What to do:** nothing at current rates. If masked planes become common, add a
  per-run validity indicator to the head input.
