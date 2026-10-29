# OmniRad CLIP pre-training → run-5 fine-tuning (2026-10-03)

The user selected option A directly: plain CLIP on OmniRad first, then run 5. DeCLIP is disabled; earlier DeCLIP comparisons added cost without a significant downstream gain (`digest/codec-declip-v1-answer.md`).

## Recipe

1. **Pre-train:** OmniRad-base + multilingual E5-base using run 5's **K8 multi-slice/multiplane pipeline**: 8 contiguous 3-slice windows per plane, all 3 planes, 336 px, random jittered windows during training, fixed cached windows at validation, and masked mean within each plane then across present planes. This is **24 encoder inputs per study**. One CLIP loss compares **32 studies/reports**, computed through gradient replay with a **physical study chunk of 1**. Both towers adapt their last **6 blocks**, for 20 epochs with gradient checkpointing. Symmetric CLIP InfoNCE, one-hot paired targets; no DeCLIP, LoRA, report bank, or label-similarity loss. Image/text/projection learning rates are 1e-5 / 2e-5 / 1e-3, matching the earlier full-block depth experiment. Original multilingual reports use the `query: ` prefix and 512-token cap. Select `best.pt` by validation mean Recall@1, then validation loss, then earliest epoch.
2. **Fine-tune:** initialize from that selected image encoder, retaining **all** learned weights, then re-freeze to train only the last **3 blocks**. Otherwise identical to run 5: 336 px, soft LM labels, 8 windows per plane, masked mean pooling, physical batch 2 × accumulation 8, 20-epoch limit, early stopping patience 5, seed 2026. Select the epoch on hard-label validation BCE.
3. **Compare:** paired study-bootstrap AUROC intervals against run 5 and its seed-2027 replicate; 2,000 shared resamples, seed 2030. Gold-58 and LM-test do not enter either stage's checkpoint selection. Gold-58 remains exploratory because these cases have been examined repeatedly.

The frozen gold-holdout cohort checksum is `d104337696a19d52d12480c1cfaae30c937916102e36a67e3124755fa20123f5`. Both radiology weights and HF base architecture have pinned revisions in the config. Both stages reuse run 5's K8 336 evaluation cache and byte-identical training series mirror. Contrastive training delegates window reads to the existing `WindowDataset`; it retains CLIP's spatial/intensity augmentation on those windows. The supervised recipe is unchanged.

## Memory method and verification

The memory method follows [gradient caching (Gao et al., 2021)](https://aclanthology.org/2021.repl4nlp-1.31/). Gradient replay first encodes every image/report chunk without keeping autograd activations, computes the global 32-study CLIP loss, and obtains gradients with respect to those embeddings and the trainable temperature. It then re-encodes each chunk and applies its embedding gradient by the chain rule. Torch CPU/CUDA RNG states are replayed so augmentation, dropout, and gradient checkpointing reproduce the first pass. The scaler, clipping, optimizer, and learning-rate scheduler advance once per full comparison batch. This differs from ordinary accumulation of independent one-study losses, which would have no negatives.

Tests compare cached gradients and loss against a reference graph, including dropout, random augmentation, scaled gradients, uneven chunks, and temperature gradients. A 32-study integration check verifies that optimizer/scheduler steps are counted per full comparison batch. Masked pooled features match run 5 under missing windows and planes. Real training DICOMs from two studies were read with both pipelines using identical seeds: all pixels and masks matched exactly. A real-weight OmniRad/E5 K8 CPU forward/backward passed at 336 px, with finite gradients in every trainable parameter. The real-weight GPU preflight passed on 2026-10-03: two full 32-study loss/optimizer steps used 3,315 MiB peak allocated memory; the batch-2 supervised handoff used 3,825 MiB. The synthetic maximum-512-token profile projected 11.70 hours of pre-training compute, excluding DICOM reads, validation and checkpoint writes. The first actual training attempt exposed 120 absent mirror series from 40 studies with reports but no usable LM labels: contrastive training has 3,478 studies, whereas supervised training has 3,438. All 2,623 missing files (1.78 GB) were copied from the unchanged source and SHA256-verified. The actual K8 reader now resolves every training series, and the formerly failing study decodes all 24 windows. The failed attempt is archived and the queue retries the same recipe. `scripts/audits/audit_clip_window_mirror.py` reproduces this missing-mirror issue and verifies coverage without loading model weights.

More coverage is a hypothesis about performance, not a guaranteed gain. The practical cost is extra encoder work and a replay forward pass. Mean pooling can weaken a focal finding surrounded by unrelated windows, and report alignment can prefer features that do not help the supervised findings. The predeclared AUROC checks below decide whether the completed two-stage experiment helped.

## Predeclared pass rule

All three must hold:

- LM-test candidate minus run 5 has a positive 95% CI lower bound.
- Gold-58 candidate minus run 5 is strictly greater than −0.024.
- LM-test gain exceeds the absolute run-5 versus seed-2027 difference.

No Kaggle submission or later queue promotion is automatic.

## Execution and reporting

The existing queue dispatcher launches this two-stage pipeline from the isolated worktree, without changing the live trainer's imported source. At the user’s request (2026-10-03 18:14 Taipei), the seed-2027 replicate was stopped after 8 completed epochs and moved after this two-stage confirmation. Its partial directory and log are archived with a `.postponed-2026-10-03-181434` suffix. The supervised trainer does not save resumable mid-run weights, so the seed replicate restarts the original recipe after confirmation. The remaining epoch/depth/resolution/backbone arms and CoAtNet follow. Confirmation scores and run-5 confidence intervals are reported immediately; the original pass rule remains pending until the seed replicate finishes. `scripts/archive/finalize_clip_comparison.py` then recomputes the unchanged seed-aware pass rule from saved predictions without retraining. A GPU preflight first checks two full **32-study** optimizer steps with K8 windows and 512 text tokens, encodes one study at a time, verifies gradients, and profiles the compact-checkpoint handoff and batch-2 K8 fine-tuning. The measured step time projects pre-training compute cost; DICOM reads, validation, and checkpoint writes add to that projection. The earlier 17–21-hour total estimate applied to the superseded single-input-per-plane design and is withdrawn. Synthetic preflight pixels and weights are discarded; real training restarts from the pinned weights and seed. The comparison batch is never silently reduced: every optimizer update is based on one 32×32 similarity matrix. Pixels stay on CPU until their chunk is encoded. DataLoader prefetch is limited to one batch per worker, with two workers, to control host RAM.

Live artifacts:

- `checkpoints/pipelines/omnirad-clip-run5-v1/status.json`: queued/running/completed/failed state.
- `checkpoints/pipelines/omnirad-clip-run5-v1/preflight.json`: GPU memory and handoff check.
- `checkpoints/contrastive/omnirad-clip-run5-v1/omnirad-base-e5-base-clip-k8-336-u6-bs32-seed2026/metrics.jsonl`: contrastive epochs.
- `checkpoints/finetune/r4a-k8-omnirad-base-soft-336-from-clip.log`: combined queue log.
- `checkpoints/finetune/r4a-k8-omnirad-base-soft-336-from-clip/report.json`: supervised scores.
- `checkpoints/pipelines/omnirad-clip-run5-v1/comparison.json`: intervals and pass checks.
- `checkpoints/pipelines/omnirad-clip-run5-v1/completion-report.md`: report written automatically on completion or failure.

When this ran, the queued config named the isolated worktree's runner by absolute path. That code is now in the repository (`src/contrastive_learning/clip_pipeline.py`, `scripts/training/run_clip_then_finetune.py`), and the committed config holds no host path: `scripts/training/finetune_supervised.py` runs both stages in-process for any config that names `contrastive_pretrain_config`, and the series mirror comes from `paths.json`. The patch that actually ran is archived at `checkpoints/pipelines/omnirad-clip-run5-v1/isolated-pipeline-code.patch`.

The completion report is automatic on disk. This session has no scheduled agent wake-up capability; generating that report does not itself send a new chat message.

## Recovery on 2026-10-04

All 20 contrastive epochs finished at 06:50 Taipei; epoch 18 was selected (mean Recall@1 0.192071). The in-process supervised stage was killed with exit 247 at 13:30 without recording an epoch; its exact cause is not recoverable from the surviving logs. The host subsequently restarted and interrupted the 30-epoch arm after epoch 20. The hard-label and seed-2027 arms have complete pathology scores and paired intervals in `run5-followups-v1.json`. These results were rendered on the fine-tuning digest page before any recovery job started. The supervised stage restarted at 23:19 from the completed selected contrastive checkpoint in a fresh process. Checkpoint reuse requires all epochs, the identical pinned contrastive recipe, and SHA256 `574b5048ebd7d9e74ca748fa84f9e17b859af91fc1caa7183716a67e1ab6f30a`. The interrupted output and logs are archived; CLIP pre-training is not rerun. The 30-epoch arm restarts its original recipe because it has no resumable weights. The queue rebuilds the follow-up snapshot and digest page before each later GPU job, and pauses if a job fails.

## Reproduce

Run both stages from the repository root on a CUDA host (the pipeline refuses any other device):

```bash
python scripts/training/finetune_supervised.py \
  --config configs/finetune/r4a-k8-omnirad-base-soft-336-from-clip.json \
  --output checkpoints/finetune/<new-name> \
  [--paths-config paths.json]
```

`scripts/training/run_clip_then_finetune.py` takes the same arguments. The fine-tuning config names the pre-training config
(`configs/contrastive/omnirad-clip-run5-v1.json`) in `contrastive_pretrain_config`. As committed, it reuses the finished
CLIP checkpoint (`reuse_completed_contrastive_checkpoint`, checked against `reuse_checkpoint_sha256`) and only reruns the
supervised stage and the comparison.

**A new pre-training seed** changes these keys together:

- In a copy of the contrastive config: `seed`, plus a new `name` and `output_dir`.
- In a copy of the fine-tuning config:
  - `name`;
  - `contrastive_pretrain_config` → the new contrastive config;
  - `init_checkpoint` → `<new output_dir>/best.pt` (the pipeline checks this);
  - a new `pipeline_status_dir`;
  - `reuse_completed_contrastive_checkpoint: false`, and drop `reuse_checkpoint_sha256`.

Keep the supervised `seed` and every other supervised key. The pipeline rejects any departure from run 5's recorded
recipe (`comparison_baseline`/config.json) apart from `name` and host settings. It also fixes the pre-training recipe:
batch 32, last 6 blocks, K8 uniform pooling and gradient replay.

**Prerequisites not in git** (see "Files not in git" under Setup in the README):

- `paths.json`:
  - `data_root`: the DICOM mount, needed for cohort reads.
  - Optionally `train_series_mirror`: a byte-identical local copy of `train_series`.
- `data/processed/contrastive-cohort-gold-holdout-v1.jsonl`, matching `manifest_sha256`.
- `data/lm_label/lm_labels_kaggle.csv` (third party).
- The pinned Hugging Face snapshots: `Snarcy/OmniRad-base`, `facebook/dinov2-base` (architecture) and
  `intfloat/multilingual-e5-base`. Both configs set `local_files_only`.
- For the comparison:
  - Run 5's output, `checkpoints/finetune/r4a-k8-omnirad-base-soft-336` (`config.json`, `predictions.npz`).
  - Optionally, the seed-2027 replicate's predictions. Without them the verdict stays pending
    (`allow_pending_seed_comparison`).
