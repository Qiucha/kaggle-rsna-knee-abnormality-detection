# Full projection baseline handoff — 2026-09-05

> Historical baseline handoff. For current results and next actions, read the
> [local handoff notes](handoff/).

Experiment: `full-multilingual-v1 / 00-projection-full-mean-512`.
Source revision: the Git commit introducing this handoff; future runs record the
actual commit, dirty status, source hash, dependencies, and input hashes in their
run manifest. Setup and GPU smoke commands: [README](../README.md).

## Completed

- Audited metadata migration preserved manifest bytes, 4,407 rows, and fixed splits.
  Manifest SHA-256: `54bd803e23acf3299daeb815d651a34249c736689e20d790f862fc71c0520b63`.
  Source CSV/report hashes, series directories, saved QC, and reader invariants passed.
- Reader rejects missing/empty/mismatched checksums. Matrix also requires the frozen hash.
- Root causes: optional checksum enforcement, training's recall-only comparison,
  and resume retaining only recall. Shared selection now uses recall, loss, earliest
  epoch; finalization and resume reject checkpoint/history disagreement. Legacy
  discrepant checkpoints are not silently relabeled or overwritten.
- Compact-checkpoint training/resume regression verified tie handling and checkpoint
  metrics with a tiny local model. This is not a GPU or pretrained-backbone smoke run.
- Versioned matrix, resolved baseline configuration, protocol, artifact SHA-256
  inventory (including pinned model snapshots), and installed dependency closure.
- Verification: 25 tests passed; Python compilation and `pip check` passed;
  eight matrix dry-run commands passed. No temporary debug instrumentation remains.

## Attempted baseline and blocker

Command:

```bash
.venv/bin/python scripts/run_contrastive_matrix.py --experiment 00-projection-full-mean-512
```

Exit status 1: `This matrix requires CUDA; use --dry-run on a CPU-only host`.
Host: Python 3.13.12, PyTorch 2.13.0+cpu, CUDA unavailable, no NVIDIA device or
`nvidia-smi` exposed. Repository filesystem has about 475 GiB free; dataset volume
has about 2.4 TiB free. Required model snapshots exist locally and are hashed.
Changing the Python wheel alone cannot supply a missing GPU device.

No baseline training epochs or new full-baseline checkpoints were produced.
Runtime, peak GPU memory, curves, and a continue/revise decision are pending.
Next action: provide a CUDA host, restore the inventoried artifacts, record the
GPU dependency environment, pass the documented smoke/reload/resume checks, and
run the baseline. Keep the locked test split unused. Independent label provenance,
patient independence, and duplicate/leakage review remain explicit limitations.

## CUDA follow-up

The earlier statement that the device lacks a GPU was incorrect. An unsandboxed
WSL host probe found an RTX 3070 with 8 GB, NVIDIA KMD 610.74 / UMD 610.53,
and a successful `cuInit(0)`. The agent sandbox hid `/dev/dxg`, while the WSL
`nvidia-smi` binary was outside PATH. The user authorized CUDA setup on this
device and required explicit permission for any further Git commit or push.
All follow-up source/configuration changes therefore remain uncommitted.

CUDA setup and smoke checks are now complete: PyTorch `2.13.0+cu126`, torchvision
`0.28.0+cu126`, 27 passing GPU-enabled tests. The smoke uses 54 training and four
validation studies from the first 64 cohort rows, batch size 32, one initial
epoch and one resumed epoch. Compact reload reproduces stored validation metrics.
Initial/resumed runs took 96.68/57.35 seconds; peak allocated GPU memory was
1,709,895,168/1,718,355,456 bytes. Full baseline launched at 10:53 UTC (18:53 Taipei).

Two additional regression fixes were necessary: validation moved embeddings to
CPU but retained the temperature scalar on CUDA; evaluation also ignored the
training manifest's `max_studies` limit. Both now have focused tests. The protocol's
loss description was corrected to match the existing full-validation-matrix loss.

The execution directory contains the driver/environment record, dependency lock,
measured command logs, and a hashed source snapshot preserving uncommitted code.

## Final status

The baseline completed all 20 epochs at 12:50 UTC on 2026-09-05. Compact validation
reload agrees exactly; epoch 20 achieves 10.66% mean Recall@1. See
[completed results](full-projection-v1-results.md) and the
[local handoff notes](handoff/), which supersede earlier
blocked/running notes. No commits or pushes were made in the CUDA follow-up.
