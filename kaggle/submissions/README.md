# Final submissions

The two submissions chosen for the private leaderboard (decided 2026-10-20). Each folder holds the path-free records of one Kaggle bundle. The weights and the exported bundle themselves stay in git-ignored `checkpoints/kaggle/<name>/`.

| Folder | What | Submission | Public | Private | LM-test | Gold-58 |
|---|---|---|---|---|---|---|
| `a30-deep3-rank/` | Per-label rank mean of A30 + u9-b1 + CLIP-init u9 | 57296356 | 0.902 | 0.889 | 0.9017 | 0.8897 |
| `r4a-k8-omb-soft-336-u9-a30/` | A30 alone: OmniRad-base, K8 windows, soft labels, 336 px, 9 blocks unfrozen, lateral oversampling + label noise, 30 epochs | 57094902 | 0.890 | 0.878 | 0.8917 | 0.8814 |

Member runs. Each run's config is `configs/finetune/<run>.json`, and its run directory is `checkpoints/finetune/<run>/`. The order is the bundle's `member-NN.pt` order:

| Folder | Members |
|---|---|
| `a30-deep3-rank/` | `r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30` (A30), `r4a-k8-omnirad-base-soft-336-u9-b1` (u9-b1), `r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-from-clip` (CLIP-init u9) |
| `r4a-k8-omb-soft-336-u9-a30/` | `r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30` (A30; same `checkpoint_sha256` as deep3's member-00) |

Files in each folder:
- `leaderboard-submission.json` is the submission record: the prediction made before scoring, the scores, `bundle_sha256` and the kernel version.
- `bundle.json` is the bundle manifest: members, ensemble rule, labels and file hashes.
- `dataset-metadata.json` and `kernel-metadata.json` are the Kaggle dataset and kernel definitions. The kernel is pinned to the Python 3.12 image (`sha256:37c64f…`), because the cp312 codec wheels fail on the newer image.

To rebuild a bundle, train the member runs from `configs/finetune/<run>.json`. Then pass each run directory to `scripts/submission/export_finetune_bundle.py --run <dir> … --output checkpoints/kaggle/<name>`. The kernel runs `notebooks/kaggle_submission.ipynb`.
