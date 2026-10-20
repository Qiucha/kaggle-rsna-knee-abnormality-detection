# Archived scripts

- `monitor_run5_queue.py`, `finalize_clip_comparison.py`, `build_run5_queue_digest.py`: chain helpers from the 2026-10-03 run-5 queue; read ps/WSL/queue files, kept for provenance.
- `ensemble-sweeps/`: the rank-ensemble sweeps and A30 inference probes behind `digest/ensemble-selection-v1`, `digest/ensemble-a30-v1.json`, `digest/a30-inference-probes-v1.json` and the CLIP-init u9 swap, copied verbatim from the git-ignored `.scratch/leaderboard-next/scratch/` on 2026-10-20. They hardcode `/home/q/...` paths and read per-run `predictions.npz` caches under `checkpoints/` and `.scratch/`, so they are provenance for the published numbers, not runnable from a fresh clone.
