# Image-only adaptation protocol — 2026-09-07

User authorized launch after review rejected promotion of impression-first.
Retain full text on the quarantined manifest (3,524 eligible train, 441 validation).
Reuse verified seed-2026 full-text baseline, selected epoch 19, mean R@1 10.6576%.

Change only `image_unfreeze_last` from 0 to 1: existing implementation trains
the final DINOv2 transformer block **and final normalization layer**, along with
the projection/fusion parameters already trained by the baseline. Text stays
frozen. Start fresh from pinned pretrained models, with matching seed, rather
than resuming the baseline's trained projection checkpoint. Image LR stays 1e-5.
Keep full reports, multilingual E5 mean pooling, 512 tokens, batch 32, AMP,
20 epochs and selection by mean R@1 descending, validation loss ascending,
earliest epoch. All other saved config values must match the baseline.

Execution: `checkpoints/contrastive/claude-translated-dfd07db-image-block-execution/`.
Its `code/` copies the prior frozen snapshot and verifies all 46 inventory hashes.
`matrix.json` freezes configuration; `run.py` is an exclusively claimed bounded
queue. New outputs: `checkpoints/contrastive/claude-translated-dfd07db-image-block-v1/`.

1. Verify baseline history, selection, checkpoint hash and zero reload deltas.
2. Run a one-epoch/64-study GPU smoke, then verify compact checkpoint reload.
3. Run seed-2026 image adaptation for 20 epochs, then exact validation reload.
4. Only if candidate mean R@1 strictly exceeds matched baseline, run fresh baseline
   and candidate at seeds 2027 and 2028 sequentially. Otherwise stop.
5. Report every attempted run, paired-study descriptive bootstrap differences and,
   if repeated, mean, sample SD and range of differences across all three seeds.

Maximum: one smoke plus five new full runs. Preserve completed runs; do not rerun
the launcher after failure. Inspect exact stage/config/history before any resume.
No text-only or both-tower run is queued. Review these results before selecting
text-only adaptation. No test evaluation, downstream label fitting, commit or push.

Previous baseline and ordering candidate log an infinite gradient norm at epoch 2
under AMP; later norms and exact reload metrics are finite. Skipped-step counts
are not available from those logs. Preserve matched training code for this
comparison and inspect learning curves/telemetry during review.

Patient identity is unresolved; only the user-identified 58 cases are ground truth,
with eight labeled validation cases and zero positive MCL. Report-derived labels
are weak supervision. Exact-pair retrieval is exploratory; template overlap is
not clinical equivalence. Bootstrap intervals after selection do not measure seed
uncertainty or correct for unknown repeated patients. Test stays locked.
