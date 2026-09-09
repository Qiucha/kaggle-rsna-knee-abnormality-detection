> Completed: all 20 epochs; best epoch 16; validation mean Recall@1 9.7506%;
> checkpoint reload has zero metric deltas. Measured training: 29.03 minutes.
> Latest translation-update decision: [assessment](translation-update-dfd07db.md).
> Earlier launch-status notes below are historical.

# First Claude-translated contrastive baseline

Remote input: `origin/config` commit `9bec1bd2becef05cc898f30c34e0ad5bd2f99ba5`
(`completed translations`), fetched on 2026-09-07 Asia/Taipei. The existing dirty
`contrastive-learning` branch was preserved; remote files were extracted by commit.

All 4,407 `LLM Translation` cells are nonempty, with unique study IDs and exact
source-report alignment after outer whitespace trimming. This is complete coverage,
not comprehensive clinical validation. Three train/validation spot checks, including
previously flagged Spanish rows 1201 and 2419, retain English findings and impression
sections. Provider model/effort provenance is not established by the CSV itself.

The derived cohort retains 1,736 source-English reports and uses Claude translations
for 2,671 other reports, following the existing translated-English loader policy.
It explicitly uses `LLM Translation`, not the Google `Translated Report` column.
All nontext row fields are asserted unchanged, including patient groups, image paths,
image QC, labels, eligibility, and the 3,526/441/440 train/validation/test split.
The original language detections are retained as metadata, including known anomalies.
See [mechanical audit](claude-translated-baseline-audit.json).

This experiment changes only report text. It uses the same pinned multilingual E5
and DINOv2-small encoders, frozen towers, projection heads, symmetric contrastive
objective, 512-token mean pooling, batch size 32, seed, optimizer, and 20 epochs as
the completed multilingual baseline. Keeping E5 isolates the text change. No CLMP,
SeLIP, MedMAP, or MedCLIP changes are included. Existing duplicate-report and
patient/label-provenance limitations remain; this is an exploratory baseline.
The test split remains locked.

Configuration: `configs/contrastive/claude-translated-9bec1bd-v1.json`.
Manifest: `data/processed/contrastive-cohort-claude-9bec1bd-v1.jsonl`.
Manifest generation is reproducible with the saved `prepare_baseline.py` under
`checkpoints/translations/config-9bec1bd/` (exclusive outputs prevent overwriting).

Execution directory: `checkpoints/contrastive/claude-translated-9bec1bd-execution/`.
`run.py` runs a one-epoch 64-row GPU smoke, validation reload, full measured training,
and full validation reload in sequence, stopping on failure. `status.json` records
stage and completion/failure; `telemetry.json` records full-run runtime and GPU memory.
`source-snapshot.tar.gz` preserves source/configuration and preparation/launcher code.
A persistent host launcher allows training to continue after this conversation turn.
Do not launch a duplicate while its PID is alive.

Run directory: `checkpoints/contrastive/claude-translated-9bec1bd-v1/00-projection-full-mean-512/`.
Read `metrics.jsonl` for epoch results, `run_manifest.json` for provenance, and
`best.pt`/`last.pt` for checkpoints. Full training status and final performance are
not implied by launch; consult the execution status and completed epoch history.
After completion, compare selected validation mean Recall@1 with the original
baseline's 10.66% (20 selected epochs, 441 validation candidates), and verify
`reload-validation.json` against the selected epoch. No test evaluation is needed.

Launch verification: GPU smoke and validation reload passed with zero metric deltas.
The full run entered `full_training`. Its saved config differs from the original
only in cohort/output paths and offline model loading. Launcher PID: 1576274.
No full-run performance result is available at this handoff.
