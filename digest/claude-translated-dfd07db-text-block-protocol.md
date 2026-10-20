# Text-block adaptation — frozen protocol, not launched

Frozen 2026-09-11 (Asia/Taipei) after the user approved the proposal in
[the text error review](english-medical-text-error-review.md) and explicitly chose to
withhold the launch. **Nothing has been run.** No checkpoint, log, telemetry or status
record exists under the output root, and no launcher was written. This document plus
`configs/contrastive/claude-translated-dfd07db-text-block-v1.json` are the whole artifact.

Matrix selection is a one-way door (`src/contrastive_learning/experiments.py`, commit
`298dec1`): once a grid runs and is finalized, `validation-selection.json` refuses to be
rewritten with different content. Freezing the description is the reversible half; the
launch is not. That is why they are separated here.

## The question

Image-only final-block adaptation is complete at three matched seeds and gained
+1.2094 pp mean R@1 (sample SD 0.6643 pp, all paired intervals spanning zero). The text
side of the same question has never been tried on the full cohort. This arm tests it,
holding everything else — cohort, objective, image tower, encoder, tokenizer — fixed.

`--text-unfreeze-last 1` unfreezes the final transformer block **and the final layer
norm** (see `freeze_then_unfreeze_last` in `model.py`; the norm sits after the adapted
block and holding it fixed fights the shift the block introduces). That is the exact
mirror of what `--image-unfreeze-last 1` did in the image arm.

## What is frozen

| | |
| --- | --- |
| Cohort | `data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl`, quarantined |
| Manifest SHA-256 | `8f6f021954e268b995dbe63daf42a2cc5f0dd92efdd27ad511b0ddbe5832f99b` |
| Text encoder | `intfloat/multilingual-e5-base` @ `d128750597153bb5987e10b1c3493a34e5a4502a`, mean pooling, `query: ` prefix |
| Image encoder | `facebook/dinov2-small` @ `ed25f3a31f01632728cabb09d1542f84ab7b0056`, **frozen** (`--image-unfreeze-last 0`) |
| Adaptation | `--text-unfreeze-last 1` — final text block + final layer norm |
| Report view | full, 512-token cap |
| Schedule | 20 epochs, batch 32, `--lr-text 2e-5`, `--lr-image 1e-5`, weight decay 0.02 |
| Seeds | 2026, 2027, 2028, all three budgeted up front |
| Selection | mean R@1 descending, validation loss ascending, earliest epoch — validation only |
| Test | locked; not evaluated, not inspected |

## Matched baselines already exist — this needs three new runs, not six

All three frozen baselines at the matched seeds are already on disk, verified
2026-09-11 to be identical to each other in every setting except the seed
(`text_unfreeze_last 0`, `image_unfreeze_last 0`, full report, 20 epochs, batch 32,
same revisions, same LRs, same cohort):

| Seed | Matched frozen baseline | `best.pt` SHA-256 | Mean R@1 |
| ---: | --- | --- | ---: |
| 2026 | `claude-translated-dfd07db-q2-v1/00-full-seed2026` | `792aed15…` | 10.6576% |
| 2027 | `claude-translated-dfd07db-image-block-v1/full-seed2027` | `82e193d4…` | 8.9569% |
| 2028 | `claude-translated-dfd07db-image-block-v1/full-seed2028` | `321d279d…` | 8.8435% |

The seed-2027 and seed-2028 baselines were produced as part of the image-adaptation
queue and are reused rather than retrained. Their R@1 values reproduce the figures the
handoff records for the frozen baseline arm.

## Cost

Three runs. By analogy with the completed 20-epoch text screens on the same cohort
(42.06 and 43.12 minutes, peak 1.181 GiB allocated / 1.955 GiB reserved on the RTX 3070),
expect roughly 45 minutes each, so about 2¼ hours of GPU time, plus reload verification.
That is an estimate from a shared GPU, not an isolated throughput measurement, and the
adapted arm backpropagates through one more transformer block than the frozen screens
did, so treat it as a floor.

## Deliberate choices worth a second look before launching

- **`--lr-text` stays at the baseline's 2e-5.** The image arm likewise kept `--lr-image`
  at the baseline's 1e-5 while unfreezing a block, so this is the matched choice. But
  note what changes: in every completed arm 2e-5 only ever drove the projection head,
  because the text backbone was frozen. Here it drives real pretrained weights for the
  first time. If you would rather adapt more gently, that is a one-line change to
  `base_args` — and it must be made *before* the first run, not after seeing a seed.
  Editing it now costs nothing and re-freezes nothing: the one-way door is the selection
  file, and no run, no output root and no `validation-selection.json` exist yet. Re-run
  the dry run afterwards and the config is frozen again on the new value.
- **No repeat rule.** `repeat_seeds` is empty and all three seeds are experiments. Two
  single-seed screens have now returned negative point estimates with intervals spanning
  zero; a conditional third seed cannot fix that, so the seeds are budgeted rather than
  earned.
- **The image cache is shared** with `full-multilingual-v1`. Image preprocessing is
  unchanged in this arm, so that is correct and saves the decode pass — but it also means
  a corrupted cache would silently affect this arm too.

## Validation performed

`.venv/bin/python scripts/run_contrastive_matrix.py --matrix
configs/contrastive/claude-translated-dfd07db-text-block-v1.json --dry-run`

The dry run is total: it loads the matrix, checks the selection policy against the code's
`SELECTION_POLICY`, reads the cohort manifest and verifies its SHA-256 against the frozen
value, and emits the exact three commands. It passed on 2026-09-11 on the CPU
environment. It trains nothing.

## To launch, when and if the user says so

Run the dry run again first; if the cohort or the selection policy has moved, it fails
rather than training something that is no longer the frozen comparison. Then the three
experiments, each in a fresh interpreter, followed by
`scripts/finalize_contrastive_matrix.py`, which writes the immutable
`validation-selection.json` with `locked_test_was_used_for_selection: false`. Do not pass
`--evaluate-locked-test`. Nothing in this document authorizes any of that.
