# Image-only final-block comparison

Status: completed / complete

Quarantined cohort, full reports, 441 validation candidates. Image final block and final normalization adapt; text stays frozen. Seed-2026 baseline is reused after verification.

| Arm | Seed | Best epoch | Mean validation R@1 |
| --- | ---: | ---: | ---: |
| baseline | 2026 | 19 | 10.6576% |
| image_block | 2026 | 18 | 11.1111% |
| baseline | 2027 | 18 | 8.9569% |
| image_block | 2027 | 18 | 10.4308% |
| baseline | 2028 | 13 | 8.8435% |
| image_block | 2028 | 17 | 10.5442% |

Paired differences (candidate minus baseline), percentage points:

```json
[
  {
    "seed": 2026,
    "mean_difference_pp": 0.45351473922902497,
    "descriptive_paired_query_bootstrap_95_ci_pp": [
      -1.7006802721088436,
      2.7210884353741496
    ]
  },
  {
    "seed": 2027,
    "mean_difference_pp": 1.473922902494331,
    "descriptive_paired_query_bootstrap_95_ci_pp": [
      -0.5668934240362812,
      3.5147392290249435
    ]
  },
  {
    "seed": 2028,
    "mean_difference_pp": 1.7006802721088436,
    "descriptive_paired_query_bootstrap_95_ci_pp": [
      -0.9070294784580499,
      4.195011337868481
    ]
  }
]
```

Intervals are descriptive after validation checkpoint selection. Unknown repeated patients and report duplication remain limitations. No locked-test or downstream evaluation. See the image-block protocol for the bounded repeat rule.

Across-seed mean difference: 1.2094 pp; sample SD: 0.6643 pp; range: [0.4535, 1.7007] pp.

## Independent verification — 2026-09-07

All six full runs passed independent checks of contiguous 20-epoch history,
checkpoint selection, checkpoint hashes, zero exact-reload metric deltas,
matched configs, 882 direction/query ranking rows and completed telemetry.
All 46 frozen source-file hashes and the cohort manifest checksum match.
Paired differences were independently recomputed and match the queue report.
See [verification artifact](claude-translated-dfd07db-image-block-audit.json).
Reproduce with `.venv/bin/python checkpoints/contrastive/claude-translated-dfd07db-image-block-execution/verify_results.py`.

Image adaptation is promising: positive differences at all three matched seeds,
mean +1.2094 percentage points, sample SD 0.6643 points, range +0.4535 to +1.7007.
Every descriptive paired-query interval includes zero. Selection and unresolved
patient overlap limit inference. The detailed image-adaptation candidate error and
subgroup review is complete — see the continuation below and
[the review](claude-translated-dfd07db-image-block-error-review.md). The earlier
impression-first error review is not an image-adaptation error review.

## Error review completed — continuation

The [candidate error and subgroup review](claude-translated-dfd07db-image-block-error-review.md) now resolves the outstanding review noted above. All twenty worst seed-2026 direction/query pairs were reviewed against retrieved paired reports, and all three seeds were sliced by report provenance, length and summary visibility with the full candidate pool.
