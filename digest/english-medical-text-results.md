# English / medical text replacement results

Queue: **completed**. Stage: `complete`.

Frozen DINOv2 and frozen text backbones; full translated-English cohort, 3,524 eligible training studies and 441 validation candidates. Only completed runs with independently checked histories, configs, checkpoint hashes, exact reloads and paired rankings appear below.

| Candidate | Seed | Selected epoch | Mean R@1 | Gain pp | Descriptive paired 95% CI pp | Runtime min | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| e5-base-v2 | 2026 | 12 | 8.8435% | -1.8141 | [-4.4218, 0.6803] | 42.06 | 1.181 / 1.955 |
| pubmedbert-base-embeddings | 2026 | 19 | 9.4104% | -1.2472 | [-3.9683, 1.3605] | 43.12 | 1.181 / 1.955 |

## Retrieval directions

| Candidate / seed | Direction | R@1 % | R@5 % | R@10 % | Mean rank | Median rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| e5-base-v2 / 2026 | image_to_text | 9.2971 | 29.9320 | 51.2472 | 19.97 | 10.00 |
| e5-base-v2 / 2026 | text_to_image | 8.3900 | 29.9320 | 49.8866 | 17.33 | 11.00 |
| pubmedbert-base-embeddings / 2026 | image_to_text | 10.4308 | 34.9206 | 53.9683 | 15.74 | 9.00 |
| pubmedbert-base-embeddings / 2026 | text_to_image | 8.3900 | 33.3333 | 55.7823 | 15.88 | 9.00 |

## Attempts

- `e5-base-v2-smoke`: completed
- `e5-base-v2-seed2026`: completed
- `pubmedbert-base-embeddings-smoke`: completed
- `pubmedbert-base-embeddings-seed2026`: completed

## Conditional repeats

{
  "e5-base-v2": false,
  "pubmedbert-base-embeddings": false
}

Across-seed statistics:

```json
[
  {
    "candidate": "e5-base-v2",
    "seeds": [
      2026
    ],
    "mean_gain_pp": -1.8140589569160999,
    "sample_sd_pp": null,
    "range_pp": [
      -1.8140589569160999,
      -1.8140589569160999
    ]
  },
  {
    "candidate": "pubmedbert-base-embeddings",
    "seeds": [
      2026
    ],
    "mean_gain_pp": -1.2471655328798186,
    "sample_sd_pp": null,
    "range_pp": [
      -1.2471655328798186,
      -1.2471655328798186
    ]
  }
]
```

## Subgroup audit

Each slice keeps all 441 retrieval candidates. Length/heading groups use that candidate tokenizer; different candidate tokenizers define different slices.

| Candidate / seed | Group | n | Baseline R@1 % | Candidate R@1 % | Gain pp |
| --- | --- | ---: | ---: | ---: | ---: |
| e5-base-v2 / 2026 | source-English | 175 | 13.4286 | 8.2857 | -5.1429 |
| e5-base-v2 / 2026 | translated | 266 | 8.8346 | 9.2105 | +0.3759 |
| e5-base-v2 / 2026 | over-512 | 45 | 12.2222 | 6.6667 | -5.5556 |
| e5-base-v2 / 2026 | within-512 | 396 | 10.4798 | 9.0909 | -1.3889 |
| e5-base-v2 / 2026 | lost-heading | 22 | 11.3636 | 6.8182 | -4.5455 |
| e5-base-v2 / 2026 | retained-heading | 242 | 9.7107 | 7.8512 | -1.8595 |
| e5-base-v2 / 2026 | no-heading | 177 | 11.8644 | 10.4520 | -1.4124 |
| pubmedbert-base-embeddings / 2026 | source-English | 175 | 13.4286 | 9.1429 | -4.2857 |
| pubmedbert-base-embeddings / 2026 | translated | 266 | 8.8346 | 9.5865 | +0.7519 |
| pubmedbert-base-embeddings / 2026 | over-512 | 19 | 10.5263 | 0.0000 | -10.5263 |
| pubmedbert-base-embeddings / 2026 | within-512 | 422 | 10.6635 | 9.8341 | -0.8294 |
| pubmedbert-base-embeddings / 2026 | lost-heading | 4 | 25.0000 | 0.0000 | -25.0000 |
| pubmedbert-base-embeddings / 2026 | retained-heading | 260 | 9.6154 | 10.5769 | +0.9615 |
| pubmedbert-base-embeddings / 2026 | no-heading | 177 | 11.8644 | 7.9096 | -3.9548 |

Full metrics, twenty worst direction/query pairs per run and paired report text are in the JSON audit (not public: contains report text). Automated extraction alone is not qualitative error review; the completed reading of all forty pairs is in [the error review](english-medical-text-error-review.md).

No test or downstream evaluation. Validation selection, two-candidate screening, report duplication, unknown repeated patients and weak labels limit inference. Study-paired bootstrap intervals are descriptive, not multiplicity- or patient-corrected. Runtime includes sharing the GPU with authorized baseCaps jobs and is not an isolated throughput benchmark. Existing AMP behavior is retained; skipped-step counts are unavailable.

Nonfinite gradient epochs per completed run: {"e5-base-v2/2026": [19], "pubmedbert-base-embeddings/2026": [7]}
