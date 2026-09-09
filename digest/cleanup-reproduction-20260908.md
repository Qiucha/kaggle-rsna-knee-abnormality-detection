# Cleanup reproduction — 2026-09-08

The cleaned pipeline reproduced the existing full-report, seed-2026 baseline exactly in a fresh 20-epoch GPU run.

Original report: [updated-translation controlled comparison](claude-translated-dfd07db-comparison-results.md).

Both runs used 3,524 training studies, 441 validation studies, frozen DINOv2 and multilingual E5, the same pinned model revisions, seed, input hashes, dependency versions, and training settings. Only the output directory changed.

| Result | Original | Rerun |
| --- | ---: | ---: |
| Selected epoch | 19 | 19 |
| Mean Recall@1 | 10.6575962% | 10.6575962% |
| Image-to-text Recall@1 | 10.4308389% | 10.4308389% |
| Text-to-image Recall@1 | 10.8843535% | 10.8843535% |
| Validation loss | 3.7097836 | 3.7097836 |

## Exact comparisons

- All logged values for all 20 epochs match.
- Every tensor in the selected checkpoint’s trained weights is identical (`torch.equal`).
- All 17 evaluation metrics have zero difference.
- All 882 query-ranking records match, including similarities, ranks, and ties.
- Bootstrap confidence intervals match exactly.
- The original checkpoint also reloads exactly under the cleaned evaluation code.
- The new checkpoint reloads with zero differences from its stored metrics.
- Original run artifacts remained unchanged; source hashes were checked before and after the run.

The original epoch-2 infinite gradient-norm entry is reproduced unchanged. This check establishes equivalence to that run, including its existing behavior.

Training, including dataset preparation, took 25.71 minutes; the complete check with both checkpoint evaluations took 29.63 minutes on the RTX 3070. Peak allocated GPU memory was 1.809 GiB. Only validation was evaluated.

## Artifacts

- [Machine-readable comparison and input/source hashes](cleanup-reproduction-20260908.json)
- [Reproduction script](../checkpoints/contrastive/cleanup-reproduction-20260908/run.py)
- [Training log](../checkpoints/contrastive/cleanup-reproduction-20260908/run.log)
- [Rerun epoch history](../checkpoints/contrastive/cleanup-reproduction-20260908/full-seed2026/metrics.jsonl)
- [Rerun checkpoint evaluation](../checkpoints/contrastive/cleanup-reproduction-20260908/full-seed2026/reload-validation.json)
- [Rerun query rankings](../checkpoints/contrastive/cleanup-reproduction-20260908/full-seed2026/rankings-validation-best.jsonl)
