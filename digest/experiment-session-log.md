# Experiment session log

Session started: 2026-09-02 (Asia/Taipei)

## Scope

The four follow-ups from `translated-report-training.html`:

- [x] Run five study-level folds for Stanford RadBERT and BiomedBERT.
- [x] Report mean ± standard deviation for image-to-text and text-to-image retrieval.
- [x] Compare frozen backbones with the final one or two blocks unfrozen.
- [x] Compare 256-token and 512-token report inputs.
- [x] Rebuild the digest with the newest results and record the next decision gate.

## Decisions and progress

- 2026-09-02: Located the prior three single-holdout runs and confirmed their configuration: five epochs, batch size 8, 20% validation, seed 2026, frozen DINOv2-small and text backbones.
- 2026-09-02: Confirmed all 58 studies and source DICOM data are available. The host has CPU-only PyTorch (no `nvidia-smi`), so experiments will use `--no-amp` and `--num-workers 0` consistently with the prior pilot.
- 2026-09-02: Identified that the current trainer only supports seeded holdouts, not mutually exclusive folds. Adding an explicit deterministic study-level fold selector before launching the five-fold matrix.
- 2026-09-02: Added `--fold-count`/`--fold-index`. The seeded splitter assigns every study to exactly one validation fold (fold sizes 12, 12, 12, 11, 11); all seven tests pass. Started the 10-run frozen-backbone matrix with seed 2026 and otherwise unchanged pilot settings.
- 2026-09-02: Stopped the first fold before epoch 1 and before any artifact was saved after observing that each process rebuilt the same in-memory DICOM cache. Added a configuration-keyed persistent tensor cache so all runs reuse identical deterministic preprocessing while training remains unchanged.
- 2026-09-02: Completed and validated all 10 cross-validation runs (five epochs and all four expected artifacts per run). Best epochs were selected by mean bidirectional R@1, with the earlier epoch winning ties, matching the existing digest.
- 2026-09-02: Stanford RadBERT five-fold R@1: image→text 18.79% ± 10.65%; text→image 15.61% ± 9.65%; bidirectional mean 17.20% ± 8.92%.
- 2026-09-02: BiomedBERT five-fold R@1: image→text 18.94% ± 7.31%; text→image 20.61% ± 7.60%; bidirectional mean 19.77% ± 6.59%.
- 2026-09-02: Selected BiomedBERT for the follow-up ablations because it has the higher cross-validated mean R@1 and lower fold-to-fold spread. Starting a controlled single-holdout comparison with both image and text towers unfreezing 0, 1, or 2 final blocks; all other settings remain fixed.
- 2026-09-02: Completed the BiomedBERT 0/1/2-block comparison on the original seed-2026 46/12 holdout. Best mean R@1 was 16.67%, 20.83%, and 20.83%, respectively. One and two blocks tied on R@1, but two blocks had stronger best-epoch text→image R@5 (58.33% versus 33.33%) and lower bidirectional mean rank (5.13 versus 6.13).
- 2026-09-02: Starting the frozen BiomedBERT 512-token run. The fresh frozen 256-token control exactly reproduced the original pilot and is reused to isolate sequence length.
- 2026-09-02: Completed the 512-token run. It peaked at 12.50% mean R@1 (image→text 16.67%, text→image 8.33%) versus 16.67% in both directions for 256 tokens. At each run's best epoch, 512 tokens improved mean R@5 (58.33% versus 54.17%), but mean rank was slightly worse (5.29 versus 5.00) and the primary R@1 endpoint did not improve.
- 2026-09-02: BiomedBERT tokenization shows why sequence handling remains relevant: 24/58 reports exceed 256 tokens, 5/58 exceed 512, median length is 236.5 tokens, p90 is 441, and maximum is 710.
- 2026-09-02: Validated 14/14 new run directories: each has a five-row metrics history plus config, best, and last checkpoints. Rebuilt `digest/translated-report-training.html` from all legacy and new metrics.
- 2026-09-02: Final verification passed: 7/7 tests, Python compilation, HTML parsing/content checks, and artifact completeness checks.
- 2026-09-02: Next decision gate: do not adopt two-block adaptation from one split. First repeat the block ablation across five folds, separate image-only from text-only adaptation, and compare 256-token full reports with section-aware extraction. Patient-level IDs and an external diagnostic evaluation remain necessary for a validity claim.

## Results

### Five-fold frozen-backbone comparison

| Text encoder | Direction | R@1 | R@5 | R@10 | Mean rank |
|---|---|---:|---:|---:|---:|
| Stanford RadBERT | Image→text | 18.79% ± 10.65% | 53.33% ± 10.67% | 89.70% ± 7.36% | 5.36 ± 1.12 |
| Stanford RadBERT | Text→image | 15.61% ± 9.65% | 55.00% ± 12.19% | 93.03% ± 7.53% | 5.37 ± 1.06 |
| BiomedBERT | Image→text | 18.94% ± 7.31% | 55.15% ± 14.56% | 93.18% ± 3.83% | 5.04 ± 0.84 |
| BiomedBERT | Text→image | 20.61% ± 7.60% | 55.61% ± 15.37% | 91.36% ± 5.91% | 5.24 ± 0.84 |

Bidirectional mean R@1: Stanford RadBERT 17.20% ± 8.92%; BiomedBERT 19.77% ± 6.59%.

### BiomedBERT block-unfreezing comparison (single holdout)

| Final blocks unfrozen per tower | Image→text R@1 | Text→image R@1 | Mean R@1 | Mean rank |
|---:|---:|---:|---:|---:|
| 0 | 16.67% | 16.67% | 16.67% | 5.00 |
| 1 | 25.00% | 16.67% | 20.83% | 6.13 |
| 2 | 25.00% | 16.67% | 20.83% | 5.13 |

### BiomedBERT token-length comparison (frozen, single holdout)

| Maximum tokens | Image→text R@1 | Text→image R@1 | Mean R@1 | Mean R@5 | Mean rank |
|---:|---:|---:|---:|---:|---:|
| 256 | 16.67% | 16.67% | 16.67% | 54.17% | 5.00 |
| 512 | 16.67% | 8.33% | 12.50% | 58.33% | 5.29 |

The 512-token run does not support increasing the default on this split. Because 41% of reports exceed 256 tokens, section-aware impression/findings extraction is still worth testing rather than simply adding context.
