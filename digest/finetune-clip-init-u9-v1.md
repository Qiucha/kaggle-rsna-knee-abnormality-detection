# Completed runs and ensemble submissions · 2026-10-17

The [latest-results card](site/pages/phase2.html#latest-results) brings the finished runs together. The detailed cards retain training BCE, validation BCE, validation AUROC, paired intervals and per-label heatmaps. LM-test is 432 studies with LM labels; gold-58 is exploratory human labeling; public is Kaggle's hidden-test public split.

## Single-run results

| Run | LM-test | Gold-58 | Epochs run / selected | Public |
| --- | ---: | ---: | ---: | ---: |
| A30 reference | 0.8917 | 0.8814 | 30 / 22 | 0.890 |
| A30 + image augmentation | 0.8889 | 0.8648 | 30 / 22 | — |
| A30 + wide slice band | 0.8849 | 0.8752 | 30 / 25 | — |
| A30 recipe + CLIP init, u9 | 0.8865 | 0.8768 | 30 / 20 | — |

A30 remains the highest-scoring single training run on LM-test. Image augmentation, the wide slice band, and CLIP initialization have no clear gain over it: all paired LM-test intervals cross zero. The 30- and 43-epoch controls and A50 remain paused, so A30's gain is still confounded by its longer training budget and oversampling.

The CLIP-init run changes only A30's starting weights to the adapted OmniRad-base checkpoint. It finished all 30 epochs, selected epoch 20, and reached validation BCE 0.4059 versus A30's 0.4199. Its LM-test difference from A30 is −0.0052 [−0.0144, +0.0046]; gold-58 is −0.0046 [−0.0240, +0.0147]. Lower validation BCE did not produce a better single-model test score. Against the older shallow CLIP-init u3 run, gold-58 gains +0.0520 [+0.0252, +0.0835], although depth and training recipe both change in that comparison.

## Completed ensemble submissions

| Ensemble | LM-test, saved predictions | Gold-58 | Public | Submission |
| --- | ---: | ---: | ---: | ---: |
| A30 + u9-b1 + from-CLIP u3 + DINOv2-S | 0.9047 | 0.8724 | 0.894 | 57202235 |
| Swap u3 for CLIP-init u9 | 0.9047 | 0.8859 | 0.899 | 57291794 |
| Deep three: A30 + u9-b1 + CLIP-init u9 | 0.9016 | 0.8897 | **0.902** | 57296356 |

The exported deep-three bundle's GPU round trip scores 0.9017 LM-test, a small numerical difference from 0.9016 on the saved predictions. The leaderboard table reports the bundle score; the paired comparison charts use saved predictions consistently. The swap and deep-three bundle exports matched each member's saved predictions within 0.0002.

Against the first ensemble, the swap changes LM-test by −0.0001 [−0.0031, +0.0029] and gold by +0.0135 [+0.0042, +0.0230]. Deep three changes LM-test by −0.0031 [−0.0084, +0.0019] and gold by +0.0173 [+0.0051, +0.0301]. The public scores improve by +0.005 and +0.003 across the two steps. Gold Lateral Meniscus rises from 0.755 to 0.778 to 0.831, near A30's 0.834.

Gold-58 predicted both public improvements, while LM-test predicted a tie and a small drop. Earlier, gold-58 predicted a drop for the first ensemble even though it improved publicly. These single submissions support using gold as a guard and a tiebreaker; they do not establish it as a replacement selector. Public-split noise has not been estimated.

## Other completed checks

- The [A30 ensemble sweep](site/pages/phase2.html#a30-ensemble-sweep) now includes all 17 historical combinations, paired intervals under two bootstrap seeds, gold-58 and Lateral Meniscus. Its 0.9063 five-member candidate used CoAtNet, which the exporter could not package. Later submissions favor deep members on the public split.
- The [CLIP-init sweep](site/pages/phase2.html#clip-init-ensemble-sweep) includes all 12 later combinations, including the five-member addition, three-member swaps, and deep-only set. Eight exporter-compatible checks are also included under the A30 sweep's full table. Only the three ensembles in the submission table have public scores.
- [Seven inference-only variants](site/pages/phase2.html#a30-inference-probes)—last-checkpoint selection, best/last averaging, two weight soups and three window-jitter TTA variants—show no clear LM-test gain over A30.

Sources: [CLIP-init snapshot](finetune-clip-init-u9-v1.json), [image augmentation](finetune-image-augmentation-v1.json), [slice band](finetune-slice-band-v1.json), [inference probes](a30-inference-probes-v1.json), [A30 sweep](ensemble-a30-v1.json), and [leaderboard history](leaderboard-history-v1.json). Submission scores come from each bundle's `checkpoints/kaggle/*/leaderboard-submission.json`; paired comparisons and histories come from the saved run artifacts.
