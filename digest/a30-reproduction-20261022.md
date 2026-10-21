# A30 reproduction — 2026-10-22

Question: does the refactored code (branch `t3/refactor-script-modules`, scripts regrouped into
folders) still train the model behind our best single Kaggle submission, A30
(`r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30`, public LB 0.890)?

Original report: [lateral rebalancing](lateral-rebalance-v1.md).

**Verdict: yes, as far as an unpinned GPU allows.** The rerun's training loss matches the record to
±0.0015 at every one of the 30 epochs, it selects the same epoch (22), and gold-58 matches (−0.0002).
The recorded checkpoint, scored by the refactored code, reproduces its saved predictions exactly, so the
evaluation path is unchanged. LM-test is 0.0064 lower. That is consistent with nondeterministic training:
the paired interval includes zero, and the rerun is as decorrelated from the record as two seeds are from
each other. One rerun cannot bound the LM-test noise, though, and the one seed-pair reference shows a
smaller gap (−0.0029).

## Setup

- Code: the branch's `src/` and `scripts/training/` as of `e00c6be` (unchanged while the run trained); its `src/contrastive_learning` differs from `origin/main` only in the moved script paths it names (matrix commands, the bundled `predict_kaggle.py`, one comment) and the added `paths.repo_path`; none is on the fine-tuning path.
  The record was trained at `bf200aa`; a match therefore also covers `origin/main`'s changes since then.
- Command: `scripts/training/finetune_supervised.py --config configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30.json`,
  the committed config. Its effective configuration differs from the record's only in naming the gold labels
  `data/gold_label.csv` instead of `data/cleaned.csv`, which are byte-identical.
- Inputs: the same cohort manifest (sha256 `d1043376…`), LM labels (`6f704a7b…`), window caches and series mirror.
  Training, validation and gold populations, oversampling draws (4,946 per epoch) and window counts are identical;
  no window-cache entry was added. RTX 3070, 1,564–1,581 s per epoch (record ~1,560), peak 5.43 GB (same).
- No CUDA determinism is pinned, so the same seed does not give the same run.

## Final scores

| | Record | Rerun | Δ (paired 95% CI) |
| --- | ---: | ---: | ---: |
| Selected epoch | 22 | 22 | |
| Validation BCE at selection | 0.4199 | 0.4163 | −0.0036 |
| LM-test macro AUROC | 0.8917 | 0.8853 | −0.0064 [−0.0140, +0.0011] |
| Gold-58 macro AUROC | 0.8814 | 0.8813 | −0.0002 [−0.0182, +0.0166] |

How big is training noise? A first rerun of the same code, cut off at epoch 12 by a host restart, and this rerun
differed over their 12 shared epochs by up to 0.0022 in training loss and 0.018 in validation loss. (The cut-off run
predates two refactor commits, `48df35f` and `e00c6be`, neither on the fine-tuning path.) Neither gives an LM-test
number. Run 5 trained with seeds 2026 and 2027 differs by −0.0029 [−0.0110, +0.0061] on LM-test; it is a different
rung, and a seed change perturbs more than same-seed nondeterminism. Pooled over all labels' scores, the rerun's LM-test
predictions correlate with the record's (r = 0.941) no more closely than the two run-5 seeds do with each other
(r = 0.945). The bootstrap intervals cover test-sample uncertainty only, not training variance.

Consequence for earlier conclusions: A30's LM-test lead over u9-b1 was +0.0083 [−0.0025, +0.0202] on the record and is
+0.0019 [−0.0090, +0.0139] on the rerun. Neither interval excludes zero. The public leaderboard (0.890 vs 0.888)
scored the record, not the rerun, and cannot separate the two values. Treat A30 vs u9-b1 as a tie on one run each.

## Evaluation path

The record's own `best.pt`, rebuilt and scored by the refactored code on the window cache, gives LM-test and gold
predictions identical to its saved `predictions.npz` (largest absolute difference 0.0; LM-test 0.8917).

## Every epoch

| Epoch | Train BCE record | rerun | Δ | Val BCE record | rerun | Δ | Val AUROC record | rerun | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5021 | 0.5026 | +0.0005 | 0.5638 | 0.5595 | -0.0043 | 0.7813 | 0.7813 | +0.0000 |
| 2 | 0.4734 | 0.4723 | -0.0011 | 0.4988 | 0.4939 | -0.0049 | 0.8274 | 0.8271 | -0.0003 |
| 3 | 0.4567 | 0.4567 | +0.0001 | 0.4840 | 0.4880 | +0.0040 | 0.8378 | 0.8372 | -0.0006 |
| 4 | 0.4432 | 0.4442 | +0.0010 | 0.4703 | 0.4865 | +0.0162 | 0.8538 | 0.8551 | +0.0013 |
| 5 | 0.4301 | 0.4294 | -0.0007 | 0.4812 | 0.4541 | -0.0270 | 0.8577 | 0.8571 | -0.0006 |
| 6 | 0.4124 | 0.4126 | +0.0002 | 0.4630 | 0.4507 | -0.0122 | 0.8499 | 0.8514 | +0.0015 |
| 7 | 0.4002 | 0.3988 | -0.0015 | 0.4454 | 0.4333 | -0.0121 | 0.8588 | 0.8628 | +0.0040 |
| 8 | 0.3880 | 0.3871 | -0.0009 | 0.4662 | 0.4663 | +0.0001 | 0.8585 | 0.8524 | -0.0062 |
| 9 | 0.3780 | 0.3780 | -0.0000 | 0.4629 | 0.4573 | -0.0056 | 0.8605 | 0.8638 | +0.0032 |
| 10 | 0.3707 | 0.3702 | -0.0004 | 0.4290 | 0.4369 | +0.0080 | 0.8657 | 0.8602 | -0.0055 |
| 11 | 0.3623 | 0.3615 | -0.0007 | 0.4377 | 0.4393 | +0.0016 | 0.8509 | 0.8566 | +0.0057 |
| 12 | 0.3556 | 0.3553 | -0.0003 | 0.4351 | 0.4266 | -0.0085 | 0.8572 | 0.8635 | +0.0063 |
| 13 | 0.3486 | 0.3479 | -0.0007 | 0.4417 | 0.4375 | -0.0042 | 0.8677 | 0.8670 | -0.0007 |
| 14 | 0.3429 | 0.3425 | -0.0005 | 0.4317 | 0.4294 | -0.0023 | 0.8649 | 0.8628 | -0.0021 |
| 15 | 0.3375 | 0.3376 | +0.0001 | 0.4328 | 0.4292 | -0.0036 | 0.8625 | 0.8618 | -0.0007 |
| 16 | 0.3334 | 0.3331 | -0.0003 | 0.4372 | 0.4220 | -0.0152 | 0.8629 | 0.8699 | +0.0070 |
| 17 | 0.3299 | 0.3292 | -0.0008 | 0.4277 | 0.4261 | -0.0016 | 0.8681 | 0.8670 | -0.0011 |
| 18 | 0.3262 | 0.3257 | -0.0005 | 0.4254 | 0.4218 | -0.0037 | 0.8698 | 0.8661 | -0.0036 |
| 19 | 0.3235 | 0.3240 | +0.0005 | 0.4313 | 0.4215 | -0.0098 | 0.8659 | 0.8670 | +0.0011 |
| 20 | 0.3220 | 0.3217 | -0.0003 | 0.4309 | 0.4247 | -0.0061 | 0.8702 | 0.8721 | +0.0019 |
| 21 | 0.3199 | 0.3203 | +0.0003 | 0.4264 | 0.4230 | -0.0033 | 0.8690 | 0.8670 | -0.0020 |
| 22 | 0.3188 | 0.3189 | +0.0000 | 0.4199 | 0.4163 | -0.0035 | 0.8706 | 0.8721 | +0.0016 |
| 23 | 0.3177 | 0.3178 | +0.0001 | 0.4276 | 0.4218 | -0.0057 | 0.8686 | 0.8725 | +0.0039 |
| 24 | 0.3171 | 0.3172 | +0.0002 | 0.4223 | 0.4194 | -0.0029 | 0.8697 | 0.8697 | -0.0001 |
| 25 | 0.3164 | 0.3163 | -0.0001 | 0.4224 | 0.4192 | -0.0032 | 0.8708 | 0.8712 | +0.0004 |
| 26 | 0.3158 | 0.3158 | -0.0000 | 0.4259 | 0.4203 | -0.0056 | 0.8682 | 0.8722 | +0.0040 |
| 27 | 0.3156 | 0.3156 | +0.0000 | 0.4246 | 0.4222 | -0.0024 | 0.8693 | 0.8715 | +0.0022 |
| 28 | 0.3152 | 0.3153 | +0.0001 | 0.4244 | 0.4209 | -0.0035 | 0.8686 | 0.8714 | +0.0029 |
| 29 | 0.3151 | 0.3152 | +0.0000 | 0.4235 | 0.4195 | -0.0041 | 0.8690 | 0.8720 | +0.0030 |
| 30 | 0.3149 | 0.3149 | -0.0000 | 0.4235 | 0.4194 | -0.0042 | 0.8691 | 0.8721 | +0.0030 |

Largest per-epoch |Δ|: training BCE 0.0015, validation BCE 0.0270 (epoch 5, where the record has a one-epoch spike),
validation AUROC 0.0070. `scripts/analysis/compare_run_history.py` prints this table and passes.

## Artifacts

Kept outside the repository and the shared checkpoint tree, in `/home/q/novo-repro/`:
`a30-refactor/` (rerun: history, report, predictions, checkpoints), `a30-refactor.interrupted-ep12/` (the cut-off
first rerun), `paired_a30.py` (paired bootstrap above) and `rescore_a30_record.py` (evaluation-path check).
