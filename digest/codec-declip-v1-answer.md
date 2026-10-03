- **Codec-pipeline replicate (2026-09-27): no performance difference, as expected.** Its inputs are bit-identical to the
  reference's, and every gap is inside run-to-run noise:
  - gold MLP 0.7670 vs 0.7659 (Δ +0.0011 [−0.0143, +0.0164]);
  - LM-test MLP 0.8303 vs 0.8336 (Δ −0.0033 [−0.0095, +0.0030]);
  - selected mean R@1 13.57% vs 13.34%, I→T 14.43% vs 13.97%, T→I 12.70% vs 12.70%.

  Per-epoch mean R@1 differs by a median of 0.0035. Epoch-1 training loss agrees to 2.3e-4, and the two runs diverge
  afterwards through CUDA nondeterminism. The codec fix is therefore safe to train with, but it is not an accuracy lever.
  All later arms run on it anyway.
- **This pair is the second baseline run.** The reference and the replicate bracket single-seed noise. On LM-test MLP
  AUROC they span 0.830–0.834, and on gold MLP 0.766–0.767.
- **DeCLIP on DINOv2-small + E5-base (2026-09-27): no gain, a null result.**
  - Gold MLP 0.7686 is just above the 0.7659–0.7670 baseline pair (Δ vs reference +0.0027 [−0.0134, +0.0194]).
  - LM-test MLP 0.8315 is inside the 0.8303–0.8336 pair (Δ −0.0021 [−0.0101, +0.0056]).
  - Selected mean R@1 13.16% is below both baselines (13.34%, 13.57%), well inside the ~0.024 single-seed spread.
  - The linear control moves in opposite directions on the two measures: gold +0.0150, LM-test −0.0091, both
    intervals spanning zero. That is consistent with noise.
  - DeCLIP cost 1.32 h against 0.86 h for plain CLIP, with 4.7 GB peak (gradient checkpointing).
  - This repeats the earlier declip-v1 finding (DeCLIP + LoRA ≈ LoRA) on the gold-holdout protocol.
- **DeCLIP on RAD-DINO + E5-base (2026-09-27): DeCLIP does not rescue RAD-DINO.** Against plain RAD-DINO with the
  same towers, the differences are nil:
  - gold MLP 0.7189 vs 0.7246 (Δ −0.0057 [−0.0216, +0.0103]);
  - LM-test MLP 0.8031 vs 0.8027 (Δ +0.0004 [−0.0064, +0.0065]).

  Against the DINOv2 baselines it is clearly worse:
  - gold MLP Δ −0.047 [−0.082, −0.012] vs the reference and −0.048 vs the replicate;
  - LM-test MLP Δ −0.031 [−0.046, −0.016] and −0.027. Both intervals exclude zero.

  Retrieval fell slightly below plain RAD-DINO (mean R@1 10.39% vs 11.84%), although the lowest validation loss
  improved (3.63 vs 3.79). It trained faster than plain RAD-DINO (2.1 h vs 3.7 h), because gradient checkpointing kept
  the peak at 6.0 GB instead of spilling past 8 GB. The RAD-DINO deficit is a property of the image tower at this
  224-pixel input, not of the objective.
- **DeCLIP on RAD-DINO + BGE-M3 · 512 (2026-09-28): a larger text tower does not close the RAD-DINO gap.** Gold MLP
  0.7339 and LM-test MLP 0.8027. Against the DINOv2 baselines:
  - gold Δ −0.032 [−0.067, +0.002] vs the reference and −0.033 [−0.066, −0.001] vs the replicate;
  - LM-test Δ −0.031 [−0.045, −0.016] and −0.028 [−0.042, −0.013].

  Against DeCLIP RAD-DINO + E5-base, BGE-M3 adds +0.015 [−0.003, +0.033] gold and nothing on LM-test (0.8027 vs 0.8031).
  That matches the BGE-M3 hint on DINOv2 (+0.009 gold, not significant) in direction only. It had the best retrieval
  of the RAD-DINO arms (mean R@1 11.55%) but the worst lowest validation loss (3.81). Training took **10.2 h** at a
  9.5 GB peak (spilling past the 8 GB GPU), against 2.1 h for the E5-base arm.
- **Bottom line.** The deployed DINOv2-small + E5-base with plain CLIP stays the reference:
  - the codec pipeline is neutral for training;
  - DeCLIP is neutral on DINOv2;
  - RAD-DINO is worse on LM-test by about 0.03 however it is trained, and BGE-M3 doesn't change that.

  The one untested RAD-DINO variable is its native 518-pixel preprocessing.
