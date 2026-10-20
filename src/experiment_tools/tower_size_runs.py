"""The four pinned tower-size arms, shared by the scripts that compare and plot them.

Image tower (DINOv2 small or base) crossed with text tower (E5 base or small), each trained
for 20 epochs at depth 6, rank 64, seed 2026. ``compare_tower_size_factorial.py`` scores the
arms on identical gold folds and ``plot_tower_size_factorial.py`` draws their curves, so the
run directories are pinned once here.
"""

ARM_RUNS = {  # arm -> run directory, relative to the repository root
    "small_image_base_text": "checkpoints/contrastive/both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026",
    "small_image_small_text": "checkpoints/contrastive/tower-size-factorial-v1/dinov2-small-e5-small-depth6-r64-seed2026",
    "base_image_base_text": "checkpoints/contrastive/tower-size-factorial-v1/dinov2-base-e5-base-depth6-r64-seed2026",
    "base_image_small_text": "checkpoints/contrastive/size-matched-holdout-v1/dinov2-base-e5-small-depth6-r64-seed2026",
}
