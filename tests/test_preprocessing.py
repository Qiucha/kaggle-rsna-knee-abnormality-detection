from __future__ import annotations

import torch

from contrastive_learning.preprocessing import apply_left_knee, augment_study_images


def test_left_knee_convention_matches_baseline_axes() -> None:
    image = torch.arange(12).reshape(3, 2, 2)
    assert torch.equal(apply_left_knee(image, "Coronal", "R"), torch.flip(image, [-1]))
    assert torch.equal(apply_left_knee(image, "Axial", "R"), torch.flip(image, [-1]))
    assert torch.equal(apply_left_knee(image, "Sagittal", "R"), torch.flip(image, [0]))
    assert torch.equal(apply_left_knee(image, "Coronal", "L"), image)


def test_augmentation_preserves_shape_dtype_and_range() -> None:
    images = torch.rand(2, 3, 3, 224, 224)
    augmented = augment_study_images(images)
    assert augmented.shape == images.shape
    assert augmented.dtype == images.dtype
    assert float(augmented.min()) >= 0
    assert float(augmented.max()) <= 1
