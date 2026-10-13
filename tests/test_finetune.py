import numpy as np
import pandas as pd
import pytest
import torch

from contrastive_learning.finetune import FinetuneModel, soft_targets, unk_supervision, validate_config
from contrastive_learning.submission import LABELS, _adapt, image_features, predict_head


def tiny_backbone():
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
        image_size=28, patch_size=14,
    )
    return Dinov2Model(config)


def test_exported_member_predicts_what_the_trained_model_predicts():
    torch.manual_seed(0)
    model = FinetuneModel(tiny_backbone(), [16], dropout=0.2).eval()
    _adapt(model.backbone, {"use_lora": False, "image_unfreeze_last": 2})
    images = torch.rand(2, 3, 3, 28, 28)
    with torch.no_grad():
        trained = model(images).sigmoid().numpy()
    deployed = predict_head(image_features(model.backbone, images).numpy(), model.export_head())
    np.testing.assert_allclose(deployed, trained, atol=1e-5)


def test_only_the_last_blocks_and_head_train():
    from contrastive_learning.model import freeze_then_unfreeze_last

    backbone = tiny_backbone()
    freeze_then_unfreeze_last(backbone, 1)
    model = FinetuneModel(backbone, [16], dropout=0.0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("backbone.encoder.layer.3.") for n in trainable)
    assert not any(n.startswith("backbone.encoder.layer.2.") for n in trainable)
    assert any(n.startswith("head.") for n in trainable)


def test_soft_targets_keep_unk_masked(tmp_path):
    uids = ["a", "b"]
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for label in LABELS:
        frame[label] = [0.94, 0.28]
    frame.to_csv(tmp_path / "lm.csv", index=False)
    masks = np.zeros((2, len(LABELS)), np.float32)
    masks[0] = 1
    values = soft_targets(tmp_path / "lm.csv", uids, masks)
    assert np.allclose(values[0], 0.94)
    assert np.allclose(values[1], 0.0)


def test_unk_cells_train_toward_the_csv_value_at_their_own_weight(tmp_path):
    uids = ["a", "b"]
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for label in LABELS:
        frame[label] = [0.94, 0.28]
    frame.to_csv(tmp_path / "lm.csv", index=False)
    masks = np.zeros((2, len(LABELS)), np.float32)
    masks[0] = 1
    targets = np.full((2, len(LABELS)), 0.5, np.float32)
    values, weights = unk_supervision(tmp_path / "lm.csv", uids, masks, targets, 0.2)
    assert np.allclose(values[0], 0.5) and np.allclose(weights[0], 1.0)
    assert np.allclose(values[1], 0.28) and np.allclose(weights[1], 0.2)


@pytest.mark.parametrize("change", [{"targets": "probability"}, {"image_unfreeze_last": 0},
                                    {"slices_per_plane": 5}, {"epochs": 0}, {"unk_weight": 1.5}])
def test_invalid_rung_configs_are_rejected(change):
    config = {"targets": "verdict", "image_unfreeze_last": 3, "slices_per_plane": 3,
              "epochs": 20, "batch_size": 16, "patience": 5, **change}
    with pytest.raises(ValueError):
        validate_config(config)


def test_jittered_slices_stay_in_their_strata_and_replay_under_a_seed():
    from contrastive_learning.preprocessing import _slice_indices, jittered_slice_indices

    draws = [jittered_slice_indices(40, (0.2, 0.8), 3, np.random.default_rng(s)) for s in range(200)]
    for indices in draws:
        assert len(indices) == 3 and list(indices) == sorted(indices)
        assert 7 <= indices[0] < 15 <= indices[1] < 23 <= indices[2] <= 31
    assert len({tuple(d) for d in draws}) > 50
    assert (jittered_slice_indices(40, (0.2, 0.8), 3, np.random.default_rng(1))
            == jittered_slice_indices(40, (0.2, 0.8), 3, np.random.default_rng(1))).all()
    # Too few slices to stratify: the fixed sampler's answer.
    assert (jittered_slice_indices(4, (0.2, 0.8), 3, np.random.default_rng(0))
            == _slice_indices(4, (0.2, 0.8), 3)).all()


def test_only_the_training_dataset_samples_slices_at_random(tmp_path):
    from contrastive_learning.data import KneeReportDataset
    from contrastive_learning.finetune import RandomSliceDataset, _dataset

    config = {"image_size": 28, "crop_mm": 130.0, "slices_per_plane": 3, "slice_band": [0.2, 0.8],
              "image_cache_dir": str(tmp_path),
              "train_slice_sampling": "random"}
    assert isinstance(_dataset([], config, training=True), RandomSliceDataset)
    assert isinstance(_dataset([], config, training=False), KneeReportDataset)
    config["train_slice_sampling"] = "fixed"
    assert isinstance(_dataset([], config, training=True), KneeReportDataset)


def test_unknown_slice_sampling_is_rejected():
    with pytest.raises(ValueError):
        validate_config({"targets": "verdict", "image_unfreeze_last": 3, "slices_per_plane": 3,
                         "epochs": 20, "batch_size": 16, "patience": 5,
                         "train_slice_sampling": "shuffled"})


def test_random_slices_read_from_the_local_mirror(tmp_path):
    from pathlib import Path

    from contrastive_learning.finetune import RandomSliceDataset

    (tmp_path / "local" / "study" / "series").mkdir(parents=True)
    # Wherever the source tree lives, the mirror is found by <study>/<series>.
    dataset = RandomSliceDataset([], {}, {"to": str(tmp_path / "local")})
    assert dataset._series(Path("/any/train_series/study/series")) == tmp_path / "local/study/series"
    with pytest.raises(FileNotFoundError):
        dataset._series(Path("/any/train_series/study/other"))


def test_the_mirror_comes_from_the_host_unless_the_rung_names_one(tmp_path):
    import json

    from contrastive_learning.finetune import with_host_mirror

    host = tmp_path / "paths.json"
    host.write_text(json.dumps({"train_series_mirror": "mirror"}))
    resolved = with_host_mirror({"seed": 1}, host)
    assert resolved["train_series_mirror"] == {"to": str(tmp_path / "mirror")}
    stated = {"train_series_mirror": {"to": "/elsewhere"}}
    assert with_host_mirror(stated, host) == stated
    host.write_text("{}")
    assert "train_series_mirror" not in with_host_mirror({"seed": 1}, host)


def test_eval_windows_sit_inside_the_strata_training_draws_from():
    from contrastive_learning.preprocessing import window_indices

    for n in range(1, 120):
        centre, valid = window_indices(n, (0.2, 0.8), 8)
        draws = [window_indices(n, (0.2, 0.8), 8, np.random.default_rng(s)) for s in range(30)]
        for indices, drawn_valid in draws:
            assert (drawn_valid == valid).all()
            # Each window is 3 contiguous slices (clipped only at the series ends).
            assert (indices >= 0).all() and (indices < n).all()
        lows = np.min([d[0][:, 1] for d in draws], axis=0)
        highs = np.max([d[0][:, 1] for d in draws], axis=0)
        assert ((lows <= centre[:, 1]) & (centre[:, 1] <= highs))[valid].all() or n < 20
        assert list(centre[valid, 1]) == sorted(set(centre[valid, 1]))


def test_short_stacks_mask_windows_instead_of_repeating_them():
    from contrastive_learning.preprocessing import window_indices

    _, valid = window_indices(60, (0.2, 0.8), 8)
    assert valid.all()
    indices, valid = window_indices(8, (0.2, 0.8), 8)
    assert 0 < valid.sum() < 8
    assert len(set(indices[valid, 1])) == valid.sum()


def test_sagittal_right_knees_reverse_window_depth_and_mask_together():
    from contrastive_learning.preprocessing import apply_left_knee_windows

    images = torch.arange(4 * 3, dtype=torch.float32).view(4, 3, 1, 1)
    valid = torch.tensor([True, True, True, False])
    flipped, flipped_valid = apply_left_knee_windows(images, valid, "Sagittal", "R")
    assert flipped.flatten().tolist() == list(range(11, -1, -1))
    assert flipped_valid.tolist() == [False, True, True, True]


@pytest.mark.parametrize("pooling", ["mean", "attention", "grouped"])
def test_masked_windows_do_not_change_predictions(pooling):
    from contrastive_learning.finetune import WindowModel

    torch.manual_seed(0)
    model = WindowModel(tiny_backbone(), [16], dropout=0.0, pooling=pooling,
                        window_groups=2 if pooling == "grouped" else None).eval()
    images = torch.rand(2, 3, 4, 3, 28, 28)
    valid = torch.ones(2, 3, 4, dtype=torch.bool)
    valid[0, 1, 2:] = False
    valid[1, 2] = False  # a whole missing plane
    noisy = images.clone()
    noisy[~valid] = torch.rand_like(noisy[~valid])
    with torch.no_grad():
        logits, attention = model(images, valid, return_attention=True)
        assert torch.allclose(logits, model(noisy, valid), atol=1e-5)
    assert logits.shape == (2, len(LABELS)) and torch.isfinite(logits).all()
    if pooling == "attention":
        assert attention.shape == (2, len(LABELS), 12)
        assert torch.allclose(attention.sum(-1), torch.ones(2, len(LABELS)))
        assert (attention[~valid.flatten(1)[:, None].expand_as(attention)] == 0).all()


def test_mean_pooling_on_one_window_per_plane_matches_r1c():
    from contrastive_learning.finetune import WindowModel

    torch.manual_seed(0)
    backbone = tiny_backbone().eval()
    windowed = WindowModel(backbone, [16], dropout=0.0, pooling="mean").eval()
    plain = FinetuneModel(backbone, [16], dropout=0.0).eval()
    plain.head.load_state_dict(windowed.head.state_dict())
    images = torch.rand(2, 3, 3, 28, 28)
    with torch.no_grad():
        assert torch.allclose(windowed(images[:, :, None], torch.ones(2, 3, 1, dtype=torch.bool)),
                              plain(images), atol=1e-5)


@pytest.mark.parametrize("change", [{"pooling": "mean"}, {"windows_per_plane": 8},
                                    {"windows_per_plane": 8, "pooling": "max", "window_cache_dir": "x"},
                                    {"grad_accumulation": 0},
                                    {"windows_per_plane": 8, "pooling": "grouped", "window_cache_dir": "x"},
                                    {"windows_per_plane": 8, "pooling": "grouped", "window_groups": 3,
                                     "window_cache_dir": "x"},
                                    {"windows_per_plane": 8, "pooling": "mean", "window_groups": 2,
                                     "window_cache_dir": "x"}])
def test_invalid_window_configs_are_rejected(change):
    config = {"targets": "verdict", "image_unfreeze_last": 3, "slices_per_plane": 3,
              "epochs": 20, "batch_size": 16, "patience": 5, **change}
    with pytest.raises(ValueError):
        validate_config(config)


def test_masked_window_counts():
    from contrastive_learning.finetune import masked_window_counts

    valid = np.ones((2, 3, 4), bool)
    valid[0, 0, 3] = False
    valid[1, 2] = False
    assert masked_window_counts(valid) == {"studies": 2, "windows": 24, "masked_windows": 5,
                                           "masked_planes": 1, "short_planes": 1}


@pytest.mark.parametrize("damage", [b"", b"\x80\x02truncated"])
def test_a_corrupt_window_cache_entry_is_rebuilt(tmp_path, monkeypatch, damage):
    from types import SimpleNamespace

    from contrastive_learning import finetune

    config = {"image_size": 28, "crop_mm": 130.0, "slices_per_plane": 3, "slice_band": [0.2, 0.8],
              "windows_per_plane": 2, "window_cache_dir": str(tmp_path / "cache")}
    record = SimpleNamespace(study_uid="s", laterality="L", slots={p: tmp_path / p for p in finetune.PLANES})
    dataset = finetune.WindowDataset([record], config, training=False)
    reads = []
    monkeypatch.setattr(dataset, "_read", lambda r, rng: reads.append(1) or (
        torch.ones(3, 2, 3, 28, 28), torch.ones(3, 2, dtype=torch.bool)))
    first = dataset[0]
    (entry,) = (tmp_path / "cache").iterdir()
    entry.write_bytes(damage)
    second = dataset[0]
    assert len(reads) == 2
    assert torch.equal(second["images"], first["images"]) and entry.stat().st_size > len(damage)


def test_only_training_loaders_keep_their_workers_alive(tmp_path):
    from types import SimpleNamespace

    from contrastive_learning.finetune import _loader

    config = {"image_size": 28, "crop_mm": 130.0, "slices_per_plane": 3, "slice_band": [0.2, 0.8],
              "windows_per_plane": 2, "window_cache_dir": str(tmp_path),
              "batch_size": 2, "num_workers": 2}
    records = [SimpleNamespace(study_uid="s", laterality="L", slots={})]
    assert _loader(records, config, training=True).persistent_workers
    assert not _loader(records, config, training=False).persistent_workers


@pytest.mark.parametrize("groups", [None, 2])
def test_exported_window_member_predicts_what_the_trained_model_predicts(groups):
    from contrastive_learning.finetune import WindowModel, export_dense_head
    from contrastive_learning.submission import window_features

    torch.manual_seed(0)
    model = WindowModel(tiny_backbone(), [16], dropout=0.2, pooling="grouped" if groups else "mean",
                        window_groups=groups).eval()
    images = torch.rand(2, 3, 4, 3, 28, 28)
    valid = torch.ones(2, 3, 4, dtype=torch.bool)
    valid[0, 1, 2:] = False
    valid[1, 2] = False  # a whole missing plane
    with torch.no_grad():
        trained = model(images, valid).sigmoid().numpy()
    deployed = predict_head(window_features(model.backbone, images, valid, groups).numpy(),
                            export_dense_head(model.head))
    np.testing.assert_allclose(deployed, trained, atol=1e-5)


def test_grouped_pooling_keeps_slice_runs_and_planes_apart():
    from contrastive_learning.submission import pool_windows

    tokens = torch.arange(3 * 4, dtype=torch.float32).reshape(1, 3, 4, 1)  # window value = its index
    valid = torch.ones(1, 3, 4, dtype=torch.bool)
    valid[0, 0, 1] = False  # masked window leaves its run's mean to the valid one
    valid[0, 2, 2:] = False  # a fully masked run pools to zero
    pooled = pool_windows(tokens, valid, groups=2)
    assert pooled.tolist() == [[0.0, 2.5, 4.5, 6.5, 8.5, 0.0]]
    assert pool_windows(tokens, valid).shape == (1, 1)


def test_oversampling_repeats_target_positives_and_masks_other_labels_on_repeats():
    from contrastive_learning.finetune import oversampled_rows

    lateral, acl = LABELS.index("Lateral Meniscus"), LABELS.index("ACL")
    hard = np.zeros((4, len(LABELS)), np.float32)
    masks = np.ones((4, len(LABELS)), np.float32)
    hard[1, lateral] = 1
    hard[2, lateral] = 1
    masks[2, lateral] = 0  # a positive verdict that is not observed does not count
    hard[3, acl] = 1
    rows, keep = oversampled_rows(hard, masks, {"labels": ["Lateral Meniscus"], "copies": 2})
    assert rows.tolist() == [0, 1, 2, 3, 1, 1]
    assert (keep[:4] == 1).all()
    assert (keep[4:, lateral] == 1).all() and keep[4:].sum() == 2
    _, kept = oversampled_rows(hard, masks, {"labels": ["Lateral Meniscus"], "copies": 1,
                                             "mask_other_labels": False})
    assert (kept == 1).all()


def test_label_weights_keep_the_mean_weight_at_one():
    from contrastive_learning.finetune import label_weight_vector

    vector = label_weight_vector({"Lateral OA": 2.0, "Synovitis": 2.0})
    assert vector.mean() == pytest.approx(1.0)
    assert vector[LABELS.index("Lateral OA")] == pytest.approx(2 * vector[LABELS.index("ACL")])


def test_label_noise_touches_only_its_columns_and_stays_in_range():
    from contrastive_learning.finetune import noisy_targets

    targets = torch.full((500, len(LABELS)), 0.08)
    columns = [LABELS.index("Lateral Meniscus"), LABELS.index("Lateral OA")]
    generator = torch.Generator().manual_seed(0)
    noisy = noisy_targets(targets, columns, 0.1, generator)
    others = [i for i in range(len(LABELS)) if i not in columns]
    assert torch.equal(noisy[:, others], targets[:, others])
    assert (noisy[:, columns] >= 0).all() and (noisy[:, columns] <= 1).all()
    assert noisy[:, columns].std() > 0.05
    assert torch.equal(targets, torch.full((500, len(LABELS)), 0.08))  # input untouched
    again = noisy_targets(targets, columns, 0.1, generator)
    assert not torch.equal(again, noisy)  # fresh noise on every draw


@pytest.mark.parametrize("change", [
    {"oversample": {"labels": ["Lateral Menisci"], "copies": 1}},
    {"oversample": {"labels": ["Lateral OA"], "copies": 0}},
    {"label_noise": {"labels": ["Lateral OA"], "sigma": 0.0}},
    {"label_weights": {"Synovitis": -1.0}},
    {"divergence_stop": {"margin": 0.0, "epochs": 3}},
    {"divergence_stop": {"margin": 0.02, "epochs": 0}},
    {"resume_from": {"run": "checkpoints/finetune/x", "lr_scale": 0.0}},
])
def test_invalid_rebalancing_configs_are_rejected(change):
    config = {"targets": "soft", "image_unfreeze_last": 3, "slices_per_plane": 3,
              "epochs": 20, "batch_size": 16, "patience": 5, **change}
    with pytest.raises(ValueError):
        validate_config(config)


def test_training_rows_read_repeats_and_report_their_position():
    from contrastive_learning.finetune import TrainingRows

    base = [{"study_uid": u} for u in "abc"]
    rows = TrainingRows(base, [0, 1, 2, 1])
    assert len(rows) == 4
    assert rows[3] == {"study_uid": "b", "row": 3}


@pytest.mark.parametrize("spec", [True, False, None, [], {"flip": True},
                                  {"zoom": -0.1}, {"shift": float("nan")},
                                  {"intensity": float("inf")}, {"rotation_degrees": 181},
                                  {"zoom": True}, {"intensity": "0.1"}])
def test_invalid_image_augmentation_is_rejected_before_training(spec):
    with pytest.raises(ValueError, match="image_augmentation"):
        validate_config({"image_augmentation": spec})


def test_disabled_augmentation_keeps_inputs_and_rng_unchanged():
    from contrastive_learning.finetune import augment_training_inputs, image_augmentation_settings

    inputs = (torch.rand(2, 3, 3, 28, 28),)
    generator = torch.Generator().manual_seed(2028)
    state = generator.get_state().clone()
    assert image_augmentation_settings({}) is None
    assert augment_training_inputs(inputs, None, generator) is inputs
    assert torch.equal(generator.get_state(), state)


def test_window_augmentation_is_coherent_masked_and_reproducible():
    from contrastive_learning.finetune import augment_training_inputs, image_augmentation_settings

    # Identical slices/windows expose accidental independent transforms along depth.
    ramp = torch.linspace(0.0, 0.8, 28 * 28).reshape(1, 1, 1, 1, 28, 28)
    images = ramp.expand(2, 3, 8, 3, 28, 28).clone()
    valid = torch.ones(2, 3, 8, dtype=torch.bool)
    valid[0, 1, 4:] = False
    valid[1, 2] = False
    before = images.clone()
    settings = image_augmentation_settings({"image_augmentation": {}})
    torch_state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(2028)
    augmented, mask = augment_training_inputs((images, valid), settings, generator)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert mask is valid and torch.equal(images, before)
    assert augmented.shape == images.shape and augmented.dtype == images.dtype
    assert augmented.min() >= 0 and augmented.max() <= 1
    assert torch.count_nonzero(augmented[~valid]) == 0
    torch.testing.assert_close(augmented[0, 0], augmented[0, 0, :1].expand_as(augmented[0, 0]))
    torch.testing.assert_close(augmented[0, 0, 0, 0], augmented[0, 0, 0, 1])
    assert not torch.equal(augmented[0, 0, 0], augmented[0, 2, 0])
    replay, _ = augment_training_inputs((images, valid), settings, torch.Generator().manual_seed(2028))
    torch.testing.assert_close(replay, augmented, rtol=0, atol=0)
    fresh, _ = augment_training_inputs((images, valid), settings, generator)
    assert not torch.equal(fresh, augmented)


def test_three_slice_augmentation_supports_nonwindowed_finetuning():
    from contrastive_learning.finetune import augment_training_inputs, image_augmentation_settings

    ramp = torch.linspace(0, 0.8, 28 * 28).reshape(1, 1, 1, 28, 28)
    images = ramp.expand(2, 3, 3, 28, 28).clone()
    (augmented,) = augment_training_inputs((images,), image_augmentation_settings({"image_augmentation": {}}),
                                          torch.Generator().manual_seed(2028))
    assert augmented.shape == images.shape and not torch.equal(images, augmented)
    torch.testing.assert_close(augmented[:, :, 0], augmented[:, :, 1])


def test_augmented_oversampling_run_transforms_only_training_and_records_the_recipe(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from contrastive_learning import finetune

    config = json.loads((Path(__file__).resolve().parents[1] / "configs/finetune" /
                         "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30.json").read_text())
    config["image_augmentation"] = {}  # an empty object turns augmentation on at its defaults
    config.update(epochs=1, patience=1, image_size=28, windows_per_plane=2, amp=False)
    monkeypatch.setattr(finetune, "ROOT", tmp_path)
    lm_csv = tmp_path / config["lm_labels_csv"]
    lm_csv.parent.mkdir(parents=True)
    lm_csv.write_text("synthetic labels\n")
    groups = {split: [SimpleNamespace(study_uid=f"{split}-{i}") for i in range(2)]
              for split in ("train", "validation", "gold")}
    hard = np.array([[0] * len(LABELS), [1] * len(LABELS)], np.float32)
    targets = {split: hard.copy() for split in groups}
    masks = {split: np.ones_like(hard) for split in groups}
    monkeypatch.setattr(finetune, "load_population", lambda *args: (groups, targets, masks, {"coverage": {}}))
    monkeypatch.setattr(finetune, "soft_targets", lambda *args: hard * 0.86 + 0.08)
    test_records = [SimpleNamespace(study_uid=f"test-{i}") for i in range(2)]
    monkeypatch.setattr(finetune, "load_test_population", lambda *args: (
        test_records, hard, np.ones_like(hard),
        {"study_uids": [r.study_uid for r in test_records], "manifest_sha256": "test-manifest"}))
    monkeypatch.setattr(finetune, "load_backbone", lambda cfg: (tiny_backbone(), {}))
    batches, forwards = {}, []

    def loader(records, cfg, *, training, generator=None, rows=None):
        count = len(rows) if rows is not None else len(records)
        images = torch.linspace(0, 0.8, count * 3 * 2 * 3 * 28 * 28).reshape(count, 3, 2, 3, 28, 28)
        valid = torch.ones(count, 3, 2, dtype=torch.bool)
        valid[:, 1, 1] = False
        images[~valid] = 0
        batches[training] = (images.clone(), valid.clone())
        return [((images, valid), list(range(count)) if training else [r.study_uid for r in records])]

    class RecordingModel(finetune.WindowModel):
        def forward(self, images, valid, **kwargs):
            forwards.append((self.training, images.detach().clone(), valid.detach().clone()))
            return super().forward(images, valid, **kwargs)

    monkeypatch.setattr(finetune, "_loader", loader)
    monkeypatch.setattr(finetune, "WindowModel", RecordingModel)
    report = finetune.run_finetune(config, tmp_path / "run", device="cpu")
    (training, transformed, mask), *evaluations = forwards
    assert training and not torch.equal(transformed, batches[True][0])
    assert torch.equal(mask, batches[True][1]) and torch.count_nonzero(transformed[~mask]) == 0
    assert evaluations
    for training, images, valid in evaluations:
        assert not training
        torch.testing.assert_close(images, batches[False][0], rtol=0, atol=0)
        assert torch.equal(valid, batches[False][1])
    saved = json.loads((tmp_path / "run/config.json").read_text())
    assert saved["image_augmentation"] == finetune.image_augmentation_settings(config)
    assert saved["label_noise"] == config["label_noise"] and saved["oversample"] == config["oversample"]
    assert report["image_augmentation"]["shared_across_windows"] is True
    assert report["oversample"]["extra_draws"] == 1
    last = torch.load(tmp_path / "run/last.pt", weights_only=False)
    assert "augmentation_rng_state" in last
