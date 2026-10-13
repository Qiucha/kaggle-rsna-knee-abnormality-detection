import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from contrastive_learning.config import TrainConfig
from contrastive_learning.data import StudyRecord
from contrastive_learning.finetune import WindowDataset, WindowModel
from contrastive_learning.gradient_cache import backward_cached_clip, encode_chunk
from contrastive_learning.inputs import make_loader
from contrastive_learning.model import ImageTower, symmetric_contrastive_loss
from contrastive_learning.window_reports import WindowReportDataset


class PixelBackbone(nn.Module):
    config = SimpleNamespace(hidden_size=3)

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))
        self.calls = []

    def forward(self, pixel_values):
        self.calls.append(len(pixel_values))
        features = pixel_values.mean((-1, -2)) * self.scale
        return SimpleNamespace(last_hidden_state=torch.stack((features, features + 1, features + 2), 1))


def test_window_features_and_masked_pooling_match_run5_with_missing_windows_and_planes():
    backbone = PixelBackbone()
    tower = ImageTower(backbone, 8, plane_pooling="uniform", windows_per_plane=8)
    run5 = WindowModel(backbone, [8], 0.0, "mean")
    images = torch.rand(2, 3, 8, 3, 2, 2, requires_grad=True)
    valid = torch.ones(2, 3, 8, dtype=torch.bool)
    valid[0, 1, 2:] = False
    valid[1, 0] = False
    pooled, expected = [], []
    tower.projection.register_forward_pre_hook(lambda _, args: pooled.append(args[0]))
    run5.head.register_forward_pre_hook(lambda _, args: expected.append(args[0]))
    result = tower(images, valid)
    run5(images, valid)
    torch.testing.assert_close(pooled[0], expected[0], rtol=0, atol=0)
    assert backbone.calls == [int(valid.sum()), int(valid.sum())]
    result[:, 0].sum().backward()
    assert torch.all(images.grad[valid].abs().sum((1, 2, 3)) > 0)
    assert torch.count_nonzero(images.grad[~valid]) == 0


def test_window_tower_rejects_study_with_no_readable_plane():
    tower = ImageTower(PixelBackbone(), 8, plane_pooling="uniform", windows_per_plane=8)
    with pytest.raises(ValueError, match="no readable"):
        tower(torch.rand(1, 3, 8, 3, 2, 2), torch.zeros(1, 3, 8, dtype=torch.bool))


def window_config(tmp_path):
    return TrainConfig(windows_per_plane=8, window_cache_dir=tmp_path / "cache",
                       image_plane_pooling="uniform", contrastive_microbatch_size=1,
                       image_size=28, batch_size=3, num_workers=0)


def records():
    return [StudyRecord(str(i), f"report {i}", {plane:Path(str(i)) for plane in
                                             ("Sagittal", "Coronal", "Axial")}, None) for i in range(7)]


def mock_window_read(path, *, windows, image_size, slice_rng=None, **kwargs):
    jitter = int(slice_rng.integers(0, 4)) if slice_rng is not None else 0
    images = torch.full((windows, 3, image_size, image_size), (int(path.name) + jitter) / 8)
    valid = torch.ones(windows, dtype=torch.bool)
    valid[-1] = False
    return images, valid


@pytest.mark.parametrize("training", [False, True])
def test_report_dataset_reuses_run5_window_sampling_cache_and_mask(tmp_path, monkeypatch, training):
    monkeypatch.setattr("contrastive_learning.finetune.read_windows", mock_window_read)
    config = window_config(tmp_path)
    settings = {**config.image_kwargs, "windows_per_plane":8,"window_cache_dir":str(config.window_cache_dir)}
    settings.pop("windows")
    run5 = WindowDataset(records(), settings, training=training)
    clip = WindowReportDataset(records(), config, training=training)
    torch.manual_seed(2026)
    expected = run5[2]
    torch.manual_seed(2026)
    actual = clip[2]
    torch.testing.assert_close(actual["images"], expected["images"], rtol=0, atol=0)
    assert torch.equal(actual["valid"], expected["valid"])
    assert actual["report"] == "report 2"
    if training:
        assert clip.windows.cache_dir is None and run5.cache_dir is None
    else:
        assert clip.windows.cache_path(records()[2]) == run5.cache_path(records()[2])


def test_window_loader_keeps_report_pairing_validity_and_partial_eval_batch(tmp_path, monkeypatch):
    monkeypatch.setattr("contrastive_learning.finetune.read_windows", mock_window_read)

    def tokenizer(texts, **kwargs):
        ids = torch.tensor([[int(text.split()[-1])] for text in texts])
        return {"input_ids":ids,"attention_mask":torch.ones_like(ids)}

    batches = list(make_loader(records(), tokenizer, window_config(tmp_path)))
    assert [len(batch["study_uid"]) for batch in batches] == [3, 3, 1]
    for batch in batches:
        assert batch["images"].shape[1:] == (3, 8, 3, 28, 28)
        assert batch["valid"].shape[1:] == (3, 8)
        torch.testing.assert_close(batch["images"][:, 0, 0, 0, 0, 0], batch["input_ids"][:, 0].float() / 8)


class TinyClip(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(3, 8), nn.Dropout(dropout), nn.Linear(8, 8))
        self.text = nn.Sequential(nn.Linear(3, 8), nn.Dropout(dropout), nn.Linear(8, 8))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_images(self, images, valid):
        pixels = images.mean((-2, -1))
        pooled = (pixels * valid[..., None]).sum((1, 2)) / valid.sum((1, 2))[:, None]
        return nn.functional.normalize(self.image(pooled), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        return nn.functional.normalize(self.text(input_ids.float()), dim=-1)


@pytest.mark.parametrize("chunk", [1, 3])
@pytest.mark.parametrize("dropout", [0.0, 0.35])
@pytest.mark.parametrize("scaled", [False, True])
def test_gradient_replay_matches_loss_every_parameter_gradient_and_rng(chunk, dropout, scaled):
    torch.manual_seed(72)
    direct = TinyClip(dropout).train()
    cached = copy.deepcopy(direct)
    batch = {"images":torch.rand(7,3,8,3,2,2),"valid":torch.ones(7,3,8,dtype=torch.bool),
             "input_ids":torch.rand(7,3),"attention_mask":torch.ones(7,3)}
    batch["valid"][0, 1, 4:] = False
    config = TrainConfig(batch_size=7, contrastive_microbatch_size=chunk, amp=False)
    device = torch.device("cpu")
    scaler = torch.amp.GradScaler("cpu", enabled=scaled, init_scale=128)
    rng = torch.get_rng_state()
    embeddings = {}
    from contrastive_learning.preprocessing import augment_study_images

    for tower in ("image", "text"):
        parts = [encode_chunk(direct, batch, start, min(start + chunk, 7), device, tower=tower,
                              augment=augment_study_images if tower=="image" else None)
                 for start in range(0,7,chunk)]
        embeddings[tower] = torch.cat(parts)
    expected_loss = symmetric_contrastive_loss(direct.logit_scale.exp() * embeddings["image"] @ embeddings["text"].T)
    scaler.scale(expected_loss).backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual_loss = backward_cached_clip(cached, batch, device, scaler, config)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)
    for (name, expected), (_, actual) in zip(direct.named_parameters(), cached.named_parameters(), strict=True):
        assert actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-5)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual_loss > 0  # one 7×7 loss, despite physical chunks of one study


def test_cached_training_steps_optimizer_and_scheduler_once_per_comparison_batch():
    from contrastive_learning.training import _optimizer, _scheduler, _train_epoch

    model = TinyClip(0.1)
    batch = {"images":torch.rand(32,3,8,3,2,2),"valid":torch.ones(32,3,8,dtype=torch.bool),
             "input_ids":torch.rand(32,3),"attention_mask":torch.ones(32,3)}
    config = TrainConfig(batch_size=32, contrastive_microbatch_size=1, amp=False)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, 2, 0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    metrics = _train_epoch(model, [batch,batch], torch.device("cpu"), optimizer, scheduler, scaler, config)
    assert scheduler.last_epoch == 2
    assert {int(state["step"]) for state in optimizer.state.values()} == {2}
    assert metrics["train_loss"] > 0


def test_windowed_evaluation_encodes_every_study_with_bounded_chunks():
    from contrastive_learning.evaluation import collect_embeddings

    model = TinyClip(0.0).eval()
    batch = {"images":torch.rand(5,3,8,3,2,2),"valid":torch.ones(5,3,8,dtype=torch.bool),
             "input_ids":torch.rand(5,3),"attention_mask":torch.ones(5,3),"study_uid":list("abcde")}
    images, texts, uids = collect_embeddings(model, [batch], torch.device("cpu"))
    torch.testing.assert_close(images, model.encode_images(batch["images"], batch["valid"]))
    torch.testing.assert_close(texts, model.encode_text(batch["input_ids"], batch["attention_mask"]))
    assert uids == list("abcde")
