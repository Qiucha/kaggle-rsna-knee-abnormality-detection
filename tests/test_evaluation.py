import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from contrastive_learning.config import TrainConfig
from contrastive_learning import evaluation


@pytest.mark.parametrize("compact", [False, True])
def test_checkpoint_evaluation_preserves_reports(tmp_path, monkeypatch, compact):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

        def encode_images(self, images):
            return images

        def encode_text(self, input_ids, attention_mask):
            return input_ids.float()

    model = TinyModel()
    config = TrainConfig(output_dir=tmp_path, batch_size=2, num_workers=0)
    checkpoint_path = tmp_path / "best.pt"
    state_key = "trainable_state_dict" if compact else "model_state_dict"
    torch.save({
        "config": config.as_dict(), "epoch": 2,
        "metrics": {"mean_r@1": 1.0}, state_key: model.state_dict(),
    }, checkpoint_path)
    model_options = []

    def build_model(*args, **kwargs):
        model_options.append(kwargs)
        return TinyModel(), None, {"image_revision": "test"}

    monkeypatch.setattr(evaluation, "build_model", build_model)
    monkeypatch.setattr(evaluation, "load_cohort", lambda *args: SimpleNamespace(split=lambda _: ["a", "b"]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Multiple batches ensure validation and offline evaluation preserve pairing/order.
    batches = [{
        "images": torch.eye(2)[i:i+1], "input_ids": torch.eye(2)[i:i+1],
        "attention_mask": torch.ones(1, 2), "study_uid": [uid],
    } for i, uid in enumerate(["a", "b"])]
    monkeypatch.setattr(evaluation, "make_loader", lambda *args, **kwargs: batches)

    result = evaluation.evaluate_checkpoint(checkpoint_path)
    assert model_options[0]["pretrained_backbones"] is compact
    assert result["metrics"] == evaluation.evaluate(model, batches, torch.device("cpu"))
    assert result["metrics"]["mean_r@1"] == 1.0
    assert result["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert result["checkpoint_epoch"] == 2
    assert result["metric_deltas"] == {"mean_r@1": 0.0}
    assert result["queries_with_tied_candidates"] == 0
    assert result["random_ranking_baseline"]["r@1"] == 0.5
    assert all(interval == [1.0, 1.0] for interval in result["bootstrap_95_ci"].values())
    output = tmp_path / "evaluation-validation-best.json"
    rankings = tmp_path / "rankings-validation-best.jsonl"
    assert json.loads(output.read_text()) == result
    rows = [json.loads(line) for line in rankings.read_text().splitlines()]
    assert [(row["query_uid"], row["direction"]) for row in rows] == [
        ("a", "image_to_text"), ("b", "image_to_text"),
        ("a", "text_to_image"), ("b", "text_to_image"),
    ]
    assert all(row["query_uid"] == row["top_uid"] for row in rows)
    with pytest.raises(FileExistsError, match="already exists"):
        evaluation.evaluate_checkpoint(checkpoint_path)


def test_checkpoint_evaluation_rebuilds_lora_adapters(tmp_path, monkeypatch):
    # A compact LoRA checkpoint stores only adapter weights; rebuilt without
    # adapters, strict=False loading would silently drop them.
    config = TrainConfig(output_dir=tmp_path, use_lora=True, lora_rank=8, lora_alpha=16.0, lora_dropout=0.1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"config": config.as_dict(), "epoch": 1, "metrics": {}, "trainable_state_dict": {}}, checkpoint_path)
    model_options = []

    def build_model(*args, **kwargs):
        model_options.append(kwargs)
        raise RuntimeError("stop after build_model")

    monkeypatch.setattr(evaluation, "build_model", build_model)
    monkeypatch.setattr(evaluation, "load_cohort", lambda *args: SimpleNamespace(split=lambda _: ["a"]))
    with pytest.raises(RuntimeError, match="stop after build_model"):
        evaluation.evaluate_checkpoint(checkpoint_path)
    assert {key: model_options[0][key] for key in ("use_lora", "lora_rank", "lora_alpha", "lora_dropout")} == {
        "use_lora": True, "lora_rank": 8, "lora_alpha": 16.0, "lora_dropout": 0.1,
    }
