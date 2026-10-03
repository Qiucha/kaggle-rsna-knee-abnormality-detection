from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from contrastive_learning.config import TrainConfig
from contrastive_learning.checkpoints import initialize_run


def test_new_run_refuses_to_reuse_an_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    config = TrainConfig(output_dir=output_dir)
    initialize_run(config)

    with pytest.raises(FileExistsError, match="already exists"):
        initialize_run(config)


def test_resume_restores_epoch_and_best_score_without_rewriting_history(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    config = TrainConfig(output_dir=output_dir, epochs=2)
    initialized = initialize_run(config)
    checkpoint_path = output_dir / "last.pt"
    torch.save(
        {
            "epoch": 1,
            "metrics": {"epoch": 1, "mean_r@1": 0.25, "validation_loss": 1.0},
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        checkpoint_path,
    )
    (output_dir / "best.pt").write_bytes(checkpoint_path.read_bytes())
    history = '{"epoch": 1, "mean_r@1": 0.25, "validation_loss": 1.0}\n'
    (output_dir / "metrics.jsonl").write_text(history, encoding="utf-8")

    resumed = initialize_run(replace(config, epochs=3, resume=checkpoint_path))

    assert initialized.start_epoch == 1
    assert resumed.start_epoch == 2
    assert resumed.best_score == 0.25
    assert resumed.checkpoint is not None
    assert (output_dir / "metrics.jsonl").read_text(encoding="utf-8") == history
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["resume_events"]) == 1


def _resumable_run(output_dir: Path, saved_config: dict) -> Path:
    """A run directory at epoch 1 whose config.json is ``saved_config``."""
    initialize_run(TrainConfig(output_dir=output_dir, epochs=2))
    (output_dir / "config.json").write_text(json.dumps(saved_config), encoding="utf-8")
    checkpoint_path = output_dir / "last.pt"
    torch.save({"epoch": 1, "metrics": {"epoch": 1, "mean_r@1": 0.25, "validation_loss": 1.0}}, checkpoint_path)
    (output_dir / "best.pt").write_bytes(checkpoint_path.read_bytes())
    (output_dir / "metrics.jsonl").write_text(
        '{"epoch": 1, "mean_r@1": 0.25, "validation_loss": 1.0}\n', encoding="utf-8"
    )
    return checkpoint_path


def test_resume_accepts_a_config_saved_before_newer_settings_existed(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    config = TrainConfig(output_dir=output_dir, epochs=2)
    older = config.as_dict()
    for key in ("use_lora", "lora_rank", "image_drop_path_rate", "use_declip", "label_similarity_weight"):
        older.pop(key)
    checkpoint_path = _resumable_run(output_dir, older)

    resumed = initialize_run(replace(config, epochs=3, resume=checkpoint_path))

    assert resumed.start_epoch == 2


def test_resume_still_refuses_a_new_setting_changed_from_its_default(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    config = TrainConfig(output_dir=output_dir, epochs=2)
    older = config.as_dict()
    older.pop("use_declip")
    checkpoint_path = _resumable_run(output_dir, older)

    with pytest.raises(ValueError, match="use_declip"):
        initialize_run(replace(config, epochs=3, use_declip=True, resume=checkpoint_path))
