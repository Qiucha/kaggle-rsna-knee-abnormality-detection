from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from contrastive_learning.config import TrainConfig
from contrastive_learning.run_state import initialize_run


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
