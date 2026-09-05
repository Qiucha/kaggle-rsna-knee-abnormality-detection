"""Validation-only checkpoint selection, shared by training and finalization."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

SELECTION_POLICY = 'mean_r@1 descending, validation_loss ascending, earliest epoch'


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, int]:
    recall = float(metrics['mean_r@1'])
    loss = float(metrics['validation_loss'])
    if not math.isfinite(recall) or not math.isfinite(loss):
        raise ValueError('Checkpoint selection requires finite recall and validation loss')
    return recall, -loss, -int(metrics['epoch'])


def is_better(candidate: dict, incumbent: dict | None) -> bool:
    key = selection_key(candidate)
    return incumbent is None or key > selection_key(incumbent)


def verified_best_epoch(history_path: Path, checkpoint_path: Path) -> dict:
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    if not rows:
        raise ValueError('Checkpoint selection requires nonempty metrics history')
    best = max(rows, key=selection_key)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != best['epoch'] or checkpoint['metrics'] != best:
        raise ValueError('best.pt does not match the selected validation epoch and metrics')
    return best
