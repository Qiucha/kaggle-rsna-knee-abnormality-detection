"""Frozen-feature supervision with missing-label masks and validation-only selection."""

from __future__ import annotations

import copy
from itertools import pairwise

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def masked_bce(logits, targets, mask, observed_rates=None):
    """Macro-average BCE across findings, ignoring unknown cells completely.

    Fixed training observation rates give an unbiased minibatch estimate of the
    full-data macro loss. Validation derives rates from its complete own split.
    """
    if observed_rates is None:
        observed_rates = mask.float().mean(0)
    if (observed_rates <= 0).any():
        raise ValueError("Each label needs observed targets")
    safe_targets = torch.where(mask.bool(), targets, torch.zeros_like(targets))
    return (
        F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
        * mask
        / observed_rates
    ).mean()


def fit_dense_head(
    train_x,
    train_y,
    train_mask,
    validation_x,
    validation_y,
    validation_mask,
    *,
    hidden_dims,
    dropout,
    learning_rate,
    weight_decay,
    batch_size,
    max_epochs,
    patience,
    min_delta,
    seed,
    device="cpu",
):
    """Fit only a dense head. No encoder or gold targets enter this function."""
    if batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    for x, y, mask in (
        (train_x, train_y, train_mask),
        (validation_x, validation_y, validation_mask),
    ):
        if x.ndim != 2 or len(x) == 0 or y.shape != mask.shape or len(x) != len(y):
            raise ValueError("Invalid supervised feature/label shapes")
        if (
            not np.isfinite(x).all()
            or not np.isin(mask, [0, 1]).all()
            or not np.isin(y[mask.astype(bool)], [0, 1]).all()
        ):
            raise ValueError(
                "Expected finite features and binary observed labels/masks"
            )
        for label in range(y.shape[1]):
            if set(np.unique(y[mask[:, label].astype(bool), label])) != {0, 1}:
                raise ValueError(f"Both classes must be observed for label {label}")
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    # Population variance matches StandardScaler, fitted only on head-training rows.
    mean = train_x.astype(np.float64).mean(0).astype(np.float32)
    scale = train_x.astype(np.float64).std(0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    x = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    mask = torch.as_tensor(train_mask, dtype=torch.float32, device=device)
    vx = torch.as_tensor(
        (validation_x - mean) / scale, dtype=torch.float32, device=device
    )
    vy = torch.as_tensor(validation_y, dtype=torch.float32, device=device)
    vm = torch.as_tensor(validation_mask, dtype=torch.float32, device=device)
    widths = [train_x.shape[1], *hidden_dims, train_y.shape[1]]
    layers = []
    for i, (a, b) in enumerate(pairwise(widths)):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    observed_rates = mask.mean(0)
    history, best_state = [], None
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        indices = torch.randperm(len(x), generator=rng).to(device)
        for start in range(0, len(x), batch_size):
            index = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(x[index]), y[index], mask[index], observed_rates)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite head training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_loss = float(masked_bce(model(x), y, mask))
            validation_loss = float(masked_bce(model(vx), vy, vm))
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite head validation loss")
        history.append(
            {
                "epoch": epoch,
                "train_masked_bce": train_loss,
                "validation_masked_bce": validation_loss,
            }
        )
        # Save the absolute minimum, while patience requires a material improvement.
        if validation_loss < best_loss:
            meaningful = validation_loss < best_loss - min_delta
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0 if meaningful else stale + 1
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    head = {
        "kind": "dense_v1",
        "mean": torch.from_numpy(mean),
        "scale": torch.from_numpy(scale),
        "layers": [
            {"weight": layer.weight.detach().cpu(), "bias": layer.bias.detach().cpu()}
            for layer in model
            if isinstance(layer, nn.Linear)
        ],
    }
    return head, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_masked_bce": best_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
    }
