#!/usr/bin/env python3
"""Render the four-arm 20-epoch training and validation curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "digest/tower-size-factorial-curves.svg"
RUNS = [
    (
        "DINOv2-small + E5-base",
        "checkpoints/contrastive/both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026",
        "#3366a0",
        "-",
    ),
    (
        "DINOv2-small + E5-small",
        "checkpoints/contrastive/tower-size-factorial-v1/dinov2-small-e5-small-depth6-r64-seed2026",
        "#3366a0",
        "--",
    ),
    (
        "DINOv2-base + E5-base",
        "checkpoints/contrastive/tower-size-factorial-v1/dinov2-base-e5-base-depth6-r64-seed2026",
        "#d06431",
        "-",
    ),
    (
        "DINOv2-base + E5-small",
        "checkpoints/contrastive/size-matched-holdout-v1/dinov2-base-e5-small-depth6-r64-seed2026",
        "#d06431",
        "--",
    ),
]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
    fields = [
        ("train_loss", "Training loss", 1.0),
        ("validation_loss", "Validation loss", 1.0),
        ("mean_r@1", "Validation mean Recall@1 (%)", 100.0),
    ]
    for label, directory, color, style in RUNS:
        rows = [
            json.loads(line)
            for line in (ROOT / directory / "metrics.jsonl").read_text().splitlines()
        ]
        if [row["epoch"] for row in rows] != list(range(1, 21)):
            raise ValueError(f"Incomplete history for {label}")
        for axis, (field, title, scale) in zip(axes, fields):
            axis.plot(
                [row["epoch"] for row in rows],
                [row[field] * scale for row in rows],
                label=label,
                color=color,
                linestyle=style,
                linewidth=2.2,
            )
            axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
            axis.set_xlim(1, 20)
            axis.set_xticks([1, 5, 10, 15, 20])
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Tower-size factorial · same cohort and 20-epoch schedule",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
