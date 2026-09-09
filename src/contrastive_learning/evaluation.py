"""Turn a model and a split into retrieval numbers, and a checkpoint into a report.

Two layers live here. The lower one is pure tensor math -- given paired image and
report embeddings, produce recall, ranks, and embedding-health diagnostics -- and
is what training calls once per epoch. The upper one, :func:`evaluate_checkpoint`,
owns the whole offline story for a saved checkpoint: rebuild the exact model the
checkpoint describes, encode a split, and write a self-describing report with
per-query rankings, confidence intervals, and the baseline a random ranker would
achieve, so a number can be judged without re-deriving its context.

Ties are treated explicitly throughout. Retrieval over medical reports produces
genuinely identical similarity scores (duplicated or templated reports), and
ranking them arbitrarily silently inflates recall. Every metric here instead
awards the *expected* credit a uniformly random tie-break would give, and the
report records that policy as ``metric_tie_policy`` so old and new numbers are
never compared under different rules by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_learning.artifacts import sha256_file, write_json, write_jsonl
from contrastive_learning.checkpoints import (
    checkpoint_config,
    has_full_weights,
    load_model_weights,
    read_checkpoint,
)
from contrastive_learning.data import move_batch
from contrastive_learning.inputs import load_cohort, make_loader
from contrastive_learning.model import build_model, symmetric_contrastive_loss


RECALL_CUTOFFS = (1, 5, 10)
METRIC_TIE_POLICY = "uniform_expected_credit"
EVALUATION_SCHEMA_VERSION = 1

# Resampling settings for the reported confidence intervals. The seeds differ per
# metric so the three intervals are not drawn from one identical resample.
_BOOTSTRAP_RESAMPLES = 5000
_BOOTSTRAP_SEEDS = {
    "image_to_text_r@1": 2026,
    "text_to_image_r@1": 2027,
    "mean_r@1": 2028,
}


# --------------------------------------------------------------------------- #
# Retrieval math
# --------------------------------------------------------------------------- #


def _tie_aware_ranks(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per query: strictly better candidates, tied candidates, and expected rank.

    Row ``i`` of ``matrix`` scores query ``i`` against all candidates, with the
    correct answer on the diagonal. ``tied`` counts the positive itself, so it is
    always at least one and safe to divide by. The expected rank places the
    positive at the midpoint of its tied block.
    """
    diagonal = matrix.diagonal().unsqueeze(1)
    greater = (matrix > diagonal).sum(dim=1)
    tied = (matrix == diagonal).sum(dim=1)
    ranks = 1 + greater + (tied - 1).float() / 2
    return greater, tied, ranks


def _expected_recall(greater: torch.Tensor, tied: torch.Tensor, k: int) -> torch.Tensor:
    """Probability the positive lands in the top ``k`` under a random tie-break.

    ``k - greater`` is how many of the top-``k`` slots the tied block can still
    reach; the fraction of the block that fits is the expected credit.
    """
    reachable = (k - greater).clamp(min=0)
    return torch.minimum(reachable, tied) / tied


def _check_paired(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> None:
    if image_embeddings.ndim != 2 or image_embeddings.shape != text_embeddings.shape:
        raise ValueError("Retrieval embeddings must be paired 2D tensors with matching shapes")


def retrieval_metrics(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
) -> dict[str, float]:
    """Recall and rank statistics in both retrieval directions over one split.

    Every study in the split is a candidate for every query, so recall depends on
    split size; :func:`evaluate_checkpoint` records the matching random baseline.
    """
    _check_paired(image_embeddings, text_embeddings)
    if not torch.isfinite(image_embeddings).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("Retrieval embeddings must be finite")
    similarities = image_embeddings.float() @ text_embeddings.float().T

    def direction(matrix: torch.Tensor, prefix: str) -> dict[str, float]:
        greater, tied, ranks = _tie_aware_ranks(matrix)
        out = {
            f"{prefix}_median_rank": float(ranks.float().median()),
            f"{prefix}_mean_rank": float(ranks.float().mean()),
        }
        for k in RECALL_CUTOFFS:
            out[f"{prefix}_r@{k}"] = float(_expected_recall(greater, tied, k).float().mean())
        return out

    metrics = direction(similarities, "image_to_text")
    metrics.update(direction(similarities.T, "text_to_image"))
    metrics["mean_r@1"] = (metrics["image_to_text_r@1"] + metrics["text_to_image_r@1"]) / 2
    return metrics


def retrieval_diagnostics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    study_uids: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return the same metrics plus one row per query, for per-study error review.

    Rows come back as all image-to-text queries followed by all text-to-image
    queries, in split order, so the two directions for one study are found at
    index ``i`` and ``i + len(study_uids)``.
    """
    _check_paired(image_embeddings, text_embeddings)
    if len(study_uids) != len(image_embeddings):
        raise ValueError("Retrieval embeddings and study_uids must describe paired examples")
    similarities = image_embeddings.float() @ text_embeddings.float().T
    rows: list[dict[str, Any]] = []
    for matrix, direction in ((similarities, "image_to_text"), (similarities.T, "text_to_image")):
        greater, tied, ranks = _tie_aware_ranks(matrix)
        # Stable argsort keeps the reported top match deterministic across runs.
        top_indices = torch.argsort(matrix, dim=1, descending=True, stable=True)[:, 0]
        for index, uid in enumerate(study_uids):
            row: dict[str, Any] = {
                "direction": direction,
                "query_uid": uid,
                "positive_uid": uid,
                "top_uid": study_uids[int(top_indices[index])],
                "positive_similarity": float(matrix[index, index]),
                "top_similarity": float(matrix[index, top_indices[index]]),
                "greater_candidates": int(greater[index]),
                "tied_candidates": int(tied[index]),
                "expected_rank": float(ranks[index]),
            }
            for k in RECALL_CUTOFFS:
                row[f"expected_r@{k}"] = float(_expected_recall(greater, tied, k)[index])
            rows.append(row)
    return retrieval_metrics(image_embeddings, text_embeddings), rows


def embedding_health_metrics(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, logit_scale: torch.Tensor
) -> dict[str, float]:
    """Diagnostics that distinguish real alignment from a collapsed embedding space.

    Recall alone cannot tell those apart. Positive similarity rising while negative
    similarity rises with it, or per-dimension standard deviation trending to zero,
    is representation collapse. ``inverse_temperature`` is clamped exactly as
    training clamps it, so the reported validation loss is comparable to train loss.
    """
    similarities = image_embeddings.float() @ text_embeddings.float().T
    diagonal = similarities.diagonal()
    off_diagonal = similarities[
        ~torch.eye(len(similarities), dtype=torch.bool, device=similarities.device)
    ]
    inverse_temperature = logit_scale.detach().to(similarities.device).float().exp().clamp(max=100)
    return {
        "validation_loss": float(symmetric_contrastive_loss(inverse_temperature * similarities)),
        "positive_similarity_mean": float(diagonal.mean()),
        "negative_similarity_mean": float(off_diagonal.mean()) if len(off_diagonal) else 0.0,
        "image_embedding_dimension_std_mean": float(image_embeddings.float().std(dim=0).mean()),
        "text_embedding_dimension_std_mean": float(text_embeddings.float().std(dim=0).mean()),
        "inverse_temperature": float(inverse_temperature),
    }


@torch.inference_mode()
def collect_embeddings(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Encode an entire split in loader order, returning embeddings on the CPU.

    Whole-split encoding is required, not an optimisation: retrieval metrics are
    defined against every candidate in the split, not within a batch.
    """
    model.eval()
    image_parts, text_parts, study_uids = [], [], []
    for batch in loader:
        images, input_ids, attention_mask = move_batch(batch, device)
        image_parts.append(model.encode_images(images).cpu())
        text_parts.append(model.encode_text(input_ids, attention_mask).cpu())
        study_uids.extend(batch["study_uid"])
    if not image_parts:
        raise RuntimeError("Evaluation loader produced no batches")
    return torch.cat(image_parts), torch.cat(text_parts), study_uids


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """One epoch's validation metrics: what checkpoint selection is scored on."""
    images, texts, _ = collect_embeddings(model, loader, device)
    metrics = retrieval_metrics(images, texts)
    metrics.update(embedding_health_metrics(images, texts, model.logit_scale))
    return metrics


# --------------------------------------------------------------------------- #
# Checkpoint reports
# --------------------------------------------------------------------------- #


def _bootstrap_interval(values: list[float], seed: int) -> list[float]:
    """Percentile bootstrap 95% interval for the mean of per-query recall."""
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(_BOOTSTRAP_RESAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    split: str = "validation",
    output_path: Path | None = None,
    rankings_path: Path | None = None,
    image_cache_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Re-score a saved checkpoint and write its report and per-query rankings.

    The checkpoint's own saved configuration defines the model, cohort, and split,
    so a report cannot silently describe a different setup than the one trained.
    Only the image cache location may be redirected, because it holds derived
    pixels rather than anything that changes the result. Backbone weights are
    loaded offline; a compact checkpoint additionally instantiates the pinned
    pretrained backbones to supply the frozen parameters it omits.

    Existing reports are never replaced without ``overwrite``: these files are the
    evidence for published comparisons.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = read_checkpoint(checkpoint_path)
    config = checkpoint_config(checkpoint)
    config.resume = None
    if image_cache_dir is not None:
        config.image_cache_dir = Path(image_cache_dir)
    records = load_cohort(config).split(split)
    if not records:
        raise ValueError(f"No eligible records in split {split!r}")

    model, tokenizer, model_info = build_model(
        config.image_model,
        config.text_model,
        config.embedding_dim,
        config.image_unfreeze_last,
        config.text_unfreeze_last,
        image_model_revision=config.image_model_revision,
        text_model_revision=config.text_model_revision,
        text_pooling=config.text_pooling,
        pretrained_backbones=not has_full_weights(checkpoint),
        local_files_only=True,
    )
    load_model_weights(model, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    images, texts, study_uids = collect_embeddings(
        model, make_loader(records, tokenizer, config), device
    )
    metrics, rankings = retrieval_diagnostics(images, texts, study_uids)
    metrics.update(embedding_health_metrics(images, texts, model.logit_scale))

    count = len(study_uids)
    image_rows, text_rows = rankings[:count], rankings[count:]
    paired_r1 = [
        (image_row["expected_r@1"] + text_row["expected_r@1"]) / 2
        for image_row, text_row in zip(image_rows, text_rows, strict=True)
    ]
    stored_metrics = checkpoint.get("metrics", {})
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric_tie_policy": METRIC_TIE_POLICY,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": split,
        "studies": count,
        "metrics": metrics,
        "stored_metrics": stored_metrics,
        # Nonzero deltas mean the metric definition or inputs moved since training
        # wrote this checkpoint; they are the signal that a rescore was needed.
        "metric_deltas": {
            key: metrics[key] - float(stored_metrics[key])
            for key in metrics.keys() & stored_metrics.keys()
        },
        "queries_with_tied_candidates": sum(row["tied_candidates"] > 1 for row in rankings),
        "random_ranking_baseline": {
            "r@1": 1 / count,
            "r@5": min(5, count) / count,
            "r@10": min(10, count) / count,
            "mean_rank": (count + 1) / 2,
        },
        "bootstrap_95_ci": {
            "image_to_text_r@1": _bootstrap_interval(
                [row["expected_r@1"] for row in image_rows],
                _BOOTSTRAP_SEEDS["image_to_text_r@1"],
            ),
            "text_to_image_r@1": _bootstrap_interval(
                [row["expected_r@1"] for row in text_rows],
                _BOOTSTRAP_SEEDS["text_to_image_r@1"],
            ),
            "mean_r@1": _bootstrap_interval(paired_r1, _BOOTSTRAP_SEEDS["mean_r@1"]),
        },
        "model_info": model_info,
    }

    stem = checkpoint_path.stem
    output_path = output_path or checkpoint_path.parent / f"evaluation-{split}-{stem}.json"
    rankings_path = rankings_path or checkpoint_path.parent / f"rankings-{split}-{stem}.jsonl"
    if not overwrite and (output_path.exists() or rankings_path.exists()):
        raise FileExistsError("Evaluation output already exists; pass overwrite=True to replace it")
    write_json(output_path, result)
    write_jsonl(rankings_path, rankings)
    return result
