from __future__ import annotations

import torch
import pytest

from contrastive_learning.data import (
    StudyRecord,
    collate_image_reports,
    read_reports,
    split_records_fold,
)
from contrastive_learning.model import symmetric_contrastive_loss
from contrastive_learning.evaluation import retrieval_diagnostics, retrieval_metrics


def test_symmetric_contrastive_loss_rewards_matching_pairs() -> None:
    matched = torch.eye(4) * 10
    ambiguous = torch.zeros(4, 4)
    assert symmetric_contrastive_loss(matched) < symmetric_contrastive_loss(ambiguous)


def test_retrieval_metrics_are_perfect_for_aligned_embeddings() -> None:
    embeddings = torch.eye(4)
    metrics = retrieval_metrics(embeddings, embeddings)
    assert metrics["image_to_text_r@1"] == 1.0
    assert metrics["text_to_image_r@1"] == 1.0
    assert metrics["image_to_text_median_rank"] == 1.0
    assert metrics["mean_r@1"] == 1.0


def test_retrieval_metrics_give_expected_credit_to_tied_candidates() -> None:
    embeddings = torch.ones(4, 3)
    metrics = retrieval_metrics(embeddings, embeddings)

    assert metrics["image_to_text_r@1"] == 0.25
    assert metrics["text_to_image_r@1"] == 0.25
    assert metrics["image_to_text_mean_rank"] == 2.5
    assert metrics["text_to_image_mean_rank"] == 2.5


def test_retrieval_diagnostics_preserve_query_ids_and_tie_counts() -> None:
    embeddings = torch.ones(3, 2)

    metrics, rows = retrieval_diagnostics(embeddings, embeddings, ["a", "b", "c"])

    assert metrics["mean_r@1"] == pytest.approx(1 / 3)
    assert len(rows) == 6
    assert rows[0]["query_uid"] == "a"
    assert rows[0]["positive_uid"] == "a"
    assert rows[0]["tied_candidates"] == 3
    assert rows[0]["expected_rank"] == 2.0


def test_contrastive_loss_rejects_unpaired_logits() -> None:
    try:
        symmetric_contrastive_loss(torch.zeros(2, 3))
    except ValueError as error:
        assert "square paired logits" in str(error)
    else:
        raise AssertionError("Expected non-square logits to fail")


def test_read_reports_accepts_configurable_text_column(tmp_path) -> None:
    path = tmp_path / "translated.csv"
    path.write_text(
        "StudyInstanceUID,Report,Translated Report\n1,original,translated text\n",
        encoding="utf-8",
    )
    assert read_reports(path, "Translated Report") == {"1": "translated text"}


def test_study_level_folds_are_disjoint_and_cover_every_study() -> None:
    records = [StudyRecord(str(index), "report", {}, None) for index in range(11)]
    validation_uids = []
    for fold_index in range(5):
        train, validation = split_records_fold(records, 5, fold_index, seed=2026)
        train_uids = {record.study_uid for record in train}
        fold_uids = {record.study_uid for record in validation}
        assert train_uids.isdisjoint(fold_uids)
        validation_uids.extend(fold_uids)
    assert sorted(validation_uids) == sorted(record.study_uid for record in records)


def test_collator_applies_the_configured_text_prefix() -> None:
    captured: list[str] = []

    def tokenizer(texts, **_kwargs):
        captured.extend(texts)
        return {
            "input_ids": torch.ones(len(texts), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
        }

    collate_image_reports(
        [{"images": torch.zeros(3, 3, 2, 2), "report": "knee report", "study_uid": "1"}],
        tokenizer=tokenizer,
        max_text_length=32,
        text_prefix="query: ",
    )

    assert captured == ["query: knee report"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_validation_health_accepts_cpu_embeddings_and_cuda_temperature() -> None:
    from contrastive_learning.evaluation import embedding_health_metrics
    metrics = embedding_health_metrics(torch.eye(3), torch.eye(3), torch.tensor(1.0, device='cuda'))
    assert metrics['validation_loss'] >= 0
