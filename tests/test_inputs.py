from dataclasses import replace
from pathlib import Path

import pytest
import torch

from contrastive_learning import data, inputs, training
from contrastive_learning.config import TrainConfig
from contrastive_learning.data import DataSummary, StudyRecord


def records(count=7):
    return [StudyRecord(
        str(i), f"Findings: finding {i}.\nImpression: result {i}.",
        {plane: Path(str(i)) for plane in ("Sagittal", "Coronal", "Axial")},
        None, split=("train" if i % 2 == 0 else "validation"),
    ) for i in range(count)]


def test_fixed_cohort_limit_and_text_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(inputs, "read_cohort_manifest", lambda _: (
        records(), {"text_policy": "multilingual_raw"}))
    config = TrainConfig(cohort_manifest=tmp_path / "cohort.jsonl", max_studies=3)
    with pytest.raises(ValueError, match="multilingual_raw"):
        inputs.load_cohort(config, for_training=True)
    config = replace(config, allow_text_policy_mismatch=True)
    train_cohort = inputs.load_cohort(config, for_training=True)
    saved_cohort = inputs.load_cohort(config)
    assert train_cohort.split("validation") == saved_cohort.split("validation")
    assert [r.study_uid for r in saved_cohort.split("validation")] == ["1"]
    assert train_cohort.summary["selected"] == 3
    assert train_cohort.summary["train"] == 2
    assert saved_cohort.split("test") == []


@pytest.mark.parametrize("fold_count,fold_index", [(None, None), (3, 1)])
def test_legacy_splits_match_training_and_evaluation(monkeypatch, fold_count, fold_index):
    pixel_checks = []

    def index(*args, **kwargs):
        pixel_checks.append(kwargs["validate_pixels"])
        return records(), DataSummary(7, 7, 7, 7, 0, 0)

    monkeypatch.setattr(inputs, "build_study_records", index)
    config = TrainConfig(data_root=Path("/mount"), fold_count=fold_count, fold_index=fold_index)
    train_cohort = inputs.load_cohort(config, for_training=True)
    saved_cohort = inputs.load_cohort(config)
    assert pixel_checks == [True, False]
    for split in ("train", "validation"):
        assert train_cohort.split(split) == saved_cohort.split(split)
    train_ids = {r.study_uid for r in train_cohort.split("train")}
    validation_ids = {r.study_uid for r in train_cohort.split("validation")}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == {str(i) for i in range(7)}
    with pytest.raises(ValueError, match="Legacy checkpoints"):
        saved_cohort.split("test")


def test_prepare_only_does_not_require_a_trainable_split(monkeypatch, tmp_path):
    monkeypatch.setattr(inputs, "build_study_records", lambda *a, **kw: (
        records(1), DataSummary(1, 1, 1, 1, 0, 0)))
    output = tmp_path / "run"
    result = training.run_training(
        TrainConfig(data_root=Path("/mount"), prepare_only=True, output_dir=output)
    )
    assert result["selected"] == 1
    assert not output.exists()


def test_loaders_preserve_pairing_report_view_and_resume_order(monkeypatch):
    monkeypatch.setattr(data, "read_slot", lambda path, **kw: torch.full((3, 2, 2), float(path.name)))
    seen_texts = []

    def tokenizer(texts, **kwargs):
        seen_texts.extend(texts)
        ids = torch.tensor([[int(text.split()[-1].rstrip("."))] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    config = TrainConfig(batch_size=3, num_workers=0, image_size=2,
                         report_view="impression", text_prefix="query: ")
    cohort = records()
    batches = list(inputs.make_loader(cohort, tokenizer, config))
    assert [len(batch["study_uid"]) for batch in batches] == [3, 3, 1]
    assert [uid for batch in batches for uid in batch["study_uid"]] == [str(i) for i in range(7)]
    assert seen_texts == [f"query: result {i}." for i in range(7)]
    for batch in batches:
        assert torch.equal(batch["images"][:, 0, 0, 0, 0], batch["input_ids"][:, 0].float())

    generator = torch.Generator().manual_seed(2026)
    loader = inputs.make_loader(cohort, tokenizer, config, training=True, generator=generator)
    list(loader)
    saved_state = generator.get_state()
    expected = [batch["study_uid"] for batch in loader]
    restored = torch.Generator().set_state(saved_state)
    resumed = inputs.make_loader(cohort, tokenizer, config, training=True, generator=restored)
    assert [batch["study_uid"] for batch in resumed] == expected
    assert sum(map(len, expected)) == 6
