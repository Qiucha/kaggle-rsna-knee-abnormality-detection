"""Behaviour of the LM-label similarity target and the loss that consumes it."""

from pathlib import Path

import pytest
import torch

from contrastive_learning import data
from contrastive_learning.config import TrainConfig
from contrastive_learning.labels import LabelTable, read_lm_labels
from contrastive_learning.model import (
    label_similarity_targets,
    soft_target_contrastive_loss,
    symmetric_contrastive_loss,
)


FINDINGS = ("ACL", "Effusion", "Fracture")


def write_labels(path: Path, rows: dict[str, list[str]]) -> Path:
    header = ["StudyInstanceUID"]
    for finding in FINDINGS:
        header += [finding, f"{finding}__conf", f"{finding}__verdict"]
    lines = [",".join(header)]
    for uid, verdicts in rows.items():
        cells = [uid]
        for verdict in verdicts:
            cells += ["0.5", "0.9", verdict]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return path


def table(rows: dict[str, list[str]], tmp_path: Path) -> LabelTable:
    return read_lm_labels(write_labels(tmp_path / "labels.csv", rows))


def test_verdicts_become_signed_vectors_and_unk_is_unobserved(tmp_path):
    loaded = table({"a": ["YES", "NO", "UNK"], "b": ["no", "", "YES"]}, tmp_path)
    assert loaded.findings == FINDINGS
    values, mask = loaded.get("a")
    assert values.tolist() == [1.0, -1.0, 0.0]
    assert mask.tolist() == [1.0, 1.0, 0.0]
    # Verdicts are matched case-insensitively; a blank is unobserved, not a NO.
    values, mask = loaded.get("b")
    assert values.tolist() == [-1.0, 0.0, 1.0]
    assert mask.tolist() == [1.0, 0.0, 1.0]


def test_unknown_study_is_unobserved_rather_than_an_error(tmp_path):
    loaded = table({"a": ["YES", "YES", "YES"]}, tmp_path)
    values, mask = loaded.get("absent")
    assert values.tolist() == [0.0, 0.0, 0.0] and mask.tolist() == [0.0, 0.0, 0.0]
    assert loaded.coverage(["a", "absent"]) == {
        "studies": 2,
        "labelled_studies": 1,
        "findings": 3,
        "mean_observed_findings": 3.0,
    }


def test_duplicate_and_unlabelled_csv_are_rejected(tmp_path):
    path = tmp_path / "dupe.csv"
    path.write_text(
        "StudyInstanceUID,ACL,ACL__conf,ACL__verdict\na,0.5,0.9,YES\na,0.5,0.9,NO\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        read_lm_labels(path)
    path.write_text("StudyInstanceUID,ACL\na,0.5\n")
    with pytest.raises(ValueError, match="verdict"):
        read_lm_labels(path)


def sample(batch: int = 6, findings: int = 5, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    mask = (torch.rand(batch, findings, generator=generator) > 0.3).float()
    signs = torch.where(torch.rand(batch, findings, generator=generator) > 0.5, 1.0, -1.0)
    return signs * mask, mask


def test_zero_weight_reproduces_the_unmodified_loss():
    values, mask = sample()
    targets = label_similarity_targets(values, mask, 0.0)
    assert torch.equal(targets, torch.eye(len(values)))
    logits = torch.randn(len(values), len(values), generator=torch.Generator().manual_seed(1)) * 4
    assert soft_target_contrastive_loss(logits, targets) == pytest.approx(
        float(symmetric_contrastive_loss(logits)), abs=1e-6
    )


@pytest.mark.parametrize("weight", [0.1, 0.3, 0.5, 0.9])
def test_targets_are_distributions_with_a_dominant_diagonal(weight):
    values, mask = sample()
    targets = label_similarity_targets(values, mask, weight)
    assert torch.allclose(targets.sum(dim=1), torch.ones(len(values)))
    assert bool((targets >= 0).all())
    # The true pair must stay strictly the largest target, or the loss would be
    # taught to rank some other study's report above the study's own.
    assert bool((targets.argmax(dim=1) == torch.arange(len(values))).all())


def test_similarity_ignores_findings_that_are_not_co_observed():
    # Rows 0 and 1 agree on every finding they share, disagreeing only where one
    # of them is unobserved; rows 0 and 2 disagree on both shared findings.
    values = torch.tensor([[1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    targets = label_similarity_targets(values, mask, 1.0)
    assert targets[0, 1] > 0.0
    assert targets[0, 2] == pytest.approx(0.0)


def test_rows_without_any_co_observation_fall_back_to_one_hot():
    values, mask = sample()
    mask[2] = 0.0
    values[2] = 0.0
    targets = label_similarity_targets(values, mask, 0.8)
    assert torch.allclose(targets[2], torch.eye(len(values))[2])


def test_unmasked_input_cannot_push_the_target_negative():
    _, mask = sample()
    rogue = torch.where(torch.rand(mask.shape, generator=torch.Generator().manual_seed(3)) > 0.5, 1.0, -1.0)
    targets = label_similarity_targets(rogue, mask, 0.5)
    assert bool((targets >= 0).all())
    assert torch.allclose(targets.sum(dim=1), torch.ones(len(mask)))


def test_weight_outside_the_unit_interval_is_rejected():
    values, mask = sample()
    with pytest.raises(ValueError, match="label_similarity_weight"):
        label_similarity_targets(values, mask, 1.5)


def test_config_rejects_a_weight_without_a_label_source():
    with pytest.raises(ValueError, match="requires --lm-label-csv"):
        TrainConfig(label_similarity_weight=0.3).validate_training()
    with pytest.raises(ValueError, match="label_similarity_weight"):
        TrainConfig(label_similarity_weight=-0.1).validate_training()
    # The defaults must stay a plain contrastive run.
    TrainConfig().validate_training()


def test_collate_carries_label_tensors_only_when_a_table_is_supplied(tmp_path):
    loaded = table({"a": ["YES", "NO", "UNK"], "b": ["NO", "NO", "YES"]}, tmp_path)
    batch = [
        {"images": torch.zeros(3, 3, 4, 4), "report": "one", "study_uid": "a"},
        {"images": torch.zeros(3, 3, 4, 4), "report": "two", "study_uid": "b"},
    ]
    labelled = [{**item, **dict(zip(("label_values", "label_mask"), loaded.get(item["study_uid"])))}
                for item in batch]

    class Tokenizer:
        def __call__(self, texts, **_):
            count = len(texts)
            return {"input_ids": torch.ones(count, 2, dtype=torch.long),
                    "attention_mask": torch.ones(count, 2, dtype=torch.long)}

    plain = data.collate_image_reports(batch, tokenizer=Tokenizer(), max_text_length=8)
    assert "label_values" not in plain and "label_mask" not in plain

    collated = data.collate_image_reports(labelled, tokenizer=Tokenizer(), max_text_length=8)
    assert collated["label_values"].shape == (2, 3)
    assert collated["label_mask"].shape == (2, 3)
    assert collated["label_values"][0].tolist() == [1.0, -1.0, 0.0]


def test_findings_can_be_restricted_to_a_subset(tmp_path):
    rows = {"a": ["YES", "NO", "UNK"], "b": ["NO", "NO", "YES"]}
    path = write_labels(tmp_path / "labels.csv", rows)
    loaded = read_lm_labels(path, ["Fracture", "ACL"])
    # Order follows the request, not the file.
    assert loaded.findings == ("Fracture", "ACL")
    values, mask = loaded.get("a")
    assert values.tolist() == [0.0, 1.0] and mask.tolist() == [0.0, 1.0]
    values, mask = loaded.get("b")
    assert values.tolist() == [1.0, -1.0] and mask.tolist() == [1.0, 1.0]
    assert loaded.coverage(["a", "b"])["findings"] == 2


def test_restriction_rejects_unknown_or_duplicated_findings(tmp_path):
    path = write_labels(tmp_path / "labels.csv", {"a": ["YES", "NO", "UNK"]})
    with pytest.raises(ValueError, match="not present"):
        read_lm_labels(path, ["ACL", "Nonexistent"])
    with pytest.raises(ValueError, match="Duplicate findings"):
        read_lm_labels(path, ["ACL", "ACL"])


def test_config_validates_the_finding_restriction():
    base = dict(lm_label_csv=Path("labels.csv"), label_similarity_weight=0.3)
    TrainConfig(**base, label_findings=["ACL"]).validate_training()
    with pytest.raises(ValueError, match="empty list"):
        TrainConfig(**base, label_findings=[]).validate_training()
    with pytest.raises(ValueError, match="duplicates"):
        TrainConfig(**base, label_findings=["ACL", "ACL"]).validate_training()
    with pytest.raises(ValueError, match="requires --lm-label-csv"):
        TrainConfig(label_findings=["ACL"]).validate_training()


def test_restricting_findings_makes_the_target_coarser(tmp_path):
    # Two studies identical on ACL but differing on Fracture are distinguishable
    # under both findings, and indistinguishable under ACL alone. That collapse is
    # the documented hazard of a short finding list.
    path = write_labels(
        tmp_path / "labels.csv",
        {"a": ["YES", "NO", "YES"], "b": ["YES", "NO", "NO"]},
    )
    both = read_lm_labels(path, ["ACL", "Fracture"])
    one = read_lm_labels(path, ["ACL"])
    pair = lambda table: label_similarity_targets(
        torch.stack([table.get("a")[0], table.get("b")[0]]),
        torch.stack([table.get("a")[1], table.get("b")[1]]),
        1.0,
    )[0, 1]
    assert pair(both) < pair(one)
    assert pair(one) == pytest.approx(0.5)  # a perfect tie: the diagonal splits evenly
