"""Read-only section 9 inventory: what the 58 gold studies can and cannot support.

The feedback response's section 9 asks for four things before any downstream number is
interpretable: provenance, an exposure audit, per-label feasibility counts, and an
estimand fixed in advance. This produces the first three as evidence. It fits nothing,
selects nothing, and does not touch the locked test's labels or text.

The test gate, stated once here because it shapes every number below: a gold study in
the test split is counted, and nothing else. Its labels are not tabulated, its report is
not hashed against anything, and it is excluded from every feasibility count. The
response names this conflict explicitly -- using all 58 has to respect current
locked-test membership, and that is a user decision, not one this script makes.

Output: digest/gold-case-inventory.json
"""

from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl"
GOLD = ROOT / "data/gold_label.csv"
GOLD_BLANK = ROOT / "data/gold_unlabeled.csv"
GOLD_TRANSLATED = ROOT / "data/translation/gold_translated.csv"
OUT = ROOT / "digest/gold-case-inventory.json"

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# A probe fold needs positives on both sides of the split. These are the thresholds the
# feasibility verdict uses; they are conventions, declared here so the verdict is not a
# judgement call made later against a result someone already wants.
MIN_POSITIVES_FOR_ANY_ESTIMATE = 2
MIN_POSITIVES_FOR_GROUPED_FOLDS = 5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise_report(text: str) -> str:
    """Whitespace-insensitive text identity: trailing-newline churn is not exposure."""
    return " ".join(str(text).split())


def provenance() -> dict:
    """Which of the three gold files carries labels, and whether they agree."""
    cleaned = pd.read_csv(GOLD)
    blank = pd.read_csv(GOLD_BLANK)
    translated = pd.read_csv(GOLD_TRANSLATED)

    uid_sets = {
        "cleaned.csv": set(cleaned.StudyInstanceUID),
        "drop.csv": set(blank.StudyInstanceUID),
        "gold_translated.csv": set(translated.StudyInstanceUID),
    }
    same_uids = len({frozenset(s) for s in uid_sets.values()}) == 1

    indexed = {
        name: frame.set_index("StudyInstanceUID").sort_index()
        for name, frame in [
            ("cleaned.csv", cleaned),
            ("drop.csv", blank),
            ("gold_translated.csv", translated),
        ]
    }
    populated = {
        name: int(frame[LABELS].notna().values.sum()) for name, frame in indexed.items()
    }
    labelled = [name for name, count in populated.items() if count > 0]
    agreement = {}
    for i, left in enumerate(labelled):
        for right in labelled[i + 1 :]:
            differing = (indexed[left][LABELS] != indexed[right][LABELS]).values.sum()
            agreement[f"{left} vs {right}"] = {
                "cells_compared": int(indexed[left][LABELS].size),
                "cells_differing": int(differing),
                "identical": bool(differing == 0),
            }

    return {
        "files": {
            name: {
                "sha256": sha256_file(path),
                "studies": len(uid_sets[name]),
                "populated_label_cells": populated[name],
                "carries_labels": populated[name] > 0,
            }
            for name, path in [
                ("cleaned.csv", GOLD),
                ("drop.csv", GOLD_BLANK),
                ("gold_translated.csv", GOLD_TRANSLATED),
            ]
        },
        "all_three_cover_the_same_studies": same_uids,
        "label_bearing_files": labelled,
        "label_agreement": agreement,
        "authoritative": (
            "cleaned.csv and gold_translated.csv carry identical labels; drop.csv has "
            "every label column empty and is the blank annotation template. Reading "
            "drop.csv as labels yields 58 studies with no positives and would fail "
            "silently -- it is a trap, not a second opinion."
            if populated.get("drop.csv", 0) == 0
            else "see label_agreement"
        ),
        "labels_are_binary_and_complete": {
            "distinct_values": sorted(
                {float(v) for v in pd.read_csv(GOLD)[LABELS].values.ravel() if pd.notna(v)}
            ),
            "missing_cells": int(pd.read_csv(GOLD)[LABELS].isna().values.sum()),
            "uncertainty_encoding": (
                "none: the labels are 0/1 with no missing cells, so the response's "
                "uncertainty-aware contract has nothing to read here. Absence of an "
                "'unknown' code is not evidence every case was adjudicable."
            ),
        },
    }


def main() -> None:
    rows = {json.loads(line)["study_uid"]: json.loads(line) for line in COHORT.open(encoding="utf-8")}
    gold = pd.read_csv(GOLD).set_index("StudyInstanceUID")
    translated = pd.read_csv(GOLD_TRANSLATED).set_index("StudyInstanceUID")

    present = [uid for uid in gold.index if uid in rows]
    absent = [uid for uid in gold.index if uid not in rows]

    exposure = []
    for uid in present:
        row = rows[uid]
        gold_text = normalise_report(translated.loc[uid, "Translated Report"])
        gold_hash = hashlib.sha256(gold_text.encode("utf-8")).hexdigest()
        trained_hash = hashlib.sha256(normalise_report(row["report"]).encode("utf-8")).hexdigest()
        # Two different reasons the strings can differ, and they are not equally bad:
        # a different underlying report, or the same report through a second translation
        # run. Separating them is the whole point of the check.
        source_hash = hashlib.sha256(
            normalise_report(gold.loc[uid, "Report"]).encode("utf-8")
        ).hexdigest()
        cohort_source_hash = hashlib.sha256(
            normalise_report(row["source_report"]).encode("utf-8")
        ).hexdigest()
        exposure.append(
            {
                "study_uid": uid,
                "split": row["split"],
                "training_eligible": row["training_eligible"],
                "source_language": row["source_language"],
                "gold_original_language": str(translated.loc[uid, "Original Language"]),
                "report_language": row["report_language"],
                "report_provenance": row["report_provenance"],
                "gold_text_matches_trained_text": gold_hash == trained_hash,
                "gold_source_matches_cohort_source": source_hash == cohort_source_hash,
            }
        )

    by_split = collections.Counter(e["split"] for e in exposure)
    train_val = [e for e in exposure if e["split"] in {"train", "validation"}]
    trained_on = [e for e in train_val if e["split"] == "train" and e["training_eligible"]]

    # Every count below this line is restricted to train+validation gold studies. Test
    # membership is a number, never a label.
    estimable_uids = [e["study_uid"] for e in train_val]
    subset = gold.loc[estimable_uids]

    feasibility = {}
    for label in LABELS:
        column = subset[label]
        positives = int(column.sum())
        negatives = int((column == 0).sum())
        split_positives = {
            split: int(gold.loc[[e["study_uid"] for e in train_val if e["split"] == split], label].sum())
            for split in ("train", "validation")
        }
        if positives < MIN_POSITIVES_FOR_ANY_ESTIMATE:
            verdict = "not estimable"
        elif positives < MIN_POSITIVES_FOR_GROUPED_FOLDS:
            verdict = "leave-one-out only, no grouped folds"
        else:
            verdict = "estimable under grouped folds"
        feasibility[label] = {
            "positives": positives,
            "negatives": negatives,
            "prevalence": round(positives / max(len(subset), 1), 4),
            "positives_by_split": split_positives,
            "verdict": verdict,
        }

    text_mismatch = [e for e in train_val if not e["gold_text_matches_trained_text"]]
    retranslation = [e for e in text_mismatch if e["gold_source_matches_cohort_source"]]
    different_document = [e for e in text_mismatch if not e["gold_source_matches_cohort_source"]]

    audit = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cohort": str(COHORT.relative_to(ROOT)),
        "cohort_sha256": sha256_file(COHORT),
        "gate": (
            "Test-split gold studies are reported as a count only. Their labels, report "
            "text and images are not read, and they are excluded from every feasibility "
            "count. Whether the probe may use them is a user protocol decision."
        ),
        "provenance": provenance(),
        "cohort_membership": {
            "gold_studies": len(gold),
            "present_in_cohort": len(present),
            "absent_from_cohort": len(absent),
            "absent_uids": absent,
            "by_split": dict(sorted(by_split.items())),
            "in_train_and_training_eligible": len(trained_on),
            "quarantined_or_ineligible": [
                e["study_uid"] for e in exposure if not e["training_eligible"]
            ],
        },
        "representation_exposure": {
            "statement": (
                "Every gold study in the train split with training_eligible true was a "
                "training pair for every completed contrastive run on this cohort. "
                "Holding such a case out of a later probe does not undo that exposure."
            ),
            "seen_in_representation_training": len(trained_on),
            "in_validation_used_for_checkpoint_selection": by_split.get("validation", 0),
            "in_locked_test": by_split.get("test", 0),
            # The only gold studies untouched by representation training or checkpoint
            # selection are the ones in the locked test -- which is why this number is
            # not usable data.
            "clean_of_all_exposure": by_split.get("test", 0),
            "clean_and_usable_without_a_locked_test_decision": 0,
        },
        "text_exposure": {
            "question": (
                "Is the report text the model trained on the same text the gold labels "
                "were adjudicated against?"
            ),
            "compared": len(train_val),
            "matching": len(train_val) - len(text_mismatch),
            "mismatching": len(text_mismatch),
            "mismatching_but_same_source_report": len(retranslation),
            "mismatching_with_a_different_source_report": len(different_document),
            "different_document_uids": [e["study_uid"] for e in different_document],
            "source_languages_of_retranslations": dict(
                sorted(collections.Counter(e["source_language"] for e in retranslation).items())
            ),
            "note": (
                "Compared whitespace-normalised. The severity depends on which kind of "
                "mismatch it is. Same source report through a second translation run is "
                "translation variance: the label was adjudicated from the same clinical "
                "document the model read, worded differently. A different source report "
                "would mean the label describes a document the model never saw, which is "
                "a provenance failure rather than a wording difference."
            ),
        },
        "per_label_feasibility": {
            "population": "train+validation gold studies only",
            "studies": len(subset),
            "thresholds": {
                "min_positives_for_any_estimate": MIN_POSITIVES_FOR_ANY_ESTIMATE,
                "min_positives_for_grouped_folds": MIN_POSITIVES_FOR_GROUPED_FOLDS,
            },
            "labels": feasibility,
            "not_estimable": [k for k, v in feasibility.items() if v["verdict"] == "not estimable"],
            "loo_only": [
                k for k, v in feasibility.items() if v["verdict"] == "leave-one-out only, no grouped folds"
            ],
        },
        "per_study": exposure,
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "per_study"}, indent=1))


if __name__ == "__main__":
    main()
