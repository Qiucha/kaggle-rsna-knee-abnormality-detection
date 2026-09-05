#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from contrastive_learning.text import select_report_view


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values)
    return {
        "count": len(values),
        "tokens_p50": float(np.percentile(array, 50)),
        "tokens_p90": float(np.percentile(array, 90)),
        "tokens_p99": float(np.percentile(array, 99)),
        "tokens_max": int(array.max()),
        "reports_over_256": int((array > 256).sum()),
        "reports_over_512": int((array > 512).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit multilingual tokenizer coverage without exporting reports.")
    parser.add_argument("--reports-csv", type=Path, default=Path("data/sample/train.csv"))
    parser.add_argument("--language-audit-csv", type=Path, default=Path("data/gold_translated.csv"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", default="query: ")
    parser.add_argument("--output", type=Path, default=Path("digest/multilingual-text-qc.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True, use_fast=True
    )
    audited_languages = {
        row["StudyInstanceUID"].strip(): (row.get("Original Language") or "und").strip().lower()
        for row in _rows(args.language_audit_csv)
    }
    reports = _rows(args.reports_csv)
    lengths: list[int] = []
    unknown_tokens = total_tokens = 0
    by_language: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in reports:
        encoded = tokenizer(args.prefix + (row.get("Report") or ""), add_special_tokens=True)
        input_ids = encoded["input_ids"]
        special = set(tokenizer.all_special_ids)
        content_ids = [token_id for token_id in input_ids if token_id not in special]
        unknown = sum(token_id == tokenizer.unk_token_id for token_id in content_ids)
        length = len(input_ids)
        lengths.append(length)
        unknown_tokens += unknown
        total_tokens += len(content_ids)
        language = audited_languages.get(row["StudyInstanceUID"].strip(), "und")
        if language != "und":
            by_language[language].append((length, unknown, len(content_ids)))

    report_views = {}
    for view in ("impression", "findings_impression"):
        selected = [select_report_view(row.get("Report") or "", view) for row in reports]
        view_lengths = [
            len(tokenizer(args.prefix + report, add_special_tokens=True)["input_ids"])
            for report in selected
        ]
        report_views[view] = {
            **_summarize(view_lengths),
            "reports_with_extracted_sections": sum(
                value.strip() != (row.get("Report") or "").strip()
                for value, row in zip(selected, reports, strict=True)
            ),
        }

    result = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "prefix": args.prefix,
        "reports_sha256": _sha256(args.reports_csv),
        "language_audit_sha256": _sha256(args.language_audit_csv),
        "full_cohort": {
            **_summarize(lengths),
            "unknown_tokens": unknown_tokens,
            "content_tokens": total_tokens,
            "unknown_token_fraction": unknown_tokens / max(total_tokens, 1),
        },
        "audited_languages": {
            language: {
                **_summarize([length for length, _, _ in values]),
                "unknown_tokens": sum(unknown for _, unknown, _ in values),
                "content_tokens": sum(total for _, _, total in values),
                "unknown_token_fraction": sum(unknown for _, unknown, _ in values)
                / max(sum(total for _, _, total in values), 1),
            }
            for language, values in sorted(by_language.items())
        },
        "report_views": report_views,
        "limitations": [
            "Only 58 studies have audited source-language labels.",
            "Tokenizer coverage does not establish medical semantic quality.",
            "The selected multilingual encoder is not specifically pretrained on radiology reports.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
