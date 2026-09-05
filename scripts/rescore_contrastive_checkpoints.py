#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score every saved best contrastive checkpoint.")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/contrastive")
    )
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=Path("checkpoints/contrastive/session-20260902/image-cache-v2"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("digest/contrastive-checkpoint-rescore.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoints = sorted(args.checkpoint_root.rglob("best.pt"))
    completed = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        output = checkpoint.parent / "evaluation-validation-best.json"
        rankings = checkpoint.parent / "rankings-validation-best.jsonl"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_contrastive.py"),
            str(checkpoint),
            "--image-cache-dir",
            str(args.image_cache_dir),
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(f"[{index}/{len(checkpoints)}] {checkpoint}", flush=True)
        if not output.exists() or args.overwrite:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        result = json.loads(output.read_text(encoding="utf-8"))
        completed.append(result)

    summary = {
        "schema_version": 1,
        "metric_tie_policy": "uniform_expected_credit",
        "checkpoint_count": len(completed),
        "checkpoints_with_metric_changes": sum(
            any(abs(delta) > 1e-8 for delta in result["metric_deltas"].values())
            for result in completed
        ),
        "queries_with_tied_candidates": sum(
            result["queries_with_tied_candidates"] for result in completed
        ),
        "results": completed,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
