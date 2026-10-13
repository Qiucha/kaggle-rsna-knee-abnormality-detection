"""Finalize the original CLIP pass rule once the postponed seed run has scored."""

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning.artifacts import read_json, write_json
from contrastive_learning.clip_pipeline import ROOT, compare_to_run5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    fine = read_json(args.config)
    directory = ROOT / fine["pipeline_status_dir"]
    while True:
        status = read_json(directory / "status.json")
        stage = status.get("stage")
        if stage in ("failed", "completed"):
            return
        if stage == "completed_pending_seed" and (ROOT / fine["comparison_seed"] / "report.json").exists():
            break
        time.sleep(60)
    try:
        comparison = compare_to_run5(fine, ROOT / "checkpoints/finetune" / fine["name"], directory)
        if comparison["seed_comparison_pending"]:
            raise RuntimeError("Seed report exists but its predictions are missing")
        status.update(stage="completed", passed=comparison["passed"], seed_comparison_pending=False,
                      updated_at=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                      seed_comparison_finalized_at=time.strftime("%Y-%m-%d %H:%M:%S %z"))
        write_json(directory / "status.json", status)
        print("Postponed seed comparison finalized", flush=True)
    except Exception:
        (directory / "seed-comparison-error.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
