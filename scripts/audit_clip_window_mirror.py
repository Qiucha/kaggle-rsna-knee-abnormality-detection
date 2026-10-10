"""Check every training-series path through the actual K8 CLIP image reader."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning.artifacts import read_json, write_json
from contrastive_learning.config import TrainConfig
from contrastive_learning.inputs import load_cohort
from contrastive_learning.window_reports import WindowReportDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = TrainConfig.from_dict(read_json(args.config))
    records = load_cohort(config, for_training=True).split("train")
    dataset = WindowReportDataset(records, config, training=True)
    failures = []
    for record in records:
        for plane, source in record.slots.items():
            try:
                path = dataset.windows._series(source)
                if not path.is_dir() or not any(path.iterdir()):
                    raise FileNotFoundError(f"Training series is empty or missing: {path}")
            except FileNotFoundError as error:
                failures.append({"study_uid": record.study_uid, "plane": plane, "error": str(error)})
    report = {"training_studies": len(records), "missing_series": len(failures), "failures": failures}
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
