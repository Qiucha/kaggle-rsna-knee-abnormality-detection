"""Pair fine-tuned rungs with R0 on LM-test and gold-58, one study bootstrap per metric.

R0's LM-test predictions are rebuilt from its bundle through the lm_proxy feature
cache, without writing into R0's run directory.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning.artifacts import write_json  # noqa: E402
from contrastive_learning.lm_proxy import (  # noqa: E402
    DEFAULTS,
    encoder_test_features,
    masked_macro_auroc,
    paired_intervals,
    run_checkpoints,
)
from contrastive_learning.submission import predict_head  # noqa: E402


def r0_predictions(run_dir, device, batch_size):
    members, uids, lm_test = run_checkpoints(run_dir), None, []
    for checkpoint, member_path, sha in members:
        features, targets, masks, population = encoder_test_features(
            checkpoint, sha, device=device, batch_size=batch_size,
            lm_labels_csv=DEFAULTS["lm_labels_csv"], gold_labels_csv=DEFAULTS["gold_labels_csv"],
            cache_dir=DEFAULTS["cache_dir"], image_cache_dir=DEFAULTS["image_cache_dir"],
        )
        head = torch.load(member_path, map_location="cpu", weights_only=False)["head"]
        lm_test.append(predict_head(features, head))
        uids = population["study_uids"]
    with np.load(run_dir / "gold-predictions.npz", allow_pickle=False) as gold:
        return {
            "lm_test_uids": list(uids), "lm_test": np.mean(lm_test, axis=0),
            "lm_test_targets": targets, "lm_test_masks": masks,
            "gold_uids": gold["study_uids"].tolist(), "gold": gold["mlp"], "gold_targets": gold["targets"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0", type=Path,
                        default=Path("checkpoints/kaggle/lm-mlp-tower-size-58-dinov2-small-e5-base"))
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    reference = r0_predictions(args.r0, args.device, args.batch_size)
    lm_test, gold = {"R0": reference["lm_test"]}, {"R0": reference["gold"]}
    for run in args.runs:
        name = json.loads((run / "config.json").read_text())["name"]
        with np.load(run / "predictions.npz", allow_pickle=False) as saved:
            if saved["lm_test_uids"].tolist() != reference["lm_test_uids"]:
                raise ValueError(f"{name}: LM-test population differs from R0")
            if saved["gold_uids"].tolist() != reference["gold_uids"]:
                raise ValueError(f"{name}: gold population differs from R0")
            lm_test[name], gold[name] = saved["lm_test"], saved["gold"]

    gold_masks = np.ones_like(reference["gold_targets"])
    report = {
        "lm_test": {
            "macro_auroc": {n: masked_macro_auroc(p, reference["lm_test_targets"], reference["lm_test_masks"])["macro_auroc"]
                            for n, p in lm_test.items()},
            "paired": paired_intervals(lm_test, reference["lm_test_targets"], reference["lm_test_masks"]),
        },
        "gold": {
            "macro_auroc": {n: masked_macro_auroc(p, reference["gold_targets"], gold_masks)["macro_auroc"]
                            for n, p in gold.items()},
            "paired": paired_intervals(gold, reference["gold_targets"], gold_masks),
        },
        "bar": "LM-test CI excludes zero and gold-58 difference >= -0.024 (ticket 03)",
    }
    write_json(args.output, report)
    print(json.dumps({k: v["macro_auroc"] for k, v in report.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
