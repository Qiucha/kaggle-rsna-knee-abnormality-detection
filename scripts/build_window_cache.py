"""Pre-build the evaluation window cache of a windowed fine-tuning rung.

Validation, LM-test and gold series are not in the local training mirror, so their
first read comes from the slow source mount. Building the cache up front keeps that
cost out of the training run. Safe to rerun: cached studies are skipped.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning import finetune
from contrastive_learning.artifacts import read_json
from contrastive_learning.lm_proxy import load_test_population
from contrastive_learning.lm_supervision import load_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    config = read_json(args.config)
    finetune.validate_config(config)
    groups, *_ = load_population(config, finetune.ROOT)
    test_rows = load_test_population(finetune.ROOT / config["cohort_manifest"],
                                     finetune.ROOT / config["lm_labels_csv"],
                                     finetune.ROOT / config["gold_labels_csv"])[0]
    for name, rows in (("validation", groups["validation"]), ("gold", groups["gold"]), ("lm_test", test_rows)):
        dataset = finetune.WindowDataset(rows, config, training=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
        for done, _ in enumerate(loader, start=1):
            if done % 50 == 0 or done == len(rows):
                print(f"{name}: {done}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
