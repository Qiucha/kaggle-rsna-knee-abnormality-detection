"""Run offline inference; works in the checkout or beside an exported bundle.

Reads ``--bundle`` (the exported ``bundle/`` directory) and the competition data under
``--data-root``; writes the submission CSV ``--output``. This file is copied verbatim
into every Kaggle bundle and runs offline there with only ``src/`` beside it, so it
imports nothing but ``contrastive_learning`` and the standard library.

    python predict_kaggle.py --bundle <bundle> --data-root <data> --output submission.csv
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(
    0, str(HERE / "src" if (HERE / "src").exists() else HERE.parents[1] / "src")
)
from contrastive_learning.submission import predict_submission


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            predict_submission(
                args.bundle,
                args.data_root,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
