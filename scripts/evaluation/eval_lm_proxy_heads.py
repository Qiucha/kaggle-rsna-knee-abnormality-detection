"""Score exported bundles with LM-verdict AUROC on the locked test split and compare them.

Writes ``<run>/lm-proxy.json`` for every run and, with --output, a comparison with
paired study-bootstrap intervals and any recorded Kaggle public scores.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning.artifacts import write_json  # noqa: E402
from contrastive_learning.lm_proxy import paired_intervals, score_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories containing bundle/ and benchmark.json")
    parser.add_argument("--output", type=Path, help="comparison JSON (paired intervals)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    reports, predictions, population = {}, {}, None
    for run in args.runs:
        report, prediction, targets, masks = score_run(run, device=args.device, batch_size=args.batch_size)
        if population is None:
            population, y, m = report["study_uids_sha256"], targets, masks
        elif report["study_uids_sha256"] != population:
            parser.error(f"{run} uses a different locked test population; runs are not comparable")
        record = run / "leaderboard-submission.json"
        public = json.loads(record.read_text()).get("public_score") if record.exists() else None
        reports[run.name] = {"lm_test_macro_auroc": report["macro_auroc"], "public_score": public, "studies": report["studies"]}
        predictions[run.name] = prediction
        print(f"{run.name}: LM-test macro AUROC {report['macro_auroc']:.4f} (public {public})", flush=True)

    if args.output:
        comparison = {
            "metric": "lm_test_macro_auroc",
            "studies": int(len(y)),
            "runs": reports,
            "paired_intervals": paired_intervals(predictions, y, m, resamples=args.resamples) if len(predictions) > 1 else {},
        }
        scored = [(r["lm_test_macro_auroc"], r["public_score"]) for r in reports.values() if r["public_score"] is not None]
        if len(scored) > 1:
            proxy, public = np.array(scored).T
            pairs = [(i, j) for i in range(len(scored)) for j in range(i + 1, len(scored)) if public[i] != public[j]]
            agree = sum(np.sign(proxy[i] - proxy[j]) == np.sign(public[i] - public[j]) for i, j in pairs)
            comparison["public_pairwise_agreement"] = f"{agree}/{len(pairs)}"
        write_json(args.output, comparison)
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
