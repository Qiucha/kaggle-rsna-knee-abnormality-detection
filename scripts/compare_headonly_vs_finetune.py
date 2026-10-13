"""Head-only readout on frozen towers against fully label-fine-tuned run-5 rungs.

Arms, all scored on the same 432 LM-test and 58 gold studies:

* ``clip-head`` / ``base-head`` -- MLP heads (3-seed mean) on the frozen CLIP-adapted or
  unadapted OmniRad tower, from ``analyze_omnirad_probe.py``.
* fine-tuned rungs -- last 3 blocks + head trained on LM labels (run 5 from OmniRad,
  ``-from-clip`` from the CLIP tower, and run 5's seed replicate).

Paired study-bootstrap intervals use 2,000 shared resamples, seed 2030, the same as
``digest/run5-followups-v1.json``.

Usage: compare_headonly_vs_finetune.py --heads DIR/head-predictions.npz --features DIR --output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contrastive_learning.lm_proxy import masked_macro_auroc, paired_intervals  # noqa: E402

FINETUNED = {
    "run5-ft": "r4a-k8-omnirad-base-soft-336",
    "run5-ft-seed2027": "r4a-k8-omnirad-base-soft-336-seed2027",
    "clip-then-ft": "r4a-k8-omnirad-base-soft-336-from-clip",
}


def aligned(uids, source_uids, values):
    order = {u: i for i, u in enumerate(source_uids)}
    return values[[order[u] for u in uids]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    heads = np.load(args.heads)
    population = np.load(args.features / "population.npz")
    result = {}
    for split, key in (("lm_test", "test"), ("gold", "gold")):
        uids = population[f"{key}_uids"]
        y = population[f"{key}_targets"]
        m = population[f"{key}_masks"]
        predictions = {
            "clip-head": aligned(uids, heads[f"{key}_uids"], heads[f"{key}_clip_mlp"]),
            "base-head": aligned(uids, heads[f"{key}_uids"], heads[f"{key}_base_mlp"]),
        }
        for name, rung in FINETUNED.items():
            path = ROOT / "checkpoints/finetune" / rung / "predictions.npz"
            if not path.exists():
                continue
            with np.load(path) as f:
                if set(f[f"{split}_uids"]) != set(uids):
                    raise ValueError(f"{rung} scored a different {split} population")
                predictions[name] = aligned(uids, f[f"{split}_uids"], f[split])
        scores = {n: masked_macro_auroc(p, y, m) for n, p in predictions.items()}
        result[split] = {
            "studies": len(uids),
            "macro_auroc": {n: s["macro_auroc"] for n, s in scores.items()},
            "per_label": {n: s["per_label"] for n, s in scores.items()},
            "paired": paired_intervals(predictions, y, m, resamples=2000, seed=2030),
        }
    args.output.write_text(json.dumps(result, indent=2))
    for split, entry in result.items():
        print(split, {n: round(v, 4) for n, v in entry["macro_auroc"].items()})
        for pair, v in entry["paired"].items():
            print(f"  {pair}: {v['difference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
