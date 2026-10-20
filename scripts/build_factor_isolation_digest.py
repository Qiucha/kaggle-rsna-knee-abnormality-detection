"""Export factor-isolation reports, cached predictions, curves and paired comparisons.

Reads the saved summary for arm definitions, verifies completed head reports against
predictions, and never extracts images or trains models. Rebuild the digest site after this.

Inputs not in git (under ``--project``, default this repository):

* ``checkpoints/contrastive/factor-isolation-v1/summary.json``: the arm definitions, written
  by the chain's own off-repo ``summarize.py``. One object per arm (``reference`` plus each
  name in ``configs/contrastive/factor-isolation-v1.json``) with ``purpose``, ``comparator``
  (the arm it is paired against) and ``delta_vs_unadapted``.
* each arm's ``metrics.jsonl``, ``config.json`` and ``best.pt`` there, and the codec
  reference run under ``checkpoints/contrastive/codec-declip-v1``.
* the LM-head runs ``checkpoints/kaggle/lm-mlp-factor-iso-58-*`` (and the codec reference
  and unadapted heads) and the LM-test feature caches in ``checkpoints/kaggle/lm-proxy-cache``.

Usage: build_factor_isolation_digest.py [--project DIR] [--digest DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.lm_proxy import paired_intervals
from contrastive_learning.lm_supervision import paired_gold_interval
from contrastive_learning.submission import LABELS
from experiment_tools.lm_heads import load_head

SOURCE = "checkpoints/contrastive/factor-isolation-v1"
# Frozen-tower heads each arm is compared against: the unadapted tower at the same input.
UNADAPTED = {
    "reference": "lm-mlp-codec-declip-58-unadapted-dinov2-small",
    "dinov2-small-uniform-224": "lm-mlp-codec-declip-58-unadapted-dinov2-small",
    "omnirad-base-lora-224": "lm-mlp-factor-iso-58-unadapted-omnirad-base-224",
}

# Appended to the md, so a reader can rerun an arm from the committed recipe.
REPRODUCE = [
    '',
    '## Reproduce',
    '',
    'From the repository root on a CUDA host, with `paths.json` naming `data_root` (and `train_series_mirror` for the K8 arm):',
    '',
    '1. Train one arm: `python scripts/run_contrastive_matrix.py --matrix configs/contrastive/factor-isolation-v1.json --experiment <arm>`.',
    '2. Fit its heads, once per config the arm lists under `lm_head_configs`: `python scripts/train_lm_head.py --config configs/kaggle/<head>.json --features checkpoints/kaggle/<head>-features.npz --output checkpoints/kaggle/<head> --device cuda`.',
    "3. Rebuild this digest: `python scripts/build_factor_isolation_digest.py`. It also reads the chain's off-repo `summary.json` (see its docstring).",
    '',
    "A new arm is a new matrix entry. Its `args` override the base recipe. Copy a head config and point `checkpoint` at the new run's `best.pt`.",
    'Not in git: the cohort manifest (checked against `manifest_sha256`), `data/lm_label/lm_labels_kaggle.csv`, and the pinned Hugging Face snapshots.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, default=ROOT, help="Root holding the checkpoints (read only)")
    parser.add_argument("--digest", type=Path, default=ROOT / "digest", help="Where the md and json are written")
    args = parser.parse_args()
    root, source = args.project, args.project / SOURCE
    destination = args.digest / "factor-isolation-v1.json"
    definitions = json.loads((source / "summary.json").read_text())
    previous = json.loads(destination.read_text()) if destination.exists() else {}
    snapshot, heads = {}, {}
    for name, definition in definitions.items():
        arm = {key: definition[key] for key in ("purpose", "comparator")}
        reference = name == "reference"
        run = (root / "checkpoints/contrastive/codec-declip-v1/dinov2-small-e5-base-512-codec-seed2026"
               if reference else source / name)
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        best = max(metrics, key=lambda row: (row["mean_r@1"], -row["validation_loss"], -row["epoch"]))
        arm["contrastive"] = {"epochs": len(metrics), "selected_epoch": best["epoch"], "mean_r@1": best["mean_r@1"]}
        config = json.loads((run / "config.json").read_text())
        arm["settings"] = {key: config.get(key) for key in ("image_model", "text_model", "image_size",
            "windows_per_plane", "image_plane_pooling", "batch_size", "image_unfreeze_last", "text_unfreeze_last",
            "use_lora", "lora_rank", "lora_alpha", "lora_dropout", "lr_image", "lr_text", "lr_projection", "seed",
            "weight_decay", "warmup_fraction", "crop_mm", "max_text_length", "epochs")}
        head = root / "checkpoints/kaggle" / ("lm-mlp-codec-declip-58-dinov2-small-e5-base-512-codec"
                                              if reference else f"lm-mlp-factor-iso-58-{name}")
        scored = load_head(head, run / "best.pt", root)
        if scored:
            heads[name] = scored
        arm["head_run"] = str(head.relative_to(root))
        arm["gold_mlp"] = scored["benchmark"]["results"]["mlp"]["macro_auroc"] if scored else None
        arm["lm_test_mlp"] = scored["proxy"]["macro_auroc"] if scored else None
        arm["gold_per_label"] = {label: value["auroc"] for label, value in
            scored["benchmark"]["results"]["mlp"]["per_label"].items()} if scored else {}
        arm["lm_test_per_label"] = scored["proxy"]["per_label"] if scored else {}
        arm["gold_positives"] = {label: int(value) for label, value in
            zip(LABELS, scored["gold"]["targets"].sum(0))} if scored else {}
        arm["lm_test_observed"] = dict(zip(LABELS, scored["proxy"]["observed"])) if scored else {}
        floor_name = UNADAPTED.get(name, f"lm-mlp-factor-iso-58-unadapted-{name}")
        floor_path = root / "checkpoints/kaggle" / floor_name / "benchmark.json"
        floor = json.loads(floor_path.read_text()) if floor_path.exists() else {}
        arm["unadapted_per_label"] = {label: value["auroc"] for label, value in
            floor.get("results", {}).get("mlp", {}).get("per_label", {}).items()}
        arm["unadapted_gold_mlp"] = floor.get("results", {}).get("mlp", {}).get("macro_auroc")
        arm["unadapted_head"] = floor_name
        arm["delta_vs_unadapted"] = definition.get("delta_vs_unadapted")
        arm["head_histories"] = {path.stem: json.loads(path.read_text())
                                 for path in sorted(head.glob("mlp-seed*-history.json")) if scored}
        snapshot[name] = arm
    for name, arm in snapshot.items():
        arm["paired"] = {"gold": {}, "lm_test": {}}
        for kind, comparator in (("reference", "reference"), ("comparator", arm["comparator"])):
            if name == comparator or name not in heads or comparator not in heads:
                continue
            candidate, baseline = heads[name], heads[comparator]
            if (not np.array_equal(candidate["gold"]["study_uids"], baseline["gold"]["study_uids"])
                    or not np.array_equal(candidate["gold"]["targets"], baseline["gold"]["targets"])
                    or candidate["population"] != baseline["population"]):
                raise ValueError(f"{name} vs {comparator}: population mismatch")
            signature = {"candidate": candidate["signature"], "baseline": baseline["signature"]}
            old = previous.get(name, {})
            if kind == "comparator" and comparator == "reference":
                values = {split: arm["paired"][split]["reference"] for split in ("gold", "lm_test")}
            elif old.get("comparison_identity", {}).get(kind) == signature:
                values = {split: old["paired"][split][kind] for split in ("gold", "lm_test")}
            else:
                print(f"paired comparison: {name} vs {comparator}", flush=True)
                values = {"gold": paired_gold_interval(candidate["gold"]["targets"], baseline["gold"]["mlp"],
                    candidate["gold"]["mlp"], resamples=2000, seed=2030),
                    "lm_test": paired_intervals({name: candidate["lm_predictions"], comparator: baseline["lm_predictions"]},
                        candidate["lm_targets"], candidate["lm_masks"])[f"{name} - {comparator}"]}
            for split in values:
                arm["paired"][split][kind] = values[split]
            arm.setdefault("comparison_identity", {})[kind] = signature
        arm["delta_vs_comparator"] = arm["paired"]["gold"].get("comparator")
    args.digest.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2) + "\n")
    lines = ["# Factor isolation: completed reports and paired comparisons", "",
        "One contrastive factor per arm; selection uses mean retrieval R@1. Frozen LM heads use three seeds.",
        "Gold-58 is exploratory. LM-test is agreement with LM verdicts on 432 non-gold held-out studies.",
        "Paired intervals use 2,000 study resamples, seed 2030. Missing head scores mean evaluation pending.", "",
        "| Arm | Mean R@1 | Gold-58 | LM-test | Δ gold vs reference [95% CI] | Δ LM-test vs reference [95% CI] |",
        "|---|---:|---:|---:|---|---|"]
    for name, arm in snapshot.items():
        def value(v):
            return "pending" if v is None else f"{v:.4f}"
        def interval(split):
            d = arm["paired"][split].get("reference")
            return f"{d['difference']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else "baseline" if name == "reference" else "pending"
        lines.append(f"| {name} | {value(arm['contrastive']['mean_r@1'])} | {value(arm['gold_mlp'])} | "
            f"{value(arm['lm_test_mlp'])} | {interval('gold')} | {interval('lm_test')} |")
    lines += ["", "The codec reference is the common anchor; the K8 matched comparator is the uniform-224 arm.",
        "A difference against the common reference is not the isolated K8 effect: it also changes plane pooling.",
        "The OmniRad swap changes model size and pretraining. No unadapted LM-test score is assumed.", "",
        "Sources: saved summary, metrics.jsonl, config.json, benchmark.json, gold-predictions.npz,",
        "bundle checksums and existing LM-test feature caches. Predictions are checked against reported AUROCs.",
        "", "[Adaptation page](site/pages/adaptation.html)",
        "[Verified data](factor-isolation-v1.json)"]
    lines += REPRODUCE
    (args.digest / "factor-isolation-v1.md").write_text("\n".join(lines) + "\n")
    arms = sum(name != "reference" for name in snapshot)
    completed = sum(name != "reference" for name in heads)
    print(f"exported {arms} contrastive arms; {completed} completed heads (plus the reference)")


if __name__ == "__main__":
    main()
