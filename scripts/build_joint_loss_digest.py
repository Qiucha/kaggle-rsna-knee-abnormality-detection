"""Snapshot the completed joint-loss run and its saved paired comparisons.

Run from the repository root after scoring with compare_joint_vs_finetune.py.
The site consumes this snapshot without loading checkpoints or training libraries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "omnirad-base-k8-soft-336-u9-b1-bank-l0.1"
JOINT = ROOT / "checkpoints" / "joint" / RUN
RUNS = {
    "run5 (label, u3)": ("Run 5 · label only, u3", "r4a-k8-omnirad-base-soft-336"),
    "run5 seed2027": ("Run 5 · seed 2027", "r4a-k8-omnirad-base-soft-336-seed2027"),
    "CLIP->label (u3)": ("Two stages · CLIP → labels, u3", "r4a-k8-omnirad-base-soft-336-from-clip"),
    "u9-b1 (label, u9)": ("Label only · u9, batch 1", "r4a-k8-omnirad-base-soft-336-u9-b1"),
    "joint (u9)": ("One stage · joint loss, u9", RUN),
}


def build_snapshot() -> Path:
    sources = {}

    def read(path: Path, *, history=False):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if history else json.loads(raw)

    comparison = read(JOINT / "comparison.json")
    config = read(JOINT / "config.json")
    manifest = read(JOINT / "run_manifest.json")
    runs = []
    for key, (title, name) in RUNS.items():
        joint = key == "joint (u9)"
        directory = JOINT if joint else ROOT / "checkpoints" / "finetune" / name
        report = read(directory / "report.json")
        history = read(directory / ("metrics.jsonl" if joint else "history.jsonl"), history=True)
        # Keep the last row for an epoch, matching the site's existing resume handling.
        by_epoch = {row["epoch"]: row for row in history}
        history = [by_epoch[e] for e in sorted(by_epoch)]
        if report["lm_test"]["studies"] != 432 or report["gold"]["studies"] != 58:
            raise ValueError(f"{key}: unexpected evaluation population")
        for split in ("lm_test", "gold"):
            if abs(report[split]["macro_auroc"] - comparison["point"][split][key]) > 1e-10:
                raise ValueError(f"{key}: comparison and report disagree on {split}")
        runs.append({
            "key": key, "name": name, "title": title, "source": str(directory.relative_to(ROOT)),
            "best_epoch": report["best_epoch"], "epochs_completed": len(history),
            "validation_masked_bce": report["validation_masked_bce"],
            "history": [{"epoch": row["epoch"],
                         "train_bce": row["train_classification_bce" if joint else "train_masked_bce"],
                         "validation_bce": row["validation_classification_bce" if joint else "validation_masked_bce"],
                         **({"validation_macro_auroc": row["validation_macro_auroc"]}
                            if "validation_macro_auroc" in row else {}),
                         **({"train_contrastive_loss": row["train_contrastive_loss"],
                             "validation_retrieval_loss": row["validation_loss"],
                             "mean_recall_at_1": row["mean_r@1"],
                             "image_to_text_r@1": row["image_to_text_r@1"],
                             "text_to_image_r@1": row["text_to_image_r@1"]} if joint else {})} for row in history],
        })
    snapshot = {
        "experiment": RUN, "primary_pair": "joint (u9) - u9-b1 (label, u9)",
        "population": {"lm_test": 432, "gold": 58, "train": 3438, "validation": 426},
        "bootstrap": {"resamples": 2000, "seed": 2030, "unit": "study", "confidence": 0.95},
        "recipe": {key: config[key] for key in (
            "image_model", "image_unfreeze_last", "windows_per_plane", "slices_per_plane",
            "image_size", "classification_targets", "classification_weight", "contrastive_weight",
            "batch_size", "grad_accumulation", "epochs", "seed", "text_model", "text_unfreeze_last")},
        "text_bank": manifest["text_bank"], "selection_policy": manifest["selection_policy"],
        "run_provenance": {"git": manifest["git"], "source_tree_sha256": manifest["source_tree_sha256"]},
        **comparison, "runs": runs, "source_sha256": sources,
    }
    output = ROOT / "digest" / "joint-loss-v1.json"
    output.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    print(build_snapshot())
