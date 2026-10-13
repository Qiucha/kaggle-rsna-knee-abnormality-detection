"""Export completed run-5 follow-ups and unfinished queue states to the digest."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning.artifacts import read_json, sha256_file, write_json
from contrastive_learning.lm_proxy import load_test_population, paired_intervals

BASELINE = "r4a-k8-omnirad-base-soft-336"
TITLES = {
    BASELINE: "Run 5: soft 336, seed 2026",
    "r4a-k8-omnirad-base-hard-336": "Hard labels at 336",
    "r4a-k8-omnirad-base-soft-336-from-clip": "K8 plain CLIP → run-5 fine-tuning",
    "r4a-k8-omnirad-base-soft-336-seed2027": "Run 5: seed 2027",
    "r4a-k8-omnirad-base-soft-336-ep30": "30-epoch limit",
    "r4a-k8-omnirad-base-soft-336-u6": "Unfreeze 6 image blocks",
    "r4a-k8-omnirad-base-soft-384": "384 px",
    "r4a-k8-dinov2-small-soft-336": "DINOv2-small soft 336",
    "r4a-k8-coatnet-384-soft-s123": "CoAtNet stages 1–3",
}


def optional(path):
    return read_json(path) if path.exists() else {}


def history(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project.resolve()
    run_dir = root / "checkpoints/finetune"
    digest = root / "digest"
    destination = digest / "run5-followups-v1.json"
    previous = optional(destination)
    queue = [line.strip().split("|") for line in (run_dir / "queue-2026-10-03.txt").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    names = [arms[0] for arms in queue]
    actual_runs = {}
    for line in (run_dir / "queue-2026-10-03.out").read_text().splitlines():
        fields = line.split()
        if len(fields) > 2 and fields[1] == "rc=0" and fields[2].startswith("ran="):
            actual_runs[fields[0]] = fields[2].removeprefix("ran=")
    process_text = subprocess.check_output(["ps", "-eo", "args="], text=True)
    completed, unfinished, predictions, hashes = [], [], {}, {}
    for name in dict.fromkeys([BASELINE, *names]):
        actual = actual_runs.get(name, name)
        path = run_dir / actual
        report = optional(path / "report.json")
        config = optional(root / "configs/finetune" / f"{actual}.json")
        epochs = history(path / "history.jsonl")
        title = TITLES.get(name, name)
        if report and (path / "predictions.npz").exists():
            completed.append({"name": name, "actual_run": actual, "title": title, "lm_test": report["lm_test"]["macro_auroc"],
                              "gold": report["gold"]["macro_auroc"], "best_epoch": report["best_epoch"],
                              "epochs_completed": len(epochs), "hours": report["seconds"] / 3600,
                              "report": str((path / "report.json").relative_to(root))})
            with np.load(path / "predictions.npz", allow_pickle=False) as saved:
                predictions[name] = {key: saved[key].copy() for key in saved.files}
            hashes[name] = sha256_file(path / "predictions.npz")
        else:
            active = any(f"configs/finetune/{arm}.json --output" in process_text
                         for arms in queue if arms[0] == name for arm in arms)
            status = "running" if active else "interrupted" if path.exists() else "queued"
            unfinished.append({"name": name, "title": title, "status": status,
                               "epochs_completed": len(epochs), "max_epochs": config.get("epochs"),
                               "report": None})
    baseline = predictions[BASELINE]
    for name, saved in predictions.items():
        for key in ("lm_test_uids", "gold_uids", "gold_targets"):
            if not np.array_equal(saved[key], baseline[key]):
                raise ValueError(f"{name}: {key} differs from run 5")
    if hashes == previous.get("predictions_sha256"):
        paired = previous["paired"]
    else:
        config = read_json(run_dir / BASELINE / "config.json")
        _, targets, masks, identity = load_test_population(root / config["cohort_manifest"],
            root / config["lm_labels_csv"], root / config["gold_labels_csv"])
        if identity["study_uids"] != baseline["lm_test_uids"].tolist():
            raise ValueError("LM-test population differs from saved predictions")
        paired = {}
        for split, labels, observed in (("lm_test", targets, masks),
                ("gold", baseline["gold_targets"], np.ones_like(baseline["gold_targets"]))):
            raw = paired_intervals({name: saved[split] for name, saved in predictions.items()}, labels, observed)
            paired[split] = {}
            for name in predictions:
                if name == BASELINE:
                    continue
                delta = raw[f"{BASELINE} - {name}"]
                paired[split][name] = {"difference": -delta["difference"],
                                      "ci95": [-delta["ci95"][1], -delta["ci95"][0]]}
    pre = root / "checkpoints/contrastive/omnirad-clip-run5-v1/omnirad-base-e5-base-clip-k8-336-u6-bs32-seed2026"
    metrics = history(pre / "metrics.jsonl")
    pre_config = optional(pre / "config.json")
    best = max(metrics, key=lambda row: (row["mean_r@1"], -row["validation_loss"], -row["epoch"])) if metrics else None
    pretraining = {"epochs_completed": len(metrics), "max_epochs": pre_config.get("epochs"),
                  "completed": len(metrics) == pre_config.get("epochs") and (pre / "best.pt").exists(),
                  "selected_epoch": best["epoch"] if best else None,
                  "best_mean_recall_at_1": best["mean_r@1"] if best else None,
                  "checkpoint": str((pre / "best.pt").relative_to(root)),
                  "checkpoint_sha256": sha256_file(pre / "best.pt") if (pre / "best.pt").exists() else None}
    seed_name = "r4a-k8-omnirad-base-soft-336-seed2027"
    noise = abs(paired["lm_test"][seed_name]["difference"]) if seed_name in predictions else None
    result = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "baseline": BASELINE,
              "completed": completed, "unfinished": unfinished, "paired": paired,
              "predictions_sha256": hashes, "absolute_lm_test_seed_difference": noise,
              "contrastive_pretraining": pretraining,
              "contrastive_confirmation": optional(root / "checkpoints/pipelines/omnirad-clip-run5-v1/comparison.json"),
              "bootstrap": {"resamples": 2000, "seed": 2030},
              "note": "LM-test uses LM labels; gold-58 is exploratory. No interrupted run has a final AUROC."}
    write_json(destination, result)
    lines = ["# Run-5 follow-ups", "", f"Updated {result['updated_at']}", "",
             "| Completed run | LM-test | Δ vs run 5 [95% CI] | Gold-58 | Best epoch |",
             "|---|---:|---|---:|---:|"]
    for run in completed:
        delta = paired["lm_test"].get(run["name"])
        text = f"{delta['difference']:+.4f} [{delta['ci95'][0]:+.4f}, {delta['ci95'][1]:+.4f}]" if delta else "baseline"
        lines.append(f"| {run['title']} | {run['lm_test']:.4f} | {text} | {run['gold']:.4f} | {run['best_epoch']} |")
    lines += ["", f"Absolute run-5 seed difference: {noise:.6f}." if noise is not None else "Seed comparison pending.", "",
              f"K8 CLIP pre-training: {len(metrics)}/{pre_config.get('epochs')} epochs; selected epoch "
              f"{pretraining['selected_epoch']}; mean retrieval Recall@1 {pretraining['best_mean_recall_at_1']}. "
              "Retrieval Recall@1 is distinct from pathology AUROC.", "", "| Unfinished run | Status | Completed epochs |",
              "|---|---|---:|"]
    lines += [f"| {run['title']} | {run['status']} | {run['epochs_completed']}/{run['max_epochs']} |" for run in unfinished]
    lines += ["", result["note"], "", "[Fine-tuning digest page](site/pages/finetune.html#run5-followups)",
              "[Machine-readable snapshot](run5-followups-v1.json)"]
    (digest / "run5-followups-v1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed": len(completed) - 1, "unfinished": len(unfinished), "seed_noise": noise,
                      "pretraining_completed": pretraining["completed"]}))


if __name__ == "__main__":
    main()
