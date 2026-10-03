"""Export windowed (R4a mean- or group-pooled) fine-tuning runs as one Kaggle bundle.

Each ``--run`` becomes a member: its model is rebuilt from the run's config and
``best.pt``, and its ``image_kwargs`` carry ``windows`` (members may differ in
backbone and image size). Each member is checked against its run's own gold
predictions before packaging. With several runs the bundle combines members by
per-label rank average (``rank_mean``), and the check also compares the deployed
rank average with the one computed from the saved predictions on the same studies.
Score the bundle on LM-test afterwards (this script does so unless ``--skip-lm-test``).

    .venv-cuda/bin/python scripts/export_finetune_bundle.py \\
        --run checkpoints/finetune/r4a-soft-224-k8-mean \\
        --run checkpoints/finetune/r4a-k8-omnirad-base-pretrained \\
        --output checkpoints/kaggle/soft-omb-rank
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.artifacts import read_json, sha256_file, write_json  # noqa: E402
from contrastive_learning.finetune import WindowModel, export_dense_head, image_kwargs, load_backbone  # noqa: E402
from contrastive_learning.lm_supervision import load_population  # noqa: E402
from contrastive_learning.submission import (  # noqa: E402
    LABELS, combine_members, export_member, extract_features, load_member, predict_head,
)
from contrastive_learning.submission_training import copy_codecs  # noqa: E402

# CPU fp32 against the run's GPU autocast can swap near-tied studies; allow one position.
RANK_POSITIONS = 1


def export_run(run: Path, member: Path) -> dict:
    """Write ``run``'s best model to ``member`` and return the run's config."""
    config = read_json(run / "config.json")
    if not config.get("windows_per_plane") or config["pooling"] not in ("mean", "grouped"):
        raise ValueError(f"{run}: only windowed mean- or group-pooled runs export through this script")
    if config.get("image_model_format") == "timm":
        raise ValueError(f"{run}: timm members need a timm-aware load_member; not supported yet")
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    backbone, adaptation = load_backbone(config)
    model = WindowModel(backbone, config["head_hidden"], config["dropout"], config["pooling"],
                        window_groups=config.get("window_groups"))
    result = model.load_state_dict(best["trainable_state_dict"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected weights: {result.unexpected_keys}")
    missing_trainable = {n for n, p in model.named_parameters() if p.requires_grad} - set(best["trainable_state_dict"])
    if missing_trainable:
        raise ValueError(f"best.pt lacks trainable weights: {sorted(missing_trainable)[:5]}")
    kwargs = {**image_kwargs(config), "windows": config["windows_per_plane"],
              **({"window_groups": config["window_groups"]} if config.get("window_groups") else {})}
    export_member(member, model.backbone.eval(), type("C", (), {"image_kwargs": kwargs}),
                  adaptation, export_dense_head(model.head))
    return config


def deploy(member: Path, rows, device: str) -> np.ndarray:
    restored, payload = load_member(member, device)
    probabilities = predict_head(extract_features(restored, rows, payload["image_kwargs"], device, 4),
                                 payload["head"])
    del restored
    return probabilities


def saved_rows(run: Path, split: str, uids: list[str]) -> np.ndarray:
    saved = np.load(run / "predictions.npz")
    order = {u: i for i, u in enumerate(saved[f"{split}_uids"])}
    return saved[split][[order[u] for u in uids]]


def check(runs, members, rows, split, device, *, ensemble) -> dict:
    """Deployed vs saved predictions on ``rows``: per member, then the combined ranks."""
    uids = [r.study_uid for r in rows]
    deployed, expected, report = [], [], {"studies": len(rows), "members": []}
    for run, member in zip(runs, members):
        started = time.perf_counter()
        deployed.append(deploy(member, rows, device))
        seconds = time.perf_counter() - started
        expected.append(saved_rows(run, split, uids))
        error = float(np.abs(deployed[-1] - expected[-1]).max())
        report["members"].append({"run": str(run), "max_probability_error": error,
                                  "seconds_per_study": seconds / len(rows)})
        print(f"{split} check, {run.name} on {len(rows)} studies: max |Δp| = {error:.5f}, "
              f"{seconds / len(rows):.2f} s/study", flush=True)
        if error > 0.02:
            raise ValueError(f"Exported member {member.name} does not reproduce {run}'s predictions")
    # Ranks depend on the study set, so the reference is recomputed on exactly these rows.
    combined, reference = combine_members(deployed, ensemble), combine_members(expected, ensemble)
    report["max_rank_error"] = float(np.abs(combined - reference).max())
    print(f"{split} check, {ensemble}: max |Δrank| = {report['max_rank_error']:.5f}", flush=True)
    if ensemble == "rank_mean" and report["max_rank_error"] * len(rows) > RANK_POSITIONS + 1e-9:
        raise ValueError("Deployed ensemble ranks differ from the saved predictions' ranks")
    return report, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="a fine-tuning run directory; repeat for an ensemble")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-studies", type=int, default=16)
    parser.add_argument("--skip-lm-test", action="store_true")
    args = parser.parse_args()
    ensemble = "rank_mean" if len(args.run) > 1 else "mean_probability"

    bundle = args.output / "bundle"
    bundle.mkdir(parents=True)
    members = [bundle / f"member-{i:02d}.pt" for i in range(len(args.run))]
    configs = [export_run(run, member) for run, member in zip(args.run, members)]
    config = configs[0]
    for other in configs[1:]:
        for key in ("cohort_manifest", "lm_labels_csv", "gold_labels_csv", "gold_splits"):
            if other[key] != config[key]:
                raise ValueError(f"Members disagree on {key}; their saved predictions are not comparable")

    groups, *_ = load_population(config, ROOT)
    gold, _ = check(args.run, members, groups["gold"][: args.check_studies], "gold", args.device,
                    ensemble=ensemble)

    codecs = copy_codecs(ROOT, bundle)
    write_json(bundle / "bundle.json", {
        "schema_version": 1,
        "labels": LABELS,
        "feature": "window_mean_cls_patch",
        "ensemble": ensemble,
        "members": [{"file": member.name, "sha256": sha256_file(member),
                     "checkpoint_sha256": sha256_file(run / "best.pt")}
                    for run, member in zip(args.run, members)],
        "head": "; ".join(f"fine-tuned {c['name']}: K={c['windows_per_plane']} windows per plane at "
                          f"{c['image_size']} px, masked mean" for c in configs),
        "codecs": codecs,
    })
    shutil.copytree(ROOT / "src" / "contrastive_learning", bundle / "src" / "contrastive_learning",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(ROOT / "scripts" / "predict_kaggle.py", bundle / "predict_kaggle.py")
    shutil.copyfile(ROOT / "requirements-kaggle.txt", bundle / "requirements.txt")
    shutil.copyfile(ROOT / "notebooks" / "kaggle_submission.ipynb", args.output / "kaggle_submission.ipynb")
    write_json(args.output / "export.json", {"runs": [str(r) for r in args.run], "configs": configs,
                                             "ensemble": ensemble, "gold_check": gold})
    if not args.skip_lm_test:
        from contrastive_learning.lm_proxy import load_test_population, masked_macro_auroc

        rows, targets, masks, _ = load_test_population(
            ROOT / config["cohort_manifest"], ROOT / config["lm_labels_csv"], ROOT / config["gold_labels_csv"])
        report, combined = check(args.run, members, rows, "lm_test", args.device, ensemble=ensemble)
        lm_test = masked_macro_auroc(combined, targets, masks)["macro_auroc"]
        seconds_per_study = sum(m["seconds_per_study"] for m in report["members"])
        print(f"bundle LM-test macro AUROC={lm_test:.6f}; {seconds_per_study:.2f} s/study over "
              f"{len(rows)} studies from source DICOMs, summed over members", flush=True)
        write_json(args.output / "lm-test.json", {"macro_auroc": lm_test, **report,
                                                  "seconds_per_study": seconds_per_study})


if __name__ == "__main__":
    main()
