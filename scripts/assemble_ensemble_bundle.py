"""Assemble exported single-member fine-tuning bundles into one multi-member Kaggle bundle.

Reuses each source bundle's reload-checked ``member-00.pt`` unchanged, so a member is
byte-identical to the one already validated (and possibly scored) on its own. Then runs
the deployed path (every member, then ``combine_members``) on LM-test and checks it
against the same combination of the runs' saved predictions. Reports s/study and the
projection to Kaggle's ~1,300 hidden-test studies against the 9 h limit.

    .venv-cuda/bin/python scripts/assemble_ensemble_bundle.py \\
        --member checkpoints/kaggle/r4a-k8-soft --member checkpoints/kaggle/r4a-k8-omnirad-base \\
        --output checkpoints/kaggle/ensemble-soft-omnirad-base
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.artifacts import read_json, sha256_file, write_json  # noqa: E402
from contrastive_learning.submission import (  # noqa: E402
    ENSEMBLES, LABELS, combine_members, extract_features, load_member, predict_head,
)
from contrastive_learning.submission_training import copy_codecs  # noqa: E402

HIDDEN_TEST_STUDIES = 1300
LIMIT_SECONDS = 9 * 3600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", type=Path, action="append", required=True,
                        help="An exported single-member bundle directory (repeat per member)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble", default="rank_mean", choices=ENSEMBLES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    if len(args.member) < 2:
        raise ValueError("An ensemble bundle needs at least two members")

    bundle = args.output / "bundle"
    bundle.mkdir(parents=True)
    members, runs = [], []
    for i, source in enumerate(args.member):
        manifest = read_json(source / "bundle" / "bundle.json")
        (entry,) = manifest["members"]
        if manifest["labels"] != LABELS or sha256_file(source / "bundle" / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Source bundle fails its own checks: {source}")
        target = bundle / f"member-{i:02d}.pt"
        shutil.copyfile(source / "bundle" / entry["file"], target)
        run = Path(read_json(source / "export.json")["run"])
        members.append({**entry, "file": target.name, "source_bundle": str(source), "run": str(run),
                        "head": manifest["head"]})
        runs.append(run)

    codecs = copy_codecs(ROOT, bundle)
    write_json(bundle / "bundle.json", {
        "schema_version": 1,
        "labels": LABELS,
        "feature": "window_mean_cls_patch",
        "ensemble": args.ensemble,
        "members": members,
        "head": f"{args.ensemble} over {len(members)} fine-tuned members: "
                + "; ".join(m["head"] for m in members),
        "codecs": codecs,
    })
    shutil.copytree(ROOT / "src" / "contrastive_learning", bundle / "src" / "contrastive_learning",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(ROOT / "scripts" / "predict_kaggle.py", bundle / "predict_kaggle.py")
    shutil.copyfile(ROOT / "requirements-kaggle.txt", bundle / "requirements.txt")
    shutil.copyfile(ROOT / "notebooks" / "kaggle_submission.ipynb", args.output / "kaggle_submission.ipynb")

    from contrastive_learning.lm_proxy import load_test_population, masked_macro_auroc

    config = read_json(runs[0] / "config.json")
    rows, targets, masks, _ = load_test_population(
        ROOT / config["cohort_manifest"], ROOT / config["lm_labels_csv"], ROOT / config["gold_labels_csv"])
    uids = [r.study_uid for r in rows]
    started = time.perf_counter()
    deployed = []
    for member in members:
        restored, payload = load_member(bundle / member["file"], args.device)
        deployed.append(predict_head(
            extract_features(restored, rows, payload["image_kwargs"], args.device, 4), payload["head"]))
        del restored, payload
    seconds = time.perf_counter() - started
    deployed = combine_members(deployed, args.ensemble)

    saved = []
    for run in runs:
        z = np.load(run / "predictions.npz")
        order = {u: i for i, u in enumerate(z["lm_test_uids"])}
        saved.append(z["lm_test"][[order[u] for u in uids]])
    expected = combine_members(saved, args.ensemble)

    lm_test = masked_macro_auroc(deployed, targets, masks)["macro_auroc"]
    reference = masked_macro_auroc(expected, targets, masks)["macro_auroc"]
    per_study = seconds / len(rows)
    projected = per_study * HIDDEN_TEST_STUDIES
    print(f"bundle LM-test macro AUROC={lm_test:.6f} (saved predictions {reference:.6f}); "
          f"{per_study:.2f} s/study, projected {projected / 3600:.2f} h for {HIDDEN_TEST_STUDIES} "
          f"studies ({projected / LIMIT_SECONDS:.0%} of the 9 h limit, local GPU)", flush=True)
    write_json(args.output / "lm-test.json", {
        "macro_auroc": lm_test, "saved_predictions_macro_auroc": reference, "ensemble": args.ensemble,
        "studies": len(rows), "seconds": seconds, "seconds_per_study": per_study,
        "projected_hidden_test_seconds": projected, "limit_seconds": LIMIT_SECONDS,
    })
    if abs(lm_test - reference) > args.tolerance:
        raise ValueError("Deployed ensemble does not reproduce the saved predictions' LM-test score")


if __name__ == "__main__":
    main()
