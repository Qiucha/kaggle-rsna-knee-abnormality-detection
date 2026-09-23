"""Benchmark frozen representations, fit final heads, and assemble Kaggle artifacts."""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read_json, sha256_file, write_atomic, write_json
from .manifest import read_cohort_manifest
from .submission import (
    LABELS,
    UID,
    benchmark_head,
    export_member,
    extract_features,
    fit_head,
    load_encoder,
    load_member,
    predict_head,
    score_predictions,
    validate_ids,
)


def training_population(manifest, labels_csv, allowed_splits):
    if not allowed_splits or not set(allowed_splits) <= {
        "train",
        "validation",
        "excluded",
    }:
        raise ValueError(
            "Only train, validation, and excluded populations are allowed; locked test is never used"
        )
    frame = pd.read_csv(labels_csv, dtype={UID: str})
    ids = validate_ids(frame)
    values = frame[LABELS].to_numpy(dtype=float)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(
            "Label file must contain complete binary 0/1 labels (not blanks or LM confidences)"
        )
    records, metadata = read_cohort_manifest(manifest)
    by_uid = {record.study_uid: record for record in records}
    kept, rows, counts = [], [], {}
    for i, uid in enumerate(ids):
        record = by_uid.get(uid)
        split = record.split if record else "absent"
        counts[split] = counts.get(split, 0) + 1
        if record and split in allowed_splits:
            kept.append(record)
            rows.append(i)
    if not kept:
        raise ValueError("No eligible labeled studies")
    return (
        kept,
        values[rows],
        {"split_counts": counts, "manifest_sha256": metadata["manifest_sha256"]},
    )


def train_bundle(
    config_path: Path, output: Path, *, device="cpu", batch_size=8, cache_dir=None
):
    started = time.perf_counter()
    config = read_json(config_path)
    root = Path(__file__).resolve().parents[2]
    checkpoints = [root / value for value in config["checkpoints"]]
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Supply at least one checkpoint, without duplicates")
    labels_path = root / config["labels_csv"]
    records, targets, population = training_population(
        root / config["cohort_manifest"],
        labels_path,
        config["population_splits"],
    )
    if population["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Population manifest checksum differs from configuration")
    output.mkdir(parents=True, exist_ok=False)
    bundle_dir = output / "bundle"
    bundle_dir.mkdir()
    members, results, all_oof = [], {}, []
    baseline_by_preprocessing = {}
    for i, checkpoint in enumerate(checkpoints):
        print(f"[{i + 1}/{len(checkpoints)}] {checkpoint.parent.name}", flush=True)
        backbone, run_config, adaptation = load_encoder(checkpoint)
        # Exposure is checked against EACH checkpoint's own frozen cohort, not just
        # against the manifest used to select labeled images.
        trained_records, trained_meta = read_cohort_manifest(
            root / run_config.cohort_manifest
        )
        own = {record.study_uid: record for record in trained_records}
        if any(record.study_uid not in own for record in records):
            raise ValueError(
                "Cannot establish encoder exposure for every labeled study"
            )
        exposure = {}
        for record in records:
            split = own[record.study_uid].split
            exposure[split] = exposure.get(split, 0) + 1
        if exposure.get("test", 0):
            raise ValueError("A labeled study is locked test for this checkpoint")
        features = extract_features(
            backbone, records, run_config.image_kwargs, device, batch_size, cache_dir
        )
        oof, folds = benchmark_head(features, targets, **config["probe"])
        all_oof.append(oof)
        name = f"member-{i:02d}"
        results[name] = {
            **score_predictions(targets, oof),
            "checkpoint": str(checkpoint.relative_to(root)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "encoder_exposure": exposure,
            "encoder_manifest_sha256": trained_meta["manifest_sha256"],
            "image_kwargs": run_config.image_kwargs,
        }
        head = fit_head(features, targets, c=config["probe"]["c"])
        member_path = bundle_dir / f"{name}.pt"
        export_member(member_path, backbone, run_config, adaptation, head)
        # A real DICOM batch must survive offline export/reload without changing predictions.
        before = predict_head(features[:1], head)
        del backbone
        reloaded, payload = load_member(member_path, device)
        reloaded_features = extract_features(
            reloaded, records[:1], payload["image_kwargs"], device, 1, cache_dir
        )
        after = predict_head(reloaded_features, payload["head"])
        np.testing.assert_allclose(before, after, atol=1e-5, rtol=1e-5)
        results[name]["reload_max_probability_error"] = float(
            np.abs(before - after).max()
        )
        del reloaded, payload
        members.append(
            {
                "file": member_path.name,
                "sha256": sha256_file(member_path),
                "checkpoint_sha256": results[name]["checkpoint_sha256"],
            }
        )
        write_atomic(
            output / f"{name}-oof.npz",
            lambda temporary, oof=oof, folds=folds: _save_npz(
                temporary,
                study_uids=np.asarray([r.study_uid for r in records]),
                targets=targets,
                probabilities=oof,
                fold_assignments=folds,
            ),
        )
        key = (
            run_config.image_model,
            run_config.image_model_revision,
            tuple(sorted(run_config.image_kwargs.items())),
        )
        if key not in baseline_by_preprocessing:
            baseline, _, _ = load_encoder(checkpoint, baseline=True)
            baseline_features = extract_features(
                baseline,
                records,
                run_config.image_kwargs,
                device,
                batch_size,
                cache_dir,
            )
            baseline_oof, _ = benchmark_head(
                baseline_features, targets, **config["probe"]
            )
            baseline_name = f"unadapted-{len(baseline_by_preprocessing):02d}"
            results[baseline_name] = {
                **score_predictions(targets, baseline_oof),
                "image_kwargs": run_config.image_kwargs,
            }
            baseline_by_preprocessing[key] = baseline_name
            del baseline
        results[name]["baseline"] = baseline_by_preprocessing[key]
    ensemble = np.mean(all_oof, axis=0)
    results["ensemble"] = score_predictions(targets, ensemble)
    write_atomic(
        output / "ensemble-oof.npz",
        lambda temporary: _save_npz(
            temporary,
            study_uids=np.asarray([r.study_uid for r in records]),
            targets=targets,
            probabilities=ensemble,
        ),
    )
    write_json(
        bundle_dir / "bundle.json",
        {
            "schema_version": 1,
            "labels": LABELS,
            "feature": "uniform_mean_cls_patch",
            "ensemble": "mean_probability",
            "members": members,
        },
    )
    shutil.copytree(
        root / "src" / "contrastive_learning",
        bundle_dir / "src" / "contrastive_learning",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(
        root / "scripts" / "predict_kaggle.py", bundle_dir / "predict_kaggle.py"
    )
    shutil.copyfile(root / "requirements-kaggle.txt", bundle_dir / "requirements.txt")
    shutil.copyfile(
        root / "notebooks" / "kaggle_submission.ipynb",
        output / "kaggle_submission.ipynb",
    )
    source_hashes = {
        str(path.relative_to(bundle_dir)): sha256_file(path)
        for path in sorted((bundle_dir / "src").rglob("*.py"))
    }
    write_json(
        output / "benchmark.json",
        {
            "config": config,
            "population": population,
            "studies": len(records),
            "labels_sha256": sha256_file(labels_path),
            "results": results,
            "seconds": time.perf_counter() - started,
            "runtime_source_sha256": source_hashes,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "torch",
                    "transformers",
                    "pydicom",
                    "scikit-learn",
                )
            },
            "python": sys.version,
            "interpretation": "Study-level per-label stratified OOF readout. Encoder exposure is reported per member. "
            "Repeated development on this small gold cohort limits generalisation claims. "
            "This is not a Kaggle leaderboard score. Patient identity is unverifiable.",
        },
    )
    print(
        f"Bundle: {bundle_dir}\nOOF ensemble macro AUROC: {results['ensemble']['macro_auroc']:.6f}",
        flush=True,
    )
    return results


def _save_npz(path, **values):
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)
