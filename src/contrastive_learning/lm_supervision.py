"""LM-label experiment over a fixed, frozen contrastive image representation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import sha256_file, write_atomic, write_json
from .labels import read_lm_labels
from .manifest import read_cohort_manifest
from .submission import LABELS, UID, extract_features, load_encoder, validate_ids


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def partition_records(records, table, gold_ids, *, gold_splits=("excluded",)):
    """Keep gold identities/groups out of both head-fitting splits."""
    gold_splits = frozenset(gold_splits)
    if not gold_splits or not gold_splits <= {"excluded", "test"}:
        raise ValueError("Gold scoring may use only excluded and test splits")
    by_uid = {r.study_uid: r for r in records}
    gold_groups = {
        r.patient_group_id
        for r in records
        if r.study_uid in gold_ids and r.patient_group_id is not None
    }
    groups = {"train": [], "validation": [], "gold": []}
    excluded = {"test": 0, "other": 0, "all_unknown": 0, "gold_group": 0}
    for record in records:
        if record.study_uid in gold_ids:
            if record.split in gold_splits:
                groups["gold"].append(record)
            elif record.split == "test":
                excluded["test"] += 1
            else:
                raise ValueError(
                    "Gold benchmark has encoder exposure; use a gold-holdout checkpoint"
                )
        elif record.split == "test":
            excluded["test"] += 1
        elif (
            record.patient_group_id is not None
            and record.patient_group_id in gold_groups
        ):
            excluded["gold_group"] += 1
        elif record.split in ("train", "validation"):
            _, mask = table.get(record.study_uid)
            if mask.sum() == 0:
                excluded["all_unknown"] += 1
            else:
                groups[record.split].append(record)
        else:
            excluded["other"] += 1
    # Keep gold order stable against the existing probe and submission benchmark.
    kept_gold = {r.study_uid: r for r in groups["gold"]}
    groups["gold"] = [kept_gold[uid] for uid in gold_ids if uid in kept_gold]
    if any(not rows for rows in groups.values()):
        raise ValueError("Train, validation, and gold populations must all be nonempty")
    group_sets = [
        {r.patient_group_id or r.study_uid for r in rows} for rows in groups.values()
    ]
    for i in range(len(group_sets)):
        for j in range(i):
            if group_sets[i] & group_sets[j]:
                raise ValueError("A study group crosses supervised partitions")
    excluded["gold_absent"] = sum(uid not in by_uid for uid in gold_ids)
    excluded["gold_locked_test"] = sum(
        uid in by_uid and by_uid[uid].split == "test" and "test" not in gold_splits
        for uid in gold_ids
    )
    excluded["gold_test_evaluated"] = sum(
        uid in by_uid and by_uid[uid].split == "test" and "test" in gold_splits
        for uid in gold_ids
    )
    return groups, excluded


def load_population(config, root):
    records, metadata = read_cohort_manifest(root / config["cohort_manifest"])
    if metadata["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Manifest checksum differs from frozen experiment config")
    lm_path = root / config["lm_labels_csv"]
    # Validate the schema explicitly rather than silently accepting misspelled verdicts.
    lm_frame = pd.read_csv(lm_path, dtype={UID: str})
    validate_ids(lm_frame)
    for label in LABELS:
        if not lm_frame[f"{label}__verdict"].isin(["YES", "NO", "UNK"]).all():
            raise ValueError(f"Unexpected verdict for {label}")
    table = read_lm_labels(lm_path, LABELS)
    gold_frame = pd.read_csv(root / config["gold_labels_csv"], dtype={UID: str})
    gold_ids = validate_ids(gold_frame)
    if not np.isin(gold_frame[LABELS].to_numpy(float), [0, 1]).all():
        raise ValueError("Gold targets must be complete and binary")
    groups, excluded = partition_records(
        records, table, gold_ids, gold_splits=config.get("gold_splits", ["excluded"])
    )
    if "expected_gold_studies" in config and len(groups["gold"]) != config["expected_gold_studies"]:
        raise ValueError("Gold scoring population differs from expected_gold_studies")
    targets, masks = {}, {}
    for split in ("train", "validation"):
        entries = [table.get(r.study_uid) for r in groups[split]]
        targets[split] = np.stack([((value.numpy() + 1) / 2) for value, _ in entries])
        masks[split] = np.stack([mask.numpy() for _, mask in entries])
        # Unknown targets are arbitrary placeholders; they contribute no loss.
        targets[split][masks[split] == 0] = 0
    gold_by_uid = gold_frame.set_index(UID)
    targets["gold"] = gold_by_uid.loc[
        [r.study_uid for r in groups["gold"]], LABELS
    ].to_numpy(np.float32)
    masks["gold"] = np.ones_like(targets["gold"])
    coverage = {
        split: {
            "studies": len(rows),
            "observed": masks[split].sum(0).astype(int).tolist(),
            "positive": (targets[split] * masks[split]).sum(0).astype(int).tolist(),
        }
        for split, rows in groups.items()
    }
    return (
        groups,
        targets,
        masks,
        {"coverage": coverage, "excluded": excluded, "lm_rows": len(lm_frame)},
    )


def prepare_features(config, cache_path, *, device, batch_size, image_cache_dir):
    if config["feature"] != "uniform_mean_cls_patch":
        raise ValueError("Unsupported frozen feature representation")
    root = Path(__file__).resolve().parents[2]
    groups, targets, masks, population = load_population(config, root)
    checkpoint = root / config["checkpoint"]
    # image_baseline: the checkpoint's pretrained image tower, before contrastive adaptation
    baseline = bool(config.get("image_baseline"))
    backbone, run_config, adaptation = (load_encoder(checkpoint, baseline=True) if baseline
                                        else load_encoder(checkpoint))
    if baseline:
        # the exported bundle member must rebuild the plain pretrained tower, not the checkpoint's LoRA layout
        adaptation = {**adaptation, "use_lora": False, "image_unfreeze_last": 0}
    _, own_metadata = read_cohort_manifest(root / run_config.cohort_manifest)
    if own_metadata["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError(
            "Supervised splits must match the frozen encoder's own manifest"
        )
    backbone.requires_grad_(False).eval()
    digest = state_digest(backbone)
    identity = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": config["manifest_sha256"],
        "lm_labels_sha256": sha256_file(root / config["lm_labels_csv"]),
        "gold_labels_sha256": sha256_file(root / config["gold_labels_csv"]),
        "image_kwargs": run_config.image_kwargs,
        "feature": "uniform_mean_cls_patch_v1",
        "study_uids": {
            split: [r.study_uid for r in rows] for split, rows in groups.items()
        },
        "preprocessing_source": {
            name: sha256_file(Path(__file__).parent / name)
            for name in ("preprocessing.py", "data.py", "lora.py")
        },
        "encoder_state_sha256": digest,
    }
    if baseline:
        identity["image_baseline"] = True
    # JSON stores tuples (notably slice_band) as lists; compare canonical values.
    identity = json.loads(json.dumps(identity, sort_keys=True))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if json.loads(str(cached["identity"])) != identity:
                raise ValueError(
                    "Feature cache identity mismatch; use a fresh --features path"
                )
            features = {split: cached[split] for split in groups}
        for split, matrix in features.items():
            if (
                matrix.shape != (len(groups[split]), 2 * backbone.config.hidden_size)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("Malformed feature cache")
        print(f"Reused validated features: {cache_path}", flush=True)
    else:
        features = {}
        for split, rows in groups.items():
            parts = []
            for start in range(0, len(rows), 64):
                parts.append(
                    extract_features(
                        backbone,
                        rows[start : start + 64],
                        run_config.image_kwargs,
                        device,
                        batch_size,
                        image_cache_dir,
                    )
                )
                print(
                    f"features {split}: {min(start + 64, len(rows))}/{len(rows)}",
                    flush=True,
                )
            features[split] = np.concatenate(parts)

        def save(temporary):
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle, identity=json.dumps(identity, sort_keys=True), **features
                )

        write_atomic(cache_path, save)
    if state_digest(backbone) != digest or any(
        p.requires_grad for p in backbone.parameters()
    ):
        raise AssertionError("Frozen encoder changed during feature extraction")
    print(json.dumps(population, indent=2), flush=True)
    return (
        features,
        targets,
        masks,
        groups,
        population,
        identity,
        backbone,
        run_config,
        adaptation,
    )


def paired_gold_interval(targets, reference, candidate, *, resamples, seed):
    """Paired study bootstrap of macro AUROC differences, conditional on fitted heads."""
    from scipy.stats import rankdata

    def macro(y, p):
        positive = y.sum(0)
        negative = len(y) - positive
        if (positive == 0).any() or (negative == 0).any():
            return None
        ranks = rankdata(p, axis=0)
        return float(
            np.mean(
                ((ranks * y).sum(0) - positive * (positive + 1) / 2)
                / (positive * negative)
            )
        )

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        index = rng.integers(0, len(targets), size=len(targets))
        a, b = (
            macro(targets[index], reference[index]),
            macro(targets[index], candidate[index]),
        )
        if a is not None and b is not None:
            deltas.append(b - a)
    if not deltas:
        raise ValueError("No valid paired bootstrap samples")
    return {
        "difference": macro(targets, candidate) - macro(targets, reference),
        "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "valid_resamples": len(deltas),
        "resamples": resamples,
        "unit": "study",
        "seed": seed,
    }


def run_experiment(config, output, prepared, *, device):
    import importlib.metadata
    import shutil
    import time

    from .submission import (
        assert_reload_matches,
        benchmark_head,
        export_member,
        load_member,
        predict_head,
        reload_tolerance,
        score_predictions,
    )
    from .supervised_head import fit_dense_head

    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    (
        features,
        targets,
        masks,
        groups,
        population,
        identity,
        backbone,
        run_config,
        adaptation,
    ) = prepared
    if (
        set(config["arms"]) != {"linear", "mlp"}
        or config["arms"]["linear"]
        or not config["arms"]["mlp"]
    ):
        raise ValueError("Declare a linear control and a nonempty MLP architecture")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Head seeds must be nonempty and unique")
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("Encoder must be frozen before fitting heads")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    selections, heads = {}, {}
    # Fit every head and freeze all epoch choices before looking at gold scores.
    for arm, hidden_dims in config["arms"].items():
        heads[arm], selections[arm] = [], []
        for seed in config["seeds"]:
            print(f"Training {arm}, seed={seed}, hidden={hidden_dims}", flush=True)
            head, selection = fit_dense_head(
                features["train"],
                targets["train"],
                masks["train"],
                features["validation"],
                targets["validation"],
                masks["validation"],
                hidden_dims=hidden_dims,
                seed=seed,
                device=device,
                **config["training"],
            )
            heads[arm].append(head)
            selections[arm].append(selection)
            path = output / f"{arm}-seed{seed}.pt"
            write_atomic(path, lambda temporary, head=head: torch.save(head, temporary))
            write_json(output / f"{arm}-seed{seed}-history.json", selection)
            print(
                f"{arm} seed={seed}: epoch={selection['best_epoch']}, "
                f"validation masked BCE={selection['validation_masked_bce']:.6f}",
                flush=True,
            )
    write_json(output / "selection.json", selections)
    encoder_after = state_digest(backbone)
    if encoder_after != identity["encoder_state_sha256"]:
        raise AssertionError("Encoder weights changed during supervised training")
    results, predictions = {}, {}
    for arm, arm_heads in heads.items():
        seed_scores = []
        for seed, head in zip(config["seeds"], arm_heads, strict=True):
            prediction = predict_head(features["gold"], head)
            seed_scores.append(
                {"seed": seed, **score_predictions(targets["gold"], prediction)}
            )
        predictions[arm] = predict_head(
            features["gold"], {"kind": "mean_heads_v1", "heads": arm_heads}
        )
        results[arm] = {
            **score_predictions(targets["gold"], predictions[arm]),
            "seeds": seed_scores,
        }
        print(
            f"Gold {arm} ensemble macro AUROC={results[arm]['macro_auroc']:.6f}",
            flush=True,
        )
    reference, folds = benchmark_head(
        features["gold"], targets["gold"], **config["gold_reference_probe"]
    )
    predictions["gold_only_logistic_oof"] = reference
    results["gold_only_logistic_oof"] = score_predictions(targets["gold"], reference)
    comparisons = {
        "mlp_minus_lm_linear": paired_gold_interval(
            targets["gold"],
            predictions["linear"],
            predictions["mlp"],
            **config["bootstrap"],
        ),
        "mlp_minus_gold_only_logistic_oof": paired_gold_interval(
            targets["gold"], reference, predictions["mlp"], **config["bootstrap"]
        ),
    }

    def save_predictions(path):
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                study_uids=np.asarray([r.study_uid for r in groups["gold"]]),
                targets=targets["gold"],
                reference_folds=folds,
                **predictions,
            )

    write_atomic(output / "gold-predictions.npz", save_predictions)
    bundle = output / "bundle"
    bundle.mkdir()
    head_ensemble = {"kind": "mean_heads_v1", "heads": heads["mlp"]}
    member_path = bundle / "member-00.pt"
    export_member(member_path, backbone, run_config, adaptation, head_ensemble)
    restored, payload = load_member(member_path, device)
    # Deployment reload is validated on real gold image pixels, independently of the feature cache.
    reloaded_features = extract_features(
        restored, groups["gold"][:1], run_config.image_kwargs, device, 1
    )
    expected = predictions["mlp"][:1]
    reloaded_predictions = predict_head(reloaded_features, payload["head"])
    assert_reload_matches(reloaded_predictions, expected, reload_tolerance(config))
    write_json(
        bundle / "bundle.json",
        {
            "schema_version": 1,
            "labels": LABELS,
            "feature": "uniform_mean_cls_patch",
            "ensemble": "mean_probability",
            "members": [
                {
                    "file": member_path.name,
                    "sha256": sha256_file(member_path),
                    "checkpoint_sha256": identity["checkpoint_sha256"],
                }
            ],
            "head": "seeded MLPs averaged after sigmoid; shared frozen encoder",
        },
    )
    shutil.copytree(
        root / "src" / "contrastive_learning",
        bundle / "src" / "contrastive_learning",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(
        root / "scripts" / "submission" / "predict_kaggle.py", bundle / "predict_kaggle.py"
    )
    shutil.copyfile(root / "requirements-kaggle.txt", bundle / "requirements.txt")
    shutil.copyfile(
        root / "notebooks" / "kaggle_submission.ipynb",
        output / "kaggle_submission.ipynb",
    )
    report = {
        "config": config,
        "population": population,
        "input_identity": identity,
        "encoder_frozen_verified": True,
        "encoder_after_sha256": encoder_after,
        "selections": {
            arm: [
                {key: value for key, value in item.items() if key != "history"}
                for item in items
            ]
            for arm, items in selections.items()
        },
        "results": results,
        "paired_comparisons": comparisons,
        "reload_max_probability_error": float(
            np.abs(reloaded_predictions - expected).max()
        ),
        "seconds_excluding_feature_extraction": time.perf_counter() - started,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "transformers", "scikit-learn")
        },
        "runtime_source_sha256": {
            str(p.relative_to(bundle)): sha256_file(p)
            for p in sorted((bundle / "src").rglob("*.py"))
        },
        "limitations": [
            "LM targets reflect report mention and extraction errors; unknown cells are unobserved, not negatives.",
            "Validation selects on pseudo labels from studies previously used for encoder validation.",
            "Gold is withheld from all head fitting and epoch selection, but repeatedly used in project development.",
            "Gold-only logistic OOF uses other gold cases as training labels; LM heads do not. Different supervision, not an architecture-only comparison.",
            "Paired bootstrap conditions on fitted heads; it is not patient-grouped or a new external test.",
            "Export uses selected train-only heads; no refit on validation or gold labels.",
        ],
    }
    write_json(output / "benchmark.json", report)
    print(f"MLP bundle: {bundle}\nComparisons: {json.dumps(comparisons)}", flush=True)
    return report
