"""Reload a finished frozen-tower LM head and verify it reproduces its saved LM-test and gold scores."""

import hashlib
import json

import numpy as np
import torch

from contrastive_learning.artifacts import sha256_file
from contrastive_learning.lm_proxy import load_test_population, masked_macro_auroc
from contrastive_learning.submission import predict_head


def load_head(path, checkpoint, root):
    """Read final reports and predictions, using only the existing LM-test feature cache."""
    required = ("benchmark.json", "gold-predictions.npz", "lm-proxy.json", "bundle/bundle.json")
    if not all((path / name).exists() for name in required):
        return None
    benchmark = json.loads((path / "benchmark.json").read_text())
    proxy = json.loads((path / "lm-proxy.json").read_text())
    sha = sha256_file(checkpoint)
    if not benchmark["encoder_frozen_verified"] or benchmark["input_identity"]["checkpoint_sha256"] != sha:
        raise ValueError(f"{path}: encoder provenance mismatch")
    cfg = benchmark["config"]
    _, targets, masks, population = load_test_population(
        root / cfg["cohort_manifest"], root / cfg["lm_labels_csv"], root / cfg["gold_labels_csv"])
    cache = root / "checkpoints/kaggle/lm-proxy-cache" / f"{sha[:16]}-test.npz"
    with np.load(cache, allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity"]))
        if (identity["checkpoint_sha256"] != sha or identity["study_uids"] != population["study_uids"]
                or identity["manifest_sha256"] != population["manifest_sha256"]
                or identity["image_kwargs"] != benchmark["input_identity"]["image_kwargs"]):
            raise ValueError(f"{path}: cached LM-test population or input mismatch")
        features = saved["features"].copy()
    if (proxy["population_identity"] != {key: value for key, value in population.items() if key != "study_uids"}
            or proxy["study_uids_sha256"] != hashlib.sha256("\n".join(population["study_uids"]).encode()).hexdigest()):
        raise ValueError(f"{path}: LM-test report provenance mismatch")
    bundle = json.loads((path / "bundle/bundle.json").read_text())
    predictions = []
    member_hashes = []
    for member in bundle["members"]:
        member_path = path / "bundle" / member["file"]
        member_sha = sha256_file(member_path)
        if member["checkpoint_sha256"] != sha or member["sha256"] != member_sha:
            raise ValueError(f"{path}: bundle checksum mismatch")
        payload = torch.load(member_path, map_location="cpu", weights_only=False)
        predictions.append(predict_head(features, payload["head"]))
        member_hashes.append(member_sha)
    lm_predictions = np.mean(predictions, axis=0)
    actual = masked_macro_auroc(lm_predictions, targets, masks)
    if not np.isclose(actual["macro_auroc"], proxy["macro_auroc"], atol=1e-7, rtol=0):
        raise ValueError(f"{path}: saved head does not reproduce LM-test score")
    with np.load(path / "gold-predictions.npz", allow_pickle=False) as saved:
        gold = {key: saved[key].copy() for key in saved.files}
    gold_actual = masked_macro_auroc(gold["mlp"], gold["targets"], np.ones_like(gold["targets"]))
    if not np.isclose(gold_actual["macro_auroc"], benchmark["results"]["mlp"]["macro_auroc"], atol=1e-7, rtol=0):
        raise ValueError(f"{path}: saved gold predictions do not reproduce score")
    return {"benchmark": benchmark, "proxy": proxy, "gold": gold, "lm_predictions": lm_predictions,
            "lm_targets": targets, "lm_masks": masks, "population": population,
            "signature": {"gold": sha256_file(path / "gold-predictions.npz"),
                          "members": member_hashes, "lm_features": sha256_file(cache),
                          "population": population}}
