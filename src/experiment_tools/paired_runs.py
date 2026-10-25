"""Paired LM-test and gold-58 comparison of finished fine-tuning runs against a matched run.

The ``scripts/analysis/compare_*`` trials all read the same artifacts (each run's
``predictions.npz``, ``report.json``, ``config.json`` and ``history.jsonl``), check that
every run scored the same studies, and report macro AUROC with study-bootstrap deltas
that share their resamples. Only the run table, the pairs and the labels of interest
differ, so those stay in the scripts and the shared mechanics live here.

``runs`` is always ``{key: (run_directory, title)}`` and the reference run, which fixes the
study populations every other run must match, is the key ``"run5"`` unless said otherwise.
"""

import json

import numpy as np

from contrastive_learning.lm_proxy import load_test_population, masked_macro_auroc, paired_intervals
from contrastive_learning.submission import LABELS

REFERENCE = "run5"


def load_predictions(runs, reference=REFERENCE):
    """Each run's saved ``predictions.npz`` arrays, after checking all runs scored the reference's studies."""
    saved = {}
    for name, (path, _) in runs.items():
        with np.load(path / "predictions.npz", allow_pickle=False) as data:
            saved[name] = {key: data[key].copy() for key in data.files}
    base = saved[reference]
    for name, data in saved.items():
        for key in ("lm_test_uids", "gold_uids", "gold_targets"):
            if not np.array_equal(data[key], base[key]):
                raise ValueError(f"{name}: {key} differs from run 5")
    return saved


def split_populations(root, runs, saved, reference=REFERENCE):
    """``{split: (targets, masks)}`` for LM-test (rebuilt from the cohort) and gold-58 (all labels observed)."""
    config = json.loads((runs[reference][0] / "config.json").read_text())
    _, targets, masks, identity = load_test_population(
        root / config["cohort_manifest"], root / config["lm_labels_csv"], root / config["gold_labels_csv"])
    if identity["study_uids"] != saved[reference]["lm_test_uids"].tolist():
        raise ValueError("LM-test population differs from saved predictions")
    gold_y = saved[reference]["gold_targets"]
    return {"lm_test": (targets, masks), "gold": (gold_y, np.ones_like(gold_y))}


def oriented_differences(raw, pairs, available=None):
    """``"first - second"`` entries of ``paired_intervals`` output, flipped when only the reverse was computed.

    With ``available``, pairs naming a run outside it are skipped instead of looked up.
    """
    out = {}
    for first, second in pairs:
        if available is not None and (first not in available or second not in available):
            continue
        key = f"{first} - {second}"
        if key in raw:
            out[key] = raw[key]
        else:
            flipped = raw[f"{second} - {first}"]
            out[key] = {"difference": -flipped["difference"], "ci95": [-flipped["ci95"][1], -flipped["ci95"][0]]}
    return out


def score_splits(predictions, populations, pairs, target_labels, available=None):
    """Point, per-label and paired scores per split, and paired scores for each label of interest alone.

    ``predictions`` is ``{split: {run key: predictions}}``. Returns the four JSON sections
    ``point``, ``paired``, ``per_label`` and ``paired_target_labels``, in that order.
    """
    result = {"point": {}, "paired": {}, "per_label": {}, "paired_target_labels": {}}
    for split, (y, m) in populations.items():
        scores = {name: masked_macro_auroc(p, y, m) for name, p in predictions[split].items()}
        result["point"][split] = {name: s["macro_auroc"] for name, s in scores.items()}
        result["per_label"][split] = {name: {label: s["per_label"].get(label) for label in LABELS}
                                      for name, s in scores.items()}
        result["paired"][split] = oriented_differences(paired_intervals(predictions[split], y, m), pairs, available)
        # Per target label: mask every other label so the "macro" is that label alone.
        result["paired_target_labels"][split] = {}
        for label in target_labels:
            only = np.zeros_like(m)
            only[:, LABELS.index(label)] = m[:, LABELS.index(label)]
            result["paired_target_labels"][split][label] = oriented_differences(
                paired_intervals(predictions[split], y, only), pairs, available)
    return result


def pick(source, keys):
    """The ``keys`` present in ``source``, in order."""
    return {k: source[k] for k in keys if k in source}


def run_records(root, runs, extras, seconds=True):
    """One summary per run: selection, history and the fields ``extras(report, config)`` adds.

    ``seconds`` includes the report's wall-clock time, ahead of the extras.
    """
    records = []
    for name, (path, title) in runs.items():
        report = json.loads((path / "report.json").read_text())
        run_config = json.loads((path / "config.json").read_text())
        history_path = path / "history.jsonl"
        history = ([json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
                   if history_path.exists() else [])
        record = {
            "key": name, "title": title, "dir": str(path.relative_to(root)),
            "best_epoch": report["best_epoch"], "validation_masked_bce": report["validation_masked_bce"],
            "epochs_completed": len(history), "history": history,
        }
        if seconds:
            record["seconds"] = report.get("seconds")
        records.append({**record, **extras(report, run_config)})
    return records


def print_report(result, widths=(28, 50, 50), label_pair=None):
    """Print point scores, paired deltas and the per-label deltas that ``label_pair(key)`` selects (default all).

    ``widths`` pads the run name, the pair name and the per-label pair name.
    """
    name_width, pair_width, label_pair_width = widths
    for split in result["point"]:
        print(f"\n{split}")
        for name, value in result["point"][split].items():
            print(f"  {name:{name_width}s} {value:.4f}")
        for key, value in result["paired"][split].items():
            low, high = value["ci95"]
            print(f"  {key:{pair_width}s} {value['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
        for label, pairs in result["paired_target_labels"][split].items():
            for key, value in pairs.items():
                if label_pair is None or label_pair(key):
                    low, high = value["ci95"]
                    print(f"  {label:18s} {key:{label_pair_width}s} {value['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
