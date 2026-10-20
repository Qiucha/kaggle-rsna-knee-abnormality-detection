#!/usr/bin/env python3
"""Build digest/size-matched-holdout-v1.{json,html}: the size-matched DINOv2-base + E5-small holdout run against the DINOv2-small one.

Reads both runs' config.json and metrics.jsonl under checkpoints/contrastive, the cached gold-probe features
digest/{lora,size-matched}-holdout-features.npz and configs/downstream/gold-probe-size-matched-holdout-v1.json, scores the gold
probe on them and writes the paired study-bootstrap comparison and the page.

    .venv/bin/python scripts/digests/build_size_matched_digest.py
"""

from __future__ import annotations

import html
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from contrastive_learning.artifacts import sha256_file
from contrastive_learning.checkpoints import verified_best_epoch
from experiment_tools.gold_probe import (
    load_gold_labels,
    paired_bootstrap,
    resolve_population,
    score_arm,
)

ROOT = Path(__file__).resolve().parents[2]
OLD_RUN = (
    ROOT
    / "checkpoints/contrastive/both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026"
)
NEW_RUN = (
    ROOT
    / "checkpoints/contrastive/size-matched-holdout-v1/dinov2-base-e5-small-depth6-r64-seed2026"
)
OLD_FEATURES = ROOT / "digest/lora-holdout-features.npz"
NEW_FEATURES = ROOT / "digest/size-matched-holdout-features.npz"
PROBE_CONFIG = ROOT / "configs/downstream/gold-probe-size-matched-holdout-v1.json"
RESULT_JSON = ROOT / "digest/size-matched-holdout-v1.json"
RESULT_HTML = ROOT / "digest/size-matched-holdout-v1.html"
OLD_KEY = (
    "unexposed_lora_6blk_r64/holdout-both-lora-depth6-r64-seed2026|uniform_mean_768"
)
OLD_FLOOR_KEY = "before_cl/dinov2-unadapted|uniform_mean_768"
NEW_FLOOR_KEY = "before_cl/dinov2-unadapted|uniform_mean_1536"
NEW_ARM = "size_matched_lora_6blk_r64/dinov2-base-e5-small-depth6-r64-seed2026"


def history(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [row["epoch"] for row in rows] != list(range(1, 21)):
        raise ValueError(f"Expected completed 20-epoch history: {path}")
    return rows


def polyline(
    rows: list[dict],
    field: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    minimum: float,
    maximum: float,
) -> str:
    span = max(maximum - minimum, 1e-9)
    return " ".join(
        f"{x0 + (row['epoch'] - 1) * width / 19:.1f},{y0 + height - (row[field] - minimum) * height / span:.1f}"
        for row in rows
    )


def chart(
    old: list[dict], new: list[dict], title: str, field: str, *, percent: bool = False
) -> str:
    values = [row[field] * (100 if percent else 1) for row in old + new]
    minimum, maximum = min(values), max(values)
    pad = max((maximum - minimum) * 0.12, 0.01)
    minimum -= pad
    maximum += pad

    def points(rows: list[dict]) -> str:
        scaled = [{**row, field: row[field] * (100 if percent else 1)} for row in rows]
        return polyline(scaled, field, 49, 20, 610, 172, minimum, maximum)

    label = "%" if percent else ""
    return f'''<figure><figcaption>{html.escape(title)}</figcaption>
      <svg viewBox="0 0 690 230" role="img" aria-label="{html.escape(title)} over 20 epochs">
      <line x1="49" y1="192" x2="659" y2="192" class="axis"/>
      <line x1="49" y1="20" x2="49" y2="192" class="axis"/>
      <text x="45" y="210">1</text><text x="645" y="210">20</text>
      <text x="4" y="29">{maximum:.2f}{label}</text><text x="4" y="192">{minimum:.2f}{label}</text>
      <polyline points="{points(old)}" class="old"/>
      <polyline points="{points(new)}" class="new"/></svg></figure>'''


def main() -> None:
    config = json.loads(PROBE_CONFIG.read_text())
    old_train_config = json.loads((OLD_RUN / "config.json").read_text())
    new_train_config = json.loads((NEW_RUN / "config.json").read_text())
    # The reference predates these serialized, inactive defaults. They do not
    # turn on DeCLIP or alter the temperature within the observed run range.
    inactive_defaults = {
        "declip_image_ss_weight",
        "declip_mv_weight",
        "declip_nn_weight",
        "declip_queue_size",
        "declip_text_mask_prob",
        "freeze_temperature",
        "max_inverse_temperature",
        "use_declip",
    }
    intentionally_changed = {
        "image_model",
        "image_model_revision",
        "text_model",
        "text_model_revision",
        "output_dir",
    }
    excluded = inactive_defaults | intentionally_changed
    if {k: v for k, v in old_train_config.items() if k not in excluded} != {
        k: v for k, v in new_train_config.items() if k not in excluded
    }:
        raise ValueError("An active training setting differs beyond the backbone swap")
    old, new = history(OLD_RUN / "metrics.jsonl"), history(NEW_RUN / "metrics.jsonl")
    uids, labels = load_gold_labels(ROOT / config["gold_labels_csv"], config["labels"])
    records, gold, population = resolve_population(
        ROOT / config["cohort_manifest"], uids, labels, ("excluded",)
    )
    expected = [record.study_uid for record in records]
    old_features = np.load(OLD_FEATURES, allow_pickle=True)
    new_features = np.load(NEW_FEATURES, allow_pickle=True)
    if (
        list(old_features["study_uids"]) != expected
        or list(new_features["study_uids"]) != expected
    ):
        raise ValueError(
            "Old and new features must use exactly the same 56 study order"
        )
    new_key = NEW_ARM + "|uniform_mean_1536"
    if old_features[OLD_KEY].shape != (56, 768) or new_features[new_key].shape != (
        56,
        1536,
    ):
        raise ValueError("Unexpected feature dimensions")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        old_score = score_arm(
            old_features[OLD_KEY], gold, config["labels"], config["probe"]
        )
        new_score = score_arm(
            new_features[new_key], gold, config["labels"], config["probe"]
        )
        old_floor = score_arm(
            old_features[OLD_FLOOR_KEY], gold, config["labels"], config["probe"]
        )
        new_floor = score_arm(
            new_features[NEW_FLOOR_KEY], gold, config["labels"], config["probe"]
        )
    if abs(old_score["macro_auroc"] - 0.7593615004455258) > 1e-9:
        raise ValueError("Baseline did not reproduce the published gold probe")
    if abs(old_floor["macro_auroc"] - 0.6439917511777699) > 1e-9:
        raise ValueError(
            "Unadapted small backbone did not reproduce the published floor"
        )
    paired = paired_bootstrap(
        gold,
        config["labels"],
        old_score["_oof_scores"],
        new_score["_oof_scores"],
        config["bootstrap"],
    )

    def best(rows: list[dict]) -> dict:
        return max(
            rows,
            key=lambda row: (row["mean_r@1"], -row["validation_loss"], -row["epoch"]),
        )

    old_best, new_best = best(old), best(new)
    if (
        verified_best_epoch(OLD_RUN / "metrics.jsonl", OLD_RUN / "best.pt")["epoch"]
        != old_best["epoch"]
    ):
        raise ValueError(
            "Existing best checkpoint does not match its validation history"
        )
    if (
        verified_best_epoch(NEW_RUN / "metrics.jsonl", NEW_RUN / "best.pt")["epoch"]
        != new_best["epoch"]
    ):
        raise ValueError(
            "Swapped best checkpoint does not match its validation history"
        )
    old_low = min(old, key=lambda row: row["validation_loss"])
    new_low = min(new, key=lambda row: row["validation_loss"])
    result = {
        "protocol": "One paired backbone swap; same 56 excluded gold studies, fixed eight-fold OOF logistic probe, same seed and hyperparameters",
        "control_audit": {
            "active_training_settings_equal_except_backbones": True,
            "inactive_defaults_newly_serialized": sorted(inactive_defaults),
            "changed_fields": sorted(intentionally_changed),
        },
        "artifacts": {
            "existing_checkpoint": str(OLD_RUN / "best.pt"),
            "existing_checkpoint_sha256": sha256_file(OLD_RUN / "best.pt"),
            "size_matched_checkpoint": str(NEW_RUN / "best.pt"),
            "size_matched_checkpoint_sha256": sha256_file(NEW_RUN / "best.pt"),
            "training_config": str(
                ROOT / "configs/contrastive/size-matched-holdout-v1.json"
            ),
            "probe_config": str(PROBE_CONFIG),
        },
        "population": population,
        "models": {
            "existing": {
                "image": "facebook/dinov2-small",
                "text": "intfloat/multilingual-e5-base",
                "feature_dimensions": 768,
                "image_backbone_parameters": 22056576,
                "text_backbone_parameters": 278043648,
            },
            "size_matched": {
                "image": "facebook/dinov2-base",
                "text": "intfloat/multilingual-e5-small",
                "feature_dimensions": 1536,
                "image_backbone_parameters": 86580480,
                "text_backbone_parameters": 117653760,
            },
        },
        "retrieval": {
            "existing_best_epoch": old_best["epoch"],
            "existing_best_mean_r1": old_best["mean_r@1"],
            "size_matched_best_epoch": new_best["epoch"],
            "size_matched_best_mean_r1": new_best["mean_r@1"],
            "at_best_checkpoint": {
                key: {"existing": old_best[key], "size_matched": new_best[key]}
                for key in (
                    "image_to_text_r@1",
                    "text_to_image_r@1",
                    "image_to_text_r@5",
                    "text_to_image_r@5",
                    "image_to_text_r@10",
                    "text_to_image_r@10",
                )
            },
        },
        "loss_trajectory": {
            "existing": {
                "minimum_validation_epoch": old_low["epoch"],
                "minimum_validation_loss": old_low["validation_loss"],
                "final_validation_loss": old[-1]["validation_loss"],
                "final_training_loss": old[-1]["train_loss"],
            },
            "size_matched": {
                "minimum_validation_epoch": new_low["epoch"],
                "minimum_validation_loss": new_low["validation_loss"],
                "final_validation_loss": new[-1]["validation_loss"],
                "final_training_loss": new[-1]["train_loss"],
            },
        },
        "gold": {
            "unadapted_small_macro_auroc": old_floor["macro_auroc"],
            "unadapted_base_macro_auroc": new_floor["macro_auroc"],
            "existing_macro_auroc": old_score["macro_auroc"],
            "size_matched_macro_auroc": new_score["macro_auroc"],
            "difference": new_score["macro_auroc"] - old_score["macro_auroc"],
            "paired_bootstrap": paired,
            "per_label": {
                label: {
                    "existing": old_score["per_label"][label]["auroc"],
                    "size_matched": new_score["per_label"][label]["auroc"],
                    "difference": new_score["per_label"][label]["auroc"]
                    - old_score["per_label"][label]["auroc"],
                    "positives": old_score["per_label"][label]["positives"],
                }
                for label in config["labels"]
            },
        },
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    delta = result["gold"]["difference"]
    ci = paired["macro_auroc_difference_ci95"]
    evidence = (
        "The paired interval excludes zero in favour of the size-matched run."
        if ci[0] > 0
        else "The paired interval excludes zero in favour of the existing run."
        if ci[1] < 0
        else "The paired interval includes zero, so this sample does not establish a gold-label advantage."
    )
    rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{v['positives']}</td><td>{v['existing']:.3f}</td><td>{v['size_matched']:.3f}</td><td class='{('up' if v['difference'] > 0 else 'down')}'>{v['difference']:+.3f}</td></tr>"
        for label, v in result["gold"]["per_label"].items()
    )
    all_charts = "".join(
        [
            chart(old, new, "Training loss", "train_loss"),
            chart(old, new, "Validation loss", "validation_loss"),
            chart(old, new, "Validation mean Recall@1", "mean_r@1", percent=True),
        ]
    )
    recall_rows = "".join(
        f"<tr><th>{html.escape(key.replace('_', ' ').replace('image to text', 'MRI → report').replace('text to image', 'Report → MRI'))}</th><td>{values['existing'] * 100:.1f}%</td><td>{values['size_matched'] * 100:.1f}%</td></tr>"
        for key, values in result["retrieval"]["at_best_checkpoint"].items()
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Does matching image and text tower size help?</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#142235;background:#f5f8fc;margin:0}}main{{max-width:1080px;margin:auto;padding:44px 24px 80px}}
h1{{font-size:clamp(2rem,4vw,3.2rem);line-height:1.1;margin:.2em 0}}h2{{margin-top:2.3em}}p{{max-width:82ch}}.lede{{font-size:1.2rem;color:#456}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:28px 0}}.card,figure,section.note{{background:white;border:1px solid #dce5ef;border-radius:15px;padding:18px;box-shadow:0 5px 22px #19324a0a}}
.card b{{display:block;font-size:1.8rem}}.muted{{color:#526678}}.old{{stroke:#747f99}}.new{{stroke:#e26b35}}.axis{{stroke:#a8b7c8}}polyline{{fill:none;stroke-width:3.4;stroke-linecap:round;stroke-linejoin:round}}figcaption{{font-weight:700;margin-bottom:8px}}svg{{width:100%;height:auto}}svg text{{font-size:13px;fill:#53687d}}
.legend span{{display:inline-block;width:1.2em;height:.22em;vertical-align:middle;margin:0 .4em 0 1em}}.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:9px 12px;border-bottom:1px solid #e5ebf2;text-align:right}}th:first-child{{text-align:left}}thead{{background:#e8eff8}}.up{{color:#14704a}}.down{{color:#a83a39}}.tablewrap{{overflow-x:auto}}code{{background:#e8eff8;padding:2px 5px;border-radius:4px}}a{{color:#1857a0}}
</style></head><body><main><p class="muted">Gold holdout experiment · 24 September 2026</p><h1>Does matching the tower sizes help?</h1>
<p class="lede">A 20-epoch, one-seed comparison of the selected DINOv2-small + multilingual-e5-base encoder with DINOv2-base + multilingual-e5-small. All training and probe settings were held fixed.</p>
<p>Backbone parameter counts move from 22.1M image / 278.0M text (12.6:1) to 86.6M image / 117.7M text (1.36:1). These are closer in size, not exactly equal. Counts come from the pinned local model architectures and exclude the contrastive projections and LoRA adapters. Model sources: <a href="https://huggingface.co/facebook/dinov2-base">DINOv2-base</a> and <a href="https://huggingface.co/intfloat/multilingual-e5-small">E5-small</a>.</p>
<div class="cards"><div class="card">Existing gold macro AUROC<b>{old_score["macro_auroc"]:.3f}</b>56 unseen gold studies</div>
<div class="card">Swapped gold macro AUROC<b>{new_score["macro_auroc"]:.3f}</b>same 56 studies and folds</div>
<div class="card">Paired difference<b>{delta:+.3f}</b>95% bootstrap interval {ci[0]:+.3f} to {ci[1]:+.3f}</div>
<div class="card">Best validation Recall@1<b>{old_best["mean_r@1"] * 100:.1f}% → {new_best["mean_r@1"] * 100:.1f}%</b>epochs {old_best["epoch"]} and {new_best["epoch"]}</div></div>
<section class="note"><h2>Conclusion</h2><p><strong>Matching tower size was not necessary for the better gold-label result in this comparison.</strong> The existing size-imbalanced pair scores {old_score["macro_auroc"]:.3f} versus {new_score["macro_auroc"]:.3f} for the closer-sized pair ({delta:+.3f} AUROC; paired 95% interval {ci[0]:+.3f} to {ci[1]:+.3f}). {html.escape(evidence)} Yet the swapped pair improves validation retrieval by {(new_best["mean_r@1"] - old_best["mean_r@1"]) * 100:+.2f} percentage points. Better image–report matching did not yield better pathology readout here.</p>
<p><strong>The larger text encoder overfitting the smaller image encoder is not demonstrated.</strong> Both runs show falling training loss and validation loss rising after its minimum; the closer-sized pair did not remove that pattern. This two-model swap cannot assign the cause to text capacity: the image backbone and readout width changed too. A four-cell image-size × text-size test would isolate those effects. This single-seed gold comparison does not establish a universal size rule.</p></section>
<h2>Learning curves</h2><p class="legend"><span style="background:#747f99"></span> Existing: DINOv2-small + E5-base <span style="background:#e26b35"></span> Swapped: DINOv2-base + E5-small</p><div class="charts">{all_charts}</div>
<p>The existing run's validation loss bottomed at {old_low["validation_loss"]:.3f} (epoch {old_low["epoch"]}) and finished at {old[-1]["validation_loss"]:.3f}, while training loss finished at {old[-1]["train_loss"]:.3f}. The swapped run bottomed at {new_low["validation_loss"]:.3f} (epoch {new_low["epoch"]}) and finished at {new[-1]["validation_loss"]:.3f}, while training loss finished at {new[-1]["train_loss"]:.3f}. These trajectories can indicate objective overfitting, but the tower responsible is unidentified.</p>
<p class="muted">Training loss is an average of in-batch optimization steps; validation loss uses the held-out retrieval set. The curves show trajectories, but their absolute train–validation gap is not a calibrated overfitting measure.</p>
<h2>Validation retrieval at each selected checkpoint</h2><p>Each row is recall over the same 433 validation pairs; the checkpoints were selected by mean bidirectional Recall@1.</p>
<div class="tablewrap"><table><thead><tr><th>Metric</th><th>Existing</th><th>Swapped</th></tr></thead><tbody>{recall_rows}</tbody></table></div>
<h2>Gold-label readout</h2><p>Each pathology uses the same standardized logistic regression, eight stratified folds, and one out-of-fold prediction per study. The macro AUROC is an unweighted average across twelve findings. {html.escape(evidence)}</p>
<p>Before contrastive adaptation, unadapted DINOv2-small scores {old_floor["macro_auroc"]:.3f} and unadapted DINOv2-base scores {new_floor["macro_auroc"]:.3f} under the same probe. Adaptation improves both image towers, but the selected smaller image backbone still leads on the gold labels.</p>
<div class="tablewrap"><table><thead><tr><th>Finding</th><th>Positive cases</th><th>Existing AUC</th><th>Swapped AUC</th><th>Δ AUC</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>What was held fixed</h2><p>Gold-excluded cohort (3,478 training and 433 validation pairs), full original reports, 512 text tokens, E5 mean pooling with <code>query: </code> prefix, 130 mm MRI crop, batch size 32, 20 epochs, seed 2026, six LoRA blocks per tower, rank 64, alpha 128, optimizer rates, and retrieval-based checkpoint selection. The 56 gold cases were excluded from encoder training and selection; two locked gold test cases were not used.</p>
<p class="muted">This is a single-seed development comparison on a repeatedly examined gold cohort. The bootstrap resamples studies, not patients; shared patient identity cannot be verified. The new model has not been tested on the competition leaderboard.</p>
<p>Artifacts: <a href="size-matched-holdout-v1.json">comparison JSON</a> · <a href="../configs/contrastive/size-matched-holdout-v1.json">training config</a> · <a href="../configs/downstream/gold-probe-size-matched-holdout-v1.json">probe config</a></p></main></body></html>"""
    RESULT_HTML.write_text(page)
    print(f"Wrote {RESULT_JSON} and {RESULT_HTML}")


if __name__ == "__main__":
    main()
