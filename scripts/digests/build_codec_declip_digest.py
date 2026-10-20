"""Build the codec-declip-v1 digest: codec-pipeline reference replicate and three DeCLIP arms.

Reads each finished contrastive run and its 58-case LM heads (arms whose heads are not done
yet are listed as pending), verifies the shared gold and LM-test populations, and writes the
results page, its JSON, and the curve figures. Rerun after every arm finishes.

The narrative "Answer" section is read from digest/codec-declip-v1-answer.md when present,
so the tables stay generated while the interpretation is written by hand.

    .venv/bin/python scripts/digests/build_codec_declip_digest.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.artifacts import sha256_file  # noqa: E402
from contrastive_learning.lm_proxy import masked_macro_auroc, paired_intervals  # noqa: E402
from contrastive_learning.lm_supervision import paired_gold_interval  # noqa: E402
from contrastive_learning.submission import LABELS  # noqa: E402
from experiment_tools.digest_arms import (  # noqa: E402
    SEEDS, contrastive, delta, f4, head_histories, lm_test_predictions, pct,
)

CONTRASTIVE = ROOT / "checkpoints/contrastive"
KAGGLE = ROOT / "checkpoints/kaggle"
DIGEST = ROOT / "digest"
NAME = "codec-declip-v1"
PAGE = DIGEST / f"{NAME}.md"
ANSWER = DIGEST / f"{NAME}-answer.md"
REFERENCE, REPLICATE = "reference", "replicate"

# Pretrained image towers under the same 58-case head protocol (config "image_baseline": true):
# the anchor for each arm's per-label change. key: (label, head run dir)
UNADAPTED = {
    "DINOv2-small": ("Unadapted DINOv2-small", "lm-mlp-codec-declip-58-unadapted-dinov2-small"),
    "RAD-DINO": ("Unadapted RAD-DINO", "lm-mlp-codec-declip-58-unadapted-rad-dino"),
}

# key: (image, text, objective, contrastive run dir, head run dir, colour, linestyle)
ARMS = {
    REFERENCE: ("DINOv2-small", "E5-base", "CLIP", "both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026",
                "lm-mlp-tower-size-58-dinov2-small-e5-base", "#222222", "--"),
    REPLICATE: ("DINOv2-small", "E5-base", "CLIP · codec pipeline",
                f"{NAME}/dinov2-small-e5-base-512-codec-seed2026",
                "lm-mlp-codec-declip-58-dinov2-small-e5-base-512-codec", "#8a8a8a", "--"),
    "rad_dino": ("RAD-DINO", "E5-base", "CLIP", "next-model-pairs-v1/rad-dino-e5-base-512-seed2026",
                 "lm-mlp-next-pairs-58-rad-dino-e5-base-512", "#c9a0c6", ":"),
    "bge_m3_512": ("DINOv2-small", "BGE-M3 · 512", "CLIP", "next-model-pairs-v1/dinov2-small-bge-m3-512-seed2026",
                   "lm-mlp-next-pairs-58-dinov2-small-bge-m3-512", "#e8b98f", ":"),
    "declip_dinov2": ("DINOv2-small", "E5-base", "DeCLIP", f"{NAME}/declip-dinov2-small-e5-base-512-seed2026",
                      "lm-mlp-codec-declip-58-declip-dinov2-small-e5-base-512", "#3f9a6b", "-"),
    "declip_rad_e5": ("RAD-DINO", "E5-base", "DeCLIP", f"{NAME}/declip-rad-dino-e5-base-512-seed2026",
                      "lm-mlp-codec-declip-58-declip-rad-dino-e5-base-512", "#a0569b", "-"),
    "declip_rad_bge": ("RAD-DINO", "BGE-M3 · 512", "DeCLIP", f"{NAME}/declip-rad-dino-bge-m3-512-seed2026",
                       "lm-mlp-codec-declip-58-declip-rad-dino-bge-m3-512", "#c0392b", "-"),
}
# Each DeCLIP arm against the plain-CLIP arm with the same towers.
CLIP_COUNTERPART = {"declip_dinov2": REFERENCE, "declip_rad_e5": "rad_dino", "declip_rad_bge": None}


def label(key):
    image, text, objective, *_ = ARMS[key]
    return f"{image} + {text} · {objective}"


def finished(key):
    run, head = ARMS[key][3], ARMS[key][4]
    return (CONTRASTIVE / run / "best.pt").exists() and (KAGGLE / head / "lm-proxy.json").exists()


def main():
    with (ROOT / "data/gold_label.csv").open(newline="") as handle:
        source_gold = {row["StudyInstanceUID"] for row in csv.DictReader(handle)}
    done = [k for k in ARMS if finished(k)]
    pending = [k for k in ARMS if k not in done]
    arms, gold, lm_test, lm_population = {}, {}, {}, None
    for key in done:
        run, head = ARMS[key][3], ARMS[key][4]
        run_dir, head_dir = CONTRASTIVE / run, KAGGLE / head
        report = json.loads((head_dir / "benchmark.json").read_text())
        if not report["encoder_frozen_verified"]:
            raise ValueError(f"Encoder verification failed for {key}")
        if report["input_identity"]["checkpoint_sha256"] != sha256_file(run_dir / "best.pt"):
            raise ValueError(f"Head for {key} was not fitted on {run}/best.pt")
        uids = report["input_identity"]["study_uids"]
        if source_gold & (set(uids["train"]) | set(uids["validation"])):
            raise ValueError(f"Gold study entered head fitting for {key}")
        with np.load(head_dir / "gold-predictions.npz", allow_pickle=False) as loaded:
            gold[key] = {name: loaded[name] for name in loaded.files}
        if set(gold[key]["study_uids"]) != source_gold or len(source_gold) != 58:
            raise ValueError(f"Scored gold studies for {key} differ from the 58 cleaned.csv cases")
        lm_test[key] = {}
        for kind in ("linear", "mlp"):
            prediction, targets, masks, population = lm_test_predictions(run_dir, head_dir, kind)
            if lm_population is None:
                lm_population, lm_targets, lm_masks = population, targets, masks
            elif population != lm_population:
                raise ValueError(f"LM-test population differs for {key}")
            lm_test[key][kind] = prediction
        proxy = json.loads((head_dir / "lm-proxy.json").read_text())["macro_auroc"]
        recomputed = masked_macro_auroc(lm_test[key]["mlp"], lm_targets, lm_masks)["macro_auroc"]
        if abs(proxy - recomputed) > 1e-6:
            raise ValueError(f"Seed-averaged MLP does not reproduce the bundle LM-test score for {key}")
        manifest = run_dir / "run_manifest.json"
        arms[key] = {
            "image": ARMS[key][0],
            "text": ARMS[key][1],
            "objective": ARMS[key][2],
            "contrastive_run": f"checkpoints/contrastive/{run}",
            "head_run": f"checkpoints/kaggle/{head}",
            "git": json.loads(manifest.read_text()).get("git") if manifest.exists() else None,
            "checkpoint_sha256": report["input_identity"]["checkpoint_sha256"],
            "contrastive": contrastive(run_dir),
            "gold_macro_auroc": {k: report["results"][k]["macro_auroc"] for k in ("linear", "mlp")},
            "gold_per_label_auroc": {k: {name: v["auroc"] for name, v in report["results"][k]["per_label"].items()}
                                     for k in ("linear", "mlp")},
            "lm_test_macro_auroc": {k: masked_macro_auroc(lm_test[key][k], lm_targets, lm_masks)["macro_auroc"]
                                    for k in ("linear", "mlp")},
            "head_selection": {
                kind: {
                    "epochs": [h["best_epoch"] for h in head_histories(head_dir, kind)],
                    "validation_masked_bce": [
                        next(r for r in h["history"] if r["epoch"] == h["best_epoch"])["validation_masked_bce"]
                        for h in head_histories(head_dir, kind)
                    ],
                }
                for kind in ("linear", "mlp")
            },
        }
    reference = gold[REFERENCE]
    for key, item in gold.items():
        if not (np.array_equal(reference["study_uids"], item["study_uids"])
                and np.array_equal(reference["targets"], item["targets"])):
            raise ValueError(f"Gold study order or targets differ for {key}")

    def gold_delta(key, base):
        if base not in gold or key == base:
            return None
        return {kind: paired_gold_interval(reference["targets"], gold[base][kind], gold[key][kind],
                                           resamples=5000, seed=2030) for kind in ("linear", "mlp")}

    def lm_delta(key, base):
        if base not in lm_test or key == base:
            return None
        return {kind: paired_intervals({key: lm_test[key][kind], base: lm_test[base][kind]},
                                       lm_targets, lm_masks)[f"{key} - {base}"] for kind in ("linear", "mlp")}

    for key in arms:
        arms[key]["gold_delta_vs_reference"] = gold_delta(key, REFERENCE)
        arms[key]["lm_test_delta_vs_reference"] = lm_delta(key, REFERENCE)
        arms[key]["gold_delta_vs_replicate"] = gold_delta(key, REPLICATE)
        arms[key]["lm_test_delta_vs_replicate"] = lm_delta(key, REPLICATE)
        counterpart = CLIP_COUNTERPART.get(key)
        if counterpart:
            arms[key]["clip_counterpart"] = counterpart
            arms[key]["gold_delta_vs_clip_counterpart"] = gold_delta(key, counterpart)
            arms[key]["lm_test_delta_vs_clip_counterpart"] = lm_delta(key, counterpart)

    unadapted = {}
    for image, (name, head) in UNADAPTED.items():
        head_dir = KAGGLE / head
        if not (head_dir / "benchmark.json").exists():
            continue
        report = json.loads((head_dir / "benchmark.json").read_text())
        if not report["config"].get("image_baseline") or not report["input_identity"].get("image_baseline"):
            raise ValueError(f"{head} is not an image_baseline head run")
        uids = report["input_identity"]["study_uids"]
        if source_gold & (set(uids["train"]) | set(uids["validation"])):
            raise ValueError(f"Gold study entered head fitting for {head}")
        adapted = next((k for k in arms if ARMS[k][0] == image), None)
        if adapted and report["input_identity"]["encoder_state_sha256"] == json.loads(
                (KAGGLE / ARMS[adapted][4] / "benchmark.json").read_text())["input_identity"]["encoder_state_sha256"]:
            raise ValueError(f"{head} features come from an adapted encoder")
        with np.load(head_dir / "gold-predictions.npz", allow_pickle=False) as loaded:
            base = {name_: loaded[name_] for name_ in loaded.files}
        if not (np.array_equal(reference["study_uids"], base["study_uids"])
                and np.array_equal(reference["targets"], base["targets"])):
            raise ValueError(f"Gold study order or targets differ for {head}")
        unadapted[image] = {
            "label": name,
            "head_run": f"checkpoints/kaggle/{head}",
            "gold_macro_auroc": {k: report["results"][k]["macro_auroc"] for k in ("linear", "mlp")},
            "gold_per_label_auroc": {k: {n: v["auroc"] for n, v in report["results"][k]["per_label"].items()}
                                     for k in ("linear", "mlp")},
        }
        for key in arms:
            if ARMS[key][0] == image:
                arms[key]["unadapted"] = image
                arms[key]["gold_delta_vs_unadapted"] = {
                    kind: paired_gold_interval(reference["targets"], base[kind], gold[key][kind],
                                               resamples=5000, seed=2030) for kind in ("linear", "mlp")}

    noise = None
    if REPLICATE in arms:
        a = {r["epoch"]: r for r in arms[REFERENCE]["contrastive"]["history"]}
        b = {r["epoch"]: r for r in arms[REPLICATE]["contrastive"]["history"]}
        common = sorted(set(a) & set(b))
        noise = {
            "epochs_compared": len(common),
            "median_abs_epoch_delta_mean_r1": float(np.median([abs(a[e]["mean_r@1"] - b[e]["mean_r@1"]) for e in common])),
            "max_abs_epoch_delta_mean_r1": float(max(abs(a[e]["mean_r@1"] - b[e]["mean_r@1"]) for e in common)),
            "median_abs_epoch_delta_validation_loss": float(np.median(
                [abs(a[e]["validation_loss"] - b[e]["validation_loss"]) for e in common])),
            "epoch1_abs_delta_train_loss": abs(a[1]["train_loss"] - b[1]["train_loss"]) if 1 in common else None,
            "selected_mean_r1_gap": arms[REPLICATE]["contrastive"]["mean_r1"] - arms[REFERENCE]["contrastive"]["mean_r1"],
        }

    summary = {
        "experiment": NAME,
        "matrix": f"configs/contrastive/{NAME}.json",
        "source": "fix/kaggle-dicom-codecs @ 35dfa0e (run from the codec worktree)",
        "gold_studies": 58,
        "lm_test_studies": int(len(lm_targets)),
        "contrastive_seed": 2026,
        "head_seeds": list(SEEDS),
        "gold_interval": "paired study bootstrap, 5,000 resamples, seed 2030",
        "lm_test_interval": "paired study bootstrap, 2,000 resamples, seed 2030",
        "pending_arms": [label(k) for k in pending],
        "replicate_noise": noise,
        "gold_positives": dict(zip(LABELS, reference["targets"].sum(0).astype(int).tolist())),
        "unadapted": unadapted,
        "arms": {k: {**v, "contrastive": {m: x for m, x in v["contrastive"].items() if m != "history"}}
                 for k, v in arms.items()},
    }
    (DIGEST / f"{NAME}.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    panels = (("train_loss", "Training loss (DeCLIP arms: composite MV + NN + SS)"),
              ("validation_loss", "Validation full-pool InfoNCE (comparable)"),
              ("image_to_text_r@1", "Image → text validation R@1"), ("text_to_image_r@1", "Text → image validation R@1"))
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for key in arms:
            *_, colour, style = ARMS[key]
            rows = arms[key]["contrastive"]["history"]
            ax.plot([r["epoch"] for r in rows], [r[metric] for r in rows], color=colour, ls=style,
                    lw=1.8 if key in (REFERENCE, REPLICATE) else 1.3, label=label(key))
            best = next(r for r in rows if r["epoch"] == arms[key]["contrastive"]["selected_epoch"])
            ax.scatter(best["epoch"], best[metric], color=colour, s=16, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Contrastive epoch")
        ax.grid(alpha=.16)
        if "r@1" in metric:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.text(.5, .99, "Contrastive training, seed 2026; dots mark the checkpoint selected by validation mean R@1",
             ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(DIGEST / f"{NAME}-contrastive-curves.svg")
    plt.close(fig)

    for kind, name in (("mlp", "MLP"), ("linear", "Linear (logistic)")):
        keys = list(arms)
        cols = 4
        rows_n = (len(keys) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, 3.4 * rows_n), sharey=True, squeeze=False)
        for ax in axes.flat[len(keys):]:
            ax.axis("off")
        for ax, key in zip(axes.flat, keys):
            for history in head_histories(KAGGLE / ARMS[key][4], kind):
                rows = history["history"]
                epochs = [r["epoch"] for r in rows]
                ax.plot(epochs, [r["train_masked_bce"] for r in rows], color="#4d84bd", alpha=.45)
                ax.plot(epochs, [r["validation_masked_bce"] for r in rows], color="#d68441", alpha=.55)
                best = next(r for r in rows if r["epoch"] == history["best_epoch"])
                ax.scatter(best["epoch"], best["validation_masked_bce"], color="#a84b2f", s=14, zorder=3)
            ax.set_title(label(key), fontsize=8)
            ax.set_xlabel("Head epoch")
            ax.set_ylabel("Masked BCE")
            ax.grid(alpha=.16)
        fig.text(.5, .995, f"{name}-head training loss (blue) and validation loss (orange) on LM labels; "
                 "dots mark selected epochs", ha="center", va="top", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(DIGEST / f"{NAME}-{kind}-head-curves.svg")
        plt.close(fig)

    PAGE.write_text(render(arms, summary, pending))


def render(arms, summary, pending):
    noise = summary["replicate_noise"]
    lines = [
        "# Codec-pipeline replicate and DeCLIP arms: training record and results",
        "",
        f"Status: **{'complete' if not pending else 'in progress'}**. "
        + ("Pending: " + "; ".join(label(k) for k in pending) + ". " if pending else "")
        + "All new arms ran from the codec-fix worktree (`fix/kaggle-dicom-codecs` @ `35dfa0e`: per-syntax decoder pins, "
        "decode counters, inference series fallback) on the gold-holdout cohort, contrastive seed 2026, and were read out "
        "with the 58-case LM-label head protocol of [lm-head-tower-size-58-v1.md](lm-head-tower-size-58-v1.md). "
        "Matrix: `configs/contrastive/codec-declip-v1.json`. Plain-CLIP comparators come from "
        "[next-model-pairs-v1.md](next-model-pairs-v1.md).",
        "",
        "**What the codec pipeline can and cannot change here.** All 13,221 selected training series are uncompressed "
        "Explicit VR Little Endian (one file per series scanned, 2026-09-27), the codec fix keeps every decodable file on "
        "its previous decoder, `PREPROCESSING_VERSION` is unchanged, and the image cache is reused. The codec-pipeline "
        "run therefore sees bit-identical inputs to the reference: its differences measure run-to-run CUDA nondeterminism, "
        "which makes it a second baseline run.",
        "",
    ]
    if ANSWER.exists():
        lines += ["## Answer", "", ANSWER.read_text().strip(), ""]
    lines += [
        "## Results",
        "",
        f"Gold AUROCs are macro over 12 labels on all 58 gold studies, which no head training, validation, or epoch selection used. "
        f"**LM-test AUROC** is masked macro AUROC against `lm_labels_kaggle.csv` verdicts on the {summary['lm_test_studies']} "
        "non-gold locked-test studies (UNK ignored). Each head is the probability average of seeds 2026–2028. "
        "Δ columns are arm minus base with a 95% paired study-bootstrap interval (gold: 5,000 resamples; LM-test: 2,000).",
        "",
        "| Image tower | Text tower | Objective | Gold linear | Gold MLP | Δ gold MLP vs reference | Δ gold MLP vs replicate "
        "| LM-test linear | LM-test MLP | Δ LM-test MLP vs reference | Δ LM-test MLP vs replicate |",
        "|---|---|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for key, a in arms.items():
        gr, gp = a["gold_delta_vs_reference"], a["gold_delta_vs_replicate"]
        lr, lp = a["lm_test_delta_vs_reference"], a["lm_test_delta_vs_replicate"]
        lines.append(
            f"| {a['image']} | {a['text']} | {a['objective']}{' (reference)' if key == REFERENCE else ''} "
            f"| {f4(a['gold_macro_auroc']['linear'])} | {f4(a['gold_macro_auroc']['mlp'])} "
            f"| {delta(gr and gr['mlp'])} | {delta(gp and gp['mlp'])} "
            f"| {f4(a['lm_test_macro_auroc']['linear'])} | {f4(a['lm_test_macro_auroc']['mlp'])} "
            f"| {delta(lr and lr['mlp'])} | {delta(lp and lp['mlp'])} |"
        )
    declip_rows = [k for k in arms if arms[k].get("clip_counterpart") in arms]
    if declip_rows:
        lines += [
            "",
            "**DeCLIP vs plain CLIP with the same towers** (MLP heads):",
            "",
            "| DeCLIP arm | CLIP counterpart | Δ gold MLP | Δ LM-test MLP |",
            "|---|---|---|---|",
        ]
        for key in declip_rows:
            a = arms[key]
            lines.append(f"| {label(key)} | {label(a['clip_counterpart'])} "
                         f"| {delta(a['gold_delta_vs_clip_counterpart']['mlp'])} "
                         f"| {delta(a['lm_test_delta_vs_clip_counterpart']['mlp'])} |")
    if noise:
        lines += [
            "",
            "### Run-to-run noise (reference vs codec-pipeline replicate)",
            "",
            f"- Selected mean R@1 gap: **{noise['selected_mean_r1_gap']:+.4f}** "
            "(earlier identical-config pair: 0.024).",
            f"- Per-epoch |Δ mean R@1| over {noise['epochs_compared']} epochs: median **{noise['median_abs_epoch_delta_mean_r1']:.4f}**, "
            f"max {noise['max_abs_epoch_delta_mean_r1']:.4f} (earlier pair: median ~0.0087).",
            f"- Per-epoch |Δ validation loss|: median {noise['median_abs_epoch_delta_validation_loss']:.4f}; "
            f"epoch-1 |Δ train loss| {noise['epoch1_abs_delta_train_loss']:.2e}.",
            "",
            "A treatment arm is only interesting if it lands outside the reference–replicate pair on both gold and LM-test.",
        ]
    lines += [
        "",
        "### Contrastive training",
        "",
        "Selected epoch follows the frozen rule: validation mean R@1 descending, then validation loss, then earliest epoch. "
        "Validation loss is plain symmetric InfoNCE over the full 433-pair pool for every arm, so it is comparable across "
        "CLIP and DeCLIP. Training loss is in-batch; for DeCLIP it is the composite objective (multi-view InfoNCE + "
        "0.5 × nearest-neighbour + 0.5 × image SimCLR), so it is **not** comparable with the CLIP arms.",
        "",
        "| Image tower | Text tower | Objective | Selected epoch | Train loss | Val loss selected / min / final | I→T R@1 | T→I R@1 "
        "| Mean R@1 | Hours | Peak GPU alloc |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key, a in arms.items():
        c = a["contrastive"]
        lines.append(
            f"| {a['image']} | {a['text']} | {a['objective']} | {c['selected_epoch']} | {c['train_loss']:.3f} "
            f"| {c['validation_loss']:.3f} / {c['validation_loss_min']:.3f} / {c['validation_loss_final']:.3f} "
            f"| {pct(c['image_to_text_r1'])} | {pct(c['text_to_image_r1'])} | {pct(c['mean_r1'])} "
            f"| {'—' if c['elapsed_hours'] is None else f'{c['elapsed_hours']:.1f}'} "
            f"| {'—' if c['peak_allocated_gb'] is None else f'{c['peak_allocated_gb']:.1f} GB'} |"
        )
    lines += [
        "",
        f"![Contrastive training loss, validation loss, and both Recall@1 directions]({NAME}-contrastive-curves.svg)",
        "",
        "### LM-label heads",
        "",
        "| Image tower | Text tower | Objective | MLP selected epochs | MLP mean val BCE | Linear selected epochs | Linear mean val BCE |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for key, a in arms.items():
        s = a["head_selection"]
        lines.append(
            f"| {a['image']} | {a['text']} | {a['objective']} | {', '.join(map(str, s['mlp']['epochs']))} "
            f"| {np.mean(s['mlp']['validation_masked_bce']):.4f} | {', '.join(map(str, s['linear']['epochs']))} "
            f"| {np.mean(s['linear']['validation_masked_bce']):.4f} |"
        )
    lines += [
        "",
        f"![MLP-head train and validation masked BCE, three seeds]({NAME}-mlp-head-curves.svg)",
        "",
        f"![Linear-head train and validation masked BCE, three seeds]({NAME}-linear-head-curves.svg)",
        "",
        "### Gold AUROC by label (MLP)",
        "",
        "| Label | " + " | ".join(label(k) for k in arms) + " |",
        "|---|" + "---:|" * len(arms),
    ]
    for name in sorted(LABELS):
        values = [arms[k]["gold_per_label_auroc"]["mlp"][name] for k in arms]
        top = max(values)
        lines.append(f"| {name} | " + " | ".join(f"**{v:.4f}**" if v == top else f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "## Training settings",
        "",
        "Shared by every arm (the reference settings): 2.5D input, 3 planes × 3 slices at 224², ImageNet normalization; "
        "LoRA on the last 6 blocks of both towers, rank 64, alpha 128, dropout 0; batch 32; 20 epochs; learning rate 1e-4 "
        "image/text and 1e-3 projection; weight decay 0.02; 10% linear warmup then cosine; 256-d projection; original "
        "full reports.",
        "",
        "| Arm | Swaps vs reference |",
        "|---|---|",
        "| Codec-pipeline replicate | None (codec worktree source) |",
        "| DeCLIP + DINOv2-small + E5-base | `--use-declip`: multi-view InfoNCE weight 1.0 (image augmentation + 15% report "
        "token masking, four cross-modal pairs), nearest-neighbour report supervision weight 0.5 with a 512-report FIFO "
        "queue, image SimCLR weight 0.5; gradient checkpointing (memory only) |",
        "| DeCLIP + RAD-DINO + E5-base | The DeCLIP settings above; image tower `microsoft/rad-dino` @ `110cbc18`, at the "
        "reference 224-pixel ImageNet-normalized input, as in the next-model-pairs RAD-DINO arm |",
        "| DeCLIP + RAD-DINO + BGE-M3 · 512 | As the row above; text tower `BAAI/bge-m3` @ `5617a9f6`, 512 tokens, mean "
        "pooling, no `query: ` prefix |",
        "",
        "## Limits",
        "",
        "- **One contrastive seed per arm.** The replicate row is the only measure of tower-training variation; the paired "
        "intervals condition on the fitted towers and heads.",
        "- **The 58 gold labels are not a clean test.** They have been examined repeatedly; treat gold differences as exploratory.",
        "- **LM-test AUROC scores against LM pseudo-labels**, like-for-like across arms but not a gold measure.",
        "- The codec pipeline is only exercised on compressed data at Kaggle inference; see "
        "`.scratch/input-preproc-hypotheses/r9-kaggle-bundle-check.md`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "# from the codec worktree (data/processed, lm_labels_kaggle.csv and checkpoints symlinked to the main checkout)",
        ".venv/bin/python scripts/training/run_contrastive_matrix.py --matrix configs/contrastive/codec-declip-v1.json --dry-run",
        "# each arm under scripts/training/run_measured_training.py, then per arm:",
        ".venv-cuda/bin/python scripts/training/train_lm_head.py --config configs/kaggle/lm-mlp-codec-declip-58-<arm>.json \\",
        "  --features checkpoints/kaggle/lm-mlp-codec-declip-58-<arm>-features.npz \\",
        "  --output checkpoints/kaggle/lm-mlp-codec-declip-58-<arm> --device cuda",
        "# unadapted anchors: the same command with <arm> = unadapted-dinov2-small, unadapted-rad-dino (image_baseline configs)",
        "# this page, its JSON and figures",
        ".venv/bin/python scripts/digests/build_codec_declip_digest.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
