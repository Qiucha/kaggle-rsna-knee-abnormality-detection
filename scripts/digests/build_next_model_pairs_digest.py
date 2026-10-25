"""Build the next-model-pairs-v1 digest: one-tower swaps against DINOv2-small + E5-base.

Reads the five contrastive runs and their 58-case LM heads, verifies the shared gold and
LM-test populations, and writes the results page, its JSON, and three curve figures.

    .venv/bin/python scripts/digests/build_next_model_pairs_digest.py
"""

from __future__ import annotations

import argparse
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
PAGE = DIGEST / "next-model-pairs-v1.md"
REFERENCE = "reference"

# key: (image, text, contrastive run dir, head run dir, colour)
ARMS = {
    REFERENCE: ("DINOv2-small", "E5-base", "both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026",
                "lm-mlp-tower-size-58-dinov2-small-e5-base", "#222222"),
    "registers": ("DINOv2-registers-small", "E5-base",
                  "next-model-pairs-v1/dinov2-registers-small-e5-base-512-seed2026",
                  "lm-mlp-next-pairs-58-dinov2-registers-small-e5-base-512", "#4d84bd"),
    "bge_m3_512": ("DINOv2-small", "BGE-M3 · 512", "next-model-pairs-v1/dinov2-small-bge-m3-512-seed2026",
                   "lm-mlp-next-pairs-58-dinov2-small-bge-m3-512", "#d68441"),
    "bge_m3_1024": ("DINOv2-small", "BGE-M3 · 1024", "next-model-pairs-v1/dinov2-small-bge-m3-1024-seed2026",
                    "lm-mlp-next-pairs-58-dinov2-small-bge-m3-1024", "#e2b33c"),
    "e5_large": ("DINOv2-small", "E5-large", "next-model-pairs-v1/dinov2-small-e5-large-512-seed2026",
                 "lm-mlp-next-pairs-58-dinov2-small-e5-large-512", "#3f9a6b"),
    "rad_dino": ("RAD-DINO", "E5-base", "next-model-pairs-v1/rad-dino-e5-base-512-seed2026",
                 "lm-mlp-next-pairs-58-rad-dino-e5-base-512", "#a0569b"),
}

# Settings shared by every trained arm; the table repeats them so each row is self-describing.
SETTINGS = ("2.5D: 3 planes × 3 slices, 224²", "LoRA last 6", "LoRA last 6", "32", "CLIP identity",
            "1e-4 image / text", "64", "128", "0", "1e-3", "0.02", "10% linear, then cosine", "256d")


def label(key):
    image, text, *_ = ARMS[key]
    return f"{image} + {text}"


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    with (ROOT / "data/gold_label.csv").open(newline="") as handle:
        source_gold = {row["StudyInstanceUID"] for row in csv.DictReader(handle)}
    arms, gold, lm_test, lm_population = {}, {}, {}, None
    for key, (_, _, run, head, _) in ARMS.items():
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
        arms[key] = {
            "image": ARMS[key][0],
            "text": ARMS[key][1],
            "contrastive_run": f"checkpoints/contrastive/{run}",
            "head_run": f"checkpoints/kaggle/{head}",
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

    for key in ARMS:
        if key == REFERENCE:
            continue
        arms[key]["gold_delta_vs_reference"] = {
            kind: paired_gold_interval(reference["targets"], reference[kind], gold[key][kind], resamples=5000, seed=2030)
            for kind in ("linear", "mlp")
        }
        arms[key]["lm_test_delta_vs_reference"] = {
            kind: paired_intervals({key: lm_test[key][kind], REFERENCE: lm_test[REFERENCE][kind]},
                                   lm_targets, lm_masks)[f"{key} - {REFERENCE}"]
            for kind in ("linear", "mlp")
        }

    summary = {
        "experiment": "next-model-pairs-v1",
        "matrix": "configs/contrastive/next-model-pairs-v1.json",
        "gold_studies": 58,
        "lm_test_studies": int(len(lm_targets)),
        "contrastive_seed": 2026,
        "head_seeds": list(SEEDS),
        "gold_interval": "paired study bootstrap, 5,000 resamples, seed 2030",
        "lm_test_interval": "paired study bootstrap, 2,000 resamples, seed 2030",
        "arms": {k: {**v, "contrastive": {m: x for m, x in v["contrastive"].items() if m != "history"}}
                 for k, v in arms.items()},
    }
    (DIGEST / "next-model-pairs-v1.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
    panels = (("train_loss", "Training in-batch loss"), ("validation_loss", "Validation full-pool loss"),
              ("image_to_text_r@1", "Image → text validation R@1"), ("text_to_image_r@1", "Text → image validation R@1"))
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for key, (*_, colour) in ARMS.items():
            rows = arms[key]["contrastive"]["history"]
            ax.plot([r["epoch"] for r in rows], [r[metric] for r in rows], color=colour,
                    lw=1.8 if key == REFERENCE else 1.3, ls="--" if key == REFERENCE else "-", label=label(key))
            best = next(r for r in rows if r["epoch"] == arms[key]["contrastive"]["selected_epoch"])
            ax.scatter(best["epoch"], best[metric], color=colour, s=16, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Contrastive epoch")
        ax.grid(alpha=.16)
        if "r@1" in metric:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.text(.5, .99, "Contrastive training, seed 2026; dots mark the checkpoint selected by validation mean R@1",
             ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(DIGEST / "next-model-pairs-v1-contrastive-curves.svg")
    plt.close(fig)

    for kind, name in (("mlp", "MLP"), ("linear", "Linear (logistic)")):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True)
        for ax, key in zip(axes.flat, ARMS, strict=True):
            for history in head_histories(KAGGLE / ARMS[key][3], kind):
                rows = history["history"]
                epochs = [r["epoch"] for r in rows]
                ax.plot(epochs, [r["train_masked_bce"] for r in rows], color="#4d84bd", alpha=.45)
                ax.plot(epochs, [r["validation_masked_bce"] for r in rows], color="#d68441", alpha=.55)
                best = next(r for r in rows if r["epoch"] == history["best_epoch"])
                ax.scatter(best["epoch"], best["validation_masked_bce"], color="#a84b2f", s=14, zorder=3)
            ax.set_title(label(key) + (" (reference)" if key == REFERENCE else ""))
            ax.set_xlabel("Head epoch")
            ax.set_ylabel("Masked BCE")
            ax.grid(alpha=.16)
        fig.text(.5, .99, f"{name}-head training loss (blue) and validation loss (orange) on LM labels; "
                 "dots mark selected epochs", ha="center", va="top", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(DIGEST / f"next-model-pairs-v1-{kind}-head-curves.svg")
        plt.close(fig)

    PAGE.write_text(render(arms, summary))


def render(arms, summary):
    cands = [k for k in arms if k != REFERENCE]
    best_gold = max(arms, key=lambda k: arms[k]["gold_macro_auroc"]["mlp"])
    best_lm = max(arms, key=lambda k: arms[k]["lm_test_macro_auroc"]["mlp"])
    clear = [k for k in cands if (lambda lo, hi: lo > 0 or hi < 0)(*arms[k]["gold_delta_vs_reference"]["mlp"]["ci95"])]
    lines = [
        "# Next image and text tower pairs: training record and results",
        "",
        "Status (2026-09-25): **complete**. Five one-tower swaps against the DINOv2-small + multilingual E5-base reference "
        "were trained (contrastive seed 2026) and read out with the 58-case LM-label head protocol from "
        "[lm-head-tower-size-58-v1.md](lm-head-tower-size-58-v1.md), using both the MLP head and the linear (logistic) control. "
        "DINOv3 is gated and has no local snapshot, so it was not run. The queue and hypotheses are in "
        "[next-model-candidates.md](next-model-candidates.md).",
        "",
        "## Answer",
        "",
        f"- **A larger text tower is the only swap with a consistent upside.** {label(best_gold)} has the highest gold MLP AUROC "
        f"(**{f4(arms[best_gold]['gold_macro_auroc']['mlp'])}** vs {f4(arms[REFERENCE]['gold_macro_auroc']['mlp'])}), and "
        f"{label(best_lm)} has the highest LM-test MLP AUROC (**{f4(arms[best_lm]['lm_test_macro_auroc']['mlp'])}** vs "
        f"{f4(arms[REFERENCE]['lm_test_macro_auroc']['mlp'])}). Both beat the reference on gold AUROC with both heads, "
        "but every one of those paired intervals includes zero.",
        "- **Longer reports did not help.** BGE-M3 at 1,024 tokens is below BGE-M3 at 512 tokens on every AUROC column "
        "and took about 5× longer to train because it spilled past the 8 GB GPU.",
        "- **Register tokens did not help.** DINOv2-with-registers is level with or slightly below the reference on every metric.",
        "- **RAD-DINO is worse.** Chest-X-ray pretraining is the only swap whose gold MLP interval excludes zero"
        + (" " if clear == ["rad_dino"] else f" (clear contrasts: {', '.join(label(k) for k in clear)}) ")
        + "— in the wrong direction. It was fed 224-pixel ImageNet-normalized input rather than its own 518-pixel "
        "preprocessing and is a ViT-B-sized tower, so this is a transfer result for this pipeline, not for RAD-DINO in general.",
        "",
        "## Results",
        "",
        f"Gold AUROCs are macro over 12 labels on all 58 gold studies, which no head training, validation, or epoch selection used. "
        f"**LM-test AUROC** is masked macro AUROC against `lm_labels_kaggle.csv` verdicts on the {summary['lm_test_studies']} "
        "non-gold locked-test studies (UNK ignored). Each head is the probability average of seeds 2026–2028. "
        "Δ columns are candidate minus reference with a 95% paired study-bootstrap interval "
        "(gold: 5,000 resamples; LM-test: 2,000).",
        "",
        "| Image tower | Text tower | Gold linear | Gold MLP | Δ gold linear | Δ gold MLP | LM-test linear | LM-test MLP | Δ LM-test MLP |",
        "|---|---|---:|---:|---|---|---:|---:|---|",
    ]
    for key, a in arms.items():
        ref = key == REFERENCE
        lines.append(
            f"| {a['image']} | {a['text']}{' (reference)' if ref else ''} "
            f"| {f4(a['gold_macro_auroc']['linear'])} | {f4(a['gold_macro_auroc']['mlp'])} "
            f"| {'—' if ref else delta(a['gold_delta_vs_reference']['linear'])} "
            f"| {'—' if ref else delta(a['gold_delta_vs_reference']['mlp'])} "
            f"| {f4(a['lm_test_macro_auroc']['linear'])} | {f4(a['lm_test_macro_auroc']['mlp'])} "
            f"| {'—' if ref else delta(a['lm_test_delta_vs_reference']['mlp'])} |"
        )
    lines += [
        "",
        "LM-test Δ for the linear heads is in [next-model-pairs-v1.json](next-model-pairs-v1.json). The MLP beats the linear "
        "control for every tower except RAD-DINO, repeating the earlier four-pair result.",
        "",
        "### Contrastive training",
        "",
        "Selected epoch follows the frozen rule: validation mean R@1 descending, then validation loss, then earliest epoch. "
        "Training loss is in-batch (32 pairs); validation loss is over the full 433-pair pool, so the two are not a matched pair. "
        "Time and peak memory come from `run_measured_training.py` telemetry; the reference predates it.",
        "",
        "| Image tower | Text tower | Selected epoch | Train loss | Val loss selected / min / final | I→T R@1 | T→I R@1 | Mean R@1 | Hours | Peak GPU alloc |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key, a in arms.items():
        c = a["contrastive"]
        lines.append(
            f"| {a['image']} | {a['text']} | {c['selected_epoch']} | {c['train_loss']:.3f} "
            f"| {c['validation_loss']:.3f} / {c['validation_loss_min']:.3f} / {c['validation_loss_final']:.3f} "
            f"| {pct(c['image_to_text_r1'])} | {pct(c['text_to_image_r1'])} | {pct(c['mean_r1'])} "
            f"| {'—' if c['elapsed_hours'] is None else f'{c['elapsed_hours']:.1f}'} "
            f"| {'—' if c['peak_allocated_gb'] is None else f'{c['peak_allocated_gb']:.1f} GB'} |"
        )
    lines += [
        "",
        "![Contrastive training loss, validation loss, and both Recall@1 directions for the six towers]"
        "(next-model-pairs-v1-contrastive-curves.svg)",
        "",
        "All three larger text towers reach lower validation-loss minima (3.31–3.35 vs 3.45) and about 2 points higher mean R@1 "
        "than the reference. That retrieval gap is near the ~2.4-point run-to-run spread seen between identical single-seed runs. "
        "E5-large's gain is almost entirely text → image. As in every earlier run, validation loss bottoms out mid-training and "
        "rises while R@1 keeps improving, so the R@1 rule selects late epochs with above-minimum validation loss. "
        "Peak allocations above 8 GB mean the run spilled into WSL shared memory: it completed, but slowly.",
        "",
        "### LM-label heads",
        "",
        "| Image tower | Text tower | MLP selected epochs | MLP mean val BCE | Linear selected epochs | Linear mean val BCE |",
        "|---|---|---|---:|---|---:|",
    ]
    for key, a in arms.items():
        s = a["head_selection"]
        lines.append(
            f"| {a['image']} | {a['text']} | {', '.join(map(str, s['mlp']['epochs']))} "
            f"| {np.mean(s['mlp']['validation_masked_bce']):.4f} | {', '.join(map(str, s['linear']['epochs']))} "
            f"| {np.mean(s['linear']['validation_masked_bce']):.4f} |"
        )
    lines += [
        "",
        "![MLP-head train and validation masked BCE for each tower, three seeds](next-model-pairs-v1-mlp-head-curves.svg)",
        "",
        "![Linear-head train and validation masked BCE for each tower, three seeds](next-model-pairs-v1-linear-head-curves.svg)",
        "",
        "The MLP heads again select epoch 2–3 and overfit afterwards; the linear heads select later and rebound less. "
        "RAD-DINO's features give the highest LM-label validation BCE for both heads, consistent with its lower AUROCs.",
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
        "Every trained arm shares the reference settings except the swapped tower (and, for 1,024, the context cap). "
        "Run configs were diffed against the reference: the only other differences are options added since it was trained, "
        "all at inactive defaults (DeCLIP off, temperature not frozen, inverse-temperature cap 100, never reached).",
        "",
        "| Pair / status | MRI Input Dimensions | Image Models | Text Models | Image Unfrozen Layers | Text Unfrozen Layers "
        "| Medical Report Languages | Batch Size | Supervise method for similarity matrix | Learning Rate | LoRA Rank | LoRA Alpha "
        "| LoRA Dropout | Projection Layer LR | Weight Decay | Warmup | Projection Layer | Local AUC (58 gold labels) "
        "| ItoT_Recall@1 | TtoI_Recall@1 |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    names = {"multilingual E5-base": "E5-base", "multilingual E5-large": "E5-large", "BGE-M3": "BGE-M3"}
    for key, a in arms.items():
        text = a["text"].split(" · ")[0]
        tokens = a["text"].split(" · ")[1] if " · " in a["text"] else "512"
        text_full = next((full for full, short in names.items() if short == text), text)
        status = "Reference · complete" if key == REFERENCE else f"{label(key)} · complete"
        img_in, img_l, txt_l, *rest = SETTINGS
        lines.append(
            f"| {status} | {img_in} | {a['image']} | {text_full} | {img_l} | {txt_l} | Original, full report; {tokens} tokens "
            f"| {' | '.join(rest)} | {f4(a['gold_macro_auroc']['mlp'])} "
            f"| {pct(a['contrastive']['image_to_text_r1'])} | {pct(a['contrastive']['text_to_image_r1'])} |"
        )
    lines += [
        "| DINOv3-small + E5-base · gated, not run | 2.5D: 3 planes × 3 slices, 224² planned | DINOv3 ViT-S/16 | multilingual E5-base "
        "| LoRA last 6 planned | LoRA last 6 planned | Original, full report; 512 tokens planned | 32 planned | CLIP identity planned "
        "| 1e-4 planned | 64 planned | 128 planned | 0 planned | 1e-3 planned | 0.02 planned | 10% linear, then cosine planned "
        "| 256d planned | — | — | — |",
        "",
        "Local AUC is the 3-seed LM-MLP head on the 58 gold studies. BGE-M3 uses mean pooling without E5's `query: ` prefix. "
        "The register model excludes its four register tokens from patch pooling in training and in exported features. "
        "RAD-DINO kept the reference 224-pixel input for a one-tower comparison.",
        "",
        "## Limits",
        "",
        "- **One contrastive seed per tower.** Identical single-seed runs have differed by ~0.024 mean R@1, and the paired "
        "intervals above condition on the fitted towers and heads, so they do not capture tower-training variation. "
        "Repeating E5-large and BGE-M3-512 with more seeds is the direct test of the text-tower result.",
        "- **The 58 gold labels are not a clean test.** They have been examined repeatedly across the project, including the "
        "two formerly reserved test cases. Treat gold differences as exploratory.",
        "- **LM-test AUROC scores against pseudo-labels** from the same LM that supervised the heads; it is like-for-like "
        "across these arms but is not a gold measure.",
        "- The contrastive validation selection was not finalized with `finalize_contrastive_matrix`; no immutable "
        "`validation-selection.json` exists for this matrix yet.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "# contrastive arms, one at a time (each under run_measured_training.py for time/memory telemetry)",
        ".venv-cuda/bin/python scripts/training/run_contrastive_matrix.py --matrix configs/contrastive/next-model-pairs-v1.json --dry-run",
        "# LM heads (linear + MLP, 3 seeds) and LM-test score, per arm",
        ".venv-cuda/bin/python scripts/training/train_lm_head.py --config configs/kaggle/lm-mlp-next-pairs-58-<arm>.json \\",
        "  --features checkpoints/kaggle/lm-mlp-next-pairs-58-<arm>-features.npz \\",
        "  --output checkpoints/kaggle/lm-mlp-next-pairs-58-<arm> --device cuda",
        "# this page, its JSON and figures",
        ".venv/bin/python scripts/digests/build_next_model_pairs_digest.py",
        "```",
        "",
        "Run artifacts are under `checkpoints/contrastive/next-model-pairs-v1/` and `checkpoints/kaggle/lm-mlp-next-pairs-58-*/`. "
        "The encoder page of the digest site plots the same contrastive curves after `scripts/digests/build_digest_site.py export`.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
