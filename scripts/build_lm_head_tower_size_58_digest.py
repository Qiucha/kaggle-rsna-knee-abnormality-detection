"""Summarize four frozen-encoder LM-head fits on the same 58 gold studies."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.lm_supervision import paired_gold_interval  # noqa: E402
from contrastive_learning.submission import LABELS  # noqa: E402

ARMS = {
    "small_image_base_text": ("DINOv2-small", "E5-base", "dinov2-small-e5-base"),
    "small_image_small_text": ("DINOv2-small", "E5-small", "dinov2-small-e5-small"),
    "base_image_base_text": ("DINOv2-base", "E5-base", "dinov2-base-e5-base"),
    "base_image_small_text": ("DINOv2-base", "E5-small", "dinov2-base-e5-small"),
}
BASE = ROOT / "checkpoints/kaggle"
DIGEST = ROOT / "digest/lm-head-tower-size-58-v1.md"


def fmt(value):
    return f"{value:.4f}"


def public_score(path):
    record = path / "leaderboard-submission.json"
    return json.loads(record.read_text()).get("public_score") if record.exists() else None


def proxy_contrast(proxy_pairs, reference, candidate):
    """Candidate minus reference from the stored LM-test paired intervals."""
    ref, cand = f"lm-mlp-tower-size-58-{ARMS[reference][2]}", f"lm-mlp-tower-size-58-{ARMS[candidate][2]}"
    if f"{cand} - {ref}" in proxy_pairs:
        return proxy_pairs[f"{cand} - {ref}"]
    flipped = proxy_pairs[f"{ref} - {cand}"]
    return {"difference": -flipped["difference"], "ci95": [-flipped["ci95"][1], -flipped["ci95"][0]]}


def main():
    reports, predictions, paths, proxies = {}, {}, {}, {}
    for name, (_, _, suffix) in ARMS.items():
        path = BASE / f"lm-mlp-tower-size-58-{suffix}"
        paths[name] = path
        reports[name] = json.loads((path / "benchmark.json").read_text())
        # Standard local proxy report, written by scripts/eval_lm_proxy_heads.py.
        proxies[name] = json.loads((path / "lm-proxy.json").read_text())
        with np.load(path / "gold-predictions.npz", allow_pickle=False) as loaded:
            predictions[name] = {key: loaded[key] for key in loaded.files}
    first = next(iter(predictions.values()))
    with (ROOT / "data/cleaned.csv").open(newline="") as handle:
        source_gold = {row["StudyInstanceUID"] for row in csv.DictReader(handle)}
    if len(source_gold) != 58 or set(first["study_uids"]) != source_gold:
        raise ValueError("Scored gold studies differ from all 58 cleaned.csv cases")
    for name, item in predictions.items():
        if not np.array_equal(first["study_uids"], item["study_uids"]):
            raise ValueError(f"Gold study order differs for {name}")
        if not np.array_equal(first["targets"], item["targets"]):
            raise ValueError(f"Gold targets differ for {name}")
        population = reports[name]["population"]
        if population["coverage"]["gold"]["studies"] != 58:
            raise ValueError(f"Expected 58 gold studies for {name}")
        if population["excluded"]["gold_test_evaluated"] != 2:
            raise ValueError(f"Expected two former test cases for {name}")
        if not reports[name]["encoder_frozen_verified"]:
            raise ValueError(f"Encoder verification failed for {name}")
        head_uids = reports[name]["input_identity"]["study_uids"]
        if source_gold & (set(head_uids["train"]) | set(head_uids["validation"])):
            raise ValueError(f"Gold study entered head fitting for {name}")

    comparisons = {
        "E5-base − E5-small, DINOv2-small": (
            "small_image_small_text", "small_image_base_text"
        ),
        "E5-base − E5-small, DINOv2-base": (
            "base_image_small_text", "base_image_base_text"
        ),
        "DINOv2-base − DINOv2-small, E5-base": (
            "small_image_base_text", "base_image_base_text"
        ),
        "DINOv2-base − DINOv2-small, E5-small": (
            "small_image_small_text", "base_image_small_text"
        ),
    }
    intervals = {
        label: paired_gold_interval(
            first["targets"], predictions[reference]["mlp"],
            predictions[candidate]["mlp"], resamples=5000, seed=2030
        )
        for label, (reference, candidate) in comparisons.items()
    }

    proxy_comparison = json.loads((ROOT / "digest/lm-proxy-eval-v2.json").read_text())
    proxy_intervals = {
        label: proxy_contrast(proxy_comparison["paired_intervals"], reference, candidate)
        for label, (reference, candidate) in comparisons.items()
    }
    first_submission = BASE / "lora-holdout-v1"
    first_proxy = json.loads((first_submission / "lm-proxy.json").read_text())
    if len({p["study_uids_sha256"] for p in [first_proxy, *proxies.values()]}) != 1:
        raise ValueError("LM-test populations differ across runs")
    lm_test_studies = first_proxy["studies"]

    old = json.loads((ROOT / "digest/tower-size-factorial-v1.json").read_text())
    summary = {
        "gold_studies": 58,
        "head_train_studies": 3438,
        "head_validation_studies": 426,
        "arms": {
            name: {
                "image": image,
                "text": text,
                "mlp_macro_auroc": reports[name]["results"]["mlp"]["macro_auroc"],
                "linear_macro_auroc": reports[name]["results"]["linear"]["macro_auroc"],
                "contrastive_validation_mean_recall_at_1": old["arms"][name]["best_mean_r1"],
                "mlp_epochs": [s["best_epoch"] for s in reports[name]["selections"]["mlp"]],
                "mlp_validation_masked_bce": [s["validation_masked_bce"] for s in reports[name]["selections"]["mlp"]],
                "checkpoint_sha256": reports[name]["input_identity"]["checkpoint_sha256"],
                "lm_test_macro_auroc": proxies[name]["macro_auroc"],
                "public_score": public_score(paths[name]),
            }
            for name, (image, text, _) in ARMS.items()
        },
        "paired_mlp_auroc_differences": intervals,
        "lm_test_studies": lm_test_studies,
        "paired_lm_test_auroc_differences": proxy_intervals,
        "first_submission": {
            "lm_test_macro_auroc": first_proxy["macro_auroc"],
            "public_score": public_score(first_submission),
        },
    }
    (ROOT / "digest/lm-head-tower-size-58-v1.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharey=True)
    for ax, (name, (image, text, _)) in zip(axes.flat, ARMS.items(), strict=True):
        path = paths[name]
        for seed in (2026, 2027, 2028):
            history = json.loads((path / f"mlp-seed{seed}-history.json").read_text())
            rows = history["history"]
            epochs = [r["epoch"] for r in rows]
            ax.plot(epochs, [r["train_masked_bce"] for r in rows], color="#4d84bd", alpha=.45)
            ax.plot(epochs, [r["validation_masked_bce"] for r in rows], color="#d68441", alpha=.55)
            best = rows[history["best_epoch"] - 1]
            ax.scatter(best["epoch"], best["validation_masked_bce"], color="#a84b2f", s=14, zorder=3)
        ax.set_title(f"{image} + {text}")
        ax.set_xlabel("Head epoch")
        ax.set_ylabel("Masked BCE")
        ax.grid(alpha=.16)
    fig.text(.5, .99, "LM-head training loss (blue) and validation loss (orange); dots mark selected epochs", ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(ROOT / "digest/lm-head-tower-size-58-curves.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharey=True)
    for ax, (name, (image, text, _)) in zip(axes.flat, ARMS.items(), strict=True):
        path = paths[name]
        for seed in (2026, 2027, 2028):
            history = json.loads((path / f"linear-seed{seed}-history.json").read_text())
            rows = history["history"]
            epochs = [r["epoch"] for r in rows]
            ax.plot(epochs, [r["train_masked_bce"] for r in rows], color="#4d84bd", alpha=.45)
            ax.plot(epochs, [r["validation_masked_bce"] for r in rows], color="#d68441", alpha=.55)
            best = rows[history["best_epoch"] - 1]
            ax.scatter(best["epoch"], best["validation_masked_bce"], color="#a84b2f", s=14, zorder=3)
        ax.set_title(f"{image} + {text}")
        ax.set_xlabel("Head epoch")
        ax.set_ylabel("Masked BCE")
        ax.grid(alpha=.16)
    fig.text(.5, .99, "Linear-head training loss (blue) and validation loss (orange); dots mark selected epochs", ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(ROOT / "digest/lm-head-tower-size-58-linear-curves.svg")
    plt.close(fig)

    lines = ["## Results", "", f"Gold AUROCs use the same 58 gold studies. The MLP column is the primary gold result; linear is the matched supervision control. **LM-test AUROC** is the standard local proxy: masked macro AUROC against LM verdicts on the {lm_test_studies} non-gold locked-test studies, which no head or checkpoint selection used. Contrastive R@1 comes from the previously selected tower checkpoints and is included for context.", "", "| Image tower | Text tower | Contrastive val mean R@1 | LM-linear gold AUROC | LM-MLP gold AUROC | LM-test AUROC | Public score | MLP selected epochs | Mean selected val BCE |", "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for name, (image, text, _) in ARMS.items():
        row = summary["arms"][name]
        public = f"{row['public_score']:.3f}" if row["public_score"] is not None else "—"
        lines.append(f"| {image} | {text} | {fmt(row['contrastive_validation_mean_recall_at_1'])} | {fmt(row['linear_macro_auroc'])} | **{fmt(row['mlp_macro_auroc'])}** | {fmt(row['lm_test_macro_auroc'])} | {public} | {', '.join(map(str, row['mlp_epochs']))} | {fmt(float(np.mean(row['mlp_validation_masked_bce'])))} |")
    gold_only = json.loads((BASE / "lora-holdout-v1/benchmark.json").read_text())
    gold_only_auc = gold_only["results"]["ensemble"]["macro_auroc"]
    matrix_headers = [
        "Index", "Local AUROC", "LM-test AUROC", "I→T R@1", "T→I R@1", "Public score",
        "MRI input", "Image model", "Text model", "Image layers adapted",
        "Text layers adapted", "Medical report language / view", "Batch size", "Similarity targets",
        "Tower LR", "LoRA rank", "LoRA alpha", "LoRA dropout",
        "Projection LR", "Weight decay", "Warmup / schedule",
        "Contrastive projection", "Pathology head supervision",
    ]
    lines += [
        "", "### Full settings and outcome matrix", "",
        "The contrastive settings below are shared except for the named backbone sizes. The contrastive **256d projection** was trained from image–report pairs in every row. Gold labels or LM verdicts trained the **separate pathology head**; they did not train that projection or change the contrastive similarity targets. Row 1 is the earlier gold-only submission and has a **56-case out-of-fold** local score. Rows 2–5 are **58-case held-out** LM-head scores, so the local AUROCs across those two populations are not a paired before/after comparison. Retrieval recalls are from the selected contrastive checkpoints, not from the pathology head.",
        "", "| " + " | ".join(matrix_headers) + " |",
        "|" + "|".join("---" for _ in matrix_headers) + "|",
    ]
    for index, name in enumerate(("small_image_base_text", *ARMS), 1):
        image, text, _ = ARMS[name]
        old_arm = old["arms"][name]
        image_model = f"{image} ({'22.1M' if image.endswith('small') else '86.6M'})"
        text_model = f"{text} ({'278M' if text.endswith('base') else '117.7M'})"
        local = (
            f"{gold_only_auc:.4f} (56 OOF)"
            if index == 1
            else f"{summary['arms'][name]['mlp_macro_auroc']:.4f} (58 held-out)"
        )
        run_public = public_score(first_submission if index == 1 else paths[name])
        run_proxy = first_proxy if index == 1 else proxies[name]
        cells = [
            str(index), local, f"{run_proxy['macro_auroc']:.4f}",
            f"{old_arm['best_image_to_text_r1']:.2%}",
            f"{old_arm['best_text_to_image_r1']:.2%}",
            f"{run_public:.3f}" if run_public is not None else "—",
            "2.5D: 3 planes × 3 slices", image_model, text_model,
            "last 6 blocks (LoRA)", "last 6 blocks (LoRA)",
            "Original language, full report; 512 tokens", "32", "CLIP identity targets",
            "1e-4 image / text", "64", "128", "0", "1e-3", "0.02",
            "10% linear / cosine", "256d",
            "56 gold labels; logistic OOF" if index == 1 else "LM verdicts; 3,438 train / 426 val; 3-seed MLP",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"LM-test AUROC uses the same {lm_test_studies} locked-test studies in every row, including row 1, so that column is a like-for-like comparison. Public scores come from each run's `leaderboard-submission.json`; see `digest/kaggle-submission-pipeline.md`. The 2.5D input stacks three sampled slices as channels for each of the sagittal, coronal, and axial planes. The size values count pretrained backbone parameters approximately; LoRA and projection parameters are additional."]
    lines += ["", "![Three seeded MLP train and validation head-loss curves for each tower pairing](lm-head-tower-size-58-curves.svg)", "", "The MLPs select early epochs. Later training loss falls while LM-label validation loss rises, showing overfitting of the supervised head after its selected epoch. Epoch selection uses validation only.", "", "### Linear control: less late overfitting, lower gold AUROC", "", "The linear heads reach their validation minima later and have a much smaller increase in validation BCE after that point. Their selected models nevertheless score below the selected MLP ensemble on all four gold comparisons. The paired intervals below compare the saved, early-stopped heads on the same 58 studies; they do not compare the deliberately overtrained final epochs.", "", "| Image / text | Linear gold AUROC | MLP minus linear AUROC | Paired 95% interval | Linear selected epochs | Mean validation BCE rise after selection: linear / MLP |", "|---|---:|---:|---:|---|---:|"]
    for name, (image, text, _) in ARMS.items():
        report = reports[name]
        comparison = report["paired_comparisons"]["mlp_minus_lm_linear"]
        low, high = comparison["ci95"]
        rebounds = {}
        for arm in ("linear", "mlp"):
            histories = [
                json.loads((paths[name] / f"{arm}-seed{seed}-history.json").read_text())
                for seed in (2026, 2027, 2028)
            ]
            rebounds[arm] = float(np.mean([
                h["history"][-1]["validation_masked_bce"]
                - h["history"][h["best_epoch"] - 1]["validation_masked_bce"]
                for h in histories
            ]))
        epochs = ", ".join(str(item["best_epoch"]) for item in report["selections"]["linear"])
        lines.append(
            f"| {image} / {text} | {report['results']['linear']['macro_auroc']:.4f} | "
            f"{comparison['difference']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
            f"{epochs} | +{rebounds['linear']:.4f} / +{rebounds['mlp']:.4f} |"
        )
    lines += ["", "![Three seeded linear-head train and validation loss curves for each tower pairing](lm-head-tower-size-58-linear-curves.svg)", "", "Linear gold AUROC ranges from 0.7312 to 0.7362 across towers, whereas the selected MLP ensembles range from 0.7504 to 0.7659. Thus the simpler head dampens the late validation-loss rebound, but does not improve gold discrimination in this experiment.", "", "### Paired differences on the 58 gold studies", "", "Positive values favor the first model named in the contrast. Intervals are 95% paired study-bootstrap percentile intervals with 5,000 resamples; they condition on the fitted checkpoints and heads.", "", "| Contrast | Δ macro AUROC | 95% interval |", "|---|---:|---:|"]
    for label, result in intervals.items():
        low, high = result["ci95"]
        lines.append(f"| {label} | {result['difference']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines += ["", f"### Paired differences on the {lm_test_studies} locked-test studies (LM-test AUROC)", "", "Same contrasts, scored against LM verdicts on the locked test split. Intervals are 95% paired study-bootstrap percentile intervals with 2,000 resamples. This proxy favors heads trained on LM verdicts over heads trained on other labels; all four arms here share one LM-head recipe, so the contrasts below are like-for-like.", "", "| Contrast | Δ LM-test AUROC | 95% interval |", "|---|---:|---:|"]
    for label, result in proxy_intervals.items():
        low, high = result["ci95"]
        lines.append(f"| {label} | {result['difference']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    labels = list(reports[next(iter(ARMS))]["results"]["mlp"]["per_label"])
    lines += ["", "### AUROC by gold label", "", "| Label | Gold positives / 58 | Small image + base text | Small image + small text | Base image + base text | Base image + small text |", "|---|---:|---:|---:|---:|---:|"]
    for label in labels:
        values = [reports[name]["results"]["mlp"]["per_label"][label]["auroc"] for name in ARMS]
        # targets columns follow the submission LABELS order; per_label keys are alphabetical
        lines.append(f"| {label} | {int(first['targets'][:, LABELS.index(label)].sum())} | " + " | ".join(fmt(value) for value in values) + " |")
    scores = {name: summary["arms"][name]["mlp_macro_auroc"] for name in ARMS}
    winner = max(scores, key=scores.get)
    winner_image, winner_text, _ = ARMS[winner]
    small_text_effect = intervals["E5-base − E5-small, DINOv2-small"]
    seed_consistent = all(
        reports[mismatched]["results"]["mlp"]["seeds"][i]["macro_auroc"]
        > reports[matched]["results"]["mlp"]["seeds"][i]["macro_auroc"]
        for matched, mismatched in (
            ("small_image_small_text", "small_image_base_text"),
            ("base_image_base_text", "base_image_small_text"),
        )
        for i in range(3)
    )
    lines += ["", "## Interpretation", "", f"The highest observed 58-case LM-head AUROC is **{fmt(scores[winner])}** from **{winner_image} + {winner_text}**. Matching image and text tower sizes is **not necessary** for the best result in this four-run comparison." if winner in ("small_image_base_text", "base_image_small_text") else f"The highest observed 58-case LM-head AUROC is **{fmt(scores[winner])}** from **{winner_image} + {winner_text}**. This comparison alone does not establish that matching sizes is necessary; compare the paired intervals above.", "", f"With DINOv2-small fixed, E5-base changes gold macro AUROC by **{small_text_effect['difference']:+.4f}** relative to E5-small (95% interval **[{small_text_effect['ci95'][0]:+.4f}, {small_text_effect['ci95'][1]:+.4f}]**). This does not show that the larger text tower overfits the smaller image tower. The LM head itself does overfit if trained past its validation-selected epoch, which is a separate effect.", "", "These are four selected contrastive checkpoints and one 58-study gold set, not replicated tower-training seeds or an external test. Gold outcomes have been consulted during the project, and the two formerly reserved test gold cases are now included. The paired intervals capture study sampling uncertainty conditional on these fitted models, not run-to-run variation. LM pseudo-label errors and partial label coverage can also limit the downstream head.", "", "## Reproduction", "", "The four configs are under `configs/kaggle/lm-mlp-tower-size-58-*.json`. Each run's `benchmark.json`, `gold-predictions.npz`, selection histories, feature cache, and exported bundle are under the corresponding `checkpoints/kaggle/lm-mlp-tower-size-58-*` directory. Rebuild this page with `PYTHONPATH=src .venv/bin/python scripts/build_lm_head_tower_size_58_digest.py`. The machine-readable comparison is `digest/lm-head-tower-size-58-v1.json`. LM-test scores are rebuilt with `.venv/bin/python scripts/eval_lm_proxy_heads.py checkpoints/kaggle/lora-holdout-v1 checkpoints/kaggle/lm-mlp-tower-size-58-*/ --output digest/lm-proxy-eval-v2.json` before this page.", ""]
    limitations_index = next(i for i, line in enumerate(lines) if line.startswith("These are four selected"))
    lines[limitations_index:limitations_index] = [
        "All four paired 95% intervals include zero. The observed ranking therefore does not establish a reliable population advantage for either tower size. Both mismatched pairings have higher point estimates than the matched pairing with the same image size, which argues against a size-matching requirement in these runs.",
        "",
        ("That mismatched-over-matched pattern holds for each of the three individual LM-head seeds within both image sizes. It does not substitute for repeating the contrastive tower training with more seeds." if seed_consistent else "The individual head-seed rankings are mixed; the ensemble comparison should be interpreted with the paired intervals above."),
        "",
        "The earlier contrastive runs also had rising validation loss late in training for all four pairings. For DINOv2-small, E5-base ended at validation loss 4.054 and E5-small at 4.119, after minima of 3.446 and 3.441 respectively. That pattern does not isolate a larger-text-specific overfit effect. Contrastive retrieval R@1 and pathology AUROC measure different tasks and give different rankings here.",
        "",
    ]
    protocol = DIGEST.read_text().split("## Results")[0]
    content = protocol + "\n".join(lines)
    DIGEST.write_text(content)
    body = markdown.markdown(content, extensions=["tables", "fenced_code"])
    style = """body{margin:0;background:#f5f4ef;color:#202a31;font:16px/1.55 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:48px 28px 80px;background:#fff;min-height:100vh}h1{font-size:2.2rem;line-height:1.18;color:#153247}h2{margin-top:2.5rem;border-top:1px solid #d9e2e4;padding-top:1rem;color:#153247}h3{margin-top:2rem;color:#315363}p{max-width:80ch}table{border-collapse:collapse;width:100%;font-size:.88rem;display:block;overflow-x:auto}th,td{padding:9px 10px;border-bottom:1px solid #dce2e2;white-space:nowrap;text-align:left}th{background:#e9f1f2;color:#173947}tr:nth-child(even){background:#f7faf9}img{width:100%;max-width:1000px}code{font-size:.86em;background:#f2f4f4;padding:.12em .25em;border-radius:3px}strong{color:#0d675d}@media(max-width:700px){main{padding:22px 15px}h1{font-size:1.7rem}}"""
    html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>LM-head tower size comparison: 58 gold studies</title><style>{style}</style></head><body><main>{body}</main></body></html>"
    (ROOT / "digest/lm-head-tower-size-58-v1.html").write_text(html)
    print(json.dumps({"scores": scores, "contrasts": intervals}, indent=2))


if __name__ == "__main__":
    main()
