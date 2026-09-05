#!/usr/bin/env python3
"""Build a self-contained HTML digest from translated-report training artifacts."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from contrastive_learning.data import KneeReportDataset, build_study_records

RUNS = {
    "UCSD/VA RadBERT": REPO / "checkpoints/contrastive/gold-translated-radbert-va",
    "Stanford RadBERT": REPO / "checkpoints/contrastive/gold-translated-stanford-radbert",
    "BiomedBERT": REPO / "checkpoints/contrastive/gold-translated-biomedbert",
}
SESSION = REPO / "checkpoints/contrastive/session-20260902"
CV_MODELS = {
    "Stanford RadBERT": "radbert",
    "BiomedBERT": "biomedbert",
}
COLORS = {
    "UCSD/VA RadBERT": "#3b82f6",
    "Stanford RadBERT": "#14b8a6",
    "BiomedBERT": "#f59e0b",
}
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def load_run(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    history = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    best = max(history, key=lambda row: (row["mean_r@1"], -row["epoch"]))
    return {"directory": directory, "config": config, "history": history, "best": best}


def load_runs() -> dict[str, dict]:
    return {name: load_run(directory) for name, directory in RUNS.items()}


def summarize_folds(folds: list[dict]) -> dict[str, tuple[float, float]]:
    keys = (
        "image_to_text_r@1", "image_to_text_r@5", "image_to_text_r@10",
        "image_to_text_mean_rank", "text_to_image_r@1", "text_to_image_r@5",
        "text_to_image_r@10", "text_to_image_mean_rank", "mean_r@1",
    )
    return {
        key: (statistics.mean(row["best"][key] for row in folds),
              statistics.stdev(row["best"][key] for row in folds))
        for key in keys
    }


def load_session() -> dict:
    cv = {}
    for name, slug in CV_MODELS.items():
        folds = [load_run(SESSION / f"cv/{slug}-fold-{index}") for index in range(5)]
        cv[name] = {"folds": folds, "summary": summarize_folds(folds)}
    unfreezing = {
        blocks: load_run(SESSION / f"unfreezing/biomedbert-blocks-{blocks}")
        for blocks in (0, 1, 2)
    }
    token_length = {
        256: unfreezing[0],
        512: load_run(SESSION / "token-length/biomedbert-512"),
    }
    return {"cv": cv, "unfreezing": unfreezing, "token_length": token_length}


def mean_sd(metric: tuple[float, float], *, percent: bool = True) -> str:
    scale = 100 if percent else 1
    return f"{metric[0] * scale:.2f} ± {metric[1] * scale:.2f}{'%' if percent else ''}"


def cv_rows(cv: dict[str, dict]) -> str:
    rows = []
    for model, result in cv.items():
        summary = result["summary"]
        for label, prefix in (("Image→text", "image_to_text"), ("Text→image", "text_to_image")):
            rows.append(
                "<tr>"
                f"<td><strong>{html.escape(model)}</strong></td><td>{label}</td>"
                f"<td>{mean_sd(summary[f'{prefix}_r@1'])}</td>"
                f"<td>{mean_sd(summary[f'{prefix}_r@5'])}</td>"
                f"<td>{mean_sd(summary[f'{prefix}_r@10'])}</td>"
                f"<td>{mean_sd(summary[f'{prefix}_mean_rank'], percent=False)}</td>"
                "</tr>"
            )
    return "".join(rows)


def ablation_rows(runs: dict[int, dict]) -> str:
    rows = []
    for value, run in runs.items():
        best = run["best"]
        avg_r5 = (best["image_to_text_r@5"] + best["text_to_image_r@5"]) / 2
        avg_rank = (best["image_to_text_mean_rank"] + best["text_to_image_mean_rank"]) / 2
        rows.append(
            "<tr>"
            f"<td><strong>{value}</strong></td><td>{best['epoch']}</td>"
            f"<td>{best['image_to_text_r@1']*100:.2f}%</td>"
            f"<td>{best['text_to_image_r@1']*100:.2f}%</td>"
            f"<td>{best['mean_r@1']*100:.2f}%</td><td>{avg_r5*100:.2f}%</td>"
            f"<td>{avg_rank:.2f}</td>"
            "</tr>"
        )
    return "".join(rows)


def data_profile() -> dict:
    with (REPO / "data/gold_translated.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lengths = [len(row["Translated Report"].split()) for row in rows]
    labels = {target: sum(int(row[target]) for row in rows) for target in TARGETS}
    return {
        "rows": len(rows),
        "languages": Counter(row["Original Language"] for row in rows),
        "lengths": lengths,
        "labels": labels,
    }


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def montage_data_uri() -> str:
    image_kwargs = {
        "image_size": 224,
        "crop_mm": 130.0,
        "slices_per_plane": 3,
        "slice_band": (0.20, 0.80),
    }
    records, _ = build_study_records(
        REPO / "data/gold_translated.csv",
        Path("/mnt/d/train_series.csv"),
        Path("/mnt/d/train_series"),
        report_column="Translated Report",
        validate_pixels=False,
        image_kwargs=image_kwargs,
        max_studies=1,
    )
    item = KneeReportDataset(records, image_kwargs)[0]
    labels = ("Sagittal", "Coronal", "Axial")
    panel_width, header, gap = 224, 34, 10
    canvas = Image.new("RGB", (panel_width * 3 + gap * 2, panel_width + header), "#07111f")
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        pixels = (item["images"][index, 1].clamp(0, 1).numpy() * 255).astype("uint8")
        plane = Image.fromarray(pixels, mode="L").convert("RGB")
        x = index * (panel_width + gap)
        canvas.paste(plane, (x, header))
        draw.text((x + 8, 10), label, fill="#e5f7ff")
        draw.rectangle((x, header, x + panel_width - 1, header + panel_width - 1), outline="#31516d", width=1)
    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def loss_svg(runs: dict[str, dict]) -> str:
    width, height = 760, 270
    left, right, top, bottom = 58, 20, 25, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [row["train_loss"] for run in runs.values() for row in run["history"]]
    low = min(values) - 0.08
    high = max(values) + 0.08

    def point(epoch: int, value: float) -> tuple[float, float]:
        x = left + (epoch - 1) / 4 * plot_w
        y = top + (high - value) / (high - low) * plot_h
        return x, y

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training loss by epoch">']
    for step in range(5):
        y = top + step / 4 * plot_h
        value = high - step / 4 * (high - low)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>')
    for epoch in range(1, 6):
        x, _ = point(epoch, low)
        parts.append(f'<text x="{x:.1f}" y="{height-18}" text-anchor="middle" class="axis">{epoch}</text>')
    for name, run in runs.items():
        coordinates = [point(row["epoch"], row["train_loss"]) for row in run["history"]]
        path = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(coordinates))
        color = COLORS[name]
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in coordinates)
    parts.append(f'<text x="{left + plot_w/2:.1f}" y="{height-2}" text-anchor="middle" class="axis-title">Epoch</text>')
    parts.append('</svg>')
    return "".join(parts)


def retrieval_bars(runs: dict[str, dict], chance: dict[str, float]) -> str:
    rows = []
    for name, run in runs.items():
        best = run["best"]
        metrics = {
            "R@1": best["mean_r@1"],
            "R@5": (best["image_to_text_r@5"] + best["text_to_image_r@5"]) / 2,
            "R@10": (best["image_to_text_r@10"] + best["text_to_image_r@10"]) / 2,
        }
        bars = "".join(
            f'<div class="metric-row"><span>{key}</span><div class="track"><i style="width:{value*100:.1f}%;background:{COLORS[name]}"></i><b style="left:{chance[key]*100:.1f}%" title="random expectation"></b></div><strong>{value*100:.1f}%</strong></div>'
            for key, value in metrics.items()
        )
        rows.append(f'<div class="bar-card"><h4>{html.escape(name)}</h4>{bars}</div>')
    return "".join(rows)


def label_bars(profile: dict) -> str:
    maximum = max(profile["labels"].values())
    return "".join(
        f'<div class="label-row"><span>{html.escape(name)}</span><div><i style="width:{count/maximum*100:.1f}%"></i></div><strong>{count}</strong></div>'
        for name, count in profile["labels"].items()
    )


def result_rows(runs: dict[str, dict]) -> str:
    rows = []
    for name, run in runs.items():
        best = run["best"]
        avg_r5 = (best["image_to_text_r@5"] + best["text_to_image_r@5"]) / 2
        avg_rank = (best["image_to_text_mean_rank"] + best["text_to_image_mean_rank"]) / 2
        history = run["history"]
        loss_change = (history[0]["train_loss"] - history[-1]["train_loss"]) / history[0]["train_loss"]
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(name)}</strong><small>{html.escape(run['config']['text_model'])}</small></td>"
            f"<td>{best['epoch']}</td><td>{best['mean_r@1']*100:.1f}%</td><td>{avg_r5*100:.1f}%</td>"
            f"<td>{avg_rank:.2f}</td><td>{history[-1]['train_loss']:.4f}</td><td>{loss_change*100:.1f}%</td>"
            "</tr>"
        )
    return "".join(rows)


def build_html() -> str:
    runs = load_runs()
    session = load_session()
    profile = data_profile()
    montage = montage_data_uri()
    n_val = next(iter(runs.values()))["config"]["val_fraction"] * profile["rows"]
    n_val = round(n_val)
    chance = {"R@1": 1 / n_val, "R@5": min(5 / n_val, 1), "R@10": min(10 / n_val, 1)}
    _, best_run = max(runs.items(), key=lambda item: item[1]["best"]["mean_r@1"])
    best = best_run["best"]
    cv_winner_name, cv_winner = max(
        session["cv"].items(), key=lambda item: item[1]["summary"]["mean_r@1"][0]
    )
    cv_winner_r1 = cv_winner["summary"]["mean_r@1"]
    stanford_r1 = session["cv"]["Stanford RadBERT"]["summary"]["mean_r@1"]
    token_256 = session["token_length"][256]["best"]
    token_512 = session["token_length"][512]["best"]
    lengths = profile["lengths"]
    languages = " · ".join(f"{html.escape(k)} {v}" for k, v in profile["languages"].most_common())
    config = next(iter(runs.values()))["config"]

    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Translated knee MRI contrastive training digest</title>
<style>
:root{{--ink:#102538;--muted:#5e7283;--paper:#f4f8fa;--card:#fff;--line:#d9e5eb;--blue:#3b82f6;--teal:#14b8a6;--amber:#f59e0b;--navy:#071a2b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
.hero{{background:radial-gradient(circle at 82% 20%,#174a61 0,transparent 28%),linear-gradient(135deg,#061827,#0d2c40);color:#effbff;padding:64px 7vw 54px}}
.eyebrow{{color:#73d7d0;text-transform:uppercase;letter-spacing:.16em;font-weight:750;font-size:12px}} h1{{font-size:clamp(34px,5vw,66px);line-height:1.02;max-width:900px;margin:14px 0 18px;letter-spacing:-.04em}} .hero p{{max-width:810px;color:#bfd2dc;font-size:18px}}
.hero-grid,.cards,.two,.three{{display:grid;gap:18px}} .hero-grid{{grid-template-columns:repeat(4,minmax(0,1fr));max-width:1000px;margin-top:34px}} .hero-stat{{border-left:2px solid #38c8bd;padding-left:14px}} .hero-stat strong{{display:block;font-size:28px}} .hero-stat span{{color:#a9bfca;font-size:12px;text-transform:uppercase;letter-spacing:.08em}}
main{{max-width:1180px;margin:0 auto;padding:44px 24px 80px}} section{{margin:0 0 50px}} h2{{font-size:27px;margin:0 0 10px;letter-spacing:-.02em}} h3{{margin:0 0 8px}} h4{{margin:0 0 12px}} .lede,.muted,small{{color:var(--muted)}} .cards{{grid-template-columns:repeat(3,1fr)}} .card,.chart,.bar-card,.callout{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 8px 28px rgba(24,56,73,.05)}}
.two{{grid-template-columns:1.2fr .8fr}} .three{{grid-template-columns:repeat(3,1fr)}} .flow{{display:flex;align-items:stretch;gap:9px;margin-top:22px;overflow-x:auto;padding-bottom:8px}} .node{{min-width:160px;flex:1;background:#fff;border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:12px;padding:15px}} .node b{{display:block;margin-bottom:5px}} .node span{{font-size:12px;color:var(--muted)}} .arrow{{align-self:center;color:#86a0ae;font-size:25px}}
.mri{{display:block;width:100%;border-radius:12px;background:#07111f}} .caption{{font-size:12px;color:var(--muted);margin-top:9px}} .pill{{display:inline-block;border-radius:999px;background:#e5f4f3;color:#12655f;padding:5px 10px;font-size:12px;font-weight:700}}
svg{{display:block;width:100%}} .grid{{stroke:#dce7ec;stroke-width:1}} .axis{{fill:#6b7f8e;font-size:11px}} .axis-title{{fill:#425b6c;font-size:12px;font-weight:700}} .legend{{display:flex;gap:20px;flex-wrap:wrap}} .legend i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.bar-card{{margin:12px 0;padding:17px}} .metric-row{{display:grid;grid-template-columns:42px 1fr 54px;gap:10px;align-items:center;margin:9px 0;font-size:12px}} .track{{height:11px;background:#edf2f5;border-radius:99px;position:relative;overflow:visible}} .track i{{height:100%;display:block;border-radius:99px}} .track b{{position:absolute;top:-3px;width:2px;height:17px;background:#182d3d}} .metric-row strong{{text-align:right}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden}} th,td{{padding:13px 14px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}} th{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:#f8fbfc}} th:first-child,td:first-child{{text-align:left}} td small{{display:block;font-size:10px;max-width:320px;overflow-wrap:anywhere}} tr:last-child td{{border-bottom:0}}
.label-row{{display:grid;grid-template-columns:135px 1fr 28px;gap:8px;align-items:center;margin:7px 0;font-size:12px}} .label-row>div{{height:8px;background:#e9f0f3;border-radius:9px}} .label-row i{{height:100%;display:block;background:linear-gradient(90deg,#3b82f6,#14b8a6);border-radius:9px}} .callout{{border-left:5px solid var(--teal);background:#f5fffd}} code{{background:#eaf0f3;padding:2px 5px;border-radius:5px;font-size:.9em}} pre{{background:#071a2b;color:#d6edf5;padding:18px;border-radius:12px;overflow:auto}} ul{{padding-left:20px}} li{{margin:7px 0}} .footer{{color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:20px}}
@media(max-width:850px){{.hero-grid,.cards,.two,.three{{grid-template-columns:1fr 1fr}} .flow{{align-items:center}} .node{{min-width:190px}}}} @media(max-width:560px){{.hero-grid,.cards,.two,.three{{grid-template-columns:1fr}} main{{padding:30px 15px 60px}} th,td{{padding:9px 7px;font-size:11px}}}}
</style></head><body>
<header class="hero"><div class="eyebrow">Training digest · updated 02 September 2026</div><h1>Knee MRI ↔ translated report alignment</h1><p>A five-fold follow-up tests the two strongest text encoders, then probes final-block adaptation and report length against a fixed DINOv2-small image architecture.</p>
<div class="hero-grid"><div class="hero-stat"><strong>58</strong><span>paired studies</span></div><div class="hero-stat"><strong>5</strong><span>study-level folds</span></div><div class="hero-stat"><strong>14</strong><span>new training runs</span></div><div class="hero-stat"><strong>{cv_winner_r1[0]*100:.1f}%</strong><span>best CV mean R@1</span></div></div></header>
<main>
<section><span class="pill">Newest outcome</span><h2>{html.escape(cv_winner_name)} leads across five folds</h2><p class="lede">Frozen BiomedBERT reached {mean_sd(cv_winner_r1)} bidirectional mean Recall@1, compared with {mean_sd(stanford_r1)} for Stanford RadBERT. This reverses the original single-split ordering and confirms that one 12-study holdout was not stable enough for model selection. Unfreezing the final one or two blocks improved single-holdout mean R@1 from 16.67% to 20.83%; extending inputs from 256 to 512 tokens reduced it to 12.50%.</p></section>

<section><h2>Visual intuition: what the model sees</h2><div class="two"><div class="card"><img class="mri" src="{montage}" alt="Representative preprocessed sagittal, coronal and axial knee MRI center slices"><div class="caption">Representative validation-path preprocessing. Each displayed plane is the middle channel of a three-slice stack; the model receives all three channels and all three planes.</div></div><div class="card"><h3>One study becomes one point</h3><p>Each plane yields DINOv2 class and mean-patch features. Learned attention fuses the three planes. The translated report becomes one medical-language-model feature. Two projection heads pull the true pair together and push other studies in the minibatch apart.</p><div class="label-row"><span>Image embedding</span><div><i style="width:78%"></i></div><strong>256d</strong></div><div class="label-row"><span>Text embedding</span><div><i style="width:78%;background:linear-gradient(90deg,#f59e0b,#ef6f6c)"></i></div><strong>256d</strong></div></div></div>
<div class="flow"><div class="node"><b>DICOM study</b><span>Choose the largest fluid-sensitive series in each plane</span></div><div class="arrow">→</div><div class="node"><b>3 plane tensors</b><span>Sagittal · coronal · axial, each 3×224×224</span></div><div class="arrow">→</div><div class="node"><b>DINOv2-small</b><span>Shared frozen encoder; CLS + mean patch tokens</span></div><div class="arrow">↔</div><div class="node" style="border-top-color:#f59e0b"><b>Medical LM</b><span>Translated report, 256-token maximum</span></div><div class="arrow">→</div><div class="node" style="border-top-color:#14b8a6"><b>InfoNCE</b><span>Bidirectional matching with learned temperature</span></div></div></section>

<section><h2>Data and split</h2><div class="cards"><div class="card"><h3>Complete modality pairing</h3><p><strong>58 / 58</strong> translated reports matched a study with sagittal, coronal, and axial series. Full validation found zero missing planes and zero sampled-slice decode failures.</p></div><div class="card"><h3>Study-disjoint holdout</h3><p><strong>46 train · 12 validation</strong>, seeded with 2026. The tables do not expose a patient identifier, so patient-disjointness cannot be proven.</p></div><div class="card"><h3>Translated text profile</h3><p>Median {percentile(lengths,.5):.0f} words; 90th percentile {percentile(lengths,.9):.0f}; maximum {max(lengths)}. Original-language mix: {languages}.</p></div></div>
<div class="two" style="margin-top:18px"><div class="card"><h3>Pathology prevalence</h3><p class="muted">Labels are shown for cohort context only; they were not used by contrastive training.</p>{label_bars(profile)}</div><div class="card"><h3>Baseline-compatible image path</h3><ol><li>Prefer fluid-sensitive series; choose the largest per plane.</li><li>Geometrically order slices and sample three over the central 20–80%.</li><li>Apply a centered 130 mm crop when pixel spacing permits.</li><li>Clip at series percentiles 1–99, scale to [0,1], resize to 224.</li><li>Normalize right knees to the left-knee convention; augment without random flips.</li></ol></div></div></section>

<section><h2>Original pilot training behavior</h2><div class="chart">{loss_svg(runs)}<div class="legend">{''.join(f'<span><i style="background:{COLORS[name]}"></i>{html.escape(name)}</span>' for name in runs)}</div></div><p class="caption">Original single-holdout pilot. Minibatch symmetric InfoNCE; all encoders were frozen. Curves are not directly comparable to a full-dataset epoch loss because the shuffled loader drops the final incomplete training batch.</p></section>

<section><h2>Original pilot: one held-out split</h2><p class="lede">Historical context only. Bars use the best epoch by mean bidirectional R@1. The black tick is the random-ranking expectation for 12 candidates: R@1 8.3%, R@5 41.7%, R@10 83.3%.</p>{retrieval_bars(runs,chance)}
<div style="overflow-x:auto;margin-top:18px"><table><thead><tr><th>Text encoder</th><th>Best epoch</th><th>Mean R@1</th><th>Mean R@5</th><th>Mean rank</th><th>Final loss</th><th>Loss reduction</th></tr></thead><tbody>{result_rows(runs)}</tbody></table></div></section>

<section><span class="pill">Newest experiment session</span><h2>Five-fold model comparison</h2><p class="lede">Each study appears in exactly one validation fold. Fold sizes are 12, 12, 12, 11, and 11; all runs use seed 2026, five epochs, batch size 8, 256 tokens, and frozen backbones. Values are unweighted fold mean ± sample standard deviation at each fold's best epoch by bidirectional R@1.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Text encoder</th><th>Direction</th><th>R@1</th><th>R@5</th><th>R@10</th><th>Mean rank</th></tr></thead><tbody>{cv_rows(session['cv'])}</tbody></table></div>
<div class="callout" style="margin-top:18px"><p><strong>BiomedBERT is the more stable candidate.</strong> Its bidirectional mean R@1 is {mean_sd(cv_winner_r1)}, versus {mean_sd(stanford_r1)} for Stanford RadBERT. The fold spread remains large relative to the difference between models, so this is a candidate-selection result rather than a definitive performance estimate.</p></div></section>

<section><h2>Final-block adaptation</h2><p class="lede">BiomedBERT was carried forward on the original seed-2026 46/12 holdout. Both towers were frozen or had their final one or two blocks unfrozen together; tower learning rates remained 1e-5 for images and 2e-5 for text.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Blocks per tower</th><th>Best epoch</th><th>Image→text R@1</th><th>Text→image R@1</th><th>Mean R@1</th><th>Mean R@5</th><th>Mean rank</th></tr></thead><tbody>{ablation_rows(session['unfreezing'])}</tbody></table></div>
<p class="caption">One and two blocks tie on the primary endpoint at 20.83%. Two blocks are the better secondary-metric tie-breaker: text→image R@5 is 58.33% rather than 33.33%, and bidirectional mean rank is 5.13 rather than 6.13. This ablation is single-split and must be repeated across folds before adopting it.</p></section>

<section><h2>Report-length ablation</h2><p class="lede">The frozen BiomedBERT control isolates maximum input length. BiomedBERT tokenization puts 24/58 reports above 256 tokens and 5/58 above 512; median length is 236.5, p90 is 441, and maximum is 710 tokens.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Maximum tokens</th><th>Best epoch</th><th>Image→text R@1</th><th>Text→image R@1</th><th>Mean R@1</th><th>Mean R@5</th><th>Mean rank</th></tr></thead><tbody>{ablation_rows(session['token_length'])}</tbody></table></div>
<p class="caption">At each run's best epoch, 512 tokens raises mean R@5 from {((token_256['image_to_text_r@5'] + token_256['text_to_image_r@5'])/2)*100:.2f}% to {((token_512['image_to_text_r@5'] + token_512['text_to_image_r@5'])/2)*100:.2f}%, but lowers the primary mean R@1 from {token_256['mean_r@1']*100:.2f}% to {token_512['mean_r@1']*100:.2f}%. More context is not automatically better on this split; section-aware findings/impression extraction is the cleaner next test.</p></section>

<section><div class="callout"><h2>Interpretation</h2><p><strong>The newest evidence favors frozen BiomedBERT for model selection and cautious two-block adaptation as a follow-up.</strong> Cross-validation overturned Stanford RadBERT's original single-split lead. The adaptation and token-length results remain exploratory because they use only one holdout and select the best epoch on that same validation set.</p><p>Retrieval measures pair identification, not diagnostic correctness. With only 11–12 candidates per fold, one match moves fold R@1 by 8.3–9.1 percentage points, and patient-disjointness still cannot be established from the available tables.</p></div></section>

<section><h2>Exact experiment configuration</h2><div class="three"><div class="card"><h3>Image</h3><p><code>{html.escape(config['image_model'])}</code><br>224×224 · three planes · final 0/1/2 blocks tested</p></div><div class="card"><h3>Text</h3><p>Stanford RadBERT and BiomedBERT in cross-validation<br>BiomedBERT 256/512-token and block ablations</p></div><div class="card"><h3>Optimization</h3><p>{config['epochs']} epochs · batch {config['batch_size']}<br>AdamW · projection LR {config['lr_projection']:.0e}<br>seed {config['seed']}</p></div></div>
<pre>source .venv/bin/activate
HF_HOME=/tmp/novo-project-hf python scripts/train_contrastive.py \\
  --reports-csv data/gold_translated.csv \\
  --report-column "Translated Report" \\
  --text-model biomedbert \\
  --epochs 5 --batch-size 8 --fold-count 5 --fold-index 0 \\
  --image-unfreeze-last 0 --text-unfreeze-last 0 \\
  --cache-images --image-cache-dir checkpoints/contrastive/session-20260902/image-cache \\
  --skip-decode-validation --no-amp \\
  --output-dir checkpoints/contrastive/session-20260902/cv/biomedbert-fold-0</pre></section>

<section><h2>Artifacts and next steps</h2><div class="two"><div class="card"><h3>Saved experiment session</h3><ul><li><code>checkpoints/contrastive/session-20260902/cv</code> · 10 runs</li><li><code>checkpoints/contrastive/session-20260902/unfreezing</code> · 3 runs</li><li><code>checkpoints/contrastive/session-20260902/token-length</code> · 1 run</li><li><code>digest/experiment-session-log.md</code> · decisions and results</li></ul><p>Every run contains <code>config.json</code>, five-row <code>metrics.jsonl</code>, <code>best.pt</code>, and <code>last.pt</code>. Preprocessed tensors are configuration-keyed and shared across runs.</p></div><div class="card"><h3>Recommended next experiment session</h3><ol><li>Repeat the 0/1/2-block BiomedBERT comparison across all five folds before adopting adaptation.</li><li>Separate image-only from text-only unfreezing to identify which tower produces any gain.</li><li>Compare the 256-token full report with impression-aware or findings-plus-impression extraction.</li><li>Add patient identifiers for patient-disjoint validation, then test diagnostic downstream utility on an external cohort.</li></ol></div></div></section>

<div class="footer">Generated from the local CSV, DICOM preprocessing, 17 legacy/new run configs, and JSONL metrics by <code>scripts/build_training_digest.py</code>. Retrieval measures pair identification, not diagnostic validity. Pathology labels were not optimization targets.</div>
</main></body></html>'''


def main() -> None:
    destination = REPO / "digest/translated-report-training.html"
    destination.write_text(build_html(), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
