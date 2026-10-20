"""Encoder choices: tower-size factorial, size-matched swap, label-similarity targets, DeCLIP, codec + DeCLIP holdout, next candidates.

Every number is read at build time from digest/*.json (plus curve snapshots under digest/site/data/curves/).
Absorbs the legacy reports tower-size-factorial-v1.html, size-matched-holdout-v1.html and
declip-alignment-v1.html; each is linked once as "original report" in its section's notes.
"""

from __future__ import annotations

import re

from .. import data
from ..shell import Page, caveat_badge, chart, esc, group_major, notice, num, section, status_badge, table
from ..sitemap import BY_SLUG
from ._encoders_util import curve_ref, md_table, section_script
from ._experiment_views import experiment_views
from ._adaptation_sections import HELPERS

SLUG = "encoders"
SUBTITLE = ("Which image and text towers to pair, and three ideas that did not change the gold readout: "
            "a size-matched swap, label-similarity targets and DeCLIP.")

FACTORIAL = "tower-size-factorial-v1.json"
SIZE_MATCHED = "size-matched-holdout-v1.json"
LS_PAIRED = "label-similarity-paired-v2.json"
LS_V2 = "label-similarity-v2.json"
DECLIP_PROBE = "downstream-probe-declip-v1.json"
CODEC = "codec-declip-v1.json"
CODEC_MD = "codec-declip-v1.md"
NEXT_MD = "next-model-candidates.md"
NEXT_RESULTS = "next-model-pairs-v1.md"
NEXT_JSON = "next-model-pairs-v1.json"
NEXT_RUNS = (
    "dinov2-registers-small-e5-base-512-seed2026",
    "dinov2-small-bge-m3-512-seed2026",
    "dinov2-small-bge-m3-1024-seed2026",
    "dinov2-small-e5-large-512-seed2026",
    "rad-dino-e5-base-512-seed2026",
)

# source: digest/label-similarity-paired-v2.json config.comparisons.pre_registered_reading
# ("A difference inside +/-0.01 macro AUROC is a null at this sample size")
NULL_BAND = 0.01

IMG_GROUPS = {
    "small": {"label": "DINOv2-small image tower", "color": "var(--g-teal)"},
    "base": {"label": "DINOv2-base image tower", "color": "var(--g-violet)"},
}


def _short(model: str | None) -> str:
    m = (model or "").split("/")[-1]
    m = m.replace("multilingual-", "")
    m = re.sub(r"^dinov2", "DINOv2", m)
    m = re.sub(r"^e5", "E5", m)
    return m


def _arm_label(a: dict) -> str:
    return f"{_short(a.get('image'))} + {_short(a.get('text'))}"


def _img_size(a: dict) -> str:
    return "base" if "base" in _short(a.get("image")) else "small"


def _txt_size(a: dict) -> str:
    return "base" if "base" in _short(a.get("text")) else "small"


def _ci(lo, hi, spec=".4f") -> str:
    return f"[{num(lo, spec, True)}, {num(hi, spec, True)}]"


def _excludes_zero(lo, hi) -> bool:
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


# --------------------------------------------------------------------------- 1. factorial


def factorial_section(ctx) -> tuple[str, dict, dict]:
    sid = "factorial"
    j = data.read_json(FACTORIAL)
    if not j:
        return section(sid, "Tower-size factorial", notice(f"<code>{FACTORIAL}</code> is missing.")), {}, {}
    arms = j["arms"]
    pop = (j.get("population") or {}).get("population_size")
    boot = j.get("bootstrap") or {}
    probe = j.get("probe") or {}

    # interaction plot: x = text tower (0 small, 1 base), one line per image tower
    series = []
    for size in ("small", "base"):
        pts = []
        for key, a in arms.items():
            if _img_size(a) == size:
                pts.append([1 if _txt_size(a) == "base" else 0, a["gold_macro_auroc"]])
        pts.sort()
        img_name = next(_short(a["image"]) for a in arms.values() if _img_size(a) == size)
        series.append({"id": f"img-{size}", "label": img_name, "group": size, "points": pts})

    def contrast_label(key: str, c: dict) -> str:
        left, right = arms[c["left"]], arms[c["right"]]
        if _img_size(left) == _img_size(right):
            return f"txt {_txt_size(right)[0].upper()}−{_txt_size(left)[0].upper()} @ img {_img_size(left)[0].upper()}"
        return f"img {_img_size(right)[0].upper()}−{_img_size(left)[0].upper()} @ txt {_txt_size(left)[0].upper()}"

    rows = []
    for key, c in (j.get("contrasts") or {}).items():
        pb = c.get("paired_bootstrap") or {}
        lo, hi = pb.get("macro_auroc_difference_ci95", [None, None])
        left, right = arms[c["left"]], arms[c["right"]]
        same_img = _img_size(left) == _img_size(right)
        long = (f"{_short(right['text'])} vs {_short(left['text'])} at {_short(left['image'])}" if same_img
                else f"{_short(right['image'])} vs {_short(left['image'])} at {_short(left['text'])}")
        rows.append({"id": key, "label": contrast_label(key, c), "long": long,
                     "group": _img_size(left) if same_img else "cross",
                     "kind": "text tower" if same_img else "image tower",
                     "est": pb.get("macro_auroc_difference_mean"), "point": c.get("point_difference"),
                     "lo": lo, "hi": hi, "p": pb.get("fraction_of_replicates_favouring_right")})
    inter = j.get("interaction") or {}
    if inter:
        rows.append({"id": "interaction", "label": "interaction", "group": "cross",
                     "kind": "interaction", "est": inter.get("bootstrap_mean"), "point": inter.get("point_difference"),
                     "lo": (inter.get("ci95") or [None, None])[0], "hi": (inter.get("ci95") or [None, None])[1], "p": None})

    single = [r for r in rows if r["id"] != "interaction"]
    clear = [r for r in single if _excludes_zero(r["lo"], r["hi"])]
    top_key, top = max(arms.items(), key=lambda kv: kv[1]["gold_macro_auroc"])
    claim = (f"{_arm_label(top)} has the highest point estimate ({num(top['gold_macro_auroc'])}), "
             f"but only {len(clear)} of {len(single)} single-tower contrasts excludes zero"
             + (f": {esc(clear[0]['long'])}, {num(clear[0]['point'], '.4f', True)}" if len(clear) == 1 else ""))

    groups = dict(IMG_GROUPS)
    groups["cross"] = {"label": "Image-size contrasts & interaction", "color": "var(--g-anchor)"}
    blob = {"groups": groups, "series": series, "contrasts": rows}
    body = (
        '<div data-role="filter"></div>\n<div class="grid-2">\n'
        f"<div>{chart('inter', 'Gold macro AUROC by text tower, one line per image tower')}</div>\n"
        f"<div>{chart('forest', 'Paired contrasts, 95% CI (S/B = small/base tower)')}</div>\n</div>\n"
        '<div data-role="legend"></div>\n'
        + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var xf = function (v) { return v === 0 ? 'E5-small' : (v === 1 ? 'E5-base' : ''); };
  Charts.lineChart(sec.querySelector('[data-chart=inter]'), d.series,
    {groups: d.groups, yLabel: 'Gold macro AUROC', yFmt: '.3f', xFmt: xf, xStep: 1, xLabel: 'Text tower', best: 'max', store: store, height: 300});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {groups: d.groups, best: 'max', fmt: '.4f', valuePrefix: 'best '});
  Charts.ciDotPlot(sec.querySelector('[data-chart=forest]'), d.contrasts,
    {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.2f', sort: false, labelWidth: 250,
     xLabel: '\\u0394 gold macro AUROC', store: store,
     tooltipExtra: function (r) { return '<div>' + r.kind + ' contrast' + (r.p === null ? '' : '; share of resamples favouring right arm ' + Charts.fmt('.1%')(r.p)) + '</div>'; }});"""))

    caption = (f"Dot = bootstrap mean; h2 and table give point differences. {pop} held-out gold studies, {probe.get('folds', '?')}-fold pooled OOF logistic probe, "
               f"{boot.get('resamples', '?')} paired study resamples; one contrastive seed per pairing (6-block LoRA r64, gold-holdout cohort).")
    tbl = table(["Pairing", "Feature width", "Best epoch", "Gold macro AUROC", "Gold macro AUPRC"],
                [[_arm_label(a), a.get("feature_dimensions"), a.get("best_epoch"), num(a["gold_macro_auroc"]),
                  num(a.get("gold_macro_auprc"))] for a in arms.values()],
                num_cols=(1, 2, 3, 4),
                row_attrs=[{"data-series-id": f"img-{_img_size(a)}", "data-group": _img_size(a)} for a in arms.values()])
    tbl += table(["Contrast", "Point Δ", "Bootstrap mean", "95% CI", "Excludes 0"],
                 [[r["label"], num(r["point"], ".4f", True), num(r["est"], ".4f", True), _ci(r["lo"], r["hi"]),
                   "yes" if _excludes_zero(r["lo"], r["hi"]) else "no"] for r in rows],
                 num_cols=(1, 2, 3), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    sm = data.read_json(SIZE_MATCHED) or {}
    g = sm.get("gold") or {}
    notes = (
        "<ul>"
        f"<li><strong>Single seed per pairing.</strong> Intervals resample studies (not patients) and condition on the four fitted "
        "checkpoints; they do not cover training-seed variation. Read the ranking as exploratory.</li>"
        f"<li>Unadapted floors differ by image tower: DINOv2-small {num(g.get('unadapted_small_macro_auroc'))}, "
        f"DINOv2-base {num(g.get('unadapted_base_macro_auroc'))} (from <code>{SIZE_MATCHED}</code>). No shared floor line is drawn.</li>"
        "<li>Feature width is 768 for DINOv2-small and 1,536 for DINOv2-base: the probe is matched, not equal-dimensional.</li>"
        "<li>This is the 56-case gold-only protocol. The 58-case LM-head re-scoring of the same four checkpoints is a different "
        f"protocol and population; it lives on <a href=\"{ctx.page('submission')}\">Heads &amp; Kaggle submission</a>.</li>"
        f"<li>Sources: <a href=\"{ctx.raw(FACTORIAL)}\">{FACTORIAL}</a>, <a href=\"{ctx.doc('tower-size-factorial-v1.md')}\">tower-size-factorial-v1.md</a>, "
        f"<a href=\"{ctx.raw('tower-size-factorial-v1.html')}\">original report</a>, "
        f"<a href=\"{ctx.raw('tower-size-factorial-curves.svg')}\">static curve figure</a>.</li>"
        "</ul>")
    kpi = {"top_label": _arm_label(top), "top": top["gold_macro_auroc"], "clear": clear, "n_single": len(single)}
    return (section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                    table_summary=f"Arms and contrasts ({len(arms)} pairings)"), {f"data-{sid}": blob}, kpi)


# --------------------------------------------------------------------------- 2. curves


def tower_curves_section(ctx) -> tuple[str, dict]:
    sid = "tower-curves"
    j = data.read_json(FACTORIAL)
    if not j:
        return section(sid, "Contrastive curves by pairing", notice(f"<code>{FACTORIAL}</code> is missing.")), {}
    series_loss, series_r1, missing, pools, min_epochs = [], [], [], set(), []
    for key, a in j["arms"].items():
        ref = curve_ref(a.get("checkpoint"))
        curves = data.load_curves(ref[0]) if ref else None
        if not ref or not curves or ref[1] not in curves["runs"]:
            missing.append(_arm_label(a))
            continue
        pools.add(curves["runs"][ref[1]].get("val_candidates"))
        common = {"id": key, "label": _arm_label(a), "group": _img_size(a),
                  "dash": "5 3" if _txt_size(a) == "small" else None}
        series_loss.append(dict(common, points=data.curve_series(curves, ref[1], "validation_loss")))
        series_r1.append(dict(common, points=data.curve_series(curves, ref[1], "mean_r@1")))
        min_epochs.append(a.get("minimum_validation_loss_epoch"))
    if not series_r1:
        return section(sid, "Contrastive curves by pairing", notice(
            "Curve snapshots are missing. Run <code>build_digest_site.py export</code>.")), {}
    if len(pools) > 1:
        return section(sid, "Contrastive curves by pairing", notice(
            "The four pairings validate on different candidate pools, so their R@1 curves are not drawn on one axis.")), {}
    pool = next(iter(pools))
    arms = j["arms"]
    top_r1 = max(arms.values(), key=lambda a: a["best_mean_r1"])
    top_auc = max(arms.values(), key=lambda a: a["gold_macro_auroc"])
    rises = all(a["final_validation_loss"] > a["minimum_validation_loss"] for a in arms.values())
    claim = (f"Retrieval ranks the pairings differently: {_arm_label(top_r1)} peaks highest on validation R@1, "
             f"{_arm_label(top_auc)} on the gold probe"
             + (f"; every pairing's validation loss bottoms out by epoch {max(e for e in min_epochs if e)} and then rises" if rises and min_epochs else ""))
    blob = {"groups": IMG_GROUPS, "loss": series_loss, "r1": series_r1}
    body = (
        '<div data-role="filter"></div>\n<div class="grid-2">\n'
        f"<div>{chart('loss', 'Validation contrastive loss by epoch')}</div>\n"
        f"<div>{chart('r1', f'Validation mean R@1 by epoch ({pool} candidates)')}</div>\n</div>\n"
        '<div data-role="legend"></div>\n'
        '<p class="legend-hint">Solid = E5-base text tower, dashed = E5-small. Legend values are peak R@1.</p>\n'
        + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.lineChart(sec.querySelector('[data-chart=loss]'), d.loss, {groups: d.groups, yLabel: 'Validation loss', yFmt: '.2f', best: 'min', store: store});
  Charts.lineChart(sec.querySelector('[data-chart=r1]'), d.r1, {groups: d.groups, yLabel: 'Validation mean R@1', yFmt: '.1%', best: 'max', store: store});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.r1, store, {groups: d.groups, best: 'max', fmt: '.1%'});"""))
    caption = (f"Per-epoch <code>metrics.jsonl</code> snapshots of the four checkpoints in <code>{FACTORIAL}</code>; "
               f"{pool}-candidate gold-holdout validation split, single seed. Checkpoints were selected on validation R@1.")
    notes = ("<ul><li>Retrieval R@1 and gold AUROC are different tasks on different populations; they share this page but never a table.</li>"
             "<li>The CUDA-nondeterminism noise floor for selected mean R@1 is about 0.024 for a same-config rerun (see the label-similarity "
             "section), larger than most gaps between these curves.</li>"
             + (f"<li>No snapshot for: {esc(', '.join(missing))}.</li>" if missing else "")
             + "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 3. size-matched


def size_matched_section(ctx) -> tuple[str, dict, dict]:
    sid = "size-matched"
    j = data.read_json(SIZE_MATCHED)
    if not j:
        return section(sid, "Size-matched swap", notice(f"<code>{SIZE_MATCHED}</code> is missing.")), {}, {}
    g, models = j["gold"], j["models"]
    pb = g["paired_bootstrap"]
    lo, hi = pb["macro_auroc_difference_ci95"]
    ex, sm = models["existing"], models["size_matched"]
    groups = {"lower": {"label": "Swap scores lower", "color": "var(--bad)"},
              "higher": {"label": "Swap scores equal or higher", "color": "var(--good)"}}
    per = [{"id": lab, "label": f"{lab} ({v['positives']}+)", "group": "lower" if v["difference"] < 0 else "higher",
            "value": v["difference"], "existing": v["existing"], "swap": v["size_matched"]}
           for lab, v in g["per_label"].items()]
    lvl_groups = {"trained": {"label": "Contrastively trained", "color": "var(--g-lora)"},
                  "floor": {"label": "Unadapted floor", "color": "var(--g-anchor)"}}
    levels = [
        {"id": "existing", "label": f"{_short(ex['image'])} + {_short(ex['text'])}", "group": "trained", "value": g["existing_macro_auroc"]},
        {"id": "swap", "label": f"{_short(sm['image'])} + {_short(sm['text'])}", "group": "trained", "value": g["size_matched_macro_auroc"]},
        {"id": "floor-small", "label": f"{_short(ex['image'])}, unadapted", "group": "floor", "value": g["unadapted_small_macro_auroc"]},
        {"id": "floor-base", "label": f"{_short(sm['image'])}, unadapted", "group": "floor", "value": g["unadapted_base_macro_auroc"]},
    ]
    n_lower = sum(1 for r in per if r["value"] < 0)
    claim = (f"Swapping to closer tower sizes ({_short(sm['image'])} + {_short(sm['text'])}) drops gold AUROC "
             f"{num(g['existing_macro_auroc'])} → {num(g['size_matched_macro_auroc'])} "
             f"({num(g['difference'], '.4f', True)}, CI {_ci(lo, hi)}) "
             + status_badge("superseded"))
    blob = {"groups": groups, "lvlGroups": lvl_groups, "per": per, "levels": levels}
    body = (
        '<div class="grid-2">\n'
        f"<div>{chart('levels', 'Gold macro AUROC (each floor labelled; no shared line)')}</div>\n"
        f"<div>{chart('per', f'Per-label AUROC change, swap − existing ({n_lower} of {len(per)} labels lower)')}</div>\n</div>\n"
        + section_script(sid, """
  Charts.barChart(sec.querySelector('[data-chart=levels]'), d.levels, {groups: d.lvlGroups, fmt: '.4f', axisFmt: '.2f', xLabel: 'Gold macro AUROC', store: store, labelWidth: 220});
  Charts.barChart(sec.querySelector('[data-chart=per]'), d.per, {groups: d.groups, fmt: '+.3f', axisFmt: '+.2f', ref: 0, xLabel: '\\u0394 AUROC', store: store, sort: 'value',
    tooltipExtra: function (r) { return '<div>existing ' + Charts.fmt('.3f')(r.existing) + ' \\u2192 swap ' + Charts.fmt('.3f')(r.swap) + '</div>'; }});"""))
    r = j["retrieval"]
    caption = (f"Same {j['population']['population_size']} gold studies and probe as the factorial; this swap became its "
               f"{_short(sm['image'])} + {_short(sm['text'])} arm. Validation R@1 moved the other way "
               f"({num(r['existing_best_mean_r1'], '.1%')} → {num(r['size_matched_best_mean_r1'], '.1%')}).")
    notes = ("<ul>"
             f"<li>Backbone parameters: existing image {ex['image_backbone_parameters']:,} / text {ex['text_backbone_parameters']:,}; "
             f"swap image {sm['image_backbone_parameters']:,} / text {sm['text_backbone_parameters']:,}.</li>"
             "<li>Single seed per arm; per-label AUROCs rest on as few as 8 positives (MCL) and are much noisier than the macro value. "
             "Row labels show positives out of 56.</li>"
             "<li>Superseded: chronologically this ran first and was then absorbed as one arm of the 2×2 factorial above.</li>"
             f"<li>Sources: <a href=\"{ctx.raw(SIZE_MATCHED)}\">{SIZE_MATCHED}</a>, "
             f"<a href=\"{ctx.raw('size-matched-holdout-probe.json')}\">size-matched-holdout-probe.json</a>, "
             f"<a href=\"{ctx.raw('size-matched-holdout-v1.html')}\">original report</a>.</li></ul>")
    tbl = table(["Label", "Positives", "Existing", "Swap", "Δ"],
                [[r_["id"], g["per_label"][r_["id"]]["positives"], num(r_["existing"]), num(r_["swap"]), num(r_["value"], ".4f", True)]
                 for r_ in per], num_cols=(1, 2, 3, 4),
                row_attrs=[{"data-series-id": r_["id"], "data-group": r_["group"]} for r_ in per])
    kpi = {"diff": g["difference"], "lo": lo, "hi": hi}
    return (section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Per-label table (12 labels)"),
            {f"data-{sid}": blob}, kpi)


# --------------------------------------------------------------------------- 4. label similarity


def _ls_run(arm_key: str) -> str:
    """'lambda01/lambda01-seed2026/uniform_mean_768' -> 'lambda01-seed2026'."""
    return data.split_key(arm_key)[1]


def _ls_label(run: str, weight: float | None) -> str:
    base = f"λ={weight:g}" if weight is not None else run
    return base + (" low-UNK" if "lowunk" in run else "")


def label_similarity_section(ctx) -> tuple[str, dict, dict]:
    sid = "label-similarity"
    paired = data.read_json(LS_PAIRED)
    v2 = data.read_json(LS_V2)
    if not paired or not v2:
        return section(sid, "Label similarity", notice("Label-similarity JSONs are missing.")), {}, {}
    weights = {a["directory"].rstrip("/").split("/")[-1]: a.get("label_similarity_weight") for a in v2["arms"].values()}
    nf = v2.get("noise_floor") or {}
    control_dir = nf.get("control_run")
    control_ref = curve_ref((control_dir or "") + "/best.pt")

    groups = {"base": {"label": "λ=0 baseline and control rerun", "color": "var(--g-anchor)"},
              "treat": {"label": "Label-similarity treatments", "color": "var(--g-holdout)"}}
    rows = []
    for p in data.paired_comparisons(paired):
        run = _ls_run(p.candidate)
        rows.append({"id": run, "label": _ls_label(run, weights.get(run)) + " − λ=0", "group": "treat",
                     "est": p.diff, "lo": p.ci[0], "hi": p.ci[1], "cand": p.macro_candidate, "base": p.macro_baseline,
                     "inBand": abs(p.diff) < NULL_BAND, "spans": not _excludes_zero(*p.ci)})
    # curves: every arm's run dir, plus the same-config control rerun
    loss, r1, pools = [], [], set()
    for arm in v2["arms"].values():
        root, run = arm["directory"].rstrip("/").split("/")[-2:]
        c = data.load_curves(root)
        if not c or run not in c["runs"]:
            continue
        pools.add(c["runs"][run].get("val_candidates"))
        w = arm.get("label_similarity_weight")
        s = {"id": run, "label": _ls_label(run, w), "group": "base" if not w else "treat"}
        loss.append(dict(s, points=data.curve_series(c, run, "validation_loss")))
        r1.append(dict(s, points=data.curve_series(c, run, "mean_r@1")))
    if control_ref:
        c = data.load_curves(control_ref[0])
        if c and control_ref[1] in c["runs"]:
            pools.add(c["runs"][control_ref[1]].get("val_candidates"))
            s = {"id": "control", "label": "λ=0 control rerun (noise only)", "group": "base", "dash": "5 3"}
            loss.append(dict(s, points=data.curve_series(c, control_ref[1], "validation_loss")))
            r1.append(dict(s, points=data.curve_series(c, control_ref[1], "mean_r@1")))
    same_pool = len(pools) == 1
    in_band = sum(1 for r in rows if r["inBand"])
    spans = sum(1 for r in rows if r["spans"])
    lam01 = next((c for k, c in (v2.get("comparisons") or {}).items() if "lambda01" in k), {})
    ahead = lam01.get("epochs_treatment_ahead_on_validation_loss")
    epochs = lam01.get("epochs_compared")
    claim = (f"Label-similarity targets are a null: {in_band} of {len(rows)} paired gold Δ fall inside ±{NULL_BAND:g} and "
             f"{spans} of {len(rows)} intervals span zero " + status_badge("null"))
    blob = {"groups": groups, "rows": rows, "loss": group_major(loss, groups),
            "r1": group_major(r1, groups) if same_pool else [], "band": NULL_BAND}
    r1_mount = (chart('r1', f"Validation mean R@1 by epoch ({next(iter(pools))} candidates)") if same_pool
                else notice("Runs use different candidate pools; R@1 not drawn on one axis."))
    body = (
        '<div data-role="filter"></div>\n'
        f"{chart('forest', 'Paired gold macro AUROC difference vs λ=0 (null band ±' + format(NULL_BAND, 'g') + ')')}\n"
        '<div class="grid-2">\n'
        f"<div>{chart('loss', 'Validation loss by epoch (comparable across arms)')}</div>\n"
        f"<div>{r1_mount}</div>\n</div>\n"
        '<div data-role="legend"></div>\n'
        + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.ciDotPlot(sec.querySelector('[data-chart=forest]'), d.rows,
    {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.3f', sort: false,
     domain: [-Math.max(0.03, d.band * 2.5), Math.max(0.03, d.band * 2.5)],
     xLabel: '\\u0394 gold macro AUROC (treatment \\u2212 \\u03bb=0)', store: store,
     tooltipExtra: function (r) { return '<div>' + Charts.fmt('.4f')(r.base) + ' \\u2192 ' + Charts.fmt('.4f')(r.cand) + '</div><div>' +
       (r.inBand ? 'inside' : 'outside') + ' the \\u00b1' + d.band + ' null band; CI ' + (r.spans ? 'spans' : 'excludes') + ' zero</div>'; }});
  Charts.lineChart(sec.querySelector('[data-chart=loss]'), d.loss, {groups: d.groups, yLabel: 'Validation loss', yFmt: '.2f', best: 'min', store: store});
  if (d.r1.length) Charts.lineChart(sec.querySelector('[data-chart=r1]'), d.r1, {groups: d.groups, yLabel: 'Validation mean R@1', yFmt: '.1%', best: 'max', store: store});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.loss, store, {groups: d.groups, best: 'min', fmt: '.3f'});"""))
    floor = ((v2.get("downstream_probe") or {}).get("macro_auroc") or {}).get(data.UNADAPTED_KEY)
    caption = (f"λ=0.1 has the lower validation loss on {ahead} of {epochs} epochs, but the gold probe does not move: it "
               f"regularises the loss, not the features. 4-block image fine-tuning, single seed; floor {num(floor)} (unadapted, this file).")
    notes = ("<ul>"
             f"<li>Pre-registered reading: a difference inside ±{NULL_BAND:g} macro AUROC is a null at this sample size "
             f"(<code>{LS_PAIRED}</code> config).</li>"
             f"<li>Noise floor: a same-config control rerun differs from the λ=0 baseline by {num(nf.get('noise_floor_mean_r@1'), '.4f')} "
             f"in selected mean R@1; the λ=0.1 treatment effect is {num(nf.get('treatment_effect_mean_r@1'), '.4f')} "
             f"({'does not exceed' if not nf.get('treatment_effect_exceeds_noise_floor') else 'exceeds'} the floor).</li>"
             "<li><code>train_loss</code> is not comparable across arms (each soft target has its own entropy floor), so it is not drawn.</li>"
             "<li>These arms are 4-block image-only fine-tuning, not the 6-block LoRA of the factorial; do not compare levels across sections.</li>"
             f"<li>Sources: <a href=\"{ctx.doc('label-similarity-v2.md')}\">label-similarity-v2.md</a>, "
             f"<a href=\"{ctx.doc('label-similarity-v1.md')}\">label-similarity-v1.md</a>, <a href=\"{ctx.raw(LS_PAIRED)}\">{LS_PAIRED}</a>, "
             f"<a href=\"{ctx.raw(LS_V2)}\">{LS_V2}</a>.</li></ul>")
    tbl = table(["Treatment", "λ=0 AUROC", "Treatment AUROC", "Δ", "95% CI", "Inside ±0.01"],
                [[r["label"], num(r["base"]), num(r["cand"]), num(r["est"], ".4f", True), _ci(r["lo"], r["hi"]),
                  "yes" if r["inBand"] else "no"] for r in rows], num_cols=(1, 2, 3, 4),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    kpi = {"max_abs": max(abs(r["est"]) for r in rows) if rows else None, "spans": spans, "n": len(rows)}
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Paired table"), {f"data-{sid}": blob}, kpi


# --------------------------------------------------------------------------- 5. DeCLIP


def _declip_label(run: str, feature: str) -> str:
    names = {
        "declip-lora-depth6-r64-seed2026": "DeCLIP + LoRA",
        "declip-frozen-seed2026": "DeCLIP frozen",
        "declip-mv-only-frozen-seed2026": "DeCLIP MV-only frozen",
        "both-lora-depth6-r64-seed2026": "LoRA (exposed)",
        "both-lora-depth6-r64-bs64-reg-seed2026": "LoRA BS64 reg (exp.)",
        "both-unfreeze6-seed2026": "Both FT 6 blk (exp.)",
    }
    feat = {"uniform_mean_768": "", "projection_256": " · proj", "plane_attention_768": " · plane attn"}.get(feature, feature)
    return f"{names.get(run, run)}{feat}"


def _declip_group(run: str) -> str:
    return "declip" if run.startswith("declip") else ("lora" if run.startswith("both-lora") else "bothft")


def declip_section(ctx) -> tuple[str, dict, dict]:
    sid = "declip"
    j = data.read_json(DECLIP_PROBE)
    if not j:
        return section(sid, "DeCLIP", notice(f"<code>{DECLIP_PROBE}</code> is missing.")), {}, {}
    floor = data.unadapted_level(j)
    groups = {"declip": {"label": "DeCLIP", "color": "var(--g-holdout)"},
              "lora": {"label": "Exposed LoRA baselines", "color": "var(--g-lora)"},
              "bothft": {"label": "Exposed full fine-tuning", "color": "var(--g-bothft)"}}
    vs = data.probe_vs_unadapted(j)
    identity, rows = [], []
    for v in vs:
        if v.after is not None and floor is not None and abs(v.after - floor) < 1e-12:
            identity.append(v)  # frozen image tower read before its projection: equals the floor by construction
            continue
    frozen_runs = {v.run for v in identity}
    for v in vs:
        if v in identity:
            continue
        # frozen towers: only the trained projection can carry signal; adapted towers: the uniform-mean readout
        want = "projection_256" if v.run in frozen_runs else "uniform_mean_768"
        if v.feature != want:
            continue
        rows.append({"id": v.run, "label": _declip_label(v.run, v.feature), "group": _declip_group(v.run),
                     "est": v.diff, "lo": v.ci[0], "hi": v.ci[1], "after": v.after})
    rows.sort(key=lambda r: -r["est"])
    after = {(v.run, v.feature): v.after for v in vs}
    dl = after.get(("declip-lora-depth6-r64-seed2026", "uniform_mean_768"))
    bl = after.get(("both-lora-depth6-r64-seed2026", "uniform_mean_768"))

    # 441-candidate R@1 curves for the same runs (exposed cohort only on this axis)
    series, pools, skipped = [], {}, []
    prov = j.get("provenance") or {}
    for r in rows:
        ref = curve_ref((prov.get(f"after_cl/{r['id']}") or {}).get("checkpoint"))
        c = data.load_curves(ref[0]) if ref else None
        if not c or ref[1] not in c["runs"]:
            skipped.append(r["label"])
            continue
        run = c["runs"][ref[1]]
        if len(run["epoch"]) > 20:
            skipped.append(f"{r['label']} ({len(run['epoch'])}-epoch schedule)")
            continue
        pools.setdefault(run.get("val_candidates"), []).append(
            {"id": r["id"], "label": r["label"].split(" · ")[0], "group": r["group"],
             "points": data.curve_series(c, ref[1], "mean_r@1")})
    pool, series = (max(pools.items(), key=lambda kv: len(kv[1])) if pools else (None, []))
    for p, s in pools.items():
        if p != pool:
            skipped.extend(x["label"] + f" ({p} candidates)" for x in s)

    claim = ((f"DeCLIP + LoRA ({num(dl)}) lands {num(abs(dl - bl), '.4f')} {'below' if dl < bl else 'above'} the exposed LoRA baseline ({num(bl)}) it extends, with no paired interval"
              if dl is not None and bl is not None else "DeCLIP versus the exposed LoRA baseline")
             + " " + caveat_badge("exposed cohort") + " " + caveat_badge("single seed") + " " + caveat_badge("development statistic"))
    blob = {"groups": groups, "rows": group_major(rows, groups), "series": group_major(series, groups)}
    body = (
        '<div data-role="filter"></div>\n<div class="grid-2">\n'
        f"<div>{chart('forest', 'Gold macro AUROC minus unadapted DINOv2, 95% CI')}</div>\n"
        f"<div>{chart('r1', f'Validation mean R@1 by epoch ({pool} candidates, exposed cohort)') if series else notice('No curve snapshots.')}</div>\n</div>\n"
        '<div data-role="legend"></div>\n'
        + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.ciDotPlot(sec.querySelector('[data-chart=forest]'), d.rows,
    {groups: d.groups, ref: 0, refLabel: 'unadapted', fmt: '+.4f', axisFmt: '+.2f', xLabel: '\\u0394 vs unadapted DINOv2', store: store, labelWidth: 250,
     tooltipExtra: function (r) { return '<div>Level ' + Charts.fmt('.4f')(r.after) + '</div>'; }});
  if (d.series.length) {
    Charts.lineChart(sec.querySelector('[data-chart=r1]'), d.series, {groups: d.groups, yLabel: 'Validation mean R@1', yFmt: '.1%', best: 'max', store: store});
    Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {groups: d.groups, best: 'max', fmt: '.1%'});
  }"""))
    pop = j.get("population") or {}
    caption = (f"{pop.get('population_size')} gold studies that were in this encoder's training/validation splits "
               f"(exposed), unadapted floor {num(floor)}; single seed, no paired DeCLIP-vs-LoRA interval. "
               f"The exposed LoRA baseline is not the holdout 0.7594 encoder.")
    notes = ("<ul>"
             "<li>Off the gold-holdout lineage: DeCLIP trained on the exposed cohort, so its gold readout is a development statistic, "
             "not a generalisation estimate. Nothing downstream builds on it.</li>"
             f"<li>Identity checks (structural zeros, not results): {esc(', '.join(_declip_label(v.run, v.feature) for v in identity))} "
             f"equal the unadapted floor exactly, because a frozen image tower read before its projection is the unadapted backbone. "
             "Frozen arms are therefore shown on their trained 256-d projection.</li>"
             "<li>The legacy report's “record” and “all-time high” wording is single-seed and unpaired and is not reproduced here.</li>"
             "<li>R@1 here uses the exposed 441-candidate pool; holdout runs (433) are never drawn on this axis."
             + (f" Not drawn: {esc('; '.join(skipped))}." if skipped else "") + "</li>"
             f"<li>Sources: <a href=\"{ctx.doc('declip-alignment-results.md')}\">declip-alignment-results.md</a>, "
             f"<a href=\"{ctx.raw(DECLIP_PROBE)}\">{DECLIP_PROBE}</a>, <a href=\"{ctx.raw('declip-alignment-v1.html')}\">original report</a>.</li></ul>")
    tbl = table(["Arm", "Readout", "Gold macro AUROC", "Δ vs unadapted", "95% CI"],
                [[_declip_label(v.run, v.feature).split(" · ")[0], v.feature, num(v.after), num(v.diff, ".4f", True), _ci(*v.ci)]
                 for v in sorted(vs, key=lambda v: (v.run, v.feature))], num_cols=(2, 3, 4),
                row_attrs=[{"data-series-id": v.run, "data-group": _declip_group(v.run)} for v in sorted(vs, key=lambda v: (v.run, v.feature))])
    return (section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                    table_summary=f"All readouts ({len(vs)}, incl. identity checks)"), {f"data-{sid}": blob}, {"declip_lora": dl, "exposed_lora": bl})


# --------------------------------------------------------------------------- 6. codec pipeline + DeCLIP (gold holdout)

CODEC_GROUPS = {"baseline": {"label": "DINOv2-small + E5-base CLIP (reference pair)", "color": "var(--g-teal)"},
                "clip": {"label": "CLIP tower swaps", "color": "var(--g-violet)"},
                "declip": {"label": "DeCLIP", "color": "var(--g-amber)"}}


def _codec_group(key: str, a: dict) -> str:
    if a["objective"].startswith("DeCLIP"):
        return "declip"
    return "baseline" if key in ("reference", "replicate") else "clip"


def _codec_short(tower: str) -> str:
    return {"DINOv2-small": "DINOv2-S", "RAD-DINO": "RAD", "E5-base": "E5", "BGE-M3 · 512": "BGE"}.get(tower, tower)


def codec_declip_section(ctx) -> tuple[str, dict, dict]:
    sid = "codec-declip"
    j = data.read_json(CODEC)
    if not j:
        return section(sid, "Codec pipeline and DeCLIP", notice(f"<code>{CODEC}</code> is missing.")), {}, {}
    arms_j = j["arms"]
    labels = list(next(iter(arms_j.values()))["gold_per_label_auroc"]["mlp"])
    baselines = {img: {"label": b["label"], "short": f"{_codec_short(img)} unadapted",
                       "per_label": b["gold_per_label_auroc"]["mlp"]}
                 for img, b in (j.get("unadapted") or {}).items()}
    arms = []
    for key, a in arms_j.items():
        ref = curve_ref(a["contrastive_run"] + "/best.pt")
        short = f"{_codec_short(a['image'])}+{_codec_short(a['text'])} {a['objective'].replace('codec pipeline', 'codec')}"
        arms.append({"id": key, "label": f"{a['image']} + {a['text']} · {a['objective']}", "short": short,
                     "group": _codec_group(key, a),
                     "curve_root": ref[0], "curve_run": ref[1], "per_label": a["gold_per_label_auroc"]["mlp"],
                     "baseline": a.get("unadapted")})
    body, blob, missing = experiment_views(sid, arms, CODEC_GROUPS, labels=labels, baselines=baselines,
                                           positives=j.get("gold_positives"),
                                           train_loss_title="Training in-batch loss (DeCLIP: composite objective)")
    no_base = sorted({a["label"] for a in arms if a["baseline"] not in baselines})

    ref, rep = arms_j.get("reference"), arms_j.get("replicate")
    lo_mlp = min(ref["lm_test_macro_auroc"]["mlp"], rep["lm_test_macro_auroc"]["mlp"]) if ref and rep else None
    hi_mlp = max(ref["lm_test_macro_auroc"]["mlp"], rep["lm_test_macro_auroc"]["mlp"]) if ref and rep else None
    def lift(image, kind="mlp"):
        ds = [a["gold_delta_vs_unadapted"][kind]["difference"] for a in arms_j.values()
              if a.get("unadapted") == image and a.get("gold_delta_vs_unadapted")]
        return f"{num(min(ds), '.3f', True)} to {num(max(ds), '.3f', True)}" if ds else None

    dino, rad = lift("DINOv2-small"), lift("RAD-DINO")
    linear = [a["gold_delta_vs_unadapted"]["linear"] for a in arms_j.values() if a.get("gold_delta_vs_unadapted")]
    linear_null = linear and not any(_excludes_zero(*d["ci95"]) for d in linear)
    claim = ("Codec replicate and DeCLIP are neutral"
             + (f"; with MLP heads, contrastive training lifts DINOv2-small {dino} gold AUROC over its pretrained tower "
                f"and leaves RAD-DINO flat ({rad})" if dino and rad else "")
             + ("; with linear heads no arm beats its pretrained tower" if linear_null else "")
             + " " + caveat_badge("single contrastive seed") + " " + caveat_badge("head-dependent"))

    def delta(a, key, kind="mlp"):
        d = (a.get(key) or {}).get(kind)
        return "—" if not d else f"{num(d['difference'], '.4f', True)} {_ci(*d['ci95'])}"

    rows, attrs = [], []
    for key, a in arms_j.items():
        c = a["contrastive"]
        rows.append([f"{a['image']} + {a['text']}", a["objective"], str(c["selected_epoch"]), num(c["mean_r1"], ".1%"),
                     num(a["gold_macro_auroc"]["mlp"]), delta(a, "gold_delta_vs_reference"),
                     delta(a, "gold_delta_vs_unadapted"), delta(a, "gold_delta_vs_unadapted", "linear"),
                     num(a["lm_test_macro_auroc"]["mlp"]),
                     delta(a, "lm_test_delta_vs_reference")])
        attrs.append({"data-series-id": key, "data-group": _codec_group(key, a)})
    tbl = table(["Pair", "Objective", "Sel. epoch", "Val mean R@1", "Gold MLP", "Δ gold vs reference [95% CI]",
                 "Δ gold MLP vs unadapted tower [95% CI]", "Δ gold linear vs unadapted tower [95% CI]",
                 f"LM-test MLP ({j['lm_test_studies']})", "Δ LM-test vs reference [95% CI]"],
                rows, num_cols=(2, 3, 4, 8), row_attrs=attrs)
    base_txt = "; ".join(f"{b['label']} gold MLP {num(b['gold_macro_auroc']['mlp'])}, linear {num(b['gold_macro_auroc']['linear'])}"
                         for b in (j.get("unadapted") or {}).values())
    caption = (f"Gold-holdout cohort, contrastive seed {j['contrastive_seed']}, 58-case LM-label heads averaged over seeds "
               f"{', '.join(map(str, j['head_seeds']))}. Heatmap rows: label (gold positives of {j['gold_studies']}); "
               "cells default to the MLP-head gold AUROC minus the same head recipe on the arm's own pretrained image tower.")
    noise = j.get("replicate_noise") or {}
    notes = ("<ul>"
             "<li>DeCLIP training loss is the composite objective (multi-view InfoNCE + 0.5 × nearest-neighbour + 0.5 × image SimCLR); "
             "compare it only within the DeCLIP group. Validation loss is plain symmetric InfoNCE over the full pool for every arm.</li>"
             f"<li>The reference and codec replicate have bit-identical inputs, so they bracket single-seed noise: LM-test MLP "
             f"{num(lo_mlp)}–{num(hi_mlp)}, selected mean R@1 gap {num(noise.get('selected_mean_r1_gap'), '.4f', True)}.</li>"
             f"<li>Unadapted anchors ({esc(base_txt) or 'none'}) use the same 58-case head recipe on features from the pretrained "
             "image tower (<code>image_baseline</code>); their LM-test score is not computed.</li>"
             "<li>The lift over the pretrained tower depends on the head: on unadapted DINOv2-small features the MLP scores below the "
             "linear head, while on adapted features it scores above it. Linear-head deltas are in the table; the heatmap shows MLP heads.</li>"
             "<li>Per-label cells rest on few gold positives; the 58 gold labels have been examined repeatedly, so treat them as exploratory.</li>"
             + (f"<li>No unadapted anchor yet for: {esc(', '.join(no_base))}.</li>" if no_base else "")
             + (f"<li>Curve snapshots pending: {esc(', '.join(missing))}.</li>" if missing else "")
             + f"<li>Sources: <a href=\"{ctx.raw(CODEC)}\">{CODEC}</a>, <a href=\"{ctx.doc(CODEC_MD)}\">original report</a>.</li></ul>")
    kpi = {"declip_gold": (arms_j.get("declip_dinov2") or {}).get("gold_delta_vs_reference")}
    return (section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                    table_summary=f"All {len(arms_j)} arms"), {f"data-{sid}": blob}, kpi)


# --------------------------------------------------------------------------- 7. next


def next_section(ctx) -> tuple[str, dict]:
    sid = "next"
    text = data.read_text(NEXT_MD)
    rows = md_table(text, "| Order")  # the table has no heading of its own; start at its header row
    if not rows:
        return section(sid, "What's next", notice(f"<code>{NEXT_MD}</code> has no candidate table.")), {}
    keys = list(rows[0].keys())
    pair_k = next((k for k in keys if k.lower().startswith("pair")), keys[1])
    q_k = next((k for k in keys if k.lower().startswith("question")), keys[2])

    def plain(cell: str) -> str:
        cell = re.sub(r"\[`?([^`\]]+)`?\]\([^)]+\)", r"\1", cell)
        return cell.replace("`", "").replace("**", "")

    tbl = table(["#", "Pair to test against the reference", "Question"],
                [[r.get(keys[0], ""), plain(r.get(pair_k, "")), plain(r.get(q_k, ""))] for r in rows])
    sources = [("reference", "DINOv2-small + E5-base", "both-lora-holdout-v1",
                "holdout-both-lora-depth6-r64-seed2026")]
    sources += [(name, name.replace("-seed2026", ""), "next-model-pairs-v1", name)
                for name in NEXT_RUNS]
    metrics = {"train": "train_loss", "val": "validation_loss",
               "itot": "image_to_text_r@1", "ttoi": "text_to_image_r@1"}
    series = {key: [] for key in metrics}
    missing = []
    for name, label, root, run in sources:
        curves = data.load_curves(root)
        if not curves or run not in curves.get("runs", {}):
            missing.append(label)
            continue
        for key, metric in metrics.items():
            points = data.curve_series(curves, run, metric)
            if points:
                series[key].append({"id": name, "label": label, "group": "reference" if name == "reference" else "candidate",
                                    "points": points})
    groups = {"reference": {"label": "Selected reference", "color": "var(--g-teal)"},
              "candidate": {"label": "Next model pairs", "color": "var(--g-violet)"}}
    claim = f"Next: {len(rows)} one-tower-at-a-time swaps against DINOv2-small + E5-base"
    results = data.read_json(NEXT_JSON)
    outcome = ""
    if results:
        arms = results["arms"]
        claim = "Tower swaps: larger text towers lead on AUROC, but no gain excludes zero; RAD-DINO is worse"
        result_rows = []
        for key, a in arms.items():
            d = a.get("gold_delta_vs_reference", {}).get("mlp")
            result_rows.append([f"{a['image']} + {a['text']}", num(a["contrastive"]["mean_r1"], ".1%"),
                                num(a["gold_macro_auroc"]["linear"]), num(a["gold_macro_auroc"]["mlp"]),
                                "reference" if d is None else f"{num(d['difference'], '.4f', True)} {_ci(*d['ci95'])}",
                                num(a["lm_test_macro_auroc"]["mlp"])])
        outcome = ("<h3>Outcome (contrastive seed 2026; 3-seed LM heads)</h3>"
                   + table(["Pair", "Val mean R@1", "Gold linear", "Gold MLP", "Δ gold MLP vs reference [95% CI]",
                            f"LM-test MLP ({results['lm_test_studies']})"],
                           result_rows, num_cols=(1, 2, 3, 5)))
    plots = "".join(f"<div>{chart(key, title) if series[key] else notice('No measured curve yet.')}</div>"
                    for key, title in (("train", "Training in-batch loss by epoch"),
                                       ("val", "Validation full-pool loss by epoch"),
                                       ("itot", "Image → text validation Recall@1"),
                                       ("ttoi", "Text → image validation Recall@1")))
    body = (outcome + tbl + f'<p>Training settings and 58-case outcome table: <a href="{ctx.doc(NEXT_RESULTS)}">'
            f'{NEXT_RESULTS}</a>.</p><div data-role="filter"></div><div class="grid-2">{plots}</div>'
            '<div data-role="legend"></div>' + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  [['train', 'Training loss', '.2f', 'min'], ['val', 'Validation loss', '.2f', 'min'],
   ['itot', 'I→T Recall@1', '.1%', 'max'], ['ttoi', 'T→I Recall@1', '.1%', 'max']].forEach(function (spec) {
    if (d[spec[0]].length) Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.itot, store,
    {groups: d.groups, best: 'max', fmt: '.1%'});"""))
    caption = (f"Queue and run protocol from <a href=\"{ctx.doc(NEXT_MD)}\">next-model-candidates.md</a>. Its reference score is the "
               "58-case LM-head protocol, not the 56-case OOF used above; compare candidates only within one protocol.")
    notes = ("<ul><li>Training loss is in-batch; validation loss is over the full candidate pool. Do not read them as a matched train/validation loss pair.</li>"
             "<li>Change one tower at a time, pin revisions, keep the gold-holdout cohort and the R@1 checkpoint rule, "
             "and run more than one contrastive seed where feasible.</li>"
             "<li>Gold differences stay exploratory: all 58 gold labels have been examined repeatedly.</li>"
             + (f"<li>Curve snapshots pending: {esc(', '.join(missing))}.</li>" if missing else "") + "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": {"groups": groups, **series}}


FACTOR_GROUPS = {"reference": {"label": "Codec reference", "color": "var(--g-anchor)"},
                 "pooling": {"label": "Uniform pooling", "color": "var(--g-violet)"},
                 "windows": {"label": "K8 windows", "color": "var(--g-teal)"},
                 "resolution": {"label": "336 px", "color": "var(--g-amber)"},
                 "model": {"label": "OmniRad-base", "color": "var(--g-rose)"}}
FACTOR_GROUP_OF = {"reference": "reference", "dinov2-small-uniform-224": "pooling",
                   "dinov2-small-k8-224": "windows", "dinov2-small-336": "resolution",
                   "omnirad-base-lora-224": "model"}


def factor_comparisons(ctx, snapshot):
    sid = "factor-isolation-comparisons"
    blob = {"groups": FACTOR_GROUPS, "paired": {}, "levels": {}}
    rows = []
    def delta(arm, split, comparator):
        value = arm["paired"][split].get(comparator)
        return (f"{num(value['difference'], '.4f', True)} {_ci(*value['ci95'])}" if value else "pending")
    for split, key in (("gold", "gold_mlp"), ("lm_test", "lm_test_mlp")):
        blob["levels"][split] = [{"id": name, "label": name, "group": FACTOR_GROUP_OF[name], "value": arm[key]}
            for name, arm in snapshot.items() if arm[key] is not None]
        blob["paired"][split] = {}
        for comparator in ("reference", "comparator"):
            blob["paired"][split][comparator] = [{"id": name, "label": name + " vs " +
                ("codec reference" if comparator == "reference" else arm["comparator"]),
                "group": FACTOR_GROUP_OF[name], "est": arm["paired"][split][comparator]["difference"],
                "lo": arm["paired"][split][comparator]["ci95"][0],
                "hi": arm["paired"][split][comparator]["ci95"][1]}
                for name, arm in snapshot.items() if comparator in arm["paired"][split]]
    for name, arm in snapshot.items():
        is_reference = name == "reference"
        rows.append([name, arm["purpose"], arm["comparator"] or "baseline",
            num(arm["gold_mlp"]) if arm["gold_mlp"] is not None else "pending",
            "baseline" if is_reference else delta(arm, "gold", "reference"),
            "baseline" if is_reference else delta(arm, "gold", "comparator"),
            num(arm["lm_test_mlp"]) if arm["lm_test_mlp"] is not None else "pending",
            "baseline" if is_reference else delta(arm, "lm_test", "reference"),
            "baseline" if is_reference else delta(arm, "lm_test", "comparator")])
    plots = '<div class="grid-2">' + ''.join('<div>' + chart(key, title) + '</div>' for key, title in (
        ("gold-level", "Gold-58 macro AUROC"), ("lm-level", "LM-test macro AUROC (432 studies)"))) + '</div>'
    plots += '<div data-role="comparator"></div><div class="grid-2">' + ''.join('<div>' + chart(key, title) + '</div>'
        for key, title in (("gold-delta", "Δ gold macro AUROC, paired 95% CI"),
                           ("lm-delta", "Δ LM-test macro AUROC, paired 95% CI"))) + '</div>'
    body = HELPERS + '<div data-role="filter"></div>' + plots + section_script(sid, """
    Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
    [['gold-level', 'gold'], ['lm-level', 'lm_test']].forEach(function (spec) {
      Charts.barChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d.levels[spec[1]],
        {groups: d.groups, fmt: '.4f', axisFmt: '.2f', store: store, labelWidth: 235});
    });
    function draw(comparator) {
      [['gold-delta', 'gold'], ['lm-delta', 'lm_test']].forEach(function (spec) {
        var el = AdaptUI.remount(sec.querySelector('[data-chart=' + spec[0] + ']'));
        var rows = d.paired[spec[1]][comparator];
        if (!rows.length) { el.textContent = 'Paired comparisons pending.'; return; }
        Charts.ciDotPlot(el, rows, {groups: d.groups, ref: 0, refLabel: 'no difference',
          fmt: '+.4f', axisFmt: '+.3f', sort: false, store: store, labelWidth: 280,
          xLabel: 'Candidate minus ' + (comparator === 'reference' ? 'codec reference' : 'matched comparator')});
      });
    }
    AdaptUI.seg(sec.querySelector('[data-role=comparator]'), 'Compare with',
      [{key: 'reference', label: 'Codec reference'}, {key: 'comparator', label: 'Matched comparator'}],
      'reference', draw);
    draw('reference');""")
    pending = [name for name, arm in snapshot.items() if arm["lm_test_mlp"] is None]
    caption = ("Reference = codec DINOv2-small + E5-base, attention pooling, one stack per plane, 224 px. "
               "The matched K8 comparator uses uniform pooling at 224 px. Paired study bootstrap: 2,000 resamples, "
               "seed 2030. Gold-58 is exploratory; LM-test measures agreement with LM verdicts.")
    if pending:
        body += '<p>Head evaluation pending: ' + esc(', '.join(pending)) + '.</p>'
    tbl = table(["Arm", "Change", "Matched comparator", "Gold-58", "Δ gold vs reference [95% CI]",
                 "Δ gold vs matched [95% CI]", "LM-test", "Δ LM-test vs reference [95% CI]",
                 "Δ LM-test vs matched [95% CI]"], rows, num_cols=(3, 6))
    return section(sid, "Factor isolation: pathology scores and paired differences", body, caption=caption,
        table_html=tbl, table_summary="All arms, reference deltas and matched comparisons"), {"data-" + sid: blob}


def factor_isolation_section(ctx) -> tuple[str, dict]:
    snapshot = data.read_json("factor-isolation-v1.json")
    if not snapshot:
        return "", {}
    groups = FACTOR_GROUPS
    arms, baselines = [], {}
    for name, arm in snapshot.items():
        floor = arm["unadapted_head"]
        if arm["unadapted_per_label"]:
            baselines.setdefault(floor, {"label": floor.removeprefix("lm-mlp-"),
                "short": "DINOv2-S unadapted" if name == "reference" else name + " unadapted",
                "per_label": arm["unadapted_per_label"]})
        arms.append({"id": name, "label": name, "short": name, "group": FACTOR_GROUP_OF[name],
            "curve_root": "codec-declip-v1" if name == "reference" else "factor-isolation-v1",
            "curve_run": "dinov2-small-e5-base-512-codec-seed2026" if name == "reference" else name,
            "per_label": arm["gold_per_label"], "baseline": floor})
    reference = snapshot["reference"]
    plots, blob, missing = experiment_views("factor-isolation", arms, groups,
        labels=list(reference["gold_per_label"]), baselines=baselines, default_mode="reference", reference_id="reference",
        positives=reference["gold_positives"], lm_test_observed=reference["lm_test_observed"],
        lm_test_per_label={name: arm["lm_test_per_label"] for name, arm in snapshot.items()})
    for metric in ("train_masked_bce", "validation_masked_bce"):
        blob[metric] = [{"id": name + "-" + seed, "label": name + " · " + seed,
            "group": FACTOR_GROUP_OF[name], "points": [[row["epoch"], row[metric]] for row in saved["history"]]}
            for name, arm in snapshot.items() for seed, saved in arm["head_histories"].items()]
    plots += ('<h3>Frozen-tower MLP head loss (three seeds)</h3><div class="grid-2">'
        + '<div>' + chart("head-train", "Head training masked BCE") + '</div>'
        + '<div>' + chart("head-val", "Head validation masked BCE") + '</div></div>'
        + section_script("factor-isolation", """
    [['head-train', 'train_masked_bce', 'Head training BCE'],
     ['head-val', 'validation_masked_bce', 'Head validation BCE']].forEach(function (spec) {
      Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[1]],
        {groups: d.groups, yLabel: spec[2], yFmt: '.3f', best: 'min', store: store});
    });"""))
    head_rows = []
    for name, arm in snapshot.items():
        for seed, saved in arm["head_histories"].items():
            selected = next(row for row in saved["history"] if row["epoch"] == saved["best_epoch"])
            head_rows.append([name, seed.removeprefix("mlp-").removesuffix("-history"), saved["best_epoch"],
                len(saved["history"]), num(selected["train_masked_bce"]), num(selected["validation_masked_bce"])])
    head_table = table(["Arm", "Head seed", "Selected epoch", "Completed epochs", "Training BCE at selection",
                        "Validation BCE at selection"], head_rows, num_cols=(2, 3, 4, 5))
    head_selection = section("factor-head-selection", "Frozen-tower heads: selected epochs and validation loss",
        '<p>Three MLP seeds per completed arm; each selects minimum validation masked BCE. ' 
        'Gold and LM-test AUROC are evaluated afterward.</p>', table_html=head_table,
        table_summary="Head fitting and selection, one row per seed")
    settings_rows = []
    for name, arm in snapshot.items():
        cfg = arm["settings"]
        settings_rows.append([name, cfg["image_model"], cfg["text_model"], cfg["image_size"], cfg["windows_per_plane"] or 1,
            cfg["image_plane_pooling"], cfg["batch_size"], cfg["image_unfreeze_last"], cfg["text_unfreeze_last"],
            f"{cfg['lora_rank']}/{cfg['lora_alpha']}", cfg["lr_image"], cfg["lr_text"], cfg["lr_projection"], cfg["weight_decay"], cfg["warmup_fraction"], cfg["seed"],
            arm["contrastive"]["selected_epoch"], num(arm["contrastive"]["mean_r@1"])])
    settings = table(["Arm", "Image model", "Text model", "Input px", "Windows/plane", "Plane pooling", "Batch",
        "Image blocks", "Text blocks", "LoRA rank/alpha", "Image LR", "Text LR", "Projection LR", "Weight decay", "Warmup fraction", "Seed", "Selected epoch", "Mean R@1"],
        settings_rows, num_cols=(3,4,6,7,8,10,11,12,13,14,15,16,17))
    notes = ("<p>Heatmaps show Δ vs the codec reference by default on both Gold-58 and LM-test; AUROC shows raw levels. "
        "Gold also offers Δ vs each arm's unadapted tower when that floor has been evaluated. "
        "Per-label deltas are point differences, not confidence intervals; paired macro intervals are in the comparison card. "
        "Missing heads and floors stay blank. Gold row counts are positives; LM row counts are observed non-UNK labels.</p>"
        "<p>One contrastive seed per arm; three head seeds. The K8 difference from codec reference changes both windows and "
        "pooling; only K8 versus uniform-224 isolates windows. OmniRad changes pretraining and size. "
        "Training loss is augmented in-batch InfoNCE; validation uses the full pool. "
        "Head selection is validation masked BCE, never gold AUROC.</p>"
        f'<p>Sources: <a href="{ctx.doc("factor-isolation-v1.md")}">results</a>, '
        f'<a href="{ctx.raw("factor-isolation-v1.json")}">verified predictions, comparisons and histories</a>.</p>')
    comparisons, paired_blob = factor_comparisons(ctx, snapshot)
    return comparisons + section("factor-isolation", "Factor isolation: curves, per-label reference deltas and head training",
        plots, notes=notes, table_html=settings, table_summary="Run settings and retrieval selection") + head_selection, {
            **paired_blob, "data-factor-isolation": blob}


COMBINED_GROUPS = {"reference": {"label": "Codec reference (224, 1 stack)", "color": "var(--g-anchor)"},
                   "windows": {"label": "K8 at 224 (factor isolation)", "color": "var(--g-teal)"},
                   "combined-omnirad": {"label": "OmniRad-base · LoRA + K8 + 336", "color": "var(--g-rose)"},
                   "combined-dinov2": {"label": "DINOv2-small · LoRA + K8 + 336", "color": "var(--g-violet)"}}


def combined_lora_section(ctx) -> tuple[str, dict]:
    snapshot = data.read_json("combined-lora-k8-336-v1.json")
    if not snapshot:
        return "", {}
    sid = "combined-lora-k8-336"
    factor = data.read_json("factor-isolation-v1.json") or {}
    rows = [[a["title"], a["status"], f"{a['epochs_completed']}/{a['max_epochs']}",
             a["selected_epoch"] or "pending", num(a["best_mean_recall_at_1"]),
             a["heads_status"], num(a["lm_test_mlp"]), num(a["gold_mlp"])] for a in snapshot["arms"]]
    arms, lm = [], {}
    for name, group, short in (("reference", "reference", "codec reference"),
                               ("dinov2-small-k8-224", "windows", "DINOv2-S K8 224")):
        if name in factor:
            arms.append({"id": name, "label": short, "short": short, "group": group,
                         "curve_root": "codec-declip-v1" if name == "reference" else "factor-isolation-v1",
                         "curve_run": "dinov2-small-e5-base-512-codec-seed2026" if name == "reference" else name,
                         "per_label": factor[name]["gold_per_label"], "baseline": None})
            lm[name] = factor[name]["lm_test_per_label"]
    for a in snapshot["arms"]:
        if a.get("gold_per_label"):
            group = "combined-omnirad" if a["name"].startswith("omnirad") else "combined-dinov2"
            arms.append({"id": a["name"], "label": a["title"] + " combined", "short": a["title"] + " combined",
                         "group": group, "curve_root": "combined-lora-k8-336-v1", "curve_run": a["name"],
                         "per_label": a["gold_per_label"], "baseline": None})
            lm[a["name"]] = a["lm_test_per_label"]
    ref = factor.get("reference", {})
    views, blob, missing = experiment_views(sid, arms, COMBINED_GROUPS,
        labels=list(ref.get("gold_per_label") or arms[-1]["per_label"]), baselines={}, default_mode="reference",
        reference_id="reference" if ref else None, positives=ref.get("gold_positives"),
        lm_test_observed=ref.get("lm_test_observed"), lm_test_per_label=lm)
    body = (f"<p>Updated {esc(snapshot['updated_at'])}. Every Phase-1 factor at once: LoRA r64/a128 on the last six "
            "blocks of both towers, K8 windows at 336 px; batch size 32, seed 2026, 20 epochs, checkpoint by mean R@1. "
            "Read by frozen-tower MLP heads (three head seeds).</p>"
            + table(["Image tower", "Contrastive", "Epochs", "Selected epoch", "Mean R@1",
                     "Frozen heads", "LM-test MLP", "Gold-58 MLP"], rows, num_cols=(3, 4, 6, 7))
            + views)
    notes = (f"<p>{esc(snapshot['note'])}</p>"
             "<p>The codec reference and the K8-224 factor-isolation arm are drawn for context: they share the LoRA "
             "recipe and head protocol but not the inputs (224 px; the reference uses one stack per plane and attention "
             "pooling). Heatmap Δ are point differences vs the codec reference, not intervals. No unadapted floor at "
             "K8 336 was scored for these heads, so there is no Δ-vs-unadapted mode here; the OmniRad unadapted floor "
             f'at K8 336 under the run-5 head recipe is on the <a href="{ctx.page("picking")}#frozen-probe">picking page</a>.</p>'
             + (f"<p>No curve snapshot for: {esc(', '.join(missing))}.</p>" if missing else "")
             + f'<p>The supervised fine-tune from the OmniRad checkpoint is on '
             f'<a href="{ctx.page("phase2")}#run5-followups">Phase 2 run-5 follow-ups</a>.</p>'
             f'<p>Sources: <a href="{ctx.doc("combined-lora-k8-336-v1.md")}">results</a>, '
             f'<a href="{ctx.raw("combined-lora-k8-336-v1.json")}">verified snapshot</a>.</p>')
    best = max((a for a in snapshot["arms"] if a["lm_test_mlp"] is not None), key=lambda a: a["lm_test_mlp"], default=None)
    claim = ("Combined CLIP + LoRA + K8 + 336" if not best else
             f"Combining every Phase-1 factor: {esc(best['title'])} leads the frozen heads at "
             f"{num(best['lm_test_mlp'])} LM-test / {num(best['gold_mlp'])} gold-58")
    return section(sid, claim, body, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- page


def build(ctx) -> Page:
    latest_section, blobs = factor_isolation_section(ctx)
    combined, combined_blob = combined_lora_section(ctx)
    blobs.update(combined_blob)
    sections = [combined, latest_section]
    s1, b1, k1 = factorial_section(ctx)
    s2, b2 = tower_curves_section(ctx)
    s3, b3, k3 = size_matched_section(ctx)
    s4, b4, k4 = label_similarity_section(ctx)
    s5, b5, k5 = declip_section(ctx)
    s6, b6, _ = codec_declip_section(ctx)
    s7, b7 = next_section(ctx)
    for s, b in ((s1, b1), (s2, b2), (s3, b3), (s4, b4), (s5, b5), (s6, b6), (s7, b7)):
        sections.append(s)
        blobs.update(b)
    latest = data.read_json("factor-isolation-v1.json") or {}
    scored = [(name, arm) for name, arm in latest.items() if arm["lm_test_mlp"] is not None]
    kpis = []
    if scored:
        lm_name, lm = max(scored, key=lambda item: item[1]["lm_test_mlp"])
        gold_name, gold = max(scored, key=lambda item: item[1]["gold_mlp"])
        kpis = [{"label": "Factor isolation: best LM-test", "value": num(lm["lm_test_mlp"]),
                 "sub": esc(lm_name) + " · 432 LM-labelled studies"},
                {"label": "Factor isolation: best gold-58", "value": num(gold["gold_mlp"]),
                 "sub": esc(gold_name) + " · exploratory"},
                {"label": "Completed factor-isolation heads", "value": f"{len(scored)-1}/{len(latest)-1}",
                 "sub": "Three head seeds per completed arm; codec reference excluded"}]
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=sections, data=blobs, kpis=kpis or None)


def summary() -> dict:
    latest = data.read_json("factor-isolation-v1.json")
    if latest:
        k8 = latest["dinov2-small-k8-224"]
        completed = sum(arm["lm_test_mlp"] is not None for name, arm in latest.items() if name != "reference")
        matched = k8.get("paired", {}).get("lm_test", {}).get("comparator")
        caveat = (f"K8 vs uniform-224 Δ {num(matched['difference'], '.4f', True)} {_ci(*matched['ci95'])}; gold exploratory"
                  if matched else f"Gold-58 {num(k8['gold_mlp'])}, exploratory; matched head pending")
        return {"stage": BY_SLUG[SLUG].number, "headline": f"Factor isolation: {completed}/4 heads evaluated",
                "value": num(k8["lm_test_mlp"]), "value_label": "K8 LM-test MLP AUROC",
                "caveat": caveat,
                "href": "#factor-isolation"}
    j = data.read_json(FACTORIAL)
    info = BY_SLUG[SLUG]
    if not j:
        return {"stage": info.number, "headline": None, "value": None, "value_label": None, "caveat": None, "href": None}
    top = max(j["arms"].values(), key=lambda a: a["gold_macro_auroc"])
    n_clear = sum(1 for c in j["contrasts"].values()
                  if _excludes_zero(*c["paired_bootstrap"]["macro_auroc_difference_ci95"]))
    return {"stage": info.number, "headline": f"Keep {_arm_label(top)}",
            "value": num(top["gold_macro_auroc"]), "value_label": "gold macro AUROC, 56-case OOF",
            "caveat": f"Single seed; {n_clear} of {len(j['contrasts'])} tower contrasts exclude zero", "href": "#factorial"}
