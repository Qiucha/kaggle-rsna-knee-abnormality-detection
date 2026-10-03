"""Image-tower adaptation page (depth, both towers, LoRA, pooling).

Phase A ships ONE complete reference section here (``depth_lora_section``); the
phase-B page agent keeps it and adds the rest. Every page agent copies its pattern:

1. Load data in Python via ``digest_site.data`` (digest/* and digest/site/data/* only).
   Missing input -> return a section whose body is ``notice(...)``, never raise.
2. Give every mark a series id that is shared by every chart, legend and table row in the
   section (here: the bare run name, e.g. ``both-lora-depth6-r64-seed2026``).
3. Put the chart data in ``Page.data["data-<section-id>"]`` (inline JSON, sanitised by the shell).
4. ``section(sid, claim, body, caption, notes, table_html)``: the h2 is a claim sentence built from
   the data (no hand-typed numbers), the caption is <= 2 lines naming the protocol, and methods,
   caveats and full tables go in collapsed ``<details>``.
5. The body holds mount points (``chart(...)``, a filter-bar div, a legend div) and one inline
   ``<script>`` that calls ``Charts.selection(section)`` once and passes that store to every chart,
   so filter / hover / solo apply to both charts and the table rows at once.
6. Colours: group hues from CSS vars (``var(--g-lora)`` ...); variants get lightness via Charts.shade
   automatically, or a dash. Never hex.
7. Never put R@1 and AUROC in one table, never put 441- and 433-candidate R@1 on one axis.
"""

from __future__ import annotations

import re

from .. import data
from ..shell import Page, chart, esc, notice, num, section, table
from ..sitemap import BY_SLUG

SLUG = "adaptation"
SUBTITLE = "How deep to adapt, which towers, full fine-tuning vs LoRA, and plane pooling."

GROUPS = {
    "lora": {"label": "Both towers, LoRA", "color": "var(--g-lora)"},
    "bothft": {"label": "Both towers, full fine-tuning", "color": "var(--g-bothft)"},
    "imgft": {"label": "Image tower only, full fine-tuning", "color": "var(--g-imgft)"},
}
CURVE_ROOTS = ["both-lora-v1", "both-block-depth-v1", "image-block-depth-v1"]
PROBE_FILE = "downstream-probe-both-block-depth-v1.json"
FEATURE = "uniform_mean_768"
MIN_EPOCHS = 10  # shorter histories are aborted runs


def group_of(run: str) -> str | None:
    if run.startswith("both-lora-"):
        return "lora"
    if run.startswith("both-unfreeze"):
        return "bothft"
    if run.startswith("original-unfreeze"):
        return "imgft"
    return None


def label_of(run: str) -> str:
    m = re.match(r"both-lora-depth(\d+)-r(\d+)(.*)-seed\d+$", run)
    if m:
        extra = m.group(3).replace("-dinov2-base", " DINOv2-base").replace("-bs", " BS").replace("-reg", " reg")
        return f"LoRA {m.group(1)} blk r{m.group(2)}{extra}"
    m = re.match(r"both-unfreeze(\d+)(-bs(\d+))?-seed\d+$", run)
    if m:
        return f"Both FT {m.group(1)} blk" + (f" BS{m.group(3)}" if m.group(3) else "")
    m = re.match(r"original-unfreeze(\d+)-seed\d+$", run)
    if m:
        return f"Image FT {m.group(1)} blk"
    return run


def _sort_key(run: str):
    nums = [int(x) for x in re.findall(r"\d+", run.split("-seed")[0])]
    return (nums, run)


def depth_lora_section(ctx=None) -> tuple[str, dict]:
    """Reference section: R@1 curves + gold-probe CI dot plot sharing one selection store."""
    sid = "depth-lora"
    probe = data.read_json(PROBE_FILE)
    curves = {root: data.load_curves(root) for root in CURVE_ROOTS}
    if not probe or not any(curves.values()):
        return section(sid, "Adaptation depth and LoRA", notice(
            "Curve snapshots or the probe JSON are missing. Run <code>build_digest_site.py export</code>.")), {}

    probe_runs = {v.run: v for v in data.probe_vs_unadapted(probe) if v.feature == FEATURE}
    # one candidate pool per axis: keep the most common validation count
    counts = [r.get("val_candidates") for c in curves.values() if c for r in c["runs"].values()]
    pool = max(set(counts), key=counts.count) if counts else None
    seeds = set()
    series, excluded, val_counts = [], [], set()
    for root in CURVE_ROOTS:
        c = curves.get(root)
        if not c:
            continue
        for run in sorted(c["runs"], key=_sort_key):
            r = c["runs"][run]
            g = group_of(run)
            # image-only arms: keep the seed that matches the single-seed both-tower sweep
            if g is None or (g == "imgft" and run not in probe_runs):
                continue
            if len(r["epoch"]) < MIN_EPOCHS:
                excluded.append(f"{label_of(run)} ({len(r['epoch'])} epochs, aborted)")
                continue
            if r.get("val_candidates") != pool:
                excluded.append(f"{label_of(run)} ({r.get('val_candidates')} validation candidates)")
                continue
            val_counts.add(r["val_candidates"])
            seeds.add(r["config"].get("seed"))
            series.append({
                "id": run, "label": label_of(run), "group": g,
                "dash": "5 3" if re.search(r"-bs\d+", run) else None,
                "points": data.curve_series(c, run, "mean_r@1"),
            })

    floor = data.unadapted_level(probe)
    rows = []
    for run, v in sorted(probe_runs.items(), key=lambda kv: _sort_key(kv[0])):
        g = group_of(run)
        if g is None:
            continue
        rows.append({"id": run, "label": label_of(run), "group": g, "est": v.diff, "lo": v.ci[0], "hi": v.ci[1],
                     "after": v.after})

    def peak(s):
        pts = [p for p in s["points"] if p[1] is not None]
        return max(pts, key=lambda p: p[1]) if pts else (None, None)

    # Claim from the intervals, not from single-seed rank order.
    ranked = sorted(rows, key=lambda r: r["est"], reverse=True)
    clear = [r for r in rows if r["lo"] > 0]
    top, second = ranked[0], ranked[1]
    overlap = top["lo"] <= second["hi"]
    claim = (f"{len(clear)} of {len(rows)} adapted arms clear unadapted DINOv2 on the gold probe"
             + ("; the leading intervals overlap, so this single-seed sweep does not rank them" if overlap
                else f"; {esc(top['label'])} separates from the next arm"))
    last_epoch = {s_["id"]: s_["points"][-1][0] for s_ in series if s_["points"]}
    typical = sorted(last_epoch.values())[len(last_epoch) // 2] if last_epoch else None
    long_runs = [f"{s_['label']} ({last_epoch[s_['id']]} epochs)" for s_ in series
                 if typical and last_epoch.get(s_["id"], 0) > typical]

    pop = (probe.get("population") or {}).get("population_size")
    boot = (probe.get("config") or {}).get("bootstrap") or {}
    seed_txt = ", ".join(str(s) for s in sorted(x for x in seeds if x is not None))
    n_val = ", ".join(str(v) for v in sorted(val_counts))
    caption = (f"Left: validation mean R@1 per epoch over {n_val} candidates, single seed ({seed_txt}). "
               f"Right: gold macro-AUROC difference vs unadapted DINOv2 (level {num(floor)}), bootstrap mean and 95% CI "
               f"over {boot.get('resamples', '?')} study resamples, {pop} exposed gold studies, uniform-mean feature.")

    blob = {"groups": GROUPS, "series": series, "probe": rows,
            "refLabel": f"unadapted DINOv2 = {num(floor)}"}

    body = f"""<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('r1', f'Validation mean R@1 by epoch (retrieval, {pool} candidates)')}</div>
<div>{chart('probe', 'Gold-probe macro AUROC minus unadapted DINOv2')}</div>
</div>
<div data-role="legend"></div>
<p class="legend-hint">Hover a line, legend entry, dot or table row to highlight it everywhere in this section; click to pin it; use the pills to filter by group.</p>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.lineChart(sec.querySelector('[data-chart=r1]'), d.series,
    {{groups: d.groups, yLabel: 'Validation mean R@1', yFmt: '.1%', best: 'max', store: store}});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store,
    {{groups: d.groups, best: 'max', fmt: '.1%'}});
  Charts.ciDotPlot(sec.querySelector('[data-chart=probe]'), d.probe,
    {{groups: d.groups, ref: 0, refLabel: d.refLabel, fmt: '+.3f', axisFmt: '+.2f',
      xLabel: '\\u0394 gold macro AUROC vs unadapted', store: store,
      tooltipExtra: function (r) {{ return '<div>Level ' + Charts.fmt('.4f')(r.after) + '</div>'; }}}});
}})();
</script>"""

    tbl = table(
        ["Run", "Group", "Gold macro AUROC", "Δ vs unadapted", "95% CI"],
        [[r["label"], GROUPS[r["group"]]["label"], num(r["after"]), num(r["est"], ".4f", signed=True),
          f"[{num(r['lo'], '.4f', True)}, {num(r['hi'], '.4f', True)}]"] for r in rows],
        num_cols=(2, 3, 4), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows],
    )
    notes = (
        "<ul>"
        f"<li>Curves: <code>metrics.jsonl</code> snapshots of {', '.join(f'<code>{r}</code>' for r in CURVE_ROOTS)}; "
        + (f"probe: <a href=\"{ctx.raw(PROBE_FILE)}\"><code>{PROBE_FILE}</code></a>; rank summary: "
           f"<a href=\"{ctx.doc('lora-adaptation-summary.md')}\">lora-adaptation-summary</a>. " if ctx else f"probe: <code>{PROBE_FILE}</code>. ")
        + "Checkpoints are selected on validation R@1, never on AUROC.</li>"
        + f"<li>Unadapted anchor in this file is {num(floor)} (depth / LoRA extraction family); the image-only files use their own "
        "floor. The BS64 regularised arm changes four things at once and cannot attribute its effect.</li>"
        "<li>The gold probe is read on studies the encoder trained on: a development statistic. The generalisation "
        "estimate for the same idea is on the probes page.</li>"
        "<li>Every arm is a single seed, so neighbouring arms are not separable; do not read the ranking as a record.</li>"
        + "<li>Gold-holdout runs validate on a different candidate pool and are never drawn on this axis."
        + (f" Not shown: {esc('; '.join(excluded))}." if excluded else "") + "</li>"
        + (f"<li>Longer schedule stretches the x-axis: {esc('; '.join(long_runs))}; the others trained {typical}.</li>" if long_runs else "")
        + "<li>Retrieval and label AUROC are different metrics on different populations; they share a section, never a table.</li>"
        "</ul>"
    )
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Probe table ({len(rows)} runs)"), {f"data-{sid}": blob}


def _kpis() -> list[dict]:
    """KPI tiles, every value computed from JSON."""
    from . import _adaptation_sections as A
    tiles = []
    comps = [data.read_json(f) for f in A.COMPARE_FILES]
    d14 = []
    for j in comps:
        for p in data.paired_comparisons(j):
            if A._classify(p)[2] == (1, 4) and p.name not in {x.name for x in d14}:
                d14.append(p)
    if d14:
        clear = sum(1 for p in d14 if p.ci[0] > 0)
        lo, hi = min(p.diff for p in d14), max(p.diff for p in d14)
        tiles.append({"label": "1 → 4 image blocks", "value": f"≥ {num(lo, '.3f', True)}",
                      "sub": f"Δ gold macro AUROC, up to {num(hi, '.3f', True)}; {clear}/{len(d14)} seeds' CIs exclude 0 (exposed)",
                      "tone": "good"})
    hv = A.holdout_values()
    if hv.get("unexposed_pool") is not None:
        tiles.append({"label": "4-block image tower, held out", "value": num(hv["unexposed_pool"]),
                      "sub": f"{hv['n_seeds']}-seed pool; {num(hv['exposed_pool'])} exposed; unadapted {num(hv['floor_u'])}"})
    if hv.get("lora_holdout") is not None:
        tiles.append({"label": "LoRA r64, gold held out", "value": num(hv["lora_holdout"]),
                      "sub": f"single seed; floor {num(hv['floor_h'])}; not paired with the exposed LoRA row"})
    mem = data.read_json(A.MEM_FILES["bs32"]) or []
    ft6 = next((r for r in mem if r.get("rank") is None and r["layer_count"] == 6), None)
    l64 = next((r for r in mem if r.get("rank") == 64 and r["layer_count"] == 6), None)
    if ft6 and l64:
        tiles.append({"label": "Trainable weights, 6 blocks", "value": f"{num(l64['trainable_fraction_pct'], '.2f')}%",
                      "sub": f"LoRA r64 vs {num(ft6['trainable_fraction_pct'], '.2f')}% full FT; same peak memory"})
    tb = []
    for f in A.TEXT_FILES[:3]:
        tb += data.paired_comparisons(data.read_json(f))
    if tb:
        tiles.append({"label": "Text block added", "value": "null",
                      "sub": f"{A._range([p.diff for p in tb], '.4f')}; {sum(1 for p in tb if p.ci[0] <= 0 <= p.ci[1])}/{len(tb)} CIs span 0"})
    return tiles


def build(ctx) -> Page:
    from . import _adaptation_sections as A
    parts = [A.depth_ladder_section(ctx), A.both_towers_section(ctx), depth_lora_section(ctx),
             A.lora_efficiency_section(ctx), A.holdout_section(ctx), A.per_label_section(ctx),
             A.plane_pooling_section(ctx), A.curves_section(ctx)]
    blob = {}
    for _, b in parts:
        blob.update(b)
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=[p[0] for p in parts], data=blob, kpis=_kpis())


def summary() -> dict:
    from . import _adaptation_sections as A
    lev = A.depth_lever()
    hv = A.holdout_values()
    if not lev:
        return {"stage": BY_SLUG[SLUG].number, "headline": "Adaptation depth", "value": None, "value_label": None,
                "caveat": None, "href": "#image-depth"}
    return {"stage": BY_SLUG[SLUG].number,
            "headline": "Depth is the lever: adapting 1 → 4 image blocks",
            "value": num(lev["min"], ".3f", True),
            "value_label": (f"gold macro AUROC, lowest of {lev['n']} seed-matched paired gains "
                            f"({lev['clear']}/{lev['n']} CIs exclude zero)"),
            "caveat": (f"Exposed gold; submitted LoRA encoder reads {num(hv.get('lora_holdout'))} held out (single seed)"
                       if hv.get("lora_holdout") is not None else "Exposed gold studies"),
            "href": "#image-depth"}
