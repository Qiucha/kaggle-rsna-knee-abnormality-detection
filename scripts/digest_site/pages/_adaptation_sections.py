"""Sections of the adaptation page other than the Phase A reference section.

Every number comes from a digest/*.json file or a curve snapshot under digest/site/data/curves/.
Each builder returns (section_html, {data_id: blob}); a missing input degrades to a notice.
"""

from __future__ import annotations

import re

from .. import data
from ..shell import chart, esc, group_major, notice, num, section, table

FEATURE = "uniform_mean_768"

# Shared inline helpers (segmented switchers and chart remounting); defined once, used by later sections.
HELPERS = """<script>
window.AdaptUI = window.AdaptUI || {
  seg: function (el, label, options, active, cb) {
    el.classList.add('filter-bar'); el.setAttribute('role', 'group'); el.setAttribute('aria-label', label);
    var t = document.createElement('span'); t.className = 'muted'; t.style.fontSize = '12px'; t.textContent = label + ':';
    el.appendChild(t);
    options.forEach(function (o) {
      var b = document.createElement('button'); b.type = 'button';
      b.className = 'pill-btn' + (o.key === active ? ' active' : '');
      b.setAttribute('data-choice', o.key); b.setAttribute('aria-pressed', o.key === active ? 'true' : 'false');
      b.textContent = o.label; el.appendChild(b);
    });
    el.addEventListener('click', function (e) {
      var b = e.target.closest('[data-choice]'); if (!b) return;
      Array.prototype.forEach.call(el.querySelectorAll('[data-choice]'), function (x) {
        var on = x === b; x.classList.toggle('active', on); x.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      cb(b.getAttribute('data-choice'));
    });
  },
  remount: function (old) {
    var n = document.createElement('div'); n.className = 'chart';
    n.setAttribute('data-chart', old.getAttribute('data-chart'));
    n.setAttribute('aria-label', old.getAttribute('aria-label') || '');
    old.parentNode.replaceChild(n, old); return n;
  }
};
</script>"""


def js(code: str, **subs) -> str:
    for k, v in subs.items():
        code = code.replace(f"__{k.upper()}__", str(v))
    return f"<script>\n(function () {{\n{code}\n}})();\n</script>"


def _seed(run: str) -> str:
    m = re.search(r"seed(\d+)", run)
    return m.group(1) if m else ""


def _span(lo, hi) -> bool:
    return lo <= 0 <= hi


def _range(vals, spec=".4f", signed=True) -> str:
    vals = sorted(vals)
    if not vals:
        return "–"
    if len(vals) == 1:
        return num(vals[0], spec, signed)
    return f"{num(vals[0], spec, signed)} to {num(vals[-1], spec, signed)}"


# =============================================================================== 1. image-only depth ladder

V3 = "downstream-probe-extended-v3.json"
COMPARE_FILES = ["downstream-extended-comparisons-v1.json", "downstream-extended-comparisons-v2.json",
                 "downstream-extended-comparisons-v3.json"]
LADDER_GROUPS = {
    "depth": {"label": "Depth", "color": "var(--g-imgft)"},
    "language": {"label": "Report language", "color": "var(--g-sky)"},
    "schedule": {"label": "Schedule", "color": "var(--g-amber)"},
    "control": {"label": "Shuffled-pairing control", "color": "var(--g-rose)"},
}


def _img_blocks(run: str) -> int | None:
    if run.startswith("original-multilingual"):
        return 1
    m = re.match(r"original-unfreeze(\d+)-seed", run)
    return int(m.group(1)) if m else None


def _classify(p: data.Paired):
    _, cand, _ = data.split_key(p.candidate)
    _, base, _ = data.split_key(p.baseline)
    s = _seed(cand)
    cb, bb = _img_blocks(cand), _img_blocks(base)
    if cb and bb and cb != bb:
        return "depth", f"{bb} → {cb} blocks, s{s}", (bb, cb)
    if cand.startswith("translated-english") and base.startswith("original-multilingual"):
        return "language", f"translated reports, s{s}", None
    if cand.startswith("original-40ep") and base.startswith("original-multilingual"):
        return "schedule", f"40 vs 20 epochs, s{s}", None
    if base.startswith("shuffled-pairing") and cand.startswith("original-unfreeze4"):
        return "control", f"true vs shuffled, s{s}", None
    return None, None, None


def depth_lever() -> dict:
    """Seed-matched 1 -> 4 image-block paired gold differences (for the Home summary)."""
    comps = {f: data.read_json(f) for f in COMPARE_FILES}
    if not all(comps.values()):
        return {}
    est, clear, seen = [], 0, set()
    for f in COMPARE_FILES:
        for p in data.paired_comparisons(comps[f]):
            if p.name in seen:
                continue
            g, _, blocks = _classify(p)
            if g != "depth" or blocks != (1, 4):
                continue
            seen.add(p.name)
            est.append(p.diff)
            clear += p.ci[0] > 0
    return {"min": min(est), "max": max(est), "n": len(est), "clear": clear} if est else {}


def depth_ladder_section(ctx) -> tuple[str, dict]:
    sid = "image-depth"
    v3 = data.read_json(V3)
    comps = {f: data.read_json(f) for f in COMPARE_FILES}
    if not v3 or not all(comps.values()):
        return section(sid, "Image-tower depth ladder", notice(f"<code>{V3}</code> or a comparisons file is missing.")), {}

    floor = data.unadapted_level(v3)
    by_seed: dict[str, list] = {}
    for a in data.probe_runs(v3):
        if a.slot != "after_cl" or a.feature != FEATURE:
            continue
        b = _img_blocks(a.run)
        if b is None:
            continue
        by_seed.setdefault(_seed(a.run), []).append([b, a.macro_auroc])
    series = []
    for s in sorted(by_seed):
        pts = sorted(by_seed[s])
        series.append({"id": f"ladder-s{s}", "label": f"Image-only FT, seed {s}", "points": [[0, floor]] + pts})

    rows, seen = [], set()
    for f in COMPARE_FILES:
        for p in data.paired_comparisons(comps[f]):
            if p.name in seen:
                continue
            g, label, blocks = _classify(p)
            if not g:
                continue
            seen.add(p.name)
            rows.append({"id": f"c-{len(rows)}", "label": label, "group": g, "est": p.diff, "lo": p.ci[0],
                         "hi": p.ci[1], "file": f, "blocks": blocks, "base": p.macro_baseline,
                         "cand": p.macro_candidate})

    gorder = list(LADDER_GROUPS)
    rows.sort(key=lambda r: (gorder.index(r["group"]), r["blocks"] or (0, 0), r["label"]))
    for i, r in enumerate(rows):
        r["id"] = f"c-{i}"
    d14 = [r for r in rows if r["blocks"] == (1, 4)]
    nulls = [r for r in rows if r["group"] in ("language", "schedule")]
    ctrl = [r for r in rows if r["group"] == "control"]
    n14_clear = sum(1 for r in d14 if r["lo"] > 0)
    nulls_span = all(_span(r["lo"], r["hi"]) for r in nulls)
    claim = (f"Depth is the lever: 1 → 4 image blocks adds {_range([r['est'] for r in d14])} gold macro AUROC at "
             f"{len(d14)} seeds ({n14_clear}/{len(d14)} intervals exclude zero)"
             + ("; report language and a doubled schedule are null" if nulls_span else ""))
    ctrl_clear = sum(1 for r in ctrl if r["lo"] > 0)
    caption = (f"Left: exposed gold macro AUROC by adapted image blocks (text frozen); 0 = unadapted DINOv2 "
               f"({num(floor)} in this file). Right: seed-matched paired differences with 95% CI; the true pairing beats a "
               f"shuffled pairing by {_range([r['est'] for r in ctrl])} ({ctrl_clear}/{len(ctrl)} exclude zero).")

    blob = {"groups": LADDER_GROUPS, "series": series, "rows": rows, "floor": floor}
    body = HELPERS + f"""<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('ladder', 'Gold macro AUROC by adapted image blocks (exposed)')}</div>
<div>{chart('contrasts', 'Paired contrasts (candidate − baseline)')}</div>
</div>
<p class="legend-hint">Lines are seeds (hover for the label); only seed 2026 has 2 and 8 blocks. Pills filter the contrasts; hover or click a row to pin it.</p>
""" + js("""
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  d.series.forEach(function (s, i) { s.color = Charts.shade('var(--g-imgft)', i, d.series.length); });
  Charts.lineChart(sec.querySelector('[data-chart=ladder]'), d.series,
    {yLabel: 'Gold macro AUROC', yFmt: '.3f', xLabel: 'Adapted image blocks', xStep: 1, best: 'max', store: store,
     refs: [{y: d.floor, label: 'unadapted DINOv2 ' + Charts.fmt('.4f')(d.floor)}]});
  Charts.ciDotPlot(sec.querySelector('[data-chart=contrasts]'), d.rows,
    {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.2f', sort: false,
     xLabel: '\\u0394 gold macro AUROC', store: store, labelWidth: 220,
     tooltipExtra: function (r) { return '<div>' + Charts.fmt('.4f')(r.base) + ' \\u2192 ' + Charts.fmt('.4f')(r.cand) + '</div>'; }});
""", sid=sid)

    tbl = table(["Contrast", "Lever", "Baseline", "Candidate", "Δ", "95% CI", "File"],
                [[r["label"], LADDER_GROUPS[r["group"]]["label"], num(r["base"]), num(r["cand"]),
                  num(r["est"], ".4f", True), f"[{num(r['lo'], '.4f', True)}, {num(r['hi'], '.4f', True)}]", r["file"]]
                 for r in rows], num_cols=(2, 3, 4, 5),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul>"
        f"<li>Levels from <a href=\"{ctx.raw(V3)}\"><code>{V3}</code></a>; contrasts from "
        + ", ".join(f"<a href=\"{ctx.raw(f)}\"><code>{f}</code></a>" for f in COMPARE_FILES)
        + f". Write-up: <a href=\"{ctx.doc('image-block-variants-v1.md')}\">image-block-variants-v1</a>.</li>"
        "<li><b>Exposure:</b> every level here is a development statistic (the encoder trained on 56 of the 58 gold "
        "studies). The pooled 3-seed 4-block level and its unexposed replication are shown together in the holdout section "
        "below; the paired contrasts share one exposure and are not corrected.</li>"
        "<li>Only 1 and 4 blocks have three seeds; 2 and 8 blocks, the 40-epoch arm and every 2/8-block contrast are single seed. "
        "Depth and learning rate are confounded (image learning rate 1e-5 at every depth).</li>"
        "<li>The single-seed +0.0413 and the earlier '~53% control' share are superseded by the 3-seed figures. "
        "The shuffled control is exposed only, scored at <code>last.pt</code>, and never re-run held out.</li>"
        f"<li>Unadapted floor is {num(floor)} in this probe family; the both-tower / LoRA files use their own anchor (next "
        "section). The two floors are never drawn on one line.</li>"
        "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Contrast table ({len(rows)} rows)"), {f"data-{sid}": blob}


# =============================================================================== 2. both towers

DEPTH_FILE = "downstream-probe-both-block-depth-v1.json"
TEXT_FILES = [f"downstream-both-vs-image-block{p}-seed{s}.json" for p in ("", "-planes") for s in (2026, 2027, 2028)]
METHOD_GROUPS = {
    "imgft": {"label": "Image only, full FT", "color": "var(--g-imgft)"},
    "bothft": {"label": "Both towers, full FT", "color": "var(--g-bothft)"},
    "lora": {"label": "Both towers, LoRA", "color": "var(--g-lora)"},
    "textblk": {"label": "Text block added (1-block matrix)", "color": "var(--g-sky)"},
}


def _method_point(run: str):
    """(series_id, label, group, blocks) for plain depth arms; None for BS / reg variants."""
    if re.search(r"-bs\d+|-reg|dinov2-base", run):
        return None
    m = re.match(r"original-unfreeze(\d+)-seed", run)
    if m:
        return "m-imgft", "Image only, full FT", "imgft", int(m.group(1))
    m = re.match(r"both-unfreeze(\d+)-seed", run)
    if m:
        return "m-bothft", "Both towers, full FT", "bothft", int(m.group(1))
    m = re.match(r"both-lora-depth(\d+)-r(\d+)-seed", run)
    if m:
        return f"m-lora-r{m.group(2)}", f"LoRA r{m.group(2)}", "lora", int(m.group(1))
    return None


def both_towers_section(ctx) -> tuple[str, dict]:
    sid = "both-towers"
    depth = data.read_json(DEPTH_FILE)
    texts = {f: data.read_json(f) for f in TEXT_FILES}
    if not depth:
        return section(sid, "Both towers", notice(f"<code>{DEPTH_FILE}</code> is missing.")), {}
    floor = data.unadapted_level(depth)
    pts: dict[str, dict] = {}
    for a in data.probe_runs(depth):
        if a.slot != "after_cl" or a.feature != FEATURE:
            continue
        mp = _method_point(a.run)
        if not mp:
            continue
        s_id, label, g, b = mp
        pts.setdefault(s_id, {"id": s_id, "label": label, "group": g, "points": []})["points"].append([b, a.macro_auroc])
    order = ["m-imgft", "m-bothft", "m-lora-r16", "m-lora-r32", "m-lora-r64"]
    series = []
    for k in order:
        if k in pts:
            s = pts[k]
            s["points"] = sorted(s["points"])
            if s["group"] == "lora":
                s["dash"] = None if k.endswith("r64") else "5 3"
            series.append(s)

    rows = []
    for f in TEXT_FILES:
        j = texts.get(f)
        for p in data.paired_comparisons(j):
            feat = data.split_key(p.candidate)[2]
            rows.append({"id": f"tb-{len(rows)}", "group": "textblk",
                         "label": f"s{_seed(p.candidate)}, {'plane attention' if 'plane' in feat else 'uniform mean'}",
                         "est": p.diff, "lo": p.ci[0], "hi": p.ci[1], "file": f, "feature": feat,
                         "base": p.macro_baseline, "cand": p.macro_candidate})
    uni = [r for r in rows if r["feature"] == FEATURE]
    span_all = all(_span(r["lo"], r["hi"]) for r in rows)
    both = pts.get("m-bothft")
    best = max(both["points"], key=lambda p: p[1]) if both else (None, None)
    claim = (f"Adapting the text block adds nothing ({_range([r['est'] for r in uni])} over {len(uni)} seeds"
             + (", every interval spans zero" if span_all else "") + ")"
             + (f"; with both towers fully fine-tuned, depth peaks at {best[0]} blocks ({num(best[1])}) on one seed" if both else ""))
    caption = (f"Left: exposed gold macro AUROC by adapted blocks per tower, seed 2026, one file (unadapted = "
               f"{num(floor)}, this file's anchor). Right: 1-block matrix, image+text block minus image block alone, "
               f"paired per seed, 95% CI.")
    blob = {"groups": METHOD_GROUPS, "series": series, "rows": rows, "floor": floor}
    body = f"""<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('methods', 'Gold macro AUROC by adapted blocks per tower (single seed 2026)')}</div>
<div>{chart('textblock', 'Text block added: paired Δ gold macro AUROC')}</div>
</div>
<div data-role="legend"></div>
""" + js("""
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.lineChart(sec.querySelector('[data-chart=methods]'), d.series,
    {groups: d.groups, yLabel: 'Gold macro AUROC', yFmt: '.3f', xLabel: 'Adapted blocks per tower', best: 'max',
     store: store, refs: [{y: d.floor, label: 'unadapted ' + Charts.fmt('.4f')(d.floor) + ' (this file)'}]});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {groups: d.groups, best: 'max', fmt: '.4f'});
  Charts.ciDotPlot(sec.querySelector('[data-chart=textblock]'), d.rows,
    {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.3f', sort: false,
     xLabel: '\\u0394 macro AUROC (both-block − image-block)', store: store});
""", sid=sid)

    tbl = table(["Series", "Blocks", "Gold macro AUROC"],
                [[s["label"], p[0], num(p[1])] for s in series for p in s["points"]], num_cols=(1, 2),
                row_attrs=[{"data-series-id": s["id"], "data-group": s["group"]} for s in series for _ in s["points"]])
    notes = (
        "<ul>"
        f"<li>Depth levels: <a href=\"{ctx.raw(DEPTH_FILE)}\"><code>{DEPTH_FILE}</code></a> (single seed 2026, exposed gold). "
        f"Text-block contrasts: {', '.join(f'<a href=\"{ctx.raw(f)}\"><code>{f}</code></a>' for f in TEXT_FILES)}. "
        f"Summary: <a href=\"{ctx.doc('both-tower-adaptation-summary.md')}\">both-tower-adaptation-summary</a>.</li>"
        f"<li>The anchor here is <b>{num(floor)}</b> (depth / LoRA extraction family), not the image-only files' floor: "
        "compare levels only within one file.</li>"
        "<li>Depth and learning rate are confounded, and every both-tower point is one seed; per-seed probe intervals are "
        "about ±0.02 to ±0.03, so neighbouring depths are not separable. Batch-size and regularised variants are left off the "
        "depth lines (they are in the heatmap and the LoRA table).</li>"
        "<li>The 1-block text-block matrix ran on the translated (dfd07db) cohort. Its image-only arm on retrieval is "
        f"written up in <a href=\"{ctx.doc('claude-translated-dfd07db-image-block-results.md')}\">claude-translated-dfd07db-image-block-results</a> "
        f"(amended; protocol <a href=\"{ctx.doc('claude-translated-dfd07db-image-block-protocol.md')}\">here</a>), "
        f"which lives on the <a href=\"{ctx.page('text')}\">text page</a>.</li>"
        "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary="Depth-line values"), {f"data-{sid}": blob}


# =============================================================================== 3. LoRA efficiency / memory

MEM_FILES = {"bs32": "lora-memory-profile.json", "bs16": "lora-memory-profile-bs16.json"}
OOM_FILE = "oom-solutions-benchmark.json"
MEM_GROUPS = {"full": {"label": "Full fine-tuning", "color": "var(--g-bothft)"},
              "lora": {"label": "LoRA", "color": "var(--g-lora)"}}


def _mem_rows(rows: list[dict], key: str) -> list[dict]:
    out = []
    for r in rows:
        lora = r.get("rank") is not None
        fits = r.get("fits_in_8gb")
        label = f"{'LoRA r' + str(r['rank']) if lora else 'Full FT'}, {r['layer_count']} blk"
        out.append({"id": f"{key}-{len(out)}", "label": label + ("" if fits else " (OOM)"), "group": "lora" if lora else "full",
                    "value": r.get("peak_allocated_mb"), "params": r.get("trainable_parameters"),
                    "pct": r.get("trainable_fraction_pct"), "status": r.get("status"), "bs": r.get("batch_size")})
    return out


def lora_efficiency_section(ctx) -> tuple[str, dict]:
    sid = "lora-memory"
    mem = {k: data.read_json(v) for k, v in MEM_FILES.items()}
    oom = data.read_json(OOM_FILE)
    if not mem["bs32"] or not oom:
        return section(sid, "LoRA efficiency", notice("Memory profile JSONs are missing.")), {}
    views = {k: _mem_rows(v, k) for k, v in mem.items() if v}
    total = next((r.get("total_vram_mb") for r in mem["bs32"] if r.get("total_vram_mb")), None)
    solutions = []
    for r in oom:
        name = r["solution"]
        lora = bool(r.get("use_lora")) or "LoRA" in name
        blocks = r.get("layer_count") or (re.search(r"(\d+)-Blk", name) and int(re.search(r"(\d+)-Blk", name).group(1)))
        how = ("grad ckpt" if "Checkpoint" in name else "save_on_cpu" if "save_on_cpu" in name else "standard")
        label = f"{'LoRA' if lora else 'Full FT'} {blocks} blk, {how}" + (f" BS{r['batch_size']}" if r.get("batch_size") not in (None, 32) else "")
        if r.get("status") == "ERROR":
            label += " (OOM)"
        elif r.get("fits_in_8gb") is False:
            label += " (> 8 GB)"
        solutions.append({"id": f"oom-{len(solutions)}", "label": label, "name": name, "group": "lora" if lora else "full",
                          "value": r.get("peak_allocated_mb"), "bs": r.get("batch_size"), "blocks": r.get("layer_count")})

    def find(rows, lora_rank, blocks):
        return next((r for r in rows if r.get("rank") == lora_rank and r["layer_count"] == blocks), None)
    ft6, l64 = find(mem["bs32"], None, 6), find(mem["bs32"], 64, 6)
    ckpt = [s for s in solutions if "Checkpointing" in s["label"] and s["value"] is not None]
    claim = "LoRA and full fine-tuning compared on memory"
    if ft6 and l64:
        claim = (f"LoRA r64 on 6 blocks trains {num(l64['trainable_fraction_pct'], '.2f')}% of the weights vs "
                 f"{num(ft6['trainable_fraction_pct'], '.2f')}% for full fine-tuning, but peaks at the same memory "
                 f"({num(l64['peak_allocated_mb'], '.0f')} vs {num(ft6['peak_allocated_mb'], '.0f')} MiB); gradient checkpointing "
                 f"is what fits 8–12 blocks in 8 GB")
    caption = (f"Left: peak allocated GPU memory per configuration, both towers, sequence length 512, against the "
               f"{num(total, '.0f')} MiB card (switch batch size). Right: remedies for 8+ blocks, batch size 32 unless marked.")
    blob = {"groups": MEM_GROUPS, "views": views, "solutions": solutions, "total": total}
    body = f"""<div data-role="filter"></div><div data-role="bs"></div>
<div class="grid-2">
<div>{chart('mem', 'Peak allocated memory (MiB)')}</div>
<div>{chart('oom', 'Out-of-memory remedies (MiB)')}</div>
</div>
""" + js("""
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var ref = {groups: d.groups, value: 'value', fmt: '.0f', axisFmt: '.0f', ref: d.total, refLabel: '8 GB card',
             xLabel: 'Peak allocated MiB', store: store, labelWidth: 250, rowHeight: 22,
             tooltipExtra: function (r) {
               var s = '';
               if (r.params) s += '<div>Trainable ' + (r.params / 1e6).toFixed(2) + ' M (' + r.pct + '%)</div>';
               if (r.bs) s += '<div>Batch size ' + r.bs + '</div>';
               return s; }};
  function drawMem(key) {
    var el = AdaptUI.remount(sec.querySelector('[data-chart=mem]'));
    Charts.barChart(el, d.views[key], ref);
  }
  var keys = Object.keys(d.views);
  AdaptUI.seg(sec.querySelector('[data-role=bs]'), 'Batch size',
    keys.map(function (k) { return {key: k, label: k.replace('bs', 'BS ')}; }), 'bs32', drawMem);
  drawMem('bs32');
  Charts.barChart(sec.querySelector('[data-chart=oom]'), d.solutions, ref);
""", sid=sid)
    rows = [r for k in views for r in views[k]]
    tbl = table(["Configuration", "Batch", "Peak MiB", "Trainable params (M)", "Trainable %", "Status"],
                [[r["label"], r["bs"], num(r["value"], ".0f"), num((r["params"] or 0) / 1e6, ".2f"), num(r["pct"], ".3f"), r["status"]]
                 for r in rows], num_cols=(1, 2, 3, 4),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul>"
        f"<li>Sources: <a href=\"{ctx.raw(MEM_FILES['bs32'])}\"><code>{MEM_FILES['bs32']}</code></a>, "
        f"<a href=\"{ctx.raw(MEM_FILES['bs16'])}\"><code>{MEM_FILES['bs16']}</code></a>, "
        f"<a href=\"{ctx.raw(OOM_FILE)}\"><code>{OOM_FILE}</code></a>. Rank write-up: "
        f"<a href=\"{ctx.doc('lora-adaptation-summary.md')}\">lora-adaptation-summary</a>.</li>"
        "<li>Peak memory is dominated by activations of the unfrozen blocks, which LoRA does not remove; the adapter saves "
        "optimizer state and gradients for the frozen weights only.</li>"
        "<li>Rows marked OOM record the allocation at the failure point; the 12-block LoRA <code>save_on_cpu</code> row "
        "errored before a peak was recorded, so it has no bar.</li>"
        "<li>These are single measurements on one 8 GB card, not repeated benchmarks.</li>"
        "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Memory profile ({len(rows)} rows)"), {f"data-{sid}": blob}


# =============================================================================== 4. holdout markers

UNEXPOSED_FILE = "downstream-probe-unexposed-v2.json"
LORA_HOLDOUT_FILE = "downstream-probe-lora-holdout-v1.json"
HOLD_GROUPS = {"imgft": {"label": "Image only, 4 blocks", "color": "var(--g-imgft)"},
               "lora": {"label": "LoRA r64, 6 blocks", "color": "var(--g-lora)"}}


def holdout_values() -> dict:
    """Pooled 4-block exposed/unexposed levels from the unexposed-v2 file, plus the LoRA holdout arm."""
    u = data.read_json(UNEXPOSED_FILE)
    h = data.read_json(LORA_HOLDOUT_FILE)
    out = {}
    if u:
        lv = {a.key: a.macro_auroc for a in data.probe_runs(u) if a.feature == FEATURE}
        ex = [v for k, v in lv.items() if k.startswith("exposed/original-unfreeze4")]
        un = [v for k, v in lv.items() if k.startswith("unexposed/holdout-unfreeze4")]
        out.update(exposed_pool=data.seed_mean(ex), unexposed_pool=data.seed_mean(un), n_seeds=len(un),
                   floor_u=data.unadapted_level(u))
    if h:
        for a in data.probe_runs(h):
            if a.feature == FEATURE and a.run == "holdout-both-lora-depth6-r64-seed2026":
                out["lora_holdout"] = a.macro_auroc
        out["floor_h"] = data.unadapted_level(h)
    return out


def holdout_section(ctx) -> tuple[str, dict]:
    sid = "holdout"
    u = data.read_json(UNEXPOSED_FILE)
    h = data.read_json(LORA_HOLDOUT_FILE)
    dep = data.read_json(DEPTH_FILE)
    if not (u and h and dep):
        return section(sid, "Held-out gold", notice("Holdout probe JSONs are missing.")), {}
    hv = holdout_values()
    exposed, unexposed = [], []

    def add(lst, v: data.VsFloor, label, group, file, floor):
        lst.append({"id": f"{sid}-{len(exposed) + len(unexposed)}", "label": label, "group": group, "est": v.diff,
                    "lo": v.ci[0], "hi": v.ci[1], "after": v.after, "floor": floor, "file": file,
                    **({"color": "var(--g-holdout)"} if lst is unexposed else {})})
    fu, fh, fd = data.unadapted_level(u), data.unadapted_level(h), data.unadapted_level(dep)
    for v in sorted(data.probe_vs_unadapted(u), key=lambda v: v.key):
        if v.feature != FEATURE:
            continue
        if v.key.startswith("exposed/original-unfreeze4"):
            add(exposed, v, f"Img 4 blk s{_seed(v.run)}", "imgft", UNEXPOSED_FILE, fu)
        elif v.key.startswith("unexposed/holdout-unfreeze4"):
            add(unexposed, v, f"Img 4 blk s{_seed(v.run)}", "imgft", UNEXPOSED_FILE, fu)
    for v in sorted(data.probe_vs_unadapted(dep), key=lambda v: ("reg" in v.run, v.run)):
        if v.feature == FEATURE and v.run in ("both-lora-depth6-r64-seed2026", "both-lora-depth6-r64-bs64-reg-seed2026"):
            add(exposed, v, "LoRA r64" + (" BS64 reg" if "reg" in v.run else ""), "lora", DEPTH_FILE, fd)
    for v in sorted(data.probe_vs_unadapted(h), key=lambda v: ("reg" in v.run, v.run)):
        if v.feature == FEATURE and v.run.startswith("holdout-both-lora"):
            add(unexposed, v, "LoRA r64" + (" BS64 reg" if "reg" in v.run else ""), "lora", LORA_HOLDOUT_FILE, fh)

    claim = (f"Held out of training, the 4-block encoder reads {num(hv.get('unexposed_pool'))} "
             f"(pooled over {hv.get('n_seeds')} seeds, vs {num(hv.get('exposed_pool'))} exposed) and LoRA r64 reads "
             f"{num(hv.get('lora_holdout'))} on one seed; the two panels are different populations, not a paired transfer estimate")
    lora_floors = " / ".join(dict.fromkeys([num(fd), num(fh)]))
    caption = (f"Gold macro AUROC minus the unadapted DINOv2 floor of the same file ({num(fu)} image-only file, {lora_floors} "
               f"LoRA files), bootstrap 95% CI. Exposed panel: gold studies seen in training (development statistic); held-out panel in amber.")
    blob = {"groups": HOLD_GROUPS, "exposed": exposed, "unexposed": unexposed}
    body = f"""<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('exposed', 'Exposed: gold studies were training pairs')}</div>
<div>{chart('unexposed', 'Unexposed: gold held out of training and selection')}</div>
</div>
""" + js("""
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var dom = [0, 0];
  d.exposed.concat(d.unexposed).forEach(function (r) { dom[0] = Math.min(dom[0], r.lo); dom[1] = Math.max(dom[1], r.hi); });
  var o = {groups: d.groups, ref: 0, refLabel: 'own-file floor', fmt: '+.4f', axisFmt: '+.2f', sort: false, domain: dom,
           xLabel: '\\u0394 vs unadapted (same file)', store: store,
           tooltipExtra: function (r) { return '<div>Level ' + Charts.fmt('.4f')(r.after) + ' \\u00b7 floor ' + Charts.fmt('.4f')(r.floor) + '</div><div class="muted">' + r.file + '</div>'; }};
  Charts.ciDotPlot(sec.querySelector('[data-chart=exposed]'), d.exposed, o);
  Charts.ciDotPlot(sec.querySelector('[data-chart=unexposed]'), d.unexposed, o);
""", sid=sid)
    allr = exposed + unexposed
    tbl = table(["Arm", "Population", "Gold macro AUROC", "Floor (file)", "Δ", "95% CI", "File"],
                [[r["label"], "exposed" if r in exposed else "unexposed", num(r["after"]), num(r["floor"]),
                  num(r["est"], ".4f", True), f"[{num(r['lo'], '.4f', True)}, {num(r['hi'], '.4f', True)}]", r["file"]]
                 for r in allr], num_cols=(2, 3, 4, 5),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in allr])
    gap = data.POOLED["gold_holdout_exposure_gap"]
    notes = (
        "<ul>"
        f"<li>Sources: <a href=\"{ctx.raw(UNEXPOSED_FILE)}\"><code>{UNEXPOSED_FILE}</code></a>, "
        f"<a href=\"{ctx.raw(LORA_HOLDOUT_FILE)}\"><code>{LORA_HOLDOUT_FILE}</code></a>, "
        f"<a href=\"{ctx.raw(DEPTH_FILE)}\"><code>{DEPTH_FILE}</code></a>. Pooled levels are seed means computed from "
        f"<code>{UNEXPOSED_FILE}</code>; the pooled exposure gap {num(gap['value'], '.4f', True)} is quoted from "
        f"<a href=\"{ctx.doc(gap['doc'])}\">gold-holdout-v1</a>.</li>"
        f"<li>Always quote the exposed {num(hv.get('exposed_pool'))} pool (not seed 2026 alone) beside its unexposed "
        f"{num(hv.get('unexposed_pool'))}; never subtract the gap from a paired difference.</li>"
        "<li>Exposed and held-out LoRA rows are different training populations, not paired runs, so the right-minus-left "
        "difference is <b>not</b> a transfer or inflation estimate. The held-out LoRA row is still development evidence: "
        "the same 56 studies have been used for repeated model selection. It is the encoder selected for the first submission.</li>"
        "<li>The BS64 regularised arm changes four things at once (LoRA dropout, drop-path, soft targets λ=0.1 with a temperature "
        "cap, 40 epochs) and cannot attribute its effect. All LoRA rows are single seed 2026.</li>"
        "<li>Held-out encoders validate retrieval on 433 candidates, not 441; their R@1 curves are in the explorer below under "
        "their own pool, never on the exposed axis. The lora-holdout file also re-reads the 4-block s2026 held-out encoder "
        f"against its {num(fh)} floor; it is the same checkpoint as the s2026 held-out image row, so it is not repeated.</li>"
        "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Holdout table ({len(allr)} rows)"), {f"data-{sid}": blob}


# =============================================================================== 5. per-label heatmap

HEAT_ARMS = [
    ("before_cl/dinov2-unadapted", "Unadapted", "anchor"),
    ("after_cl/both-lora-depth6-r16-seed2026", "LoRA6 r16", "lora"),
    ("after_cl/both-lora-depth6-r32-seed2026", "LoRA6 r32", "lora"),
    ("after_cl/both-lora-depth6-r64-seed2026", "LoRA6 r64", "lora"),
    ("after_cl/both-lora-depth6-r64-bs64-seed2026", "LoRA6 BS64", "lora"),
    ("after_cl/both-lora-depth8-r64-seed2026", "LoRA8 r64", "lora"),
    ("after_cl/both-lora-depth12-r64-seed2026", "LoRA12 r64", "lora"),
    ("after_cl/both-unfreeze2-seed2026", "Both 2", "bothft"),
    ("after_cl/both-unfreeze4-seed2026", "Both 4", "bothft"),
    ("after_cl/both-unfreeze6-seed2026", "Both 6", "bothft"),
    ("after_cl/both-unfreeze8-seed2026", "Both 8", "bothft"),
    ("after_cl/original-unfreeze2-seed2026", "Img 2", "imgft"),
    ("after_cl/original-unfreeze4-seed2026", "Img 4", "imgft"),
    ("after_cl/original-unfreeze8-seed2026", "Img 8", "imgft"),
]


def per_label_section(ctx) -> tuple[str, dict]:
    sid = "per-label"
    dep = data.read_json(DEPTH_FILE)
    if not dep:
        return section(sid, "Per-label AUROC", notice(f"<code>{DEPTH_FILE}</code> is missing.")), {}
    labels = data.probe_labels(dep)
    arms = {f"{a.slot}/{a.run}": a for a in data.probe_runs(dep) if a.feature == FEATURE}
    base = arms.get("before_cl/dinov2-unadapted")
    cols = [(k, lab, g) for k, lab, g in HEAT_ARMS if k in arms]
    level = [[arms[k].per_label.get(l) for k, _, _ in cols] for l in labels]
    delta = [[(arms[k].per_label.get(l) - base.per_label.get(l)) if k != "before_cl/dinov2-unadapted" else None
              for k, _, _ in cols] for l in labels]
    pos = base.positives
    rows = [f"{l} ({pos.get(l)})" for l in labels]
    r64 = "after_cl/both-lora-depth6-r64-seed2026"
    up = sum(1 for l in labels if arms[r64].per_label[l] > base.per_label[l]) if r64 in arms else None
    pvals = [pos.get(l) for l in labels]
    claim = (f"LoRA r64 raises {up} of {len(labels)} labels above unadapted DINOv2, but each label rests on only "
             f"{min(pvals)}–{max(pvals)} positive studies, so per-label rankings are fragile")
    caption = (f"Out-of-fold AUROC per label (row: label, positive studies of {dep['population']['population_size']}) × arm "
               f"(single seed 2026, exposed, {DEPTH_FILE}). Outlined cell = row maximum. Pills filter the arm columns.")
    blob = {"groups": {k: v for k, v in METHOD_GROUPS.items() if k != "textblk"}, "rows": rows, "labels": labels,
            "cols": [c[1] for c in cols], "colGroups": [c[2] for c in cols], "level": level, "delta": delta}
    body = f"""<div data-role="filter"></div><div data-role="mode"></div>
{chart('heat', 'Per-label gold AUROC by arm')}
""" + js("""
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var mode = 'level';
  function draw() {
    var g = store.get().group;
    var idx = d.cols.map(function (c, j) { return j; }).filter(function (j) {
      return !g || g === 'all' || d.colGroups[j] === g || d.colGroups[j] === 'anchor'; });
    if (mode === 'delta') idx = idx.filter(function (j) { return d.colGroups[j] !== 'anchor'; });
    var m = (mode === 'level' ? d.level : d.delta).map(function (row) { return idx.map(function (j) { return row[j]; }); });
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var o = {rows: d.rows, cols: idx.map(function (j) { return d.cols[j]; }), rowIds: d.labels, store: store,
             fmt: mode === 'level' ? '.2f' : '+.2f', ariaLabel: 'per-label AUROC heatmap'};
    if (mode === 'delta') { o.diverging = true; o.peakPerRow = false; }
    Charts.heatmap(el, m, o);
  }
  AdaptUI.seg(sec.querySelector('[data-role=mode]'), 'Cells', [{key: 'level', label: 'AUROC'}, {key: 'delta', label: '\\u0394 vs unadapted'}],
    'level', function (k) { mode = k; draw(); });
  var last = null;
  store.on(function (st) { if (st.group !== last) { last = st.group; draw(); } });
""", sid=sid)
    notes = (
        "<ul>"
        f"<li>Source: <a href=\"{ctx.raw(DEPTH_FILE)}\"><code>{DEPTH_FILE}</code></a> <code>results[*].per_label</code>, "
        f"uniform-mean features; all columns from one extraction so folds and studies match. Unadapted anchor here is {num(base.macro_auroc)}.</li>"
        "<li>Per-label differences are simple level differences; per-label bootstrap intervals are wide on "
        f"{min(pvals)}–{max(pvals)} positives. Do not read single-cell maxima as records.</li>"
        "<li>The held-out LoRA encoder is a different population and is not a column here.</li>"
        "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# =============================================================================== 6. plane pooling

PP_COMP = "plane-pooling-comparisons-v1.json"
PP_ANALYSIS = "plane-pooling-analysis.json"
PP_GROUPS = {"paired": {"label": "Uniform − attention (4 blocks)", "color": "var(--g-imgft)"},
             "feature": {"label": "Readout feature (1-block runs)", "color": "var(--g-violet)"},
             "weights": {"label": "Attention weight", "color": "var(--g-amber)"}}
PP_FEATURES = [("uniform_mean_768", "Uniform mean"), ("concat_planes_2304", "Concat 3 planes (2304)"),
               ("attention_pooled_768", "Learned attention"), ("attention_no_embeddings_768", "Attention, no plane emb."),
               ("plane_sagittal_768", "Sagittal only"), ("plane_coronal_768", "Coronal only"), ("plane_axial_768", "Axial only")]


def plane_pooling_section(ctx) -> tuple[str, dict]:
    sid = "plane-pooling"
    comp = data.read_json(PP_COMP)
    ana = data.read_json(PP_ANALYSIS)
    if not comp or not ana:
        return section(sid, "Plane pooling", notice("Plane-pooling JSONs are missing.")), {}
    paired = [{"id": f"pp-{i}", "label": f"seed {_seed(p.candidate)}", "group": "paired", "est": p.diff, "lo": p.ci[0],
               "hi": p.ci[1], "base": p.macro_baseline, "cand": p.macro_candidate}
              for i, p in enumerate(sorted(data.paired_comparisons(comp), key=lambda p: p.candidate))]
    res = ana.get("results") or {}
    feats = []
    for key, lab in PP_FEATURES:
        vals = {k.split("/")[0]: v["macro_auroc"] for k, v in res.items() if k.endswith("/" + key)}
        if vals:
            feats.append({"id": f"feat-{key}", "label": lab, "group": "feature", "est": data.seed_mean(list(vals.values())),
                          "lo": min(vals.values()), "hi": max(vals.values()),
                          "seeds": {_seed(k): v for k, v in sorted(vals.items())}})
    w = ana.get("plane_attention_weights") or {}
    order = next(iter(w.values()), {}).get("plane_order", ["Sagittal", "Coronal", "Axial"])
    weights = []
    for i, plane in enumerate(order):
        vals = [x["mean_weight"][i] for x in w.values()]
        weights.append({"id": f"w-{plane}", "label": plane, "group": "weights", "value": data.seed_mean(vals),
                        "seeds": {_seed(k): x["mean_weight"][i] for k, x in sorted(w.items())}})
    uniform_w = next(iter(w.values()), {}).get("uniform_weight")
    n_pos = sum(1 for r in paired if r["est"] > 0)
    n_clear = sum(1 for r in paired if r["lo"] > 0)
    rule_pass = n_pos == len(paired) and n_clear == len(paired)
    claim = (f"Training with uniform plane pooling beat learned attention in {n_pos}/{len(paired)} seeds "
             f"({_range([r['est'] for r in paired])}), but {n_clear}/{len(paired)} intervals exclude zero: "
             + ("the pre-registered rule is met" if rule_pass else "null by the pre-registered rule"))
    axial = max(weights, key=lambda r: r["value"]) if weights else None
    caption = (f"Left: paired Δ gold macro AUROC, 4-block runs, 95% CI. Right: readout by feature and mean learned plane weight "
               f"(uniform = {num(uniform_w, '.3f')}), older 1-block translated runs, mean of {len(w)} seeds"
               + (f"; attention up-weights the {axial['label'].lower()} plane." if axial else "."))
    blob = {"groups": PP_GROUPS, "paired": paired, "feats": feats, "weights": weights, "uniform": uniform_w}
    body = f"""<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('pp-paired', 'Uniform pooling − learned attention (paired)')}</div>
<div>{chart('pp-feat', 'Probe AUROC by readout feature (3 seeds)')}{chart('pp-w', 'Mean learned attention weight per plane')}</div>
</div>
""" + js("""
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  function seeds(r) { return '<div>' + Object.keys(r.seeds).map(function (s) { return 's' + s + ' ' + Charts.fmt('.4f')(r.seeds[s]); }).join(' \\u00b7 ') + '</div>'; }
  Charts.ciDotPlot(sec.querySelector('[data-chart=pp-paired]'), d.paired,
    {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.2f', sort: false, store: store,
     xLabel: '\\u0394 gold macro AUROC',
     tooltipExtra: function (r) { return '<div>' + Charts.fmt('.4f')(r.base) + ' \\u2192 ' + Charts.fmt('.4f')(r.cand) + '</div>'; }});
  Charts.ciDotPlot(sec.querySelector('[data-chart=pp-feat]'), d.feats,
    {groups: d.groups, fmt: '.4f', axisFmt: '.2f', store: store, sort: false, estLabel: 'Seed mean',
     xLabel: 'Gold macro AUROC (dot = seed mean, bar = seed range)', tooltipExtra: seeds, rowHeight: 22});
  Charts.barChart(sec.querySelector('[data-chart=pp-w]'), d.weights,
    {groups: d.groups, fmt: '.3f', axisFmt: '.2f', store: store, ref: d.uniform, refLabel: 'uniform 1/3',
     xLabel: 'Mean weight', tooltipExtra: seeds, rowHeight: 22});
""", sid=sid)
    tbl = table(["Seed", "Attention (baseline)", "Uniform", "Δ", "95% CI"],
                [[r["label"], num(r["base"]), num(r["cand"]), num(r["est"], ".4f", True),
                  f"[{num(r['lo'], '.4f', True)}, {num(r['hi'], '.4f', True)}]"] for r in paired], num_cols=(1, 2, 3, 4),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in paired])
    notes = (
        "<ul>"
        f"<li>Pre-registration: <a href=\"{ctx.doc('plane-pooling-protocol.md')}\">plane-pooling-protocol</a> (amended: its "
        "addendum corrects a mis-attributed number and the plane order Sagittal/Coronal/Axial). Rule: all three seeds same sign "
        f"<b>and</b> each interval excludes zero. Result: <a href=\"{ctx.doc('plane-pooling-v1.md')}\">plane-pooling-v1</a>; "
        f"data <a href=\"{ctx.raw(PP_COMP)}\"><code>{PP_COMP}</code></a>.</li>"
        "<li>Reading: consistent direction, individually underpowered, <b>not established</b>. No post-hoc pooled test is "
        "claimed; the per-seed rule was too strict for effects of 0.01–0.02, a lesson later protocols fixed with a pooled estimand. "
        "The uniform arm also removes the attention parameters, so the two changes are not isolated.</li>"
        f"<li>Right-hand charts: <a href=\"{ctx.raw(PP_ANALYSIS)}\"><code>{PP_ANALYSIS}</code></a>, built on the older "
        "<code>claude-translated-dfd07db-image-block-v1</code> 1-block checkpoints, not the 4-block runs on the left; "
        "levels are not comparable across the two panels.</li>"
        "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary="Paired contrasts"), {f"data-{sid}": blob}


# =============================================================================== 7. curve explorer

CURVE_SOURCES = [
    # (root, group, pool-key)
    ("image-block-cohort-compare-v1", None, None),
    ("image-block-depth-v1", "imgft", None),
    ("image-block-epochs-v1", "nulls", None),
    ("image-block-control-v1", "nulls", None),
    ("plane-pooling-v1", "nulls", None),
    ("both-block-depth-v1", "bothft", None),
    ("both-lora-v1", "lora", None),
    ("both-lora-holdout-v1", "holdout", None),
]
CURVE_GROUPS = {
    "imgft": {"label": "Image only, full FT", "color": "var(--g-imgft)"},
    "nulls": {"label": "Controls and nulls", "color": "var(--g-rose)"},
    "bothft": {"label": "Both towers, full FT", "color": "var(--g-bothft)"},
    "lora": {"label": "Both towers, LoRA", "color": "var(--g-lora)"},
    "holdout": {"label": "Gold held out (LoRA)", "color": "var(--g-holdout)"},
}
CURVE_METRICS = [("mean_r@1", "Mean R@1", ".1%", "max"), ("validation_loss", "Validation loss", ".3f", "min"),
                 ("train_loss", "Train loss", ".3f", "min"), ("inverse_temperature", "Inverse temperature", ".1f", "last")]


def _curve_label(root: str, run: str) -> str:
    from .adaptation import label_of
    s = _seed(run)
    if root == "image-block-cohort-compare-v1":
        return ("Img 1 blk" if run.startswith("original") else "Img 1 blk, translated") + f" s{s}"
    if root == "image-block-depth-v1":
        m = re.match(r"original-unfreeze(\d+)", run)
        return f"Img {m.group(1)} blk s{s}"
    if root == "image-block-epochs-v1":
        return f"Img 1 blk, 40 ep s{s}"
    if root == "image-block-control-v1":
        return f"Shuffled pairing 4 blk s{s}"
    if root == "plane-pooling-v1":
        return f"Uniform pooling 4 blk s{s}"
    if root == "both-lora-holdout-v1":
        return "Held out: " + label_of(run.replace("holdout-", ""))
    return label_of(run)


def curves_section(ctx) -> tuple[str, dict]:
    sid = "curves"
    series, skipped = [], []
    for root, group, _ in CURVE_SOURCES:
        c = data.load_curves(root)
        if not c:
            skipped.append(f"{root} (snapshot missing)")
            continue
        for run in sorted(c["runs"], key=lambda x: ([int(n) for n in re.findall(r"\d+", x.split("-seed")[0])], x)):
            r = c["runs"][run]
            g = group or ("imgft" if run.startswith("original") else "nulls")
            if len(r["epoch"]) < 10:
                skipped.append(f"{_curve_label(root, run)} ({len(r['epoch'])} epochs, aborted)")
                continue
            pts = {m: data.curve_series(c, run, m) for m, *_ in CURVE_METRICS}
            series.append({"id": f"{root}/{run}", "label": _curve_label(root, run), "group": g,
                           "pool": r.get("val_candidates"), "dash": "5 3" if re.search(r"-bs\d+|40ep|-reg", run) else None,
                           "metrics": pts, "root": root})
    if not series:
        return section(sid, "Training curves", notice("No curve snapshots. Run <code>build_digest_site.py export</code>.")), {}
    pools = sorted({s["pool"] for s in series if s["pool"]}, key=lambda p: -sum(1 for s in series if s["pool"] == p))
    main = pools[0]
    n_main = sum(1 for s in series if s["pool"] == main)
    peak = max((s for s in series if s["pool"] == main),
               key=lambda s: max((p[1] for p in s["metrics"]["mean_r@1"] if p[1] is not None), default=-1))
    peak_v = max(p[1] for p in peak["metrics"]["mean_r@1"] if p[1] is not None)
    claim = (f"Training curves for {len(series)} adaptation runs: on the {main}-candidate pool the highest validation R@1 is "
             f"{num(peak_v, '.1%')} ({esc(peak['label'])}, single seed); switch the metric to see the loss divergence")
    caption = ("Per-epoch validation metrics from the checkpoint snapshots. Candidate pools are separate views: exposed runs "
               f"validate on {main} candidates, held-out runs on {', '.join(str(p) for p in pools[1:]) or '—'}; never one axis.")
    blob = {"groups": CURVE_GROUPS, "series": group_major(series, CURVE_GROUPS), "pools": pools,
            "metrics": [{"key": k, "label": lab, "fmt": f, "best": b} for k, lab, f, b in CURVE_METRICS]}
    body = f"""<div data-role="metric"></div><div data-role="pool"></div><div data-role="filter"></div>
{chart('curve', 'Per-epoch curve')}
<div data-role="legend"></div>
<p class="legend-hint">Hover a line or legend entry to highlight it; click to pin, and click more to compare several at once; pills filter by group. Dashed = batch-size, 40-epoch or regularised variant.</p>
""" + js("""
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  var metric = d.metrics[0], pool = d.pools[0];
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var title = sec.querySelector('.chart-title');
  function draw() {
    var list = d.series.filter(function (s) { return s.pool === pool; }).map(function (s) {
      return {id: s.id, label: s.label, group: s.group, dash: s.dash, points: s.metrics[metric.key]}; });
    var el = AdaptUI.remount(sec.querySelector('[data-chart=curve]'));
    title.textContent = metric.label + ' by epoch (' + pool + ' validation candidates, ' + list.length + ' runs)';
    Charts.lineChart(el, list, {groups: d.groups, yLabel: metric.label, yFmt: metric.fmt, best: metric.best, store: store});
    Charts.legend(sec.querySelector('[data-role=legend]'), list, store, {groups: d.groups, best: metric.best, fmt: metric.fmt});
  }
  AdaptUI.seg(sec.querySelector('[data-role=metric]'), 'Metric', d.metrics.map(function (m) { return {key: m.key, label: m.label}; }),
    metric.key, function (k) { metric = d.metrics.filter(function (m) { return m.key === k; })[0]; draw(); });
  AdaptUI.seg(sec.querySelector('[data-role=pool]'), 'Candidate pool', d.pools.map(function (p) {
      return {key: String(p), label: p + (p === d.pools[0] ? ' (exposed)' : ' (gold held out)')}; }),
    String(pool), function (k) { pool = +k; store.set({group: 'all', solo: null, hover: null}); draw(); });
  draw();
""", sid=sid)
    notes = (
        "<ul>"
        f"<li>Snapshots of <code>checkpoints/contrastive/&lt;root&gt;/*/metrics.jsonl</code> under <code>digest/site/data/curves/</code>: "
        + ", ".join(f"<code>{r}</code>" for r, *_ in CURVE_SOURCES) + ".</li>"
        f"<li>This explorer replaces two legacy pages: the <a href=\"{ctx.raw('both-block-depth-v1.html')}\">original report</a> "
        f"for the both-block / LoRA sweep and the <a href=\"{ctx.raw('image-block-variants-curves.html')}\">original report</a> "
        "for the image-block variants. The first carries 'record' and '0% inflation' wording that this site does not endorse "
        f"(raw record: <a href=\"{ctx.doc('both-block-depth-results.md')}\">both-block-depth-results</a>).</li>"
        "<li><b>Train loss is not comparable to validation loss</b> (batch-32 augmented vs full-pool clean), nor across batch "
        "sizes or regularised arms; read its shape, not its level. Losses compare only at a matched temperature; R@1 is "
        "temperature-invariant. Why validation loss rises while R@1 holds is on the "
        f"<a href=\"{ctx.page('overfitting')}\">overfitting page</a>.</li>"
        "<li>AUROC has no per-epoch curve (only best/last checkpoints are probed). Checkpoints are selected on validation R@1. "
        "Both-tower and LoRA runs are single seed 2026, so neighbouring curves are not separable.</li>"
        + (f"<li>Not shown: {esc('; '.join(skipped))}.</li>" if skipped else "")
        + "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}
