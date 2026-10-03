"""Stage 3: report text, translation and the text tower.

Every number on this page is read at build time from digest/*.json or the digest/site/data/
snapshots (curves, text-block-v1.json, text-hubs.json). Sections:

  forest    every text-side comparison vs its matched frozen baseline (hero)
  timeline  translation pipeline 8e3bd68 -> 9bec1bd -> dfd07db -> quarantine, with val curves
  seeds     seed-paired slope chart (frozen vs adapted) for the image and text final blocks
  tokens    tokenizer length profile per text encoder (always with n)
  subgroups subgroup heatmaps (image block x 3 seeds, text encoders, impression-first)
  pairs     worst-pair explorer (replaces the 118 KB error-review packet)
  hubs      retrieval-hub concentration vs a random null
  gold      original-vs-translated gold-label probe (AUROC; kept apart from every R@1 chart)
"""

from __future__ import annotations

from .. import data
from ..shell import Page, chart, esc, notice, num, section, table
from ..sitemap import BY_SLUG

SLUG = "text"
SUBTITLE = "Translating reports, reordering them, and swapping or adapting the text tower."

F_COMPARE = "claude-translated-dfd07db-comparison-audit.json"
F_IMAGE = "claude-translated-dfd07db-image-block-audit.json"
F_IMAGE_REVIEW = "claude-translated-dfd07db-image-block-error-review.json"
F_ENC = "english-medical-text-results.json"
F_PRE = "english-medical-text-preflight.json"
F_GOLD_1 = "downstream-original-vs-translated-v1.json"
F_GOLD_3 = "downstream-extended-comparisons-v1.json"
F_GOOGLE = "config-translation-audit.json"
F_BASE_AUDIT = "claude-translated-baseline-audit.json"
F_UPDATE = "translation-update-dfd07db-audit.json"
F_INPUT = "claude-translated-dfd07db-input-audit.json"
F_ORDER = "claude-translated-dfd07db-ordering-audit.json"
F_DUP = "claude-translated-dfd07db-duplicate-verification.json"

LEVERS = {
    "view": {"label": "Report view", "color": "var(--g-amber)"},
    "image": {"label": "Image final block", "color": "var(--g-bothft)"},
    "textenc": {"label": "Text encoder swap", "color": "var(--g-violet)"},
    "texttower": {"label": "Text final block", "color": "var(--g-teal)"},
}
MOJIBAKE = "14 of 4,407"  # source: digest/english-medical-text-error-review.md:232
ENC_LABEL = {"e5-base-v2": "e5-base-v2", "pubmedbert-base-embeddings": "PubMedBERT"}

# One JS helper block for the whole page (Charts has no exported resize helper or custom marks).
HELPERS = r"""<style>
.tp-panel{border:1px solid var(--border);border-radius:8px;padding:10px 12px;background:var(--plot-bg);font-size:13px;margin-top:8px}
.tp-panel h4{margin:0 0 4px;font-size:13px;color:var(--text-strong)}
.tp-panel .muted{font-size:12px}
.tp-reports{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:8px}
@media(max-width:700px){.tp-reports{grid-template-columns:1fr}}
.tp-reports pre{white-space:pre-wrap;word-break:break-word;font-size:11.5px;line-height:1.45;max-height:320px;overflow:auto;margin:4px 0 0;
  padding:8px;border:1px solid var(--border-subtle);border-radius:6px;background:var(--card);font-family:var(--mono)}
.tp-toggle{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px}
.tp-toggle button{font:inherit;font-size:12px;padding:3px 10px;border-radius:999px;border:1px solid var(--border);background:var(--card);color:var(--text);cursor:pointer}
.tp-toggle button[aria-pressed=true]{background:var(--text-strong);color:var(--card);border-color:var(--text-strong)}
.chart .mark.highlighted line.tp-line{stroke-width:3.2}
.chart .mark.highlighted circle.tp-node{stroke:var(--text-strong);stroke-width:2}
.chart .tp-small{font-size:10.5px;fill:var(--muted)}
</style>
<script>
window.TP = (function () {
  var S = Charts.svg;
  function responsive(el, render) {
    var last = -1;
    function run(force) {
      var w = Math.round(el.clientWidth || 0) || 840;
      if (!force && w === last) return;
      last = w; render(w);
    }
    run(true);
    if (window.ResizeObserver) {
      var p = null;
      new ResizeObserver(function () { if (p) cancelAnimationFrame(p); p = requestAnimationFrame(function () { p = null; run(false); }); }).observe(el);
    }
    return function () { run(true); };
  }
  function hookMark(g, store, id, tip, html, onClick) {
    g.addEventListener('pointerenter', function () { if (!store.get().solo) store.set({hover: id}); });
    g.addEventListener('pointermove', function (e) { if (tip) tip.show(e, html()); });
    g.addEventListener('pointerleave', function () { if (tip) tip.hide(); store.set({hover: null}); });
    g.addEventListener('click', function () { if (onClick) onClick(); else store.toggleSolo(id); });
  }
  return {S: S, responsive: responsive, hookMark: hookMark};
})();
</script>"""


def _pp(v, d=2, signed=True):
    return num(v, f".{d}f", signed=signed) + " pp"


def _peak(curves, run):
    pts = [p for p in data.curve_series(curves, run, "mean_r@1") if p[1] is not None]
    return max(pts, key=lambda p: p[1]) if pts else (None, None)


# --------------------------------------------------------------------------- loaders


def _load():
    return {
        "cmp": data.read_json(F_COMPARE), "img": data.read_json(F_IMAGE), "rev": data.read_json(F_IMAGE_REVIEW),
        "enc": data.read_json(F_ENC), "pre": data.read_json(F_PRE), "tb": data.site_data("text-block-v1.json"),
        "hubs": data.site_data("text-hubs.json"),
    }


def _pool(d) -> int:
    return (d.get("rev") or {}).get("candidate_pool") or 441


# --------------------------------------------------------------------------- 1. forest


def forest_rows(d):
    rows = []
    cmp_, img, enc, tb = d["cmp"], d["img"], d["enc"], d["tb"]
    base2026 = None
    if cmp_:
        full = next(r for r in cmp_["runs"] if r["view"] == "full")
        base2026 = full["mean_r@1"] * 100
        p = cmp_["paired_differences"][0]
        rows.append({"id": "imp-first", "label": "Impression-first", "group": "view", "est": p["mean_difference_pp"],
                     "lo": p["descriptive_paired_query_bootstrap_95_ci_pp"][0],
                     "hi": p["descriptive_paired_query_bootstrap_95_ci_pp"][1], "seed": p["seed"], "base": base2026,
                     "note": "Single seed; report order only"})
    if img:
        bases = {r["seed"]: r["mean_r@1"] * 100 for r in img["runs"] if r["arm"] == "baseline"}
        for p in img["paired_differences"]:
            ci = p["descriptive_paired_query_bootstrap_95_ci_pp"]
            rows.append({"id": f"img-{p['seed']}", "label": f"Image blk s{p['seed']}", "group": "image",
                         "est": p["mean_difference_pp"], "lo": ci[0], "hi": ci[1], "seed": p["seed"],
                         "base": bases.get(p["seed"]), "note": "Final DINOv2 block + norm unfrozen"})
        rows.append({"id": "img-mean", "label": "Image blk mean †", "group": "image", "est": img["mean_difference_pp"],
                     "lo": None, "hi": None, "seed": "3 seeds", "base": None,
                     "note": f"Mean of 3 seeds, SD {num(img['sample_sd_pp'], '.2f')} pp; no pooled CI"})
    if enc:
        for r in enc["results"]:
            ci = r["paired"]["descriptive_paired_query_bootstrap_95_ci_pp"]
            rows.append({"id": f"enc-{r['candidate']}", "label": ENC_LABEL.get(r["candidate"], r["candidate"]),
                         "group": "textenc", "est": r["paired"]["mean_difference_pp"], "lo": ci[0], "hi": ci[1],
                         "seed": r["seed"], "base": base2026, "note": "Frozen text encoder swapped; single seed"})
    if tb:
        for s in tb["seeds"]:
            rows.append({"id": f"tb-{s['seed']}", "label": f"Text blk s{s['seed']} †", "group": "texttower",
                         "est": s["difference_pp"], "lo": None, "hi": None, "seed": s["seed"],
                         "base": s["baseline_mean_r@1_pp"], "note": "Validation-only readout; no CI exists"})
        rows.append({"id": "tb-mean", "label": "Text blk mean †", "group": "texttower",
                     "est": tb["mean_difference_pp"], "lo": None, "hi": None, "seed": "3 seeds", "base": None,
                     "note": f"Mean of 3 seeds, SD {num(tb['sd_difference_pp'], '.2f')} pp; not independently verified"})
    return rows, base2026


def forest_section(ctx, d):
    sid = "forest"
    rows, base2026 = forest_rows(d)
    if not rows:
        return section(sid, "Text-side comparisons", notice("Comparison audits are missing.")), {}
    pool = _pool(d)
    one = 100 / pool
    with_ci = [r for r in rows if r["lo"] is not None]
    clear = [r for r in with_ci if r["lo"] > 0 or r["hi"] < 0]
    no_ci = [r for r in rows if r["lo"] is None]
    best = max(rows, key=lambda r: r["est"])
    claim = (f"{len(clear)} of {len(with_ci)} text-side comparisons with an interval exclude zero; "
             f"the image final block leads at {_pp(next((r['est'] for r in rows if r['id'] == 'img-mean'), best['est']))} "
             f"(3-seed mean) and the text final block trails at "
             f"{_pp(next((r['est'] for r in rows if r['id'] == 'tb-mean'), None))}")
    vals = [v for r in rows for v in (r["est"], r["lo"], r["hi"]) if v is not None] + [-one, one]
    span = max(vals) - min(vals)
    dom = [min(vals) - span * 0.04, max(vals) + span * 0.04]
    blob = {"groups": LEVERS, "rows": rows, "dom": dom, "one": one,
            "refLabel": f"0 = matched frozen baseline (s2026 = {num(base2026 / 100 if base2026 else None, '.2%')})"}
    body = f"""{HELPERS}
<div data-role="filter"></div>
{chart('forest', f'Δ validation mean R@1 vs matched frozen baseline (pp, {pool} candidates)')}
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var el = sec.querySelector('[data-chart=forest]');
  Charts.ciDotPlot(el, d.rows, {{groups: d.groups, sort: false, ref: 0, refLabel: d.refLabel, fmt: '+.2f', axisFmt: '+.1f',
    domain: d.dom, labelWidth: 170, xLabel: 'Δ mean R@1 (pp)', store: store, band: [-d.one, d.one], bandLabel: '±1/441',
    tooltipExtra: function (r) {{
      return '<div>' + Charts.esc(r.note) + '</div>' + (r.base ? '<div>Frozen baseline ' + r.base.toFixed(2) + '%</div>' : '') +
        (r.lo === null ? '<div class="muted">No confidence interval</div>' : '');
    }}}});
}})();
</script>"""
    caption = (f"Dot = paired Δ, whisker = descriptive study-paired bootstrap 95% CI (single-seed rows only). "
               f"Shaded band = ±1/{pool} = ±{num(one, '.4f')} pp (one study counted in both directions; one direction-query moves mean R@1 by half that). † = no CI (seed means, text-block runs).")
    tbl = table(["Comparison", "Lever", "Seed", "Frozen baseline R@1", "Δ (pp)", "95% CI (pp)"],
                [[r["label"], LEVERS[r["group"]]["label"], r["seed"],
                  num(r["base"] / 100, ".2%") if r["base"] else "–", num(r["est"], ".4f", True),
                  f"[{num(r['lo'], '.2f', True)}, {num(r['hi'], '.2f', True)}]" if r["lo"] is not None else "none"]
                 for r in rows], num_cols=(3, 4, 5),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul>"
        f"<li><b>Every interval includes zero</b>, including the image block, the one lever that trended positive. "
        f"{len(no_ci)} rows (seed means and the text-block runs) have no interval at all.</li>"
        "<li>Intervals are descriptive: study-paired bootstrap after validation checkpoint selection, no multiplicity or "
        "patient-cluster correction; a single seed says nothing about seed variance.</li>"
        f"<li>Each direction's R@1 moves on a 1/{pool} grid ({num(one, '.4f')} pp per hit, the shaded band = one study in both directions); mean R@1 averages "
        f"the two directions, so it moves in {num(one / 2, '.4f')} pp steps. Equal values across seeds are grid collisions.</li>"
        "<li>Zero is the <em>matched</em> frozen baseline: seeds 2027 and 2028 compare against their own frozen runs, "
        "not against the seed-2026 value.</li>"
        "<li><b>Text final block:</b> the protocol "
        f"(<a href=\"{ctx.doc('claude-translated-dfd07db-text-block-protocol.md')}\">text-block protocol</a>) still says "
        "“not launched”; that is stale. The three runs exist and are read here from the "
        "<code>digest/site/data/text-block-v1.json</code> snapshot of "
        "<code>checkpoints/contrastive/claude-translated-dfd07db-text-block-v1/validation-selection.json</code>. "
        "These results were <b>not independently verified</b>, unlike the image-block and encoder arms, and seed 2026 carries "
        "most of the negative mean.</li>"
        f"<li>Sources: <a href=\"{ctx.doc('claude-translated-dfd07db-comparison-results.md')}\">impression-first results</a>, "
        f"<a href=\"{ctx.doc('claude-translated-dfd07db-image-block-results.md')}\">image-block results</a>, "
        f"<a href=\"{ctx.doc('english-medical-text-results.md')}\">text-encoder results</a>.</li>"
        "<li>Exact-pair retrieval is not clinical accuracy. The locked test split was never evaluated.</li>"
        "</ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Comparison table ({len(rows)} rows)"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 2. timeline


def timeline_section(ctx, d):
    sid = "timeline"
    pool = _pool(d)
    c9 = data.load_curves("claude-translated-9bec1bd-v1")
    cd = data.load_curves("claude-translated-dfd07db-v1")
    cq = data.load_curves("claude-translated-dfd07db-q2-v1")
    cm = data.load_curves("full-multilingual-v1")
    if not (c9 and cd and cq):
        return section(sid, "Translation pipeline", notice("Curve snapshots for the translation runs are missing.")), {}
    g = data.read_json(F_GOOGLE) or {}
    ba = data.read_json(F_BASE_AUDIT) or {}
    up = data.read_json(F_UPDATE) or {}
    ia = data.read_json(F_INPUT) or {}
    oa = data.read_json(F_ORDER) or {}
    dup = data.read_json(F_DUP) or []
    gt = (g.get("columns") or {}).get("Translated Report", {})
    gr = (g.get("columns") or {}).get("Report", {})
    changed = up.get("changed_rows") or []
    prov = ba.get("text_provenance") or {}

    def node(nid, label, commit, curves, run, detail, status):
        ep, v = _peak(curves, run) if curves else (None, None)
        return {"id": nid, "label": label, "commit": commit, "value": v, "epoch": ep,
                "hits": round(v * 2 * pool) if v is not None else None, "detail": detail, "status": status,
                "run": f"{curves['source']}/{run}" if curves else None}

    nodes = [
        node("google", "Google", "8e3bd68", None, None,
             f"Teammate's Google Cloud translations reviewed; no training run on them is in the digest. Zero-width-space rows "
             f"{gt.get('zero_width_space_rows')}, “trunk” {gt.get('word_trunk_rows')}, “neat” {gt.get('word_neat_rows')}; "
             f"studies in exact-duplicate report groups {gr.get('studies_in_duplicate_groups')} → "
             f"{gt.get('studies_in_duplicate_groups')}. Decision: not adopted unchecked.", "not adopted"),
        node("c9bec1bd", "9bec1bd", "9bec1bd", c9, "00-projection-full-mean-512",
             f"First complete Claude translation: {ba.get('complete_translations')} cells, "
             f"{prov.get('translated_english')} translated + {prov.get('source_english')} source-English. "
             "Kept labelled 9bec1bd (never relabelled as dfd07db).", "historical"),
        node("cdfd07db", "dfd07db", "dfd07db", cd, "00-projection-full-mean-512",
             f"{len(changed)} manual translation edits "
             f"({sum(1 for r in changed if r.get('split') == 'train')} train, "
             f"{sum(1 for r in changed if r.get('split') == 'validation')} validation) required a fresh baseline. "
             f"Input audit: {((ia.get('text') or {}).get('validation') or {}).get('over_512')} of "
             f"{((ia.get('text') or {}).get('validation') or {}).get('studies')} validation reports over 512 tokens; "
             f"{len(ia.get('exact_cached_tensor_duplicate_groups') or [])} identical-image pair found.", "full-cohort anchor"),
        node("cq2", "dfd07db-q2", "dfd07db + quarantine", cq, "00-full-seed2026",
             f"The {len(dup)} identical-image studies (laterality conflict) quarantined, leaving "
             f"{((oa.get('counts') or {}).get('train') or {}).get('studies')} train studies. "
             "This full-text run is THE frozen baseline for every comparison on this page.", "baseline"),
    ]
    ref_ep, ref = _peak(cm, "00-projection-full-mean-512") if cm else (None, None)
    bvals = [r["mean_r@1"] for r in (d["img"] or {}).get("runs", []) if r["arm"] == "baseline"]
    base_swing = (max(bvals) - min(bvals)) * 100 if bvals else None
    vals = [n for n in nodes if n["value"] is not None]
    lo, hi = min(vals, key=lambda n: n["value"]), max(vals, key=lambda n: n["value"])
    claim = (f"Claude-translated baselines span {num(lo['value'], '.2%')}–{num(hi['value'], '.2%')} "
             f"({lo['hits']}–{hi['hits']} of {2 * pool} direction-queries, one seed each), level with the multilingual original"
             + (f" ({num(ref, '.2%')})" if ref else "") + "; the Google corpus was never adopted")
    series = []
    for n in nodes:
        if n["value"] is None:
            continue
        curves = {"c9bec1bd": c9, "cdfd07db": cd, "cq2": cq}[n["id"]]
        run = n["run"].rsplit("/", 1)[1]
        series.append({"id": n["id"], "label": n["label"], "group": "claude", "points": data.curve_series(curves, run, "mean_r@1")})
    if cm:
        series.append({"id": "orig", "label": "Multilingual original", "group": "original", "dash": "5 3",
                       "points": data.curve_series(cm, "00-projection-full-mean-512", "mean_r@1")})
    groups = {"claude": {"label": "Claude translation", "color": "var(--g-teal)"},
              "original": {"label": "Multilingual original", "color": "var(--g-anchor)"},
              "google": {"label": "Google (not trained)", "color": "var(--g-rose)"}}
    blob = {"nodes": nodes, "series": series, "groups": groups, "ref": ref, "pool": pool}
    body = f"""<div class="grid-2">
<div>{chart('steps', 'Best validation mean R@1 at each translation step (click a step)')}<div class="tp-panel" data-role="detail"></div></div>
<div>{chart('curves', f'Validation mean R@1 by epoch ({pool} candidates)')}</div>
</div>
<div data-role="legend"></div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec), S = TP.S;
  var el = sec.querySelector('[data-chart=steps]'), tip = Charts.tooltip(el), detail = sec.querySelector('[data-role=detail]');
  var pct = Charts.fmt('.2%');
  function colorOf(n) {{ return n.value === null ? 'var(--g-rose)' : 'var(--g-teal)'; }}
  function show(id) {{
    var n = d.nodes.filter(function (x) {{ return x.id === id; }})[0] || d.nodes[d.nodes.length - 1];
    detail.innerHTML = '';
    var h = document.createElement('h4');
    h.textContent = n.label + ' (' + n.commit + ') · ' + n.status + (n.value !== null ? ' · ' + pct(n.value) + ', ' + n.hits + '/' + (2 * d.pool) + ' direction-queries, epoch ' + n.epoch : '');
    var p = document.createElement('div'); p.textContent = n.detail;
    detail.appendChild(h); detail.appendChild(p);
    if (n.run) {{ var c = document.createElement('div'); c.className = 'muted'; c.textContent = 'Run: ' + n.run; detail.appendChild(c); }}
  }}
  TP.responsive(el, function (W) {{
    var old = el.querySelector(':scope > svg'); if (old) old.remove();
    var H = W < 520 ? 250 : 290, pad = {{l: W < 520 ? 48 : 56, r: 18, t: 22, b: 54}};
    var vs = d.nodes.filter(function (n) {{ return n.value !== null; }}).map(function (n) {{ return n.value; }});
    if (d.ref) vs.push(d.ref);
    var e = Charts.extent(vs, 0.35), y = Charts.scale(e, [H - pad.b, pad.t]);
    var x = function (i) {{ return pad.l + 20 + i * (W - pad.l - pad.r - 40) / (d.nodes.length - 1); }};
    var s = S('svg', {{viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'translation step chart'}});
    s.appendChild(S('rect', {{class: 'plot-bg', x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b}}));
    var ax = S('g', {{class: 'axis'}});
    y.ticks(4).forEach(function (t) {{
      ax.appendChild(S('line', {{class: 'gridline', x1: pad.l, x2: W - pad.r, y1: y(t), y2: y(t)}}));
      ax.appendChild(S('text', {{x: pad.l - 6, y: y(t) + 4, 'text-anchor': 'end'}}, Charts.fmt('.1%')(t)));
    }});
    s.appendChild(ax);
    if (d.ref) {{
      s.appendChild(S('line', {{class: 'ref-line', x1: pad.l, x2: W - pad.r, y1: y(d.ref), y2: y(d.ref)}}));
      s.appendChild(S('text', {{class: 'ref-label', x: pad.l + 6, y: y(d.ref) - 5, 'text-anchor': 'start'}}, 'multilingual original ' + pct(d.ref)));
    }}
    var prev = null, path = '';
    d.nodes.forEach(function (n, i) {{
      if (n.value === null) return;
      if (prev) path += 'H' + x(i).toFixed(1) + 'V' + y(n.value).toFixed(1);
      else path = 'M' + x(i).toFixed(1) + ',' + y(n.value).toFixed(1);
      prev = n;
    }});
    s.appendChild(S('path', {{d: path, style: {{fill: 'none', stroke: 'var(--g-teal)', 'stroke-width': 1.6, opacity: 0.6}}}}));
    d.nodes.forEach(function (n, i) {{
      var g = S('g', {{class: 'mark', 'data-series-id': n.id, 'data-group': n.value === null ? 'google' : 'claude', style: {{cursor: 'pointer'}}}});
      var cy = n.value === null ? H - pad.b - 14 : y(n.value);
      g.appendChild(S('rect', {{class: 'hit', x: x(i) - 30, y: pad.t, width: 60, height: H - pad.t - pad.b + 40}}));
      g.appendChild(S('circle', {{class: 'tp-node', cx: x(i), cy: cy, r: 7,
        style: n.value === null ? {{fill: 'var(--card)', stroke: colorOf(n), 'stroke-width': 2, 'stroke-dasharray': '3 2'}} : {{fill: colorOf(n)}}}}));
      g.appendChild(S('text', {{class: 'row-label', x: x(i), y: cy - 12, 'text-anchor': 'middle'}}, n.value === null ? 'not adopted' : pct(n.value)));
      g.appendChild(S('text', {{class: 'row-label', x: x(i), y: H - pad.b + 18, 'text-anchor': 'middle'}}, n.label));
      g.appendChild(S('text', {{class: 'tp-small', x: x(i), y: H - pad.b + 32, 'text-anchor': 'middle'}}, n.value === null ? 'no run' : n.hits + '/' + (2 * d.pool) + ' hits'));
      TP.hookMark(g, store, n.id, tip, function () {{ return '<div class="tt-title">' + Charts.esc(n.label + ' · ' + n.commit) + '</div><div>' + Charts.esc(n.status) + '</div>'; }},
        function () {{ store.toggleSolo(n.id); }});
      s.appendChild(g);
    }});
    el.appendChild(s);
    Charts.applySelection(sec, store.get());
  }});
  store.on(function (st) {{ show(st.solo || st.hover); }});
  Charts.lineChart(sec.querySelector('[data-chart=curves]'), d.series, {{groups: d.groups, yLabel: 'Validation mean R@1', yFmt: '.1%', best: 'max', store: store}});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {{groups: d.groups, best: 'max', fmt: '.2%'}});
}})();
</script>"""
    caption = (f"Each translated point is one seed-2026 run on the same {pool} validation studies; the gaps are a few "
               "retrieval hits, inside single-seed noise. Click a step for its decision; hover to match its curve.")
    notes = (
        "<ul>"
        f"<li>Values are the best epoch of each snapshot curve (<code>digest/site/data/curves/</code>). The multilingual original "
        f"and the quarantined baseline landing on the same hit count is a grid collision, not a reproduction.</li>"
        + f"<li>The frozen baseline itself moves {_pp(base_swing, signed=False)} across seeds (seed section below), so no step here is a measured improvement.</li>"
        "<li>Translation coverage is complete but not clinically validated (only train/validation spot checks); the model alias was not pinned.</li>"
        f"<li>Decisions: <a href=\"{ctx.doc('config-translation-review.md')}\">Google corpus review</a> "
        f"(API facts: <a href=\"{ctx.doc('translation-api-primary-sources.md')}\">primary sources</a>), "
        f"<a href=\"{ctx.doc('claude-translated-baseline.md')}\">9bec1bd baseline</a>, "
        f"<a href=\"{ctx.doc('translation-update-dfd07db.md')}\">dfd07db update</a>, "
        f"<a href=\"{ctx.doc('claude-translated-dfd07db-results.md')}\">dfd07db anchor</a>, "
        f"<a href=\"{ctx.doc('claude-translated-dfd07db-continuation.md')}\">quarantine and continuation</a>.</li>"
        "<li>The 58-study pilot page <code>translated-report-training.html</code> is not this translation work; it lives on the pipeline page.</li>"
        "</ul>")
    tbl = table(["Step", "Commit", "Status", "Best mean R@1", "Hits", "Epoch"],
                [[n["label"], n["commit"], n["status"], num(n["value"], ".2%") if n["value"] is not None else "no run in digest",
                  n["hits"] if n["hits"] is not None else "–", n["epoch"] if n["epoch"] is not None else "–"] for n in nodes],
                num_cols=(3, 4, 5), row_attrs=[{"data-series-id": n["id"], "data-group": "google" if n["value"] is None else "claude"}
                                                for n in nodes])
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Step table"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 3. seeds


def seeds_section(ctx, d):
    sid = "seeds"
    img, tb = d["img"], d["tb"]
    if not (img and tb):
        return section(sid, "Seed-paired arms", notice("Image-block audit or text-block snapshot missing.")), {}
    series = []
    bases = {r["seed"]: r["mean_r@1"] for r in img["runs"] if r["arm"] == "baseline"}
    for r in img["runs"]:
        if r["arm"] != "image_block":
            continue
        series.append({"id": f"img-{r['seed']}", "label": f"Image blk s{r['seed']}", "group": "image",
                       "points": [[0, bases[r["seed"]]], [1, r["mean_r@1"]]]})
    for s in tb["seeds"]:
        series.append({"id": f"tb-{s['seed']}", "label": f"Text blk s{s['seed']}", "group": "texttower", "dash": "5 3",
                       "points": [[0, s["baseline_mean_r@1_pp"] / 100], [1, s["candidate_mean_r@1_pp"] / 100]]})
    bvals = list(bases.values())
    swing = (max(bvals) - min(bvals)) * 100
    cmp_word = "more than" if swing > abs(img["mean_difference_pp"]) else "less than"
    claim = (f"The frozen baseline alone swings {_pp(swing, signed=False)} across three seeds "
             f"({num(min(bvals), '.2%')}–{num(max(bvals), '.2%')}), {cmp_word} the {_pp(img['mean_difference_pp'])} "
             "image-block mean effect")
    groups = {k: LEVERS[k] for k in ("image", "texttower")}
    blob = {"series": series, "groups": groups}
    body = f"""<div data-role="filter"></div>
{chart('slope', 'Validation mean R@1, frozen vs adapted, paired by seed')}
<div data-role="legend"></div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.lineChart(sec.querySelector('[data-chart=slope]'), d.series, {{groups: d.groups, yLabel: 'Validation mean R@1', yFmt: '.1%',
    xLabel: 'Arm', xStep: 1, xFmt: function (v) {{ return v === 0 ? 'Frozen' : v === 1 ? 'Adapted' : ''; }}, best: 'last', store: store}});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {{groups: d.groups, best: 'last', fmt: '.2%', valuePrefix: 'adapted '}});
}})();
</script>"""
    caption = ("Left end = matched frozen baseline of that seed, right end = adapted run (best validation epoch). "
               "Solid = image final block, dashed = text final block; both share the same three frozen baselines.")
    notes = (
        "<ul><li>Image block: every seed goes up, but each per-seed CI spans zero (forest above). Text block: every seed goes down "
        "or stays level; no CI exists and the runs were not independently verified.</li>"
        + "<li>Text-block differences in direction-queries of " + str(2 * _pool(d)) + ": "
        + ", ".join(f"s{s['seed']} {round(s['difference_pp'] * 2 * _pool(d) / 100):+d}" for s in tb["seeds"])
        + "; seed 2026 carries most of the mean.</li>"
        f"<li>Sources: <code>{F_IMAGE}</code> runs[]; <code>digest/site/data/text-block-v1.json</code> "
        "(matched baselines from <code>configs/contrastive/claude-translated-dfd07db-text-block-v1.json</code>).</li>"
        f"<li>Protocols: <a href=\"{ctx.doc('claude-translated-dfd07db-image-block-protocol.md')}\">image block</a>, "
        f"<a href=\"{ctx.doc('claude-translated-dfd07db-text-block-protocol.md')}\">text block</a> (says “not launched”; stale).</li>"
        "<li>AMP logged nonfinite gradients in some epochs of every text-block seed; skipped-step counts are unavailable.</li></ul>")
    tbl = table(["Run", "Frozen R@1", "Adapted R@1", "Δ (pp)"],
                [[s["label"], num(s["points"][0][1], ".2%"), num(s["points"][1][1], ".2%"),
                  num((s["points"][1][1] - s["points"][0][1]) * 100, ".2f", True)] for s in series],
                num_cols=(1, 2, 3), row_attrs=[{"data-series-id": s["id"], "data-group": s["group"]} for s in series])
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Per-seed table"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 4. tokens


def tokens_section(ctx, d):
    sid = "tokens"
    pre = d["pre"]
    if not pre:
        return section(sid, "Tokenizer lengths", notice("Preflight JSON missing.")), {}
    names = {"multilingual-e5-baseline": "multilingual-E5", "e5-base-v2": "e5-base-v2",
             "pubmedbert-base-embeddings": "PubMedBERT"}
    colors = {"multilingual-e5-baseline": "var(--g-anchor)", "e5-base-v2": "var(--g-violet)",
              "pubmedbert-base-embeddings": "var(--g-rose)"}
    arms, groups = [], {}
    for a in pre["arms"]:
        groups[a["name"]] = {"label": names.get(a["name"], a["name"]), "color": colors.get(a["name"], "var(--g-anchor)")}
        arms.append({"id": a["name"], "label": names.get(a["name"], a["name"]), "group": a["name"],
                     "splits": {k: {"n": v["studies"], "over": v["over_512"], "q": v["token_quantiles"],
                                    "lost": v["last_summary_heading_not_fully_retained"],
                                    "withHead": v["reports_with_summary_heading"]} for k, v in a["splits"].items()}})
    val = [(a["label"], a["splits"]["validation"]["over"], a["splits"]["validation"]["n"]) for a in arms]
    claim = ("The tokenizer decides how much of a report survives the 512-token cap: "
             + " / ".join(f"{o}" for _, o, _ in val) + f" of {val[0][2]} validation reports overflow for "
             + " / ".join(n for n, _, _ in val))
    blob = {"arms": arms, "groups": groups}
    body = f"""<div data-role="filter"></div>
<div class="tp-toggle" data-role="split"><button type="button" data-split="validation" aria-pressed="true">Validation</button><button type="button" data-split="train" aria-pressed="false">Train</button></div>
{chart('quant', 'Report length in tokens per tokenizer (range min–max, bar median–p90, tick p95)')}
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec), S = TP.S;
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var el = sec.querySelector('[data-chart=quant]'), tip = Charts.tooltip(el), split = 'validation';
  var rerender = TP.responsive(el, function (W) {{
    var old = el.querySelector(':scope > svg'); if (old) old.remove();
    var narrow = W < 560, labelW = narrow ? 92 : 130, valW = narrow ? 84 : 150, rowH = 46;
    var pad = {{l: labelW + 8, r: valW + 8, t: 24, b: 40}}, H = pad.t + pad.b + d.arms.length * rowH;
    var mx = 0; d.arms.forEach(function (a) {{ mx = Math.max(mx, a.splits[split].q.max); }});
    var x = Charts.scale([0, mx * 1.03], [pad.l, W - pad.r]);
    var s = S('svg', {{viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'token length profile'}});
    s.appendChild(S('rect', {{class: 'plot-bg', x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b}}));
    var ax = S('g', {{class: 'axis'}});
    x.ticks(narrow ? 3 : 6).forEach(function (t) {{
      ax.appendChild(S('line', {{class: 'gridline', x1: x(t), x2: x(t), y1: pad.t, y2: H - pad.b}}));
      ax.appendChild(S('text', {{x: x(t), y: H - pad.b + 15, 'text-anchor': 'middle'}}, String(t)));
    }});
    ax.appendChild(S('text', {{class: 'axis-title', x: (pad.l + W - pad.r) / 2, y: H - 6, 'text-anchor': 'middle'}}, 'Tokens per report (' + split + ')'));
    s.appendChild(ax);
    s.appendChild(S('line', {{class: 'ref-line', x1: x(512), x2: x(512), y1: pad.t - 6, y2: H - pad.b}}));
    s.appendChild(S('text', {{class: 'ref-label', x: x(512), y: pad.t - 9, 'text-anchor': 'middle'}}, '512-token cap'));
    d.arms.forEach(function (a, i) {{
      var sp = a.splits[split], q = sp.q, cy = pad.t + i * rowH + rowH / 2;
      var col = (d.groups[a.group] || {{}}).color;
      var g = S('g', {{class: 'mark', 'data-series-id': a.id, 'data-group': a.group}});
      g.appendChild(S('rect', {{class: 'hit', x: 0, y: cy - rowH / 2, width: W, height: rowH}}));
      g.appendChild(S('text', {{class: 'row-label', x: labelW, y: cy + 4, 'text-anchor': 'end'}}, a.label));
      g.appendChild(S('line', {{class: 'tp-line', x1: x(q.min), x2: x(q.max), y1: cy, y2: cy, style: {{stroke: col, 'stroke-width': 1.5}}}}));
      g.appendChild(S('rect', {{x: x(q.median), y: cy - 8, width: Math.max(1, x(q.p90) - x(q.median)), height: 16, rx: 2,
        style: {{fill: 'color-mix(in srgb, ' + col + ' 45%, var(--card))', stroke: col}}}}));
      g.appendChild(S('line', {{x1: x(q.p95), x2: x(q.p95), y1: cy - 10, y2: cy + 10, style: {{stroke: col, 'stroke-width': 2}}}}));
      g.appendChild(S('circle', {{cx: x(q.median), cy: cy, r: 4, style: {{fill: col}}}}));
      g.appendChild(S('text', {{class: 'row-label num', x: W - 4, y: cy, 'text-anchor': 'end', style: {{'font-family': 'var(--mono)', 'font-size': '11px'}}}},
        sp.over + '/' + sp.n + (narrow ? '' : ' over 512')));
      g.appendChild(S('text', {{class: 'tp-small', x: W - 4, y: cy + 14, 'text-anchor': 'end'}}, sp.lost + ' lost heading'));
      TP.hookMark(g, store, a.id, tip, function () {{
        return '<div class="tt-title">' + Charts.esc(a.label) + ' · ' + split + ' (n=' + sp.n + ')</div>' +
          '<div>min ' + q.min + ' · median ' + q.median + ' · p90 ' + Math.round(q.p90) + ' · p95 ' + Math.round(q.p95) + ' · max ' + q.max + '</div>' +
          '<div>' + sp.over + ' of ' + sp.n + ' over 512 tokens</div><div>' + sp.lost + ' of ' + sp.withHead + ' with a summary heading lose it past the cap</div>';
      }});
      s.appendChild(g);
    }});
    el.appendChild(s);
    Charts.applySelection(sec, store.get());
  }});
  sec.querySelector('[data-role=split]').addEventListener('click', function (e) {{
    var b = e.target.closest('button[data-split]'); if (!b) return;
    split = b.getAttribute('data-split');
    sec.querySelectorAll('[data-role=split] button').forEach(function (x) {{ x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); }});
    rerender();
  }});
}})();
</script>"""
    caption = ("Right column: reports over 512 tokens / n, and reports whose last summary heading falls past the cap. "
               "Token counts include the model prefix; toggle train / validation.")
    rows, attrs = [], []
    for a in arms:
        for k in ("train", "validation"):
            sp = a["splits"][k]
            rows.append([a["label"], k, sp["n"], sp["q"]["median"], round(sp["q"]["p95"]), sp["q"]["max"], sp["over"], sp["lost"]])
            attrs.append({"data-series-id": a["id"], "data-group": a["group"]})
    tbl = table(["Tokenizer", "Split", "n", "Median", "p95", "Max", "Over 512", "Lost heading"], rows,
                num_cols=(2, 3, 4, 5, 6, 7), row_attrs=attrs)
    notes = (
        "<ul><li>Tokenizer-defined subgroups (over-512, lost heading) are <b>not comparable across encoders</b>: the same label "
        "covers different studies and very different n. Always read n with the rate.</li>"
        "<li>Heading visibility is a lexical heuristic, not clinical summary completeness. Heading-loss counts differ "
        "slightly between audits (different cohorts and heuristics).</li>"
        f"<li>Source: <code>{F_PRE}</code>; protocol <a href=\"{ctx.doc('english-medical-text-protocol.md')}\">english-medical-text protocol</a>; "
        f"why these encoders: <a href=\"{ctx.doc('english-medical-text-encoder-research.md')}\">encoder research</a>.</li>"
        f"<li>Impression-first reordering (<code>{F_ORDER}</code>) moved late summaries to the front of the input to beat "
        "the cap; it changed R@1 by " + (_pp(d["cmp"]["paired_differences"][0]["mean_difference_pp"]) if d["cmp"] else "–")
        + " overall (forest above).</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Length table"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 5. subgroups


def subgroups_section(ctx, d):
    sid = "subgroups"
    rev, enc, cmp_ = d["rev"], d["enc"], d["cmp"]
    if not (rev and enc and cmp_):
        return section(sid, "Subgroups", notice("Subgroup JSONs missing.")), {}
    seeds = sorted({g["seed"] for g in rev["groups"]})
    order = []
    for g in rev["groups"]:
        if g["group"] not in order:
            order.append(g["group"])
    by = {(g["seed"], g["group"]): g for g in rev["groups"]}
    img = {"rows": [f"{g} (n={by[(seeds[0], g)]['n']})" for g in order], "cols": [f"s{s}" for s in seeds],
           "ids": [f"img:{g}" for g in order],
           "m": [[by[(s, g)]["difference_pp"] for s in seeds] for g in order]}
    encs = enc["results"]
    eorder = [s["group"] for s in encs[0]["subgroups"]]
    ebys = [{s["group"]: s for s in r["subgroups"]} for r in encs]
    def elabel(g):
        ns = [b[g]["n"] for b in ebys]
        return f"{g} (n={ns[0]})" if len(set(ns)) == 1 else f"{g} (n={'/'.join(str(n) for n in ns)})"
    encm = {"rows": [elabel(g) for g in eorder], "cols": [ENC_LABEL.get(r["candidate"], r["candidate"]) for r in encs],
            "ids": [f"enc:{g}" for g in eorder], "m": [[b[g]["difference_pp"] for b in ebys] for g in eorder]}
    sg = cmp_["subgroups"]
    imp = {"rows": [f"{k.replace('_', '-')} (n={v['studies']})" for k, v in sg.items()], "cols": ["s2026"],
           "ids": [f"imp:{k}" for k in sg], "m": [[v["difference_pp"] for v in sg.values()]]}
    imp["m"] = [[v["difference_pp"]] for v in sg.values()]
    CLAMP = 5
    # sign flips across seeds for the image block
    flips = [g for g in order if any(by[(s, g)]["difference_pp"] > 0 for s in seeds)
             and any(by[(s, g)]["difference_pp"] < 0 for s in seeds)]
    lang_worst = all(b["source-English"]["difference_pp"] < b["translated"]["difference_pp"] for b in ebys
                     if "source-English" in b and "translated" in b)
    claim = (f"Subgroup gains are not stable: {len(flips)} of {len(order)} image-block slices change sign across seeds"
             + (", and both text-encoder swaps lose most on source-English reports" if lang_worst else ""))
    blob = {"img": img, "enc": encm, "imp": imp, "clamp": CLAMP}
    body = f"""<div class="grid-2">
<div>{chart('img', 'Image final block: Δ R@1 (pp) by subgroup and seed')}</div>
<div>{chart('enc', 'Text encoder swap: Δ R@1 (pp), seed 2026 (n = e5 / PubMedBERT)')}
{chart('imp', 'Impression-first vs full text: Δ R@1 (pp), seed 2026')}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  [['img', d.img], ['enc', d.enc], ['imp', d.imp]].forEach(function (p) {{
    Charts.heatmap(sec.querySelector('[data-chart=' + p[0] + ']'), p[1].m, {{rows: p[1].rows, cols: p[1].cols, rowIds: p[1].ids,
      fmt: '+.1f', diverging: true, domain: [-d.clamp, d.clamp], peakPerRow: false, labelWidth: 190, store: store,
      ariaLabel: 'subgroup difference heatmap'}});
  }});
}})();
</script>"""
    pm = next((b for r, b in zip(encs, ebys) if r["candidate"].startswith("pubmedbert")), {})
    pl, po = pm.get("lost-heading", {}), pm.get("over-512", {})
    caption = (f"Green = adapted/candidate better, red = worse; colour saturates at ±{CLAMP} pp (PubMedBERT's lost-heading "
               f"{_pp(pl.get('difference_pp'), 1)} is {pl.get('n')} studies). n in each row label; tokenizer-defined rows have encoder-specific n.")
    notes = (
        "<ul><li>Slices are post-selection, small and overlapping, with no multiplicity or patient-cluster correction. "
        "“Language” is report provenance (source-English vs translated), not a language detector.</li>"
        f"<li>PubMedBERT's over-512 R@1 of {num((po.get('candidate_r1_pct') or 0) / 100, '.0%')} is "
        f"{round((po.get('candidate_r1_pct') or 0) * (po.get('n') or 0) / 100)} of {po.get('n')} studies; its lost-heading row covers {pl.get('n')} studies.</li>"
        "<li>Impression-first subgroup names come from a different audit (late-summary = heading past the cap); do not "
        "merge its rows with the image-block rows.</li>"
        f"<li>Sources: <code>{F_IMAGE_REVIEW}</code> groups[]; <code>{F_ENC}</code> results[].subgroups; "
        f"<code>{F_COMPARE}</code> subgroups. Reviews: "
        f"<a href=\"{ctx.doc('claude-translated-dfd07db-image-block-error-review.md')}\">image-block error review</a>, "
        f"<a href=\"{ctx.doc('english-medical-text-error-review.md')}\">text-encoder error review</a> "
        "(amended: its 2026-09-11 follow-up overturns finding 5).</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 6. worst pairs


def pairs_section(ctx, d):
    sid = "pairs"
    rev, enc = d["rev"], d["enc"]
    if not (rev and enc):
        return section(sid, "Worst pairs", notice("Worst-pair JSONs missing.")), {}
    pool = _pool(d)

    def pack(prefix, pairs):
        out = []
        for i, w in enumerate(pairs):
            f = w.get("features") or {}
            out.append({"id": f"{prefix}-{i}", "dir": w["direction"], "br": w["baseline_rank"], "cr": w["candidate_rank"],
                        "group": f.get("language", "unknown"), "length": f.get("length"), "summary": f.get("summary"),
                        "tokens": f.get("tokens"), "q": w.get("query_report", ""), "t": w.get("top_report", ""),
                        "exact": w.get("exact_report_match")})
        return out

    runs = [{"key": "img", "label": "Image block (s2026)", "cand": "adapted", "pairs": pack("img", rev["worst_pairs"])}]
    for r in enc["results"]:
        runs.append({"key": r["candidate"], "label": f"{ENC_LABEL.get(r['candidate'], r['candidate'])} (s{r['seed']})",
                     "cand": ENC_LABEL.get(r["candidate"], r["candidate"]), "pairs": pack(r["candidate"], r["worst_pairs"])})
    allp = [p for r in runs for p in r["pairs"]]
    already_bad = sum(1 for p in allp if p["br"] > pool / 4)
    rest = [p for p in allp if p["br"] <= pool / 4]
    pushed = sum(1 for p in rest if p["cr"] > p["br"])
    claim = (f"{already_bad} of {len(allp)} worst-ranked pairs already sat outside the frozen baseline's top {pool // 4}; "
             + (f"all {len(rest)} others were pushed down by the candidate" if pushed == len(rest)
                else f"{pushed} of the other {len(rest)} were pushed down by the candidate"))
    groups = {"translated": {"label": "Translated report", "color": "var(--g-teal)"},
              "source-English": {"label": "Source-English report", "color": "var(--g-amber)"}}
    blob = {"runs": runs, "groups": groups, "pool": pool}
    body = f"""<div class="tp-toggle" data-role="run"></div>
<div data-role="filter"></div>
{chart('ranks', f'Rank of the true partner (1 = retrieved, {pool} = last): frozen baseline → candidate')}
<div class="tp-panel" data-role="pair"></div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec), S = TP.S;
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var el = sec.querySelector('[data-chart=ranks]'), tip = Charts.tooltip(el), panel = sec.querySelector('[data-role=pair]');
  var runBar = sec.querySelector('[data-role=run]'), cur = d.runs[0];
  d.runs.forEach(function (r) {{
    var b = document.createElement('button'); b.type = 'button'; b.textContent = r.label; b.setAttribute('data-run', r.key);
    b.setAttribute('aria-pressed', r === cur ? 'true' : 'false'); runBar.appendChild(b);
  }});
  function dirLabel(p) {{ return p.dir === 'text_to_image' ? 'text → image' : 'image → text'; }}
  function showPair(id) {{
    var g = store.get().group;
    var vis = cur.pairs.filter(function (x) {{ return !g || g === 'all' || x.group === g; }});
    var p = cur.pairs.filter(function (x) {{ return x.id === id; }})[0] || vis[0] || cur.pairs[0];
    panel.innerHTML = '';
    var h = document.createElement('h4');
    h.textContent = cur.label + ' · ' + dirLabel(p) + ' · rank ' + p.br + ' → ' + p.cr;
    var m = document.createElement('div'); m.className = 'muted';
    m.textContent = p.group + ', ' + p.length + ', ' + p.summary + ', ' + p.tokens + ' tokens' +
      (p.exact === true ? ' · retrieved report is an exact text match' : p.exact === false ? ' · not an exact text match' : '');
    var grid = document.createElement('div'); grid.className = 'tp-reports';
    [['Query report', p.q], [p.dir === 'text_to_image' ? 'Paired report of the retrieved image' : 'Retrieved report', p.t]].forEach(function (x) {{
      var box = document.createElement('div'); var b = document.createElement('b'); b.textContent = x[0];
      var pre = document.createElement('pre'); pre.textContent = x[1] || '(missing)';
      box.appendChild(b); box.appendChild(pre); grid.appendChild(box);
    }});
    panel.appendChild(h); panel.appendChild(m); panel.appendChild(grid);
  }}
  var rerender = TP.responsive(el, function (W) {{
    var old = el.querySelector(':scope > svg'); if (old) old.remove();
    var narrow = W < 560, H = narrow ? 300 : 340, pad = {{l: narrow ? 44 : 60, r: narrow ? 44 : 60, t: 24, b: 30}};
    var x0 = pad.l + 10, x1 = W - pad.r - 10;
    var y = Charts.scale([1, d.pool], [pad.t, H - pad.b]);
    var s = S('svg', {{viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'rank slope chart'}});
    s.appendChild(S('rect', {{class: 'plot-bg', x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b}}));
    var ax = S('g', {{class: 'axis'}});
    [1, 100, 200, 300, 400, d.pool].forEach(function (t) {{
      ax.appendChild(S('line', {{class: 'gridline', x1: pad.l, x2: W - pad.r, y1: y(t), y2: y(t)}}));
      ax.appendChild(S('text', {{x: pad.l - 6, y: y(t) + 4, 'text-anchor': 'end'}}, String(t)));
    }});
    ax.appendChild(S('text', {{x: x0, y: H - 8, 'text-anchor': 'start'}}, 'frozen baseline'));
    ax.appendChild(S('text', {{x: x1, y: H - 8, 'text-anchor': 'end'}}, cur.cand));
    s.appendChild(ax);
    cur.pairs.forEach(function (p) {{
      var col = (d.groups[p.group] || {{}}).color || 'var(--g-anchor)';
      var g = S('g', {{class: 'mark', 'data-series-id': p.id, 'data-group': p.group, style: {{cursor: 'pointer'}}}});
      g.appendChild(S('line', {{x1: x0, x2: x1, y1: y(p.br), y2: y(p.cr), style: {{stroke: 'transparent', 'stroke-width': 10}}}}));
      g.appendChild(S('line', {{class: 'tp-line', x1: x0, x2: x1, y1: y(p.br), y2: y(p.cr), 'stroke-dasharray': p.dir === 'text_to_image' ? '5 3' : null,
        style: {{stroke: col, 'stroke-width': 1.6}}}}));
      g.appendChild(S('circle', {{class: 'tp-node', cx: x0, cy: y(p.br), r: 3, style: {{fill: col}}}}));
      g.appendChild(S('circle', {{class: 'tp-node', cx: x1, cy: y(p.cr), r: 3, style: {{fill: col}}}}));
      TP.hookMark(g, store, p.id, tip, function () {{
        return '<div class="tt-title">' + Charts.esc(dirLabel(p)) + '</div><div>rank ' + p.br + ' → ' + p.cr + '</div><div>' +
          Charts.esc(p.group + ', ' + p.summary + ', ' + p.tokens + ' tokens') + '</div><div class="muted">click to read both reports</div>';
      }});
      s.appendChild(g);
    }});
    el.appendChild(s);
    Charts.applySelection(sec, store.get());
  }});
  store.on(function (st) {{ showPair(st.solo || st.hover); }});
  runBar.addEventListener('click', function (e) {{
    var b = e.target.closest('button[data-run]'); if (!b) return;
    cur = d.runs.filter(function (r) {{ return r.key === b.getAttribute('data-run'); }})[0];
    runBar.querySelectorAll('button').forEach(function (x) {{ x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); }});
    store.set({{solo: null, hover: null}});
    rerender(); showPair(null);
  }});
}})();
</script>"""
    caption = ("The 20 worst candidate ranks per run (solid = image → text, dashed = text → image). Click a line to read "
               "the query report beside the report that was ranked first; in text → image rows that is the retrieved image's paired report.")
    notes = (
        "<ul><li>Exact-pair retrieval is not clinical accuracy: a wrong “partner” can be a clinically similar report. No clinical "
        "adjudication was done. Image-block worst pairs are seed 2026 only.</li>"
        "<li>The error reviews read the dominant failure as template / register matching over findings, with laterality "
        f"switched in some pairs. Mojibake in {MOJIBAKE} source-English records is upstream data, left unrepaired to keep the "
        "manifest hash.</li>"
        f"<li>This explorer replaces the reading packet <a href=\"{ctx.doc('english-medical-text-error-review-packet.md')}\">"
        "english-medical-text-error-review-packet.md</a> (same 40 encoder pairs, from "
        f"<code>{F_ENC}</code> results[].worst_pairs), plus the 20 image-block pairs from <code>{F_IMAGE_REVIEW}</code>.</li>"
        "<li>Patient identity is unknown (no PatientID); duplicate or template reports span splits.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 7. hubs


def hubs_section(ctx, d):
    sid = "hubs"
    hubs = d["hubs"]
    if not hubs:
        return section(sid, "Retrieval hubs", notice("Hub snapshot <code>text-hubs.json</code> missing.")), {}
    null = hubs["random_null"]
    p05 = null["distinct p05/p50"][0]
    p99 = null["max p50/p95/p99"][2]
    names = {"baseline multilingual-e5": "mE5 (base)", "e5-base-v2": "e5-base-v2", "pubmedbert": "PubMedBERT"}
    rows = []
    for r in hubs["runs"]:
        dr = "t2i" if r["direction"] == "text_to_image" else "i2t"
        rows.append({"id": f"{r['run']}|{dr}", "label": f"{names.get(r['run'], r['run'])} {dr}", "group": r["direction"],
                     "distinct": r["distinct_rank1_targets"], "hub": r["biggest_hub"], "top2": r["top_two"],
                     "queries": r["queries"], "hits": r["r1_hits"]})
    t2i = [r for r in rows if r["group"] == "text_to_image"]
    claim = (f"Every run funnels text → image queries into hub studies: {min(r['distinct'] for r in t2i)}–"
             f"{max(r['distinct'] for r in t2i)} distinct rank-1 targets vs ≥ {p05} expected at random")
    groups = {"text_to_image": {"label": "text → image", "color": "var(--g-violet)"},
              "image_to_text": {"label": "image → text", "color": "var(--g-sky)"}}
    blob = {"rows": rows, "groups": groups, "p05": p05, "p99": p99}
    body = f"""<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('distinct', 'Distinct studies ever ranked first (of 441 queries)')}</div>
<div>{chart('hub', 'Queries captured by the single biggest hub')}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  function extra(r) {{ return '<div>' + r.hits + ' of ' + r.queries + ' queries hit at R@1; top two hubs take ' + r.top2 + '</div>'; }}
  Charts.barChart(sec.querySelector('[data-chart=distinct]'), d.rows, {{groups: d.groups, value: 'distinct', fmt: function (v) {{ return String(v); }},
    ref: d.p05, refLabel: 'random p05 = ' + d.p05, labelWidth: 150, store: store, estLabel: 'Distinct targets', tooltipExtra: extra}});
  Charts.barChart(sec.querySelector('[data-chart=hub]'), d.rows, {{groups: d.groups, value: 'hub', fmt: function (v) {{ return String(v); }},
    ref: d.p99, refLabel: 'random p99 = ' + d.p99, labelWidth: 150, store: store, estLabel: 'Biggest hub', tooltipExtra: extra}});
}})();
</script>"""
    caption = (f"Monte-Carlo random null: fewer than {p05} distinct targets and a hub above {p99} queries are both below "
               "the 5th / above the 99th percentile. Validation rankings of the selected checkpoints, seed 2026.")
    notes = (
        "<ul><li>Hubs appear in every run, including the multilingual-E5 baseline, but the hub studies differ per encoder. "
        "This is the follow-up that overturned finding 5 of the text-encoder error review (hubs are not PubMedBERT-specific).</li>"
        "<li>Consistency check: R@1 hits averaged over both directions divided by 441 reproduce each run's mean R@1.</li>"
        "<li>Source: <code>digest/site/data/text-hubs.json</code>, exported by <code>scripts/analyze_text_screen_followups.py</code> "
        f"from saved rankings. Discussion: <a href=\"{ctx.doc('english-medical-text-error-review.md')}\">text-encoder error review</a>.</li></ul>")
    tbl = table(["Run / direction", "Queries", "R@1 hits", "Distinct rank-1 targets", "Biggest hub", "Top two"],
                [[r["label"], r["queries"], r["hits"], r["distinct"], r["hub"], r["top2"]] for r in rows],
                num_cols=(1, 2, 3, 4, 5), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Hub table"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 8. gold probe (AUROC)


def gold_section(ctx, d):
    sid = "gold"
    j1, j3 = data.read_json(F_GOLD_1), data.read_json(F_GOLD_3)
    if not (j1 and j3):
        return section(sid, "Gold-label probe", notice("Original-vs-translated probe JSONs missing.")), {}
    rows = []
    for p in data.paired_comparisons(j3):
        if "translated-english" not in p.candidate or "original-multilingual" not in p.baseline:
            continue
        _, run, feat = data.split_key(p.candidate)
        seed = run.rsplit("seed", 1)[-1]
        rows.append({"id": f"cc-uniform-{seed}", "label": f"Uniform mean s{seed}", "group": "uniform", "est": p.diff,
                     "lo": p.ci[0], "hi": p.ci[1], "cand": p.macro_candidate, "base": p.macro_baseline})
    for p in data.paired_comparisons(j1):
        _, run, feat = data.split_key(p.candidate)
        if feat != "plane_attention_768":
            continue
        seed = run.rsplit("seed", 1)[-1]
        rows.append({"id": f"cc-plane-{seed}", "label": f"Plane attention s{seed}", "group": "plane", "est": p.diff,
                     "lo": p.ci[0], "hi": p.ci[1], "cand": p.macro_candidate, "base": p.macro_baseline})
    if not rows:
        return section(sid, "Gold-label probe", notice("No original-vs-translated comparisons found.")), {}
    n = (j3.get("population") or {}).get("population_size")
    boot = (j3.get("config") or {}).get("bootstrap") or {}
    uni = [r for r in rows if r["group"] == "uniform"]
    spans = all(r["lo"] <= 0 <= r["hi"] for r in rows)
    claim = ("Translating reports does not change what the image tower learns for gold labels: "
             + " / ".join(num(r["est"], ".4f", True) for r in uni) + " macro AUROC across seeds"
             + (", every interval spanning zero" if spans else ""))
    groups = {"uniform": {"label": "Uniform-mean feature (3 seeds)", "color": "var(--g-bothft)"},
              "plane": {"label": "Learned plane pooling (1 seed)", "color": "var(--g-imgft)"}}
    blob = {"rows": rows, "groups": groups}
    body = f"""<div data-role="filter"></div>
{chart('gold', f'Gold macro AUROC: translated-English minus original multilingual reports ({n} studies)')}
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.ciDotPlot(sec.querySelector('[data-chart=gold]'), d.rows, {{groups: d.groups, sort: false, ref: 0, refLabel: 'no difference',
    fmt: '+.4f', axisFmt: '+.3f', labelWidth: 170, xLabel: 'Δ gold macro AUROC (translated − original)', store: store,
    tooltipExtra: function (r) {{ return '<div>translated ' + Charts.fmt('.4f')(r.cand) + ' · original ' + Charts.fmt('.4f')(r.base) + '</div>'; }}}});
}})();
</script>"""
    caption = (f"Seed-matched image-block runs that differ only in report text (image-block-cohort-compare). "
               f"Pooled out-of-fold AUROC over {n} exposed gold studies, {boot.get('resamples', '?')} study bootstrap resamples, 95% CI.")
    tbl = table(["Contrast", "Translated", "Original", "Δ AUROC", "95% CI"],
                [[r["label"], num(r["cand"]), num(r["base"]), num(r["est"], ".4f", True),
                  f"[{num(r['lo'], '.4f', True)}, {num(r['hi'], '.4f', True)}]"] for r in rows],
                num_cols=(1, 2, 3, 4), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul><li>This is a <b>different metric on a different population</b> from every chart above: label AUROC on the "
        f"{n}-study gold probe ({(j3.get('population') or {}).get('gold_studies')} gold studies minus "
        f"{((j3.get('population') or {}).get('split_counts') or {}).get('test')} locked test studies), not retrieval R@1. It never shares a table with R@1.</li>"
        "<li>Plane pooling null: the learned plane-attention row reads no better than the uniform mean; it is one seed.</li>"
        "<li>These runs (<code>image-block-cohort-compare-v1</code>) are a separate experiment from the forest's image-block arms; "
        "their encoders trained on the probed studies, so this is a development statistic.</li>"
        "<li>Resampling studies is not resampling patients (study ≠ patient; anonymisation is per study).</li>"
        "<li>Text-block runs cannot change the image tower: on the gold probe their image features are identical to unadapted "
        "DINOv2 (a structural zero, an identity check), so they are not plotted.</li>"
        f"<li>Sources: <a href=\"{ctx.raw(F_GOLD_3)}\"><code>{F_GOLD_3}</code></a> (3 seeds, uniform mean), "
        f"<a href=\"{ctx.raw(F_GOLD_1)}\"><code>{F_GOLD_1}</code></a> (seed 2026 plane attention); write-up "
        f"<a href=\"{ctx.doc('downstream-original-vs-translated-v1.md')}\">original vs translated</a> (amended by its own "
        f"3-seed update) and <a href=\"{ctx.doc('image-block-variants-v1.md')}\">image-block variants</a>. More probes on the "
        f"<a href=\"{ctx.page('probes')}\">probes page</a>.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Probe table"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- page


def _kpis(d):
    tiles = []
    cmp_, img, tb = d["cmp"], d["img"], d["tb"]
    pool = _pool(d)
    if cmp_:
        full = next(r for r in cmp_["runs"] if r["view"] == "full")
        tiles.append({"label": "Text baseline R@1", "value": num(full["mean_r@1"], ".2%"),
                      "sub": f"quarantined full text, seed {full['seed']}, {round(full['mean_r@1'] * 2 * pool)}/{2 * pool} direction-queries"})
    tiles.append({"label": "R@1 grid", "value": num(100 / pool, ".2f") + " pp", "sub": f"one hit of {pool} in one direction; mean R@1 steps {num(50 / pool, '.2f')} pp"})
    rows, _ = forest_rows(d)
    with_ci = [r for r in rows if r["lo"] is not None]
    clear = [r for r in with_ci if r["lo"] > 0 or r["hi"] < 0]
    tiles.append({"label": "CIs excluding zero", "value": f"{len(clear)} of {len(with_ci)}", "sub": "descriptive paired bootstrap"})
    if img:
        tiles.append({"label": "Image final block", "value": _pp(img["mean_difference_pp"]),
                      "sub": f"3-seed mean, SD {num(img['sample_sd_pp'], '.2f')}; every CI spans 0", "tone": None})
    if tb:
        tiles.append({"label": "Text final block", "value": _pp(tb["mean_difference_pp"]),
                      "sub": "3-seed mean, no CI, not independently verified"})
    return tiles


def build(ctx) -> Page:
    d = _load()
    sections, blobs = [], {}
    for fn in (forest_section, timeline_section, seeds_section, tokens_section, subgroups_section, pairs_section,
               hubs_section, gold_section):
        s, b = fn(ctx, d)
        sections.append(s)
        blobs.update(b)
    if HELPERS not in sections[0]:  # forest degraded: keep helpers available for later sections
        sections.insert(0, HELPERS)
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=sections, data=blobs, kpis=_kpis(d))


def summary() -> dict:
    cmp_ = data.read_json(F_COMPARE)
    img = data.read_json(F_IMAGE)
    value = None
    pool = (data.read_json(F_IMAGE_REVIEW) or {}).get("candidate_pool") or 441
    if cmp_:
        full = next(r for r in cmp_["runs"] if r["view"] == "full")
        value = num(full["mean_r@1"], ".2%")
    caveat = None
    if img:
        caveat = (f"No text-side lever beat it; image final block {_pp(img['mean_difference_pp'])} "
                  "(3 seeds), every CI spans zero")
    return {"stage": BY_SLUG[SLUG].number, "headline": "Claude-translated, quarantined full-text baseline",
            "value": value, "value_label": f"validation mean R@1 ({pool} candidates)", "caveat": caveat,
            "href": None}
