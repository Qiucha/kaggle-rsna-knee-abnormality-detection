"""Overfitting, regularisation and closed gates (stage 5).

Story: memorisation measured at matched N -> drop-path 0.05 -> six-rate sweep (amended) ->
convergence audit (absorbs legacy convergence-curves.html) -> 40-epoch budget arm (ambiguous) ->
crop150 re-baseline and probe (not adopted) -> closed gates (val loss vs R@1, duplicates / false
negatives, hard negatives) -> takeaway.

Reads only digest/*.json and digest/site/data/curves/*. Every number shown is computed from
those files at build time, except the pooled crop150 AUROC difference (data.POOLED).

Caveats that shape the charts:
- train_loss is in-batch-32 with augmentation; validation_loss is the full 441x441 matrix.
  They are never compared, and train_loss is not comparable across drop-path arms: shape only.
- Retrieval R@1 (441 candidates) and gold AUROC are drawn in separate charts and tables.
- crop150 uses different pixels and its own unadapted floor (0.6529 vs 0.6438); floors are
  labelled, never differenced.
- The 20 -> 40 epoch level comparison is unpaired (direction only); the estimand is D40.
"""

from __future__ import annotations

import re
from statistics import mean

from .. import data
from ..shell import Page, caveat_badge, chart, esc, notice, num, section, status_badge, table
from ..sitemap import BY_SLUG

SLUG = "overfitting"
SUBTITLE = ("The model memorises its training pairs; every regulariser and input change tried here cut "
            "memorisation and cost retrieval, and three loss-side levers were closed by gates.")

SEEDS = ["2026", "2027", "2028"]
RATES = ["0", "0.005", "0.01", "0.02", "0.05", "0.1"]
MISSING = "Source JSON or curve snapshot missing; run <code>build_digest_site.py export</code>."

# ---------------------------------------------------------------------------- shared JS helper
# A local dumbbell (two-point row) chart. Used by several sections; defined once, first section.
HELPER_JS = r"""
<script>
window.OF = window.OF || (function () {
  'use strict';
  var C = window.Charts;
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  /* rows: [{id, label, group?, a, b, color?}]; opts: {groups, store, fmt, axisFmt, aName, bName, xLabel,
     domain, refs:[{x,label}], valueOf(r)->text, tooltipExtra(r)->html, labelWidth} */
  function dumbbell(el, rows, opts) {
    opts = opts || {};
    var store = opts.store;
    el.classList.add('chart');
    var f = C.fmt(opts.fmt || '.3f'), af = C.fmt(opts.axisFmt || opts.fmt || '.2f');
    var tip = C.tooltip(el);
    rows.forEach(function (r) {
      var g = opts.groups && r.group && opts.groups[r.group];
      r._color = r.color || (g && g.color) || 'var(--g-anchor)';
    });
    var svgEl = null, lastGroup = null, lastW = -1;
    function render(W) {
      if (svgEl) svgEl.remove();
      var st = store.get(); lastGroup = st.group;
      var list = rows.filter(function (r) { return !(st.group && st.group !== 'all' && r.group && r.group !== st.group); });
      var narrow = W < 560, rowH = 24;
      var labelW = Math.min(opts.labelWidth || 210, Math.round(W * (narrow ? 0.34 : 0.28)));
      var valW = narrow ? 52 : 70;
      var aN = opts.aName || 'a', bN = opts.bName || 'b';
      var k0 = narrow ? 6 : labelW + 14, kOff = k0 + 24 + aN.length * 6.2;
      var twoRows = kOff + 12 + bN.length * 6.2 > W - 4;
      var pad = { l: labelW + 8, r: valW + 8, t: twoRows ? 46 : 30, b: 40 };
      var H = pad.t + pad.b + Math.max(1, list.length) * rowH;
      var vals = [];
      rows.forEach(function (r) { vals.push(r.a, r.b); });
      (opts.refs || []).forEach(function (r) { vals.push(r.x); });
      var dom = opts.domain || C.extent(vals, 0.06);
      var x = C.scale(dom, [pad.l, W - pad.r]);
      svgEl = C.svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': opts.ariaLabel || 'dumbbell chart' });
      svgEl.appendChild(C.svg('rect', { class: 'plot-bg', x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b }));
      var ax = C.svg('g', { class: 'axis' });
      x.ticks(Math.max(2, Math.min(6, Math.floor((W - pad.l - pad.r) / 70)))).forEach(function (t) {
        ax.appendChild(C.svg('line', { class: 'gridline', x1: x(t), x2: x(t), y1: pad.t, y2: H - pad.b }));
        ax.appendChild(C.svg('text', { x: x(t), y: H - pad.b + 15, 'text-anchor': 'middle' }, af(t)));
      });
      if (opts.xLabel) ax.appendChild(C.svg('text', { class: 'axis-title', x: (pad.l + W - pad.r) / 2, y: H - 6, 'text-anchor': 'middle' }, opts.xLabel));
      svgEl.appendChild(ax);
      // key: hollow = a, filled = b
      var key = C.svg('g', { class: 'axis' });
      key.appendChild(C.svg('circle', { cx: k0, cy: 12, r: 4.5, style: { fill: 'var(--card)', stroke: 'var(--axis-strong)', 'stroke-width': 1.8 } }));
      key.appendChild(C.svg('text', { x: k0 + 9, y: 16 }, aN));
      var bx = twoRows ? k0 : kOff, by = twoRows ? 28 : 12;
      key.appendChild(C.svg('circle', { cx: bx, cy: by, r: 4.5, style: { fill: 'var(--axis-strong)' } }));
      key.appendChild(C.svg('text', { x: bx + 9, y: by + 4 }, bN));
      svgEl.appendChild(key);
      (opts.refs || []).forEach(function (r) {
        svgEl.appendChild(C.svg('line', { class: 'ref-line', x1: x(r.x), x2: x(r.x), y1: pad.t, y2: H - pad.b }));
        if (r.label) svgEl.appendChild(C.svg('text', { class: 'ref-label', x: x(r.x) + 4, y: pad.t + 11, 'text-anchor': x(r.x) > W * 0.6 ? 'end' : 'start' }, r.label));
      });
      var maxChars = Math.floor(labelW / 6.4);
      list.forEach(function (r, i) {
        var cy = pad.t + i * rowH + rowH / 2;
        var g = C.svg('g', { class: 'mark', 'data-series-id': r.id, 'data-group': r.group || null });
        g.appendChild(C.svg('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width: W, height: rowH }));
        var lab = String(r.label);
        g.appendChild(C.svg('text', { class: 'row-label', x: labelW, y: cy + 4, 'text-anchor': 'end' }, lab.length > maxChars ? lab.slice(0, maxChars - 1) + '…' : lab));
        if (isNum(r.a) && isNum(r.b)) g.appendChild(C.svg('line', { x1: x(r.a), x2: x(r.b), y1: cy, y2: cy, style: { stroke: r._color, 'stroke-width': 3, 'stroke-linecap': 'round', opacity: 0.55 } }));
        if (isNum(r.a)) g.appendChild(C.svg('circle', { cx: x(r.a), cy: cy, r: 5, style: { fill: 'var(--card)', stroke: r._color, 'stroke-width': 2 } }));
        if (isNum(r.b)) g.appendChild(C.svg('circle', { class: 'dot-est', cx: x(r.b), cy: cy, r: 5, style: { fill: r._color } }));
        var vtxt = opts.valueOf ? opts.valueOf(r) : f(r.b - r.a);
        g.appendChild(C.svg('text', { class: 'row-label num', x: W - 4, y: cy + 4, 'text-anchor': 'end', style: { 'font-family': 'var(--mono)', 'font-size': '11px' } }, vtxt));
        g.addEventListener('pointerenter', function () { if (!store.get().solo) store.set({ hover: r.id }); });
        g.addEventListener('pointermove', function (evt) {
          var html = '<div class="tt-title">' + C.esc(r.label) + '</div>' +
            '<div>' + C.esc(opts.aName || 'a') + ': <b>' + C.esc(f(r.a)) + '</b></div>' +
            '<div>' + C.esc(opts.bName || 'b') + ': <b>' + C.esc(f(r.b)) + '</b></div>';
          if (opts.tooltipExtra) html += opts.tooltipExtra(r);
          tip.show(evt, html);
        });
        g.addEventListener('pointerleave', function () { tip.hide(); store.set({ hover: null }); });
        g.addEventListener('click', function () { store.toggleSolo(r.id); });
        svgEl.appendChild(g);
      });
      el.appendChild(svgEl);
      C.applySelection(store.section, store.get());
    }
    function run(force) {
      var w = Math.round(el.clientWidth || 0) || 840;
      if (!force && w === lastW) return;
      lastW = w; render(w);
    }
    run(true);
    if (window.ResizeObserver) new ResizeObserver(function () { run(false); }).observe(el);
    store.on(function (st) { if (st.group !== lastGroup) run(true); });
  }
  /* Metric switcher: buttons [data-metric] inside `bar`; calls draw(key) on a fresh mount each time. */
  function switcher(bar, host, keys, labels, draw) {
    bar.classList.add('filter-bar');
    keys.forEach(function (k, i) {
      var b = C.h('button', { type: 'button', class: 'pill-btn', 'data-metric': k, text: labels[i] });
      bar.appendChild(b);
    });
    function pick(k) {
      Array.prototype.forEach.call(bar.querySelectorAll('[data-metric]'), function (b) {
        var on = b.getAttribute('data-metric') === k;
        b.classList.toggle('active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      while (host.firstChild) host.removeChild(host.firstChild);
      var mount = C.h('div', { class: 'chart' });
      host.appendChild(mount);
      draw(k, mount);
    }
    bar.addEventListener('click', function (e) {
      var b = e.target.closest('[data-metric]');
      if (b) pick(b.getAttribute('data-metric'));
    });
    pick(keys[0]);
  }
  return { dumbbell: dumbbell, switcher: switcher };
})();
</script>
"""

SWITCH_CSS = ("<style>.of-switch{margin:0 0 6px}.of-switch .pill-btn{font-size:12px}"
              ".of-status{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:-4px 0 10px;font-size:12.5px;color:var(--muted)}"
              "</style>")


def status_line(*badges: str, text: str = "") -> str:
    return f'<p class="of-status">{"".join(badges)}{f"<span>{text}</span>" if text else ""}</p>'


def pct(v: float) -> str:
    return f"{v:+.0%}".replace("-", "−")


def ci_txt(ci, spec=".4f") -> str:
    return f"[{num(ci[0], spec, True)}, {num(ci[1], spec, True)}]"


def seed_of(key: str) -> str:
    m = re.search(r"seed(\d{4})", key)
    return m.group(1) if m else key


# ---------------------------------------------------------------------------- curves


def _seed_mean_curve(runs: list[dict], metric: str) -> list[list[float]]:
    """Mean over seeds at every epoch that all runs share (None where any seed lacks a value)."""
    if not runs:
        return []
    per = []
    for r in runs:
        vals = r["metrics"].get(metric)
        if vals is None:
            return []
        per.append(dict(zip(r["epoch"], vals)))
    epochs = sorted(set.intersection(*[set(p) for p in per]))
    out = []
    for e in epochs:
        v = [p[e] for p in per]
        out.append([e, mean(v) if all(isinstance(x, (int, float)) for x in v) else None])
    return out


def _arm_runs():
    """Arm id -> (label, group, shade, [run dicts]) for the 20-epoch convergence view."""
    depth = data.load_curves("image-block-depth-v1")
    sweep = data.load_curves("drop-path-sweep-v1")
    dp05 = data.load_curves("drop-path-v1")
    crop = data.load_curves("crop150-v1")
    if not (depth and sweep and dp05 and crop):
        return None

    def pick(c, pat):
        return [c["runs"][k] for k in sorted(c["runs"]) if re.fullmatch(pat, k)]

    arms = [
        ("base", "Baseline (rate 0, crop 130)", "baseline", None, pick(depth, r"original-unfreeze4-seed\d+")),
        ("dp-0005", "Drop-path 0.005", "droppath", 0, pick(sweep, r"droppath0005-unfreeze4-seed\d+")),
        ("dp-001", "Drop-path 0.01", "droppath", 1, pick(sweep, r"droppath001-unfreeze4-seed\d+")),
        ("dp-002", "Drop-path 0.02", "droppath", 2, pick(sweep, r"droppath002-unfreeze4-seed\d+")),
        ("dp-005", "Drop-path 0.05", "droppath", 3, pick(dp05, r"droppath005-unfreeze4-seed\d+")),
        ("dp-010", "Drop-path 0.1", "droppath", 4, pick(sweep, r"droppath010-unfreeze4-seed\d+")),
        ("crop150", "Crop 150 mm (rate 0)", "crop", None, pick(crop, r"crop150-unfreeze4-seed\d+")),
    ]
    return arms


# audit arm key (convergence-audit.json per_arm) -> our arm id
AUDIT_ARM = {
    "baseline (crop130, 4 blocks)": "base", "drop-path 0.005": "dp-0005", "drop-path 0.01": "dp-001",
    "drop-path 0.02": "dp-002", "drop-path 0.05": "dp-005", "drop-path 0.1": "dp-010", "crop150mm": "crop150",
}
CONV_GROUPS = {
    "baseline": {"label": "Baseline", "color": "var(--g-anchor)"},
    "droppath": {"label": "Drop-path rates", "color": "var(--g-teal)"},
    "crop": {"label": "Crop 150 mm", "color": "var(--g-violet)"},
}


# ---------------------------------------------------------------------------- 1. memorisation


def memorisation_section(ctx):
    sid = "memorisation"
    m = data.read_json("matched-n-loss-audit.json")
    if not m:
        return section(sid, "Memorisation measured at matched candidate count", notice(MISSING)), {}, {}
    rows, curves = [], []
    for x in m:
        s = seed_of(x["checkpoint"])
        mn = x["matched_n"]
        tr = mn["train_random_subsets"]
        rows.append({"id": f"seed{s}", "label": f"Seed {s} (epoch {x['epoch']})", "a": mn["validation_mean_r@1"],
                     "b": tr["mean_r@1_mean"], "trl": tr["loss_mean"], "vl": mn["validation_loss"],
                     "tau": x["inverse_temperature"], "n": mn["n"],
                     "tau_val": x["temperature_sweep"]["validation_loss_minimising_inverse_temperature"],
                     "tau_tr": x["temperature_sweep"]["train_matched_n_loss_minimising_inverse_temperature"],
                     "r1_flat": len(set(x["temperature_invariance_of_r@1"].values())) == 1})
        grid = x["temperature_sweep"]["grid"]
        taus = sorted(grid, key=float)
        curves.append({"id": f"val-{s}", "label": f"Validation, seed {s}", "group": "val",
                       "points": [[float(t), grid[t]["validation"]] for t in taus]})
        curves.append({"id": f"train-{s}", "label": f"Train (N=441 subsets), seed {s}", "group": "train",
                       "points": [[float(t), grid[t]["train_matched_n"]] for t in taus]})
    tr_r1 = mean(r["b"] for r in rows)
    va_r1 = mean(r["a"] for r in rows)
    ratio = tr_r1 / va_r1
    tr_l = mean(r["trl"] for r in rows)
    va_l = mean(r["vl"] for r in rows)
    n = rows[0]["n"]
    rows.append({"id": "seed-mean", "label": "Mean of 3 seeds", "a": va_r1, "b": tr_r1, "color": "var(--text-strong)"})
    flat = all(r.get("r1_flat", True) for r in rows)
    groups = {"val": {"label": "Validation loss", "color": "var(--g-holdout)"},
              "train": {"label": "Train loss, matched N", "color": "var(--g-sky)"}}
    claim = (f"The 4-block model retrieves its own training pairs {ratio:.1f}× better than unseen ones at the same "
             f"N={n} candidates")
    body = f"""{HELPER_JS}{SWITCH_CSS}
<div class="grid-2">
<div>{chart('r1', f'Recall@1 at N={n}: validation vs training subsets')}</div>
<div>{chart('tau', 'Matched-N loss vs inverse temperature (τ⁻¹)')}<div data-role="legend"></div></div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  OF.dumbbell(sec.querySelector('[data-chart=r1]'), d.rows, {{store: store, fmt: '.3f', axisFmt: '.2f',
    aName: 'validation R@1', bName: 'train R@1 (N=441 subsets)', xLabel: 'Mean Recall@1 over 441 candidates',
    valueOf: function (r) {{ return (r.b / r.a).toFixed(1) + '\\u00d7'; }},
    tooltipExtra: function (r) {{ return r.trl ? '<div>Loss: train ' + r.trl.toFixed(2) + ' vs validation ' + r.vl.toFixed(2) + '</div>' : ''; }}}});
  Charts.lineChart(sec.querySelector('[data-chart=tau]'), d.curves, {{groups: d.groups, store: store,
    xLabel: 'Inverse temperature τ⁻¹', yLabel: 'InfoNCE loss, N=441', yFmt: '.2f', best: 'min'}});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.curves, store, {{groups: d.groups, best: 'min', fmt: '.2f'}});
}})();
</script>"""
    caption = (f"Three 4-block seeds, best checkpoints; train R@1 is the mean over 200 random {n}-study training "
               f"subsets. Matched-N loss: train {num(tr_l, '.2f')} vs validation {num(va_l, '.2f')} (seed mean). "
               f"Source: <a href=\"{ctx.doc('matched-n-loss-audit.md')}\">matched-n-loss-audit</a>.")
    tbl = table(["Seed", "Val R@1", "Train R@1 (N=441)", "Ratio", "Val loss", "Train loss (N=441)",
                 "Learned τ⁻¹", "Loss-minimising τ⁻¹ (val / train)"],
                [[r["label"], num(r["a"]), num(r["b"]), f"{r['b'] / r['a']:.2f}×", num(r["vl"], ".3f"),
                  num(r["trl"], ".3f"), num(r["tau"], ".2f"), f"{num(r['tau_val'], '.1f')} / {num(r['tau_tr'], '.1f')}"]
                 for r in rows[:-1]],
                num_cols=range(1, 8), row_attrs=[{"data-series-id": r["id"]} for r in rows[:-1]])
    notes = (
        "<ul>"
        "<li>This replaces the withdrawn level inference of the val-loss audit (see the gate section below): the "
        "gap is measured directly, both splits encoded with augmentation off and scored at the same N.</li>"
        "<li>Losses are compared only at a matched τ. The logged train_loss (in-batch 32, augmented) and "
        "validation_loss (full 441×441) are two different tasks and are never set side by side on this page.</li>"
        f"<li>Recall@1 is τ-invariant{' (identical at all 8 τ values tested in every seed)' if flat else ''}: moving τ "
        "reshapes the loss curves on the right but cannot change retrieval.</li>"
        "<li>Validation R@1 here is over 441 exposed-cohort candidates; never compared with 433-candidate "
        "gold-holdout R@1.</li>"
        "</ul>"
    )
    kpi = {"ratio": ratio, "tr_r1": tr_r1, "va_r1": va_r1, "tr_l": tr_l, "va_l": va_l, "n": n}
    return (section(sid, claim, body, caption, notes, tbl, "Per-seed table"),
            {f"data-{sid}": {"rows": rows, "curves": curves, "groups": groups}}, kpi)


# ---------------------------------------------------------------------------- 2. drop-path 0.05


def droppath05_section(ctx):
    sid = "drop-path-005"
    j = data.read_json("drop-path-5e-2-arm.json")
    if not j:
        return section(sid, "Drop-path 0.05", notice(MISSING)), {}, {}
    groups = {"baseline": {"label": "Baseline", "color": "var(--g-anchor)"},
              "droppath": {"label": "Drop-path 0.05", "color": "var(--g-teal)"}}
    rows, forest = [], []
    gaps = {"baseline": [], "droppath": []}
    for s in SEEDS:
        sd = j["seeds"][s]
        for arm, key in (("baseline", "baseline"), ("droppath", "drop_path")):
            mn = sd["matched_n"][key]
            gaps[arm].append(mn["r@1_gap"])
            rows.append({"id": f"{arm}-{s}", "label": f"{groups[arm]['label']}, seed {s}", "group": arm,
                         "a": mn["validation_r@1"], "b": mn["train_r@1_mean"], "gap": mn["r@1_gap"]})
        v = sd["validation_r@1"]
        forest.append({"id": f"diff-{s}", "label": f"Seed {s}", "group": "droppath", "est": v["difference"],
                       "lo": v["paired_query_bootstrap_95_ci"][0], "hi": v["paired_query_bootstrap_95_ci"][1]})
    p = j["pooled"]
    forest.append({"id": "diff-pooled", "label": "Pooled (3 seeds)", "group": "droppath",
                   "est": p["mean_difference"], "lo": p["bootstrap_95_ci"][0], "hi": p["bootstrap_95_ci"][1],
                   "color": "var(--text-strong)"})
    g0, g1 = mean(gaps["baseline"]), mean(gaps["droppath"])
    change = g1 / g0 - 1
    claim = (f"Drop-path 0.05 cut the memorisation gap by {abs(change):.0%}, yet validation R@1 moved "
             f"{num(p['mean_difference'], '.4f', True)}: less memorisation bought no generalisation")
    body = f"""{status_line(status_badge('superseded'), text='Superseded in scope by the six-rate sweep (2026-09-15), confirmed in substance.')}
<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('gap', 'Train vs validation R@1 at N=441 (matched τ)')}</div>
<div>{chart('diff', 'Paired Δ validation R@1, drop-path minus baseline')}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  OF.dumbbell(sec.querySelector('[data-chart=gap]'), d.rows, {{groups: d.groups, store: store, fmt: '.3f', axisFmt: '.2f',
    aName: 'validation R@1', bName: 'train R@1', xLabel: 'Mean Recall@1, N=441',
    valueOf: function (r) {{ return 'gap ' + Charts.fmt('.3f')(r.gap); }}}});
  Charts.ciDotPlot(sec.querySelector('[data-chart=diff]'), d.forest, {{groups: d.groups, store: store, sort: false,
    ref: 0, refLabel: 'no change', fmt: '+.4f', axisFmt: '+.2f', xLabel: 'Δ validation R@1 (441 candidates)'}});
}})();
</script>"""
    caption = (f"Memorisation gap (train − validation R@1, N=441) {num(g0, '.4f', True)} → {num(g1, '.4f', True)}; "
               f"R@1 difference bootstrapped over validation queries, pooled 95% CI {ci_txt(p['bootstrap_95_ci'])}, "
               f"{'3/3 seeds same sign' if p['same_sign_across_seeds'] else 'seeds disagree in sign'}. "
               f"<a href=\"{ctx.doc('drop-path-5e-2-arm.md')}\">drop-path-5e-2-arm</a>.")
    tbl = table(["Arm, seed", "Val R@1", "Train R@1 (N=441)", "Gap"],
                [[r["label"], num(r["a"]), num(r["b"]), num(r["gap"], ".4f", True)] for r in rows],
                num_cols=(1, 2, 3), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul>"
        f"<li>Pre-registered prediction was {esc(p.get('prediction_was', ''))}; the result is the opposite sign.</li>"
        "<li>Matched-N shares are at the matched τ (mean of the two arms' learned τ⁻¹); the own-τ rows in the JSON agree.</li>"
        "<li>No downstream probe exists for drop-path checkpoints, so there is no AUROC reading for this arm.</li>"
        "<li>Budget: both arms trained 20 epochs. The convergence audit later showed the regularised arms were "
        "still improving at the budget, which is why the sweep carries an amendment.</li>"
        "</ul>"
    )
    return (section(sid, claim, body, caption, notes, tbl, "Per-seed memorisation table"),
            {f"data-{sid}": {"rows": rows, "forest": forest, "groups": groups}},
            {"change": change, "diff": p["mean_difference"], "ci": p["bootstrap_95_ci"]})


# ---------------------------------------------------------------------------- 3. sweep


def sweep_section(ctx):
    sid = "drop-path-sweep"
    j = data.read_json("drop-path-sweep-v1.json")
    b = data.read_json("budget-40ep-v1.json")
    if not j:
        return section(sid, "Drop-path sweep", notice(MISSING)), {}, {}
    groups = {"baseline": {"label": "Rate 0 (baseline)", "color": "var(--g-anchor)"},
              "droppath": {"label": "Drop-path rates, 20 epochs", "color": "var(--g-teal)"},
              "budget": {"label": "Rate 0.1, 40-epoch budget arm", "color": "var(--g-amber)"}}
    dumb, forest, tbl_rows = [], [], []
    rates = [r for r in RATES if r in j["rates"]]
    for i, r in enumerate(rates):
        x = j["rates"][r]
        g = "baseline" if r == "0" else "droppath"
        rid = f"rate-{r}"
        dumb.append({"id": rid, "label": f"Rate {r}", "group": g, "a": x["mean_validation_r@1"],
                     "b": x["mean_matched_n_train_r@1"], "gap": x["mean_matched_n_r@1_gap"]})
        pv = x.get("paired_vs_baseline")
        if pv:
            forest.append({"id": rid, "label": f"Rate {r}, 20 ep", "group": g, "est": pv["mean_difference"],
                           "lo": pv["bootstrap_95_ci"][0], "hi": pv["bootstrap_95_ci"][1],
                           "same": pv["same_sign_across_seeds"], "share": pv["share_of_resamples_favouring_arm"]})
        tbl_rows.append((r, x, pv))
    if b:
        pb = b["pooled"]
        forest.append({"id": "budget-d40", "label": "Rate 0.1, 40 ep", "group": "budget",
                       "est": pb["D40"], "lo": pb["bootstrap_95_ci"][0], "hi": pb["bootstrap_95_ci"][1],
                       "same": pb["same_sign_across_seeds"], "share": pb.get("share_of_resamples_favouring_rate0.1")})
    for k, row in enumerate(r_ for r_ in dumb if r_["group"] == "droppath"):
        row["shade"] = k
    first, last = j["rates"][rates[0]], j["rates"][rates[-1]]
    gap_change = last["mean_matched_n_r@1_gap"] / first["mean_matched_n_r@1_gap"] - 1
    excl = [f for f in forest if f["group"] == "droppath" and (f["hi"] < 0 or f["lo"] > 0)]
    claim = (f"Across six rates the gap falls {abs(gap_change):.0%} while R@1 only falls: no knee, and "
             f"{len(excl)} of {sum(1 for f in forest if f['group'] == 'droppath')} rates exclude zero")
    body = f"""{status_line(status_badge('amended'), caveat_badge('AMENDED 2026-09-15: closed only at a 20-epoch budget'),
                             text='Rate is confounded with budget; the 40-epoch arm reads ambiguous.')}
<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('dose', 'Dose-response: validation vs train R@1 per rate (N=441)')}</div>
<div>{chart('forest', 'Paired Δ validation R@1 vs rate 0, per rate')}</div>
</div>
<p class="legend-hint">Hover a rate in either chart to highlight it in both; pills filter by arm.</p>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  d.dumb.forEach(function (r) {{ if (r.shade !== undefined) r.color = Charts.shade(d.groups[r.group].color, r.shade, d.nRates); }});
  d.forest.forEach(function (r) {{ if (r.group === 'droppath') {{ var m = d.dumb.filter(function (x) {{ return x.id === r.id; }})[0]; if (m) r.color = m.color; }} }});
  OF.dumbbell(sec.querySelector('[data-chart=dose]'), d.dumb, {{groups: d.groups, store: store, fmt: '.3f', axisFmt: '.2f',
    aName: 'validation R@1', bName: 'train R@1', xLabel: 'Seed-mean Recall@1, N=441',
    valueOf: function (r) {{ return 'gap ' + Charts.fmt('.3f')(r.gap); }}}});
  Charts.ciDotPlot(sec.querySelector('[data-chart=forest]'), d.forest, {{groups: d.groups, store: store, sort: false,
    ref: 0, refLabel: 'rate 0', fmt: '+.4f', axisFmt: '+.2f', xLabel: 'Δ validation R@1 (441 candidates)',
    tooltipExtra: function (r) {{ return '<div>' + (r.same ? '3/3 seeds same sign' : 'seeds disagree in sign') + '</div>'; }}}});
}})();
</script>"""
    caption = (f"Three seeds per rate, 20 epochs, R@1 over 441 candidates. Gap {num(first['mean_matched_n_r@1_gap'], '.4f', True)} → "
               f"{num(last['mean_matched_n_r@1_gap'], '.4f', True)}; R@1 {num(first['mean_validation_r@1'])} → "
               f"{num(last['mean_validation_r@1'])}. The last forest row is the 40-epoch budget re-run of rate 0.1. "
               f"<a href=\"{ctx.doc('drop-path-sweep-v1.md')}\">drop-path-sweep-v1</a>.")
    tbl = table(["Rate", "Val R@1", "Train R@1 (N=441)", "Gap", "Matched-N loss gap", "τ⁻¹", "Δ R@1 vs rate 0", "95% CI", "Seeds same sign"],
                [[r, num(x["mean_validation_r@1"]), num(x["mean_matched_n_train_r@1"]), num(x["mean_matched_n_r@1_gap"], ".4f", True),
                  num(x["mean_matched_n_loss_gap"], ".3f", True), num(x["mean_inverse_temperature"], ".2f"),
                  num(pv["mean_difference"], ".4f", True) if pv else "–", ci_txt(pv["bootstrap_95_ci"]) if pv else "–",
                  ("yes" if pv["same_sign_across_seeds"] else "no") if pv else "–"] for r, x, pv in tbl_rows],
                num_cols=range(1, 8), row_attrs=[{"data-series-id": f"rate-{r}", "data-group": "baseline" if r == "0" else "droppath"}
                                                 for r, _, _ in tbl_rows])
    notes = (
        "<ul>"
        "<li><b>Amended 2026-09-15.</b> The first write-up closed stochastic depth at 4 blocks. The convergence audit "
        "then found rate 0.1 still improving at epoch 20, so the sweep is closed only at a 20-epoch budget; the "
        "budget arm resolved it in part (ambiguous). The 20-epoch rate-0.1 row overstates the harm by roughly a third.</li>"
        "<li>Matched-N loss gaps are at a matched τ; train_loss levels are not comparable across drop-path arms "
        "(drop-path is active in the arm's train-mode forward pass).</li>"
        "<li>No drop-path checkpoint has a downstream probe; this is a retrieval-only result.</li>"
        "</ul>"
    )
    return (section(sid, claim, body, caption, notes, tbl, "Per-rate table"),
            {f"data-{sid}": {"dumb": dumb, "forest": forest, "groups": groups,
                             "nRates": sum(1 for r in dumb if r["group"] == "droppath")}},
            {"gap_change": gap_change, "d20": j["rates"][rates[-1]]["paired_vs_baseline"]["mean_difference"]})


# ---------------------------------------------------------------------------- 4. convergence


CONV_METRICS = [
    ("validation_loss", "Validation loss", "Validation loss (full 441×441)", ".2f", "min"),
    ("mean_r@1", "Validation R@1", "Validation mean R@1 (441 candidates)", ".1%", "max"),
    ("inverse_temperature", "Inverse temperature", "Learned inverse temperature τ⁻¹", ".1f", "last"),
    ("train_loss", "Train loss (shape only)", "Train loss, in-batch 32 (shape only)", ".2f", "last"),
]


def convergence_section(ctx):
    sid = "convergence"
    arms = _arm_runs()
    audit = data.read_json("convergence-audit.json")
    if not arms or not audit:
        return section(sid, "Convergence within the 20-epoch budget", notice(MISSING)), {}, {}
    series = {}
    for m, *_ in CONV_METRICS:
        lst = []
        for aid, label, g, shade, runs in arms:
            s = {"id": aid, "label": label, "group": g, "points": _seed_mean_curve(runs, m)}
            if shade is not None:
                s["shade"] = shade
                s["shadeOf"] = 5
            if g == "crop":
                s["dash"] = "6 3"
            lst.append(s)
        series[m] = lst
    bars = []
    for key, a in audit["per_arm"].items():
        aid = AUDIT_ARM.get(key)
        if not aid:
            continue
        bars.append({"id": aid, "label": next(x[1] for x in arms if x[0] == aid), "group": a["family"].replace("drop_path", "droppath"),
                     "value": a["mean_val_rise_from_min"], "falling": a["n_val_still_falling"], "n": a["n_runs"],
                     "minep": a["mean_val_loss_min_epoch"], "bestep": a["mean_best_epoch"], "boundary": a["n_best_at_boundary"]})
    order = [x[0] for x in arms]
    bars.sort(key=lambda r: order.index(r["id"]))
    for i, r in enumerate(x for x in bars if x["group"] == "droppath"):
        r["shade"] = i
    pa = audit["per_arm"]
    d10, base, crop = pa["drop-path 0.1"], pa["baseline (crop130, 4 blocks)"], pa["crop150mm"]
    claim = (f"Drop-path 0.1 was still improving at epoch 20 (validation loss falling in {d10['n_val_still_falling']}/"
             f"{d10['n_runs']} seeds); crop150 converged by epoch {crop['mean_val_loss_min_epoch']:.0f} and then overfit")
    metric_keys = [m[0] for m in CONV_METRICS]
    body = f"""{status_line(caveat_badge('train loss: shape only, not comparable to validation loss or across drop-path arms'))}
<div data-role="filter"></div>
<div class="grid-2">
<div><div class="chart-title">Seed-mean training curves, 20 epochs</div><div class="of-switch" data-role="metric"></div><div data-role="host"></div></div>
<div>{chart('rise', 'Validation-loss rise from its minimum to epoch 20')}</div>
</div>
<div data-role="legend"></div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  OF.switcher(sec.querySelector('[data-role=metric]'), sec.querySelector('[data-role=host]'), d.metrics.map(function (m) {{ return m.key; }}),
    d.metrics.map(function (m) {{ return m.short; }}), function (k, mount) {{
      var m = d.metrics.filter(function (x) {{ return x.key === k; }})[0];
      Charts.lineChart(mount, d.series[k], {{groups: d.groups, store: store, yLabel: m.label, yFmt: m.fmt, best: m.best}});
      var lg = sec.querySelector('[data-role=legend]');
      Charts.legend(lg, d.series[k], store, {{groups: d.groups, best: m.best, fmt: m.fmt}});
    }});
  Charts.barChart(sec.querySelector('[data-chart=rise]'), d.bars, {{groups: d.groups, store: store, fmt: '.3f', axisFmt: '.2f',
    xLabel: 'Final minus minimum validation loss (seed mean)',
    tooltipExtra: function (r) {{ return '<div>Val loss still falling: ' + r.falling + '/' + r.n + ' seeds</div><div>Val-loss minimum at epoch ' +
      r.minep.toFixed(1) + '; best R@1 at ' + r.bestep.toFixed(1) + '</div>'; }}}});
}})();
</script>"""
    caption = (f"Seed means of three runs per arm. Rise from minimum: baseline {num(base['mean_val_rise_from_min'], '.3f')}, "
               f"rate 0.1 {num(d10['mean_val_rise_from_min'], '.3f')}, crop150 {num(crop['mean_val_rise_from_min'], '.3f')}. "
               f"Absorbs the <a href=\"{ctx.raw('convergence-curves.html')}\">original convergence report</a>.")
    tbl = table(["Arm", "Best R@1 (mean)", "Best epoch (mean)", "Best at boundary", "Val loss still falling",
                 "Val-loss min epoch", "Rise from min"],
                [[r["label"], num(pa[k]["mean_best_r@1"]), num(pa[k]["mean_best_epoch"], ".1f"),
                  f"{pa[k]['n_best_at_boundary']}/{pa[k]['n_runs']}", f"{pa[k]['n_val_still_falling']}/{pa[k]['n_runs']}",
                  num(pa[k]["mean_val_loss_min_epoch"], ".1f"), num(pa[k]["mean_val_rise_from_min"], ".3f")]
                 for k in pa if AUDIT_ARM.get(k) for r in [next(b_ for b_ in bars if b_["id"] == AUDIT_ARM[k])]],
                num_cols=range(1, 7), row_attrs=[{"data-series-id": AUDIT_ARM[k], "data-group": pa[k]["family"].replace("drop_path", "droppath")}
                                                  for k in pa if AUDIT_ARM.get(k)])
    diag = audit.get("diagnostics", {})
    notes = (
        "<ul>"
        + "".join(f"<li>{esc(c)}</li>" for c in audit.get("caveats", []))
        + f"<li>Diagnostics: best at boundary = {esc(diag.get('best_at_boundary', ''))}; still falling = "
          f"{esc(diag.get('val_still_falling', ''))}.</li>"
        "<li>Curves are seed means (per-epoch mean over the three seeds); per-seed histories are in the snapshots under "
        "<code>digest/site/data/curves/</code>. AUROC has no per-epoch curve (only best/last checkpoints are saved).</li>"
        f"<li>Legacy page: <a href=\"{ctx.raw('convergence-curves.html')}\">original report</a> (static; six charts and the "
        "diagnostic table reproduced here). Audit data: "
        f"<a href=\"{ctx.raw('convergence-audit.json')}\">convergence-audit.json</a>.</li>"
        "</ul>"
    )
    blob = {"groups": CONV_GROUPS, "series": series, "bars": bars,
            "metrics": [{"key": k, "short": s, "label": l, "fmt": f, "best": b} for k, s, l, f, b in CONV_METRICS]}
    return (section(sid, claim, body, caption, notes, tbl, "Diagnostic table (convergence audit)"),
            {f"data-{sid}": blob}, {})


# ---------------------------------------------------------------------------- 5. budget arm


def budget_section(ctx):
    sid = "budget"
    b = data.read_json("budget-40ep-v1.json")
    sweep = data.read_json("drop-path-sweep-v1.json")
    curves = data.load_curves("budget-40ep-v1")
    if not b or not sweep:
        return section(sid, "The 40-epoch budget arm", notice(MISSING)), {}, {}
    groups = {"rate0": {"label": "Rate 0", "color": "var(--g-anchor)"},
              "rate01": {"label": "Rate 0.1", "color": "var(--g-teal)"},
              "est": {"label": "Paired estimate", "color": "var(--g-amber)"}}
    sd = b["secondary_difference_of_differences"]
    pb = b["pooled"]
    forest = [
        {"id": "d20", "label": "D20 (20 epochs)", "group": "est", "est": sd["D20"], "lo": sd["D20_ci"][0], "hi": sd["D20_ci"][1]},
        {"id": "d40", "label": "D40 (40 epochs)", "group": "est", "est": pb["D40"],
         "lo": pb["bootstrap_95_ci"][0], "hi": pb["bootstrap_95_ci"][1], "color": "var(--text-strong)"},
    ]
    series = []
    if curves:
        for g, pat, label in (("rate0", r"budget40-rate0-unfreeze4-seed\d+", "Rate 0, 40 ep"),
                              ("rate01", r"budget40-rate010-unfreeze4-seed\d+", "Rate 0.1, 40 ep")):
            runs = [curves["runs"][k] for k in sorted(curves["runs"]) if re.fullmatch(pat, k)]
            series.append({"id": g, "label": label, "group": g, "points": _seed_mean_curve(runs, "mean_r@1")})
    # unpaired level comparison (direction only)
    lv20 = {"rate0": sweep["rates"]["0"]["mean_validation_r@1"], "rate01": sweep["rates"]["0.1"]["mean_validation_r@1"]}
    lv40 = {"rate0": mean(b["seeds"][s]["rate0.0_r@1"] for s in SEEDS), "rate01": mean(b["seeds"][s]["rate0.1_r@1"] for s in SEEDS)}
    verdict = "AMBIGUOUS"
    claim = (f"At 40 epochs the rate-0.1 deficit shrinks to {num(pb['D40'], '.4f', True)} but its CI grazes zero: "
             f"{verdict} by the pre-registered rule")
    body = f"""{status_line(caveat_badge('AMBIGUOUS: partial or ambiguous by the pre-registered rule, do not round'))}
<div class="grid-2">
<div>{chart('forest', 'Paired Δ validation R@1, rate 0.1 minus rate 0')}</div>
<div>{chart('curve', 'Seed-mean validation R@1, 40-epoch schedule')}<div data-role="legend"></div></div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.ciDotPlot(sec.querySelector('[data-chart=forest]'), d.forest, {{groups: d.groups, store: store, sort: false,
    ref: 0, refLabel: 'no harm', fmt: '+.4f', axisFmt: '+.2f', xLabel: 'Δ validation R@1 (441 candidates)'}});
  if (d.series.length) {{
    Charts.lineChart(sec.querySelector('[data-chart=curve]'), d.series, {{groups: d.groups, store: store,
      yLabel: 'Validation mean R@1', yFmt: '.1%', best: 'max'}});
    Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {{groups: d.groups, best: 'max', fmt: '.1%'}});
  }} else {{ sec.querySelector('[data-chart=curve]').innerHTML = '<div class="notice">40-epoch curve snapshot missing.</div>'; }}
}})();
</script>"""
    caption = (f"Both arms retrained at 40 epochs, 3 seeds, bootstrapped over validation queries: D40 95% CI "
               f"{ci_txt(pb['bootstrap_95_ci'])}. Shift from D20 {num(sd['shift'], '.4f', True)} "
               f"(~{sd['share_of_20ep_deficit_recovered']:.0%} of the deficit; under-powered, not a test). "
               f"<a href=\"{ctx.doc('budget-40ep-v1.md')}\">budget-40ep-v1</a>.")
    rule = pb.get("pre_registered_rule", {})
    tbl = table(["Seed", "Rate 0 R@1", "Rate 0.1 R@1", "Difference", "Selected epoch (0 / 0.1)", "τ⁻¹ (0 / 0.1)"],
                [[s, num(b["seeds"][s]["rate0.0_r@1"]), num(b["seeds"][s]["rate0.1_r@1"]),
                  num(b["seeds"][s]["difference"], ".4f", True),
                  f"{b['seeds'][s]['rate0.0_epoch']} / {b['seeds'][s]['rate0.1_epoch']}",
                  f"{num(b['seeds'][s]['rate0.0_inverse_temperature'], '.1f')} / {num(b['seeds'][s]['rate0.1_inverse_temperature'], '.1f')}"]
                 for s in SEEDS], num_cols=(1, 2, 3))
    notes = (
        "<ul>"
        f"<li>Verdict (verbatim): {esc(pb.get('verdict', ''))}</li>"
        f"<li>Pre-registered rule: harm persists if {esc(rule.get('harm_persists', ''))}; harm did not reproduce if "
        f"{esc(rule.get('harm_did_not_reproduce', ''))}; otherwise {esc(rule.get('otherwise', ''))}.</li>"
        f"<li>Difference of differences: {esc(sd.get('caveat', ''))}</li>"
        f"<li>Unpaired level comparison, direction only: rate 0 {num(lv20['rate0'])} (20 ep) → {num(lv40['rate0'])} "
        f"(40 ep); rate 0.1 {num(lv20['rate01'])} → {num(lv40['rate01'])}. Different runs and schedules, so these "
        "levels are not paired and are not differenced here.</li>"
        "<li>Both arms share the stretched cosine schedule, which cancels in the pair. A fixed epoch budget is not "
        "neutral for a regulariser.</li>"
        "</ul>"
    )
    return (section(sid, claim, body, caption, notes, tbl, "Per-seed 40-epoch table"),
            {f"data-{sid}": {"forest": forest, "series": series, "groups": groups}},
            {"d40": pb["D40"], "ci": pb["bootstrap_95_ci"]})


# ---------------------------------------------------------------------------- 6. crop150


def crop_section(ctx):
    sid = "crop150"
    mem = data.read_json("crop150-memorisation.json")
    comp = data.read_json("crop150-vs-crop130-comparisons-v1.json")
    probe = data.read_json("downstream-probe-crop150-v1.json")
    if not mem or not comp:
        return section(sid, "Crop150", notice(MISSING)), {}, {}
    groups = {"crop130": {"label": "Crop 130 mm (default)", "color": "var(--g-anchor)"},
              "crop150": {"label": "Crop 150 mm", "color": "var(--g-violet)"}}
    dumb = []
    for s in SEEDS:
        for arm in ("crop130", "crop150"):
            x = mem["seeds"][s]["matched_n"][arm]
            dumb.append({"id": f"{arm}-{s}", "label": f"{arm[:4].title()} {arm[4:]} mm, seed {s}", "group": arm,
                         "a": x["validation_r@1"], "b": x["train_r@1_mean"] if "train_r@1_mean" in x else x.get("train_r@1"),
                         "gap": x.get("r@1_gap")})
    forest = []
    for p in data.paired_comparisons(comp):
        s = seed_of(p.candidate)
        forest.append({"id": f"auc-{s}", "label": f"Seed {s}", "group": "crop150", "est": p.diff, "lo": p.ci[0], "hi": p.ci[1],
                       "a130": p.macro_baseline, "a150": p.macro_candidate})
    pooled = data.POOLED["crop150_vs_crop130"]
    forest.append({"id": "auc-pooled", "label": "Pooled (3 seeds, md only)", "group": "crop150", "est": pooled["value"],
                   "lo": None, "hi": None, "color": "var(--text-strong)"})
    floors = {}
    if probe:
        for k, r in (probe.get("results") or {}).items():
            m_ = re.match(r"before_cl/dinov2-unadapted@crop(\d+)mm/uniform_mean_768", k)
            if m_:
                floors[m_.group(1)] = r["macro_auroc"]
    pc = mem["pooled"]
    n_same = sum(1 for f in forest[:-1] if f["est"] < 0)
    n_excl = sum(1 for f in forest[:-1] if f["hi"] < 0)
    claim = (f"Crop150 narrowed the gap but lowered R@1 and gold AUROC ({num(pooled['value'], '.4f', True)} pooled, "
             f"{n_same}/{len(forest) - 1} seeds same sign): not adopted")
    floor_txt = " · ".join(f"unadapted floor @{k} mm {num(v)}" for k, v in sorted(floors.items()))
    body = f"""{status_line(status_badge('not-adopted'), text='The config.py default stays at 130 mm. Different pixels: compare only through paired contrasts.')}
<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('mem', 'Retrieval: train vs validation R@1 at N=441')}</div>
<div>{chart('auc', 'Gold probe: Δ macro AUROC, crop150 minus crop130')}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  OF.dumbbell(sec.querySelector('[data-chart=mem]'), d.dumb, {{groups: d.groups, store: store, fmt: '.3f', axisFmt: '.2f',
    aName: 'validation R@1', bName: 'train R@1', xLabel: 'Mean Recall@1, N=441',
    valueOf: function (r) {{ return 'gap ' + Charts.fmt('.3f')(r.gap); }}}});
  Charts.ciDotPlot(sec.querySelector('[data-chart=auc]'), d.forest, {{groups: d.groups, store: store, sort: false,
    ref: 0, refLabel: 'no change', fmt: '+.4f', axisFmt: '+.2f', xLabel: 'Δ gold macro AUROC (56 exposed studies)',
    tooltipExtra: function (r) {{ return r.a130 ? '<div>crop130 ' + r.a130.toFixed(4) + ' → crop150 ' + r.a150.toFixed(4) + '</div>' :
      '<div>Pooled value from crop150-probe-v1.md (no per-study scores stored)</div>'; }}}});
}})();
</script>"""
    caption = (f"Left, retrieval (441 candidates): pooled gap {num(pc['crop130']['r@1_gap'], '.4f', True)} → "
               f"{num(pc['crop150']['r@1_gap'], '.4f', True)}, val R@1 {num(pc['crop130']['validation_r@1'])} → "
               f"{num(pc['crop150']['validation_r@1'])}. Right, gold AUROC, seed-matched, {n_excl}/{len(forest) - 1} CIs "
               f"exclude 0; {floor_txt} (separate floors, not differenced). "
               f"<a href=\"{ctx.doc('crop150-probe-v1.md')}\">crop150-probe-v1</a>.")
    tbl_r1 = table(["Crop, seed", "Val R@1", "Train R@1 (N=441)", "Gap"],
                   [[r["label"], num(r["a"]), num(r["b"]), num(r["gap"], ".4f", True)] for r in dumb],
                   num_cols=(1, 2, 3), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in dumb])
    tbl_auc = table(["Seed", "crop130 macro AUROC", "crop150 macro AUROC", "Δ", "95% CI"],
                    [[f["label"], num(f.get("a130")), num(f.get("a150")), num(f["est"], ".4f", True),
                      ci_txt((f["lo"], f["hi"])) if f["lo"] is not None else "–"] for f in forest],
                    num_cols=(1, 2, 3, 4), row_attrs=[{"data-series-id": f["id"], "data-group": "crop150"} for f in forest])
    tables = f"<h3>Retrieval (R@1, 441 candidates)</h3>{tbl_r1}<h3>Gold probe (macro AUROC, 56 studies)</h3>{tbl_auc}"
    notes = (
        "<ul>"
        f"<li>Re-baseline (<a href=\"{ctx.doc('crop150-rebaseline-v1.md')}\">crop150-rebaseline-v1</a>): convergence "
        "checked 2026-09-15, the reading stands (crop150 converged, then overfit). Its seed-spread collapse is provisional.</li>"
        f"<li>The pooled {num(pooled['value'], '.4f', True)} is quoted from <code>{esc(pooled['source'])}</code>; per-study "
        "out-of-fold scores are not stored, so it has no CI here.</li>"
        "<li>Gold AUROC levels are development statistics: the 56 studies were in training. The unexposed 4-block "
        f"estimate is on the <a href=\"{ctx.page('probes')}\">probes page</a>. Retrieval and AUROC are separate charts "
        "and tables by rule.</li>"
        "<li>Each crop is read against its own unadapted floor; the two gains-over-floor are not differenced. "
        "Dilution (extra field is muscle and subcutaneous tissue) is the better-supported explanation.</li>"
        "<li>The drop-path vs crop 'exchange rate' is a loose comparison: different interventions, different pixels.</li>"
        "</ul>"
    )
    return (section(sid, claim, body, caption, notes, tables, "Retrieval and probe tables (kept separate)"),
            {f"data-{sid}": {"dumb": dumb, "forest": forest, "groups": groups}}, {"pooled": pooled["value"]})


# ---------------------------------------------------------------------------- 7. gates: val loss vs R@1


VL_FAMILIES = {
    "image-block-cohort-compare-v1": ("cohort", "Cohort compare", "var(--g-sky)"),
    "image-block-depth-v1": ("depth", "Depth", "var(--g-teal)"),
    "image-block-epochs-v1": ("epochs", "40 epochs", "var(--g-amber)"),
    "image-block-control-v1": ("control", "Shuffled-pairing control", "var(--g-rose)"),
    "plane-pooling-v1": ("pooling", "Plane pooling", "var(--g-violet)"),
}


def vl_label(run: str) -> str:
    """Short run label: 'depth4 s2026', 'multi s2027', 'shuffled s2028' ..."""
    seed = seed_of(run)
    for pat, name in ((r"original-multilingual", "multilingual"), (r"translated-english", "translated"),
                      (r"original-unfreeze(\d+)", r"depth\1"), (r"original-40ep", "40 ep"),
                      (r"shuffled-pairing", "shuffled"), (r"uniform-unfreeze4", "uniform pool")):
        m_ = re.match(pat, run)
        if m_:
            return f"{m_.expand(name)} s{seed}"
    return run


def valloss_section(ctx):
    sid = "gate-val-loss"
    j = data.read_json("val-loss-vs-retrieval-audit.json")
    if not j:
        return section(sid, "Validation loss vs retrieval", notice(MISSING)), {}, {}
    groups = {v[0]: {"label": v[1], "color": v[2]} for v in VL_FAMILIES.values()}
    rows, bars = [], []
    for r in j["runs"]:
        root, run = r["run"].split("/", 1)
        fam = VL_FAMILIES.get(root, ("other", root, "var(--g-anchor)"))[0]
        lab = vl_label(run)
        rows.append({"id": r["run"], "label": lab, "group": fam, "a": r["best_val_loss_epoch"], "b": r["best_mean_r@1_epoch"],
                     "last": r["last_epoch"], "dvl": r["d_val_loss"], "dr1": r["d_mean_r@1"], "dit": r["d_inverse_temperature"]})
        bars.append({"id": r["run"], "label": lab, "group": fam, "value": r["d_mean_r@1"]})
    n_after = j["n_r1_peaks_after_val_loss_min"]
    n = j["n_runs_analysed"]
    ctrl = [r for r in rows if r["group"] == "control"]
    claim = (f"R@1 peaks after the validation-loss minimum in {n_after} of {n} runs, so validation loss is not a "
             f"selection signal; this audit's 'no overfitting' conclusion is withdrawn")
    body = f"""{status_line(status_badge('amended'), caveat_badge('AMENDED 2026-09-13: conclusion 1 withdrawn — the model overfits'),
                             text='Conclusions 2 to 5 (the late-window disagreement) stand.')}
<div data-role="filter"></div>
<div class="grid-2">
<div>{chart('epochs', 'Epoch of the val-loss minimum vs epoch of best R@1')}</div>
<div>{chart('dr1', 'Late-window Δ validation R@1 (val-loss minimum → last epoch)')}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  OF.dumbbell(sec.querySelector('[data-chart=epochs]'), d.rows, {{groups: d.groups, store: store, fmt: '.0f', axisFmt: '.0f',
    aName: 'val-loss minimum', bName: 'best R@1', xLabel: 'Epoch', domain: [0, 41],
    valueOf: function (r) {{ return (r.b - r.a > 0 ? '+' : '') + (r.b - r.a) + ' ep'; }},
    tooltipExtra: function (r) {{ return '<div>Late window: Δ val loss ' + Charts.fmt('+.3f')(r.dvl) + ', Δ R@1 ' +
      Charts.fmt('+.4f')(r.dr1) + ', Δ τ⁻¹ ' + Charts.fmt('+.2f')(r.dit) + '</div><div>Last epoch ' + r.last + '</div>'; }}}});
  Charts.barChart(sec.querySelector('[data-chart=dr1]'), d.bars, {{groups: d.groups, store: store, fmt: '+.4f', axisFmt: '+.2f',
    xLabel: 'Δ mean R@1 over the late window'}});
}})();
</script>"""
    fams = len({r["group"] for r in rows})
    longr = sum(1 for r in rows if r["last"] > 20)
    caption = (f"{n} single runs from {fams} families (20 epochs; {longr} run{'s' if longr != 1 else ''} longer). The shuffled-pairing control "
               f"({len(ctrl)} seeds) peaks early and gains nothing late: the true-pairing late gain is real signal. "
               f"<a href=\"{ctx.doc('val-loss-vs-retrieval-audit.md')}\">val-loss-vs-retrieval-audit</a>.")
    tbl = table(["Run", "Val-loss min epoch", "Best R@1 epoch", "Last", "Δ val loss", "Δ R@1", "Δ τ⁻¹"],
                [[r["id"], r["a"], r["b"], r["last"], num(r["dvl"], ".3f", True), num(r["dr1"], ".4f", True),
                  num(r["dit"], ".2f", True)] for r in rows],
                num_cols=range(1, 7), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul>"
        "<li><b>AMENDED 2026-09-13.</b> Conclusion 1 claimed the memorisation premise did not survive. It does: the "
        "matched-N audit (first section) measured the level gap directly. This page measured late-window deltas and "
        "inferred the level from log(N) at chance, which was not a measurement.</li>"
        "<li>Still standing: do not motivate regularisation from this observation alone; validation loss is not a "
        "usable selection signal; log median rank beside mean rank. '18/18 worse at the final epoch' is near-tautological.</li>"
        "<li>The shuffled control rules out pure memorisation of the late gain, not a mixture.</li>"
        "<li>validation_loss is the full 441×441 InfoNCE and train_loss is in-batch 32 (the evaluation.py docstring is wrong).</li>"
        f"<li>Companion scale-free ranks: <a href=\"{ctx.raw('val-loss-rank-depth.json')}\">val-loss-rank-depth.json</a>.</li>"
        "</ul>"
    )
    return (section(sid, claim, body, caption, notes, tbl, f"Per-run table ({n} runs)"),
            {f"data-{sid}": {"rows": rows, "bars": bars, "groups": groups}}, {})


# ---------------------------------------------------------------------------- 8. gates: duplicates / FN


def duplicates_section(ctx):
    sid = "gate-duplicates"
    dup = data.read_json("duplicate-confusion-audit.json")
    fn = data.read_json("false-negative-rate-audit.json")
    if not dup or not fn:
        return section(sid, "False negatives and duplicates", notice(MISSING)), {}, {}
    groups = {"train": {"label": "Train, exact dup.", "color": "var(--g-sky)"},
              "validation": {"label": "Validation, exact dup.", "color": "var(--g-anchor)"},
              "tfidf": {"label": "TF-IDF > 0.9, batch 32", "color": "var(--g-amber)"},
              "t2i": {"label": "text → image", "color": "var(--g-sky)"},
              "i2t": {"label": "image → text", "color": "var(--g-teal)"}}
    eb = dup["exact_duplicate_bound"]
    metrics = {"pairs": ("expected_duplicate_pairs_per_batch", "Expected duplicate pairs per batch", ".2f"),
               "share": ("share_of_batches_with_at_least_one", "Share of batches with ≥1 duplicate pair", ".0%")}
    series = {}
    for mk, (field, _, _) in metrics.items():
        lst = []
        for split in ("train", "validation"):
            bb = eb[split]["by_batch_size"]
            lst.append({"id": f"exact-{split}", "label": groups[split]["label"], "group": split,
                        "points": [[int(k).bit_length() - 1, bb[k][field]] for k in sorted(bb, key=int)]})
        t = fn["tfidf"]["thresholds"]["0.9"]
        lst.append({"id": "tfidf", "label": groups["tfidf"]["label"], "group": "tfidf",
                    "points": [[int(fn["population"]["batch_size"]).bit_length() - 1, t["expected_collisions_per_batch_of_32" if mk == "pairs" else "p_batch_has_any"]]]})
        series[mk] = lst
    cut = "0.90"
    dumb = []
    for s, r in dup["runs"].items():
        for dname, dd in r["directions"].items():
            g = "t2i" if dname == "text_to_image" else "i2t"
            c = dd["cuts"][cut]
            dumb.append({"id": f"{s}-{g}", "label": f"s{seed_of(s)} {'txt→img' if g == 't2i' else 'img→txt'}", "group": g, "a": dd["r@1"],
                         "b": c["duplicate_tolerant_r@1"], "enr": c["enrichment_over_background"], "miss": c["share_of_misses"]})
    gains = [r["b"] - r["a"] for r in dumb]
    b32 = eb["train"]["by_batch_size"]["32"]
    ceiling = eb["validation"]["text_to_image_r@1_ceiling"]
    claim = (f"Exact duplicate reports are rare at batch 32 ({num(b32['expected_duplicate_pairs_per_batch'], '.3f')} pairs, "
             f"{b32['share_of_batches_with_at_least_one']:.1%} of batches) and tolerating near-duplicates lifts R@1 by at "
             f"most {num(max(gains), '.3f', True)}")
    body = f"""{status_line(status_badge('amended'), caveat_badge('False-negative audit amended 2026-09-13: the rate stands, the detector finding is void'))}
<div class="grid-2">
<div><div class="chart-title">In-batch duplicate collisions vs batch size (train cohort)</div><div class="of-switch" data-role="metric"></div><div data-role="host"></div><div data-role="legend"></div></div>
<div>{chart('tol', f'Validation R@1 vs duplicate-tolerant R@1 (TF-IDF cut {cut})')}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  OF.switcher(sec.querySelector('[data-role=metric]'), sec.querySelector('[data-role=host]'), ['pairs', 'share'],
    ['Pairs per batch', 'Batches with ≥1 pair'], function (k, mount) {{
      Charts.lineChart(mount, d.series[k], {{groups: d.groups, store: store, xLabel: 'Batch size (log scale)',
        xFmt: function (v) {{ return String(Math.pow(2, v)); }},
        yLabel: d.metrics[k][1], yFmt: d.metrics[k][2], best: 'last'}});
      Charts.legend(sec.querySelector('[data-role=legend]'), d.series[k], store, {{groups: d.groups, best: 'last', fmt: d.metrics[k][2], valuePrefix: ''}});
    }});
  OF.dumbbell(sec.querySelector('[data-chart=tol]'), d.dumb, {{groups: d.groups, store: store, fmt: '.3f', axisFmt: '.2f',
    aName: 'R@1', bName: 'duplicate-tolerant R@1', xLabel: 'Validation Recall@1 (441 candidates)',
    valueOf: function (r) {{ return Charts.fmt('+.3f')(r.b - r.a); }},
    tooltipExtra: function (r) {{ return '<div>Enrichment over background ' + r.enr.toFixed(1) + '\\u00d7; share of misses ' + (r.miss * 100).toFixed(1) + '%</div>'; }}}});
}})();
</script>"""
    caption = (f"Left: exact-hash duplicate arithmetic (not a measurement) plus the TF-IDF rate at batch 32. Right: three "
               f"4-block seeds × two directions. Exact duplicates cap validation text→image R@1 at {num(ceiling)}. "
               f"<a href=\"{ctx.doc('duplicate-confusion-audit.md')}\">duplicate-confusion-audit</a> · "
               f"<a href=\"{ctx.doc('false-negative-rate-audit.md')}\">false-negative-rate-audit</a>.")
    th = fn["tfidf"]["thresholds"]
    e5 = fn["e5_full_report"]["in_batch_collisions"]
    tbl = table(["Similarity cut", "TF-IDF collisions / batch of 32", "TF-IDF P(batch has any)", "e5 collisions / batch of 32"],
                [[c, num(th[c]["expected_collisions_per_batch_of_32"], ".3f") if c in th else "–",
                  num(th[c]["p_batch_has_any"], ".1%") if c in th else "–",
                  num(e5[c]["expected_collisions_per_batch_of_32"], ".2f") if c in e5 else "–"]
                 for c in sorted(set(th) | set(e5), key=float)], num_cols=(1, 2, 3))
    notes = (
        "<ul>"
        "<li><b>False-negative audit, amended 2026-09-13.</b> The rate finding stands and is firmer (exact "
        "<code>report_sha256</code> duplicates). The detector finding is void: an exact, free detector exists. Read "
        "the two together before re-proposing or re-closing this lever.</li>"
        f"<li>e5 similarity is unusable as a detector (template-dominated: {esc(fn['e5_full_report']['verdict'][:60])}…). "
        "TF-IDF is lexical, so in a multilingual cohort its rate is a lower bound.</li>"
        "<li>Batch size, not target shape, is the binding constraint: at a larger batch this gate should be re-run, "
        "not assumed. A multi-positive loss at batch 128 is a proposal, not a result.</li>"
        "<li>Duplicate-tolerant R@1 must not be selected on. The gains sit within the ~0.02 seed spread. The hubness "
        "cause is not established here; the large normal-knee cluster may be a labelling story rather than a loss story.</li>"
        "</ul>"
    )
    return (section(sid, claim, body, caption, notes, tbl, "False-negative detector table (TF-IDF vs e5)"),
            {f"data-{sid}": {"series": series, "dumb": dumb, "groups": groups,
                             "metrics": {k: list(v) for k, v in metrics.items()}}}, {})


# ---------------------------------------------------------------------------- 9. gates: hard negatives


def hardneg_section(ctx):
    sid = "gate-hard-negative"
    j = data.read_json("hard-negative-gate-data.json")
    if not j:
        return section(sid, "Hard-negative gate", notice(MISSING)), {}, {}
    groups = {"gate": {"label": "Batch 32 (the gate)", "color": "var(--g-amber)"},
              "n441": {"label": "Matched N=441 (not the gate)", "color": "var(--g-anchor)"},
              "full": {"label": "Full train 3,526 (not the gate)", "color": "var(--g-imgft)"}}
    rows = []
    for ck, g in j["train_batch_32_as_the_gate_specified"].items():
        s = seed_of(ck)
        rows.append({"id": f"gate-{s}", "label": f"Batch 32, seed {s}", "group": "gate", "value": g["top1_share_mean"], "top3": g["top3_share_mean"]})
    for ck, g in j["seeds"].items():
        s = seed_of(ck)
        for key, grp, lab in (("train_matched_n_441", "n441", "N=441"), ("train_full_3526", "full", "N=3,526")):
            x = g[key]
            rows.append({"id": f"{grp}-{s}", "label": f"{lab}, seed {s}", "group": grp, "value": x["top1_share_mean"],
                         "top3": x["top3_share_mean"], "top10": x.get("top10_share_mean")})
    rows.sort(key=lambda r: ["gate", "n441", "full"].index(r["group"]))
    gate = [r["value"] for r in rows if r["group"] == "gate"]
    claim = (f"At batch 32 the hardest negative already carries {min(gate):.0%}–{max(gate):.0%} of the softmax mass, so "
             f"hard-negative reweighting has nothing to amplify")
    body = f"""{status_line(caveat_badge('COMPUTED, NOT A DECISION: the arm is not authorised'), text='Gate verdict is batch-32 specific.')}
<div data-role="filter"></div>
{chart('share', 'Top-1 negative share of the off-diagonal softmax mass')}
<script>
(function () {{
  var sec = document.getElementById('{sid}'), d = Charts.readJSON('data-{sid}'), store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.barChart(sec.querySelector('[data-chart=share]'), d.rows, {{groups: d.groups, store: store, fmt: '.1%', axisFmt: '.0%',
    xLabel: 'Top-1 share at the learned temperature', domain: [0, 1],
    tooltipExtra: function (r) {{ return '<div>Top-3 share ' + (r.top3 * 100).toFixed(1) + '%</div>' +
      (r.top10 ? '<div>Top-10 share ' + (r.top10 * 100).toFixed(1) + '%</div>' : ''); }}}});
}})();
</script>"""
    caption = ("Training-split embeddings, augmentation off, softmax over negatives only. Only the batch-32 rows answer the "
               "gate; the other candidate counts are the same quantity at a different N. "
               f"<a href=\"{ctx.raw('hard-negative-gate-data.json')}\">hard-negative-gate-data.json</a> · "
               f"<a href=\"{ctx.doc('matched-n-loss-audit.md')}\">matched-n-loss-audit §Scope</a>.")
    tbl = table(["Condition, seed", "Top-1 share", "Top-3 share"],
                [[r["label"], num(r["value"], ".1%"), num(r["top3"], ".1%")] for r in rows], num_cols=(1, 2),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul>"
        f"<li>{esc(j.get('note', ''))}</li>"
        f"<li>Gate reading: {esc(j.get('gate_reading', ''))}</li>"
        f"<li>{esc(j.get('why_the_n441_numbers_are_not_the_gate', ''))}</li>"
        f"<li>{esc(j.get('what_would_change_it', ''))}</li>"
        "</ul>"
    )
    return (section(sid, claim, body, caption, notes, tbl, "Per-seed shares"),
            {f"data-{sid}": {"rows": rows, "groups": groups}}, {})


# ---------------------------------------------------------------------------- 10. takeaway


def takeaway_section(ctx, k):
    sid = "takeaway"
    fov = data.read_json("fov-and-depth-audit.json")
    unseen = (fov or {}).get("unseen_slice_share")
    items = []
    if k.get("sweep"):
        items.append(f"<div><b>Stochastic depth, six rates.</b> Gap {pct(k['sweep']['gap_change'])}, R@1 only falls. "
                     f"Closed at a 20-epoch budget; at 40 epochs D40 {num(k['budget']['d40'], '.4f', True)} (ambiguous).</div>")
    if k.get("crop"):
        items.append(f"<div><b>Wider crop (150 mm).</b> Converged, then overfit; R@1 and gold AUROC "
                     f"({num(k['crop']['pooled'], '.4f', True)} pooled) both lower. Not adopted.</div>")
    items.append("<div><b>Loss-side gates.</b> False-negative cancellation, soft duplicate targets and hard-negative "
                 "reweighting were closed at batch 32 before costing GPU time; each should be re-run if the batch grows.</div>")
    nxt = (f" The untried direction adds information: the {unseen:.1%} of slices the pipeline never opens"
           if isinstance(unseen, (int, float)) else " The untried direction adds information (more slices and series)")
    claim = "Every lever that removed or perturbed information cut memorisation and cost retrieval"
    body = (f'<div class="caveat-strip">{"".join(items)}</div>'
            f'<p class="lede">The memorisation gap is real, but it is not what stands between this project and better '
            f'numbers.{nxt} (<a href="{ctx.doc("acquisition-metadata-trust.md")}">acquisition-metadata-trust</a>).</p>')
    notes = ("<ul><li>Projection-head dropout and random erasing remain untested; any future regularisation arm must train to "
             "a convergence criterion or report the budget as a co-varying factor.</li>"
             "<li>All retrieval results here are study-level, single cohort (441 validation candidates); patient "
             "independence is unverifiable from this data.</li></ul>")
    return section(sid, claim, body, notes=notes), {}


# ---------------------------------------------------------------------------- page


def build(ctx) -> Page:
    sections, blobs, k = [], {}, {}
    for name, fn_ in (("mem", memorisation_section), ("dp05", droppath05_section), ("sweep", sweep_section),
                      ("conv", convergence_section), ("budget", budget_section), ("crop", crop_section),
                      ("vl", valloss_section), ("dup", duplicates_section), ("hn", hardneg_section)):
        sec, blob, kv = fn_(ctx)
        sections.append(sec)
        blobs.update(blob)
        if kv:
            k[name] = kv
    if "mem" not in k:  # the helper script lives in the first section; keep it available
        sections[0] = HELPER_JS + SWITCH_CSS + sections[0]
    sec, _ = takeaway_section(ctx, k)
    sections.append(sec)
    kpis = []
    if "mem" in k:
        kpis.append({"label": "Train ÷ validation R@1", "value": f"{k['mem']['ratio']:.1f}×",
                     "sub": f"N={k['mem']['n']}: {num(k['mem']['tr_r1'], '.3f')} vs {num(k['mem']['va_r1'], '.3f')}", "tone": "bad"})
    if "dp05" in k:
        kpis.append({"label": "Drop-path 0.05: gap / Δ R@1", "value": pct(k['dp05']['change']),
                     "sub": f"Δ val R@1 {num(k['dp05']['diff'], '.4f', True)}"})
    if "sweep" in k:
        kpis.append({"label": "Rate 0.1 @ 20 ep, Δ R@1", "value": num(k["sweep"]["d20"], ".4f", True),
                     "sub": "amended: closed only at 20 epochs"})
    if "budget" in k:
        kpis.append({"label": "Rate 0.1 @ 40 ep, Δ R@1", "value": num(k["budget"]["d40"], ".4f", True),
                     "sub": f"AMBIGUOUS, CI {ci_txt(k['budget']['ci'])}"})
    if "crop" in k:
        kpis.append({"label": "Crop150, Δ gold AUROC", "value": num(k["crop"]["pooled"], ".4f", True),
                     "sub": "pooled 3 seeds; not adopted"})
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=sections, data=blobs, kpis=kpis or None)


def summary() -> dict:
    m = data.read_json("matched-n-loss-audit.json")
    value = None
    if m:
        tr = mean(x["matched_n"]["train_random_subsets"]["mean_r@1_mean"] for x in m)
        va = mean(x["matched_n"]["validation_mean_r@1"] for x in m)
        value = f"{tr / va:.1f}×"
    return {"stage": BY_SLUG[SLUG].number,
            "headline": "The model memorises; drop-path and a wider crop cut memorisation but only cost retrieval",
            "value": value, "value_label": "train ÷ validation R@1 at N=441",
            "caveat": "Drop-path closed only at a 20-epoch budget (40-epoch arm ambiguous); crop150 not adopted.",
            "href": "#memorisation"}
