"""Stage 2: data, gold set and integrity.

Sections (numbers read from digest/*.json at build time):
  integrity  duplicate funnel: fingerprinted -> header groups -> confirmed -> pixel-verified -> identical / cleared
  gold       58 gold studies: exposure waffle (as of 2026-09-11) + per-label positives by split (56-study population)
  leak       protocol-leak check: series count vs each gold label, AUROC per label (58-study population)
  inputs     crop retention curve (130 current, 150 NOT ADOPTED) + FOV histogram by plane
  slots      acquisition slot fill (21 plane / weighting / fat-sat slots) with the 80% line
  length     report token-length percentiles vs the 512-token limit
Gates (false-negative, hard-negative, val-loss, duplicate-confusion) live on the overfitting page.
"""

from __future__ import annotations

from .. import data as D
from ..shell import Page, chart, esc, notice, num, section, table
from ..sitemap import BY_SLUG

SLUG = "data"
SUBTITLE = ("What the 4,407 studies are, how clean they are, what the model actually sees of each one, and what "
            "the 58 gold-labelled studies can and cannot support.")

# Detectable-effect bound of the leak check exists only in markdown. source: digest/acquisition-metadata-trust.md:104
LEAK_POWER = (0.149, 0.197)

PLANE_GROUPS = {
    "Sagittal": {"label": "Sagittal", "color": "var(--g-teal)"},
    "Coronal": {"label": "Coronal", "color": "var(--g-violet)"},
    "Axial": {"label": "Axial", "color": "var(--g-amber)"},
}


def _pct(v, d=1):
    return num(v, f".{d}%")


def _sid_js(js: str, sid: str) -> str:
    return js.replace("__SID__", sid)


# --------------------------------------------------------------------------- integrity funnel


def integrity_section(ctx) -> tuple[str, dict]:
    sid = "integrity"
    scr = D.read_json("acquisition-duplicate-screen.json")
    pix = D.read_json("duplicate-group-pixel-verification.json")
    gate = D.read_json("gated-duplicate-resolution.json")
    cov = D.read_json("integrity-coverage-audit.json")
    if not (scr and pix and gate and cov):
        return section(sid, "Integrity", notice("An integrity JSON is missing.")), {}
    s1, s2 = scr["stage1"], scr["stage2"]
    verified_studies = sum(g["members"] for g in pix["detail"])
    gated_studies = sum(g["members"] for g in pix["gated"])
    cleared = sum(1 for g in pix["gated"] if "CLEARED" in str((g.get("resolved") or {}).get("verdict", "")))
    steps = [
        {"id": "f-all", "label": "Studies fingerprinted (header screen)", "n": scr["studies_fingerprinted"],
         "sub": "series-constant acquisition tags + slice count, per selected plane", "tone": "neutral"},
        {"id": "f-s1", "label": f"In stage-1 candidate groups ({s1['candidate_groups']} groups)",
         "n": s1["studies_in_candidate_groups"], "sub": f"largest group {s1['largest_candidate_group']} studies", "tone": "neutral"},
        {"id": "f-s2", "label": f"In stage-2 confirmed groups ({s2['confirmed_groups']} groups)",
         "n": s2["studies_in_confirmed_groups"], "sub": f"{s2['groups_crossing_splits']} groups cross splits, "
                                                         f"{s2['groups_involving_test']} involves test", "tone": "warn"},
        {"id": "f-pix", "label": f"Pixel-verified ({pix['groups_examined']} groups)", "n": verified_studies,
         "sub": "SHA-256 of every decoded slice", "tone": "neutral"},
        {"id": "f-same", "label": f"Pixel-identical ({pix['groups_confirmed_identical']} group), quarantined",
         "n": pix["studies_in_confirmed_groups"],
         "sub": f"train-only; {pix['confirmed_studies_still_training_eligible']} still training-eligible", "tone": "bad"},
        {"id": "f-gate", "label": f"Gated train/test group, decoded and cleared", "n": gated_studies,
         "sub": "different pixels on all 3 planes; test reports compared by hash only, never read", "tone": "good"},
    ]
    pg = cov["patient_grouping"]
    split = pg["split_sizes"]
    claim = (f"One pixel-identical pair in {scr['studies_fingerprinted']:,} studies, already quarantined; "
             f"the only train/test candidate was cleared")
    blob = {"steps": steps}
    body = """<div class="chart funnel" data-chart="funnel"></div>
<style>
#integrity .funnel{display:grid;gap:6px}
#integrity .fn-row{display:grid;grid-template-columns:minmax(0,1fr) 64px;gap:4px 10px;align-items:center;cursor:default}
#integrity .fn-bar{height:22px;border-radius:5px;background:color-mix(in srgb,var(--g-anchor) 55%,var(--card));min-width:4px}
#integrity .fn-row.warn .fn-bar{background:color-mix(in srgb,var(--g-amber) 70%,var(--card))}
#integrity .fn-row.bad .fn-bar{background:color-mix(in srgb,var(--bad) 70%,var(--card))}
#integrity .fn-row.good .fn-bar{background:color-mix(in srgb,var(--good) 70%,var(--card))}
#integrity .fn-n{font-family:var(--mono);font-weight:700;color:var(--text-strong);text-align:right}
#integrity .fn-lab{grid-column:1/-1;font-size:12.5px;color:var(--text)}
#integrity .fn-lab small{color:var(--muted);display:block}
#integrity .fn-row.dimmed{opacity:.35}
</style>
<script>
(function () {
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  var el = sec.querySelector('[data-chart=funnel]');
  var max = Math.log10(d.steps[0].n + 1);
  d.steps.forEach(function (s) {
    var row = Charts.h('div', {class: 'fn-row ' + s.tone, 'data-series-id': s.id, title: s.label});
    row.appendChild(Charts.h('div', {class: 'fn-lab'}, Charts.esc(s.label) + '<small>' + Charts.esc(s.sub) + '</small>'));
    var bar = Charts.h('div', {class: 'fn-bar'}); bar.style.width = (100 * Math.log10(s.n + 1) / max).toFixed(1) + '%';
    row.appendChild(bar);
    row.appendChild(Charts.h('div', {class: 'fn-n', text: s.n.toLocaleString('en-US')}));
    row.addEventListener('pointerenter', function () { if (!store.get().solo) store.set({hover: s.id}); });
    row.addEventListener('pointerleave', function () { store.set({hover: null}); });
    row.addEventListener('click', function () { store.toggleSolo(s.id); });
    el.appendChild(row);
  });
  Charts.applySelection(sec, store.get());
})();
</script>"""
    caption = (f"Bar length on a log scale. Study-level only: {pg['distinct_group_ids']:,} distinct PatientIDs for "
               f"{pg['rows']:,} studies is an anonymisation artefact, so patient independence is unverifiable. "
               f"Splits {split['train']:,} / {split['validation']} / {split['test']} (train / validation / test).")
    rows = []
    for g in pix["detail"]:
        rows.append([g["members"], ", ".join(g["splits"]), "yes" if g["crosses_splits"] else "no",
                     "yes" if g["pixels_identical_on_every_plane"] else "no",
                     "yes" if g["reports_conflict"] else "no",
                     "quarantined" if g["pixels_identical_on_every_plane"] else "not a pixel duplicate"])
    for g in pix["gated"]:
        rows.append([g["members"], ", ".join(g["splits"]), "yes",
                     "yes" if gate["pixels_identical_on_every_plane"] else "no", "–",
                     "cleared 2026-09-11 (gated decision)"])
    tbl = table(["Studies", "Splits", "Crosses splits", "Pixel-identical", "Reports conflict", "Outcome"], rows,
                num_cols=(0,))
    rc = cov["report_collisions"]["translated_report"]
    notes = (
        "<ul>"
        "<li><b>Study-level, not patient-level.</b> Every retrieval and probe figure on this site is study-disjoint only; "
        "it cannot be described as patient-disjoint (the known duplicate pair carries two different PatientIDs).</li>"
        f"<li>“Cleared” means “not a pixel duplicate”. The {s2['studies_in_confirmed_groups']} studies of the confirmed "
        "header groups, including the cleared train/test pair, remain the only patient-leakage candidates. The screen "
        "only finds duplicates that agree on every stage-1 tag.</li>"
        f"<li>Test boundary: {cleared} gated group decoded under an explicit user decision; its reports were compared by "
        "SHA-256 and never read, and no model or metric was run on it.</li>"
        f"<li>Report text collisions (translated): {rc['groups']} groups, {rc['studies_involved']} studies, "
        f"{rc['groups_crossing_splits']} crossing splits. Their effect on retrieval is audited on the "
        f"<a href=\"{ctx.page('overfitting')}\">overfitting &amp; gates page</a> (duplicate-confusion audit).</li>"
        f"<li>Sources: <a href=\"{ctx.raw('acquisition-duplicate-screen.json')}\">acquisition-duplicate-screen.json</a>, "
        f"<a href=\"{ctx.raw('duplicate-group-pixel-verification.json')}\">duplicate-group-pixel-verification.json</a> "
        "(amended 2026-09-11: gated group resolved), "
        f"<a href=\"{ctx.raw('gated-duplicate-resolution.json')}\">gated-duplicate-resolution.json</a>, "
        f"<a href=\"{ctx.doc('integrity-coverage-audit.md')}\">integrity-coverage-audit.md</a>. The earlier "
        f"<a href=\"{ctx.doc('full-projection-v1-data-review.md')}\">baseline data review</a> is superseded by it.</li>"
        "</ul>"
    )
    return section(sid, claim, body.replace("__SID__", sid), caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Pixel-verification groups ({len(rows)})"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- gold set


def gold_section(ctx) -> tuple[str, dict, dict]:
    sid = "gold"
    g = D.read_json("gold-case-inventory.json")
    if not g:
        return section(sid, "Gold set", notice("gold-case-inventory.json missing.")), {}, {}
    cm, ex = g["cohort_membership"], g["representation_exposure"]
    feas = g["per_label_feasibility"]
    labels = []
    for name, v in feas["labels"].items():
        labels.append({"id": "g-" + name.lower().replace(" ", "-").replace("'", ""), "label": name,
                       "train": v["positives_by_split"].get("train", 0), "val": v["positives_by_split"].get("validation", 0),
                       "neg": v["negatives"], "pos": v["positives"], "prev": v["prevalence"], "verdict": v["verdict"]})
    by_split = cm["by_split"]
    waffle = (["train"] * by_split.get("train", 0) + ["validation"] * by_split.get("validation", 0)
              + ["test"] * by_split.get("test", 0))
    exposed = ex["seen_in_representation_training"] + ex["in_validation_used_for_checkpoint_selection"]
    zero_val = [l["label"] for l in labels if l["val"] == 0]
    minpos = min(labels, key=lambda l: l["pos"])
    th = feas["thresholds"]
    claim = (f"As of 2026-09-11, {exposed} of {cm['gold_studies']} gold studies had been seen by the original-cohort encoders; all "
             f"{len(labels)} labels are estimable, but {', '.join(zero_val) or 'no label'} has no validation positives")
    blob = {"waffle": waffle, "labels": labels, "studies": feas["studies"], "th": th}
    body = """<div class="grid-2 gold-grid">
<div><div class="chart-title">58 gold studies by split, as of 2026-09-11</div>
<div class="chart waffle" data-chart="waffle"></div>
<div class="wf-key"><span class="k train">train: seen in contrastive training</span><span class="k validation">validation: used for checkpoint selection</span><span class="k test">locked test</span></div></div>
<div>__CHART__</div>
</div>
<style>
#gold .waffle{display:grid;grid-template-columns:repeat(10,minmax(0,26px));gap:4px;margin:6px 0 10px}
#gold .wf{aspect-ratio:1;border-radius:4px;background:var(--g-anchor)}
#gold .wf.validation{background:var(--g-amber)}
#gold .wf.test{background:var(--card);border:2px solid var(--g-lora)}
#gold .wf.dimmed{opacity:.2}
#gold .wf-key{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--muted)}
#gold .wf-key .k:before{content:"";display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px;background:var(--g-anchor)}
#gold .wf-key .validation:before{background:var(--g-amber)}
#gold .wf-key .test:before{background:var(--card);border:2px solid var(--g-lora);box-sizing:border-box}
#gold .stk rect.seg-train{fill:var(--g-anchor)}
#gold .stk rect.seg-val{fill:var(--g-amber)}
#gold .stk rect.seg-neg{fill:var(--border)}
</style>
<script>
(function () {
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  var w = sec.querySelector('[data-chart=waffle]');
  d.waffle.forEach(function (s, i) {
    var c = Charts.h('div', {class: 'wf ' + s, 'data-series-id': 'split-' + s, title: s});
    c.addEventListener('pointerenter', function () { if (!store.get().solo) store.set({hover: 'split-' + s}); });
    c.addEventListener('pointerleave', function () { store.set({hover: null}); });
    w.appendChild(c);
  });
  var el = sec.querySelector('[data-chart=stack]');
  var tip = Charts.tooltip(el), svgEl = null;
  function render() {
    if (svgEl) svgEl.remove();
    var W = el.clientWidth || 520, narrow = W < 460, labelW = narrow ? 104 : 124, rowH = 22;
    var pad = {l: labelW + 6, r: 10, t: 8, b: 34}, H = pad.t + pad.b + d.labels.length * rowH;
    var x = Charts.scale([0, d.studies], [pad.l, W - pad.r]);
    svgEl = Charts.svg('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'positives per label', class: 'stk'});
    var ax = Charts.svg('g', {class: 'axis'});
    x.ticks(5).forEach(function (t) {
      ax.appendChild(Charts.svg('line', {class: 'gridline', x1: x(t), x2: x(t), y1: pad.t, y2: H - pad.b}));
      ax.appendChild(Charts.svg('text', {x: x(t), y: H - pad.b + 14, 'text-anchor': 'middle'}, String(t)));
    });
    ax.appendChild(Charts.svg('text', {class: 'axis-title', x: (pad.l + W - pad.r) / 2, y: H - 4, 'text-anchor': 'middle'}, 'Gold studies (of ' + d.studies + ')'));
    [d.th.min_positives_for_grouped_folds].forEach(function (t) {
      ax.appendChild(Charts.svg('line', {class: 'ref-line', x1: x(t), x2: x(t), y1: pad.t - 4, y2: H - pad.b}));
    });
    svgEl.appendChild(ax);
    d.labels.forEach(function (l, i) {
      var y = pad.t + i * rowH, g = Charts.svg('g', {class: 'mark', 'data-series-id': l.id});
      g.appendChild(Charts.svg('text', {class: 'row-label', x: labelW, y: y + rowH / 2 + 4, 'text-anchor': 'end'}, l.label));
      var x0 = 0;
      [['seg-train', l.train], ['seg-val', l.val], ['seg-neg', l.neg]].forEach(function (p) {
        g.appendChild(Charts.svg('rect', {class: p[0], x: x(x0), y: y + 3, width: Math.max(0, x(x0 + p[1]) - x(x0)), height: rowH - 6}));
        x0 += p[1];
      });
      g.addEventListener('pointermove', function (e) {
        tip.show(e, '<div class="tt-title">' + Charts.esc(l.label) + '</div><div>Positives ' + l.pos + ' (train ' + l.train +
          ', validation ' + l.val + ') · negatives ' + l.neg + '</div><div>Prevalence ' + Charts.fmt('.1%')(l.prev) +
          ' · ' + Charts.esc(l.verdict) + '</div>');
        if (!store.get().solo) store.set({hover: l.id});
      });
      g.addEventListener('pointerleave', function () { tip.hide(); store.set({hover: null}); });
      g.addEventListener('click', function () { store.toggleSolo(l.id); });
      svgEl.appendChild(g);
    });
    el.appendChild(svgEl);
    Charts.applySelection(sec, store.get());
  }
  render();
  if (window.ResizeObserver) { var last = el.clientWidth; new ResizeObserver(function () { if (el.clientWidth !== last) { last = el.clientWidth; render(); } }).observe(el); }
})();
</script>"""
    body = body.replace("__CHART__", chart("stack", f"Positives per label: train, validation, then negatives ({feas['studies']} studies)")
                        + '<div class="wf-key" style="flex-direction:row;flex-wrap:wrap;gap:12px"><span class="k train">train positives</span>'
                          '<span class="k validation">validation positives</span><span>grey: negatives · dashed line: '
                          f'{th["min_positives_for_grouped_folds"]} positives (grouped folds)</span></div>')
    caption = (f"Waffle: {cm['gold_studies']} studies, snapshot 2026-09-11 against the original-cohort encoders. Bars: "
               f"{feas['population']} ({feas['studies']} studies); fewest positives: {esc(minpos['label'])} ({minpos['pos']}).")
    tbl = table(["Label", "Positives", "Train", "Validation", "Negatives", "Prevalence", "Verdict"],
                [[l["label"], l["pos"], l["train"], l["val"], l["neg"], _pct(l["prev"]), l["verdict"]] for l in labels],
                num_cols=(1, 2, 3, 4, 5), row_attrs=[{"data-series-id": l["id"]} for l in labels])
    te = g["text_exposure"]
    notes = (
        "<ul>"
        f"<li><b>Exposure changed after this snapshot.</b> The gold-holdout encoders (and the submitted model) excluded all "
        f"{feas['studies']} train+validation gold studies from contrastive training and selection; the {by_split.get('test', 0)} "
        "test-split gold studies were later scored in the 58-case LM-head run and are no longer untouched. See the "
        f"<a href=\"{ctx.page('probes')}\">probes page</a> and <a href=\"{ctx.doc('gold-holdout-v1.md')}\">gold-holdout-v1</a>.</li>"
        "<li>Any probe on exposed gold studies is a development statistic, not a generalisation estimate.</li>"
        f"<li>Text exposure: {te['matching']} of {te['compared']} trained texts match the adjudicated text; "
        f"{te['mismatching_but_same_source_report']} are retranslations of the same source report.</li>"
        "<li>Labels are binary with no missing cells, so there is no uncertainty encoding: absence of an unknown code does "
        "not show every case was adjudicable. <code>drop.csv</code> is a blank template; always assert a non-zero positive count.</li>"
        "<li>Population: these counts are over the 56 train+validation gold studies (for example ACL 23). The protocol-leak "
        "check below counts all 58 (ACL 24); the two are never merged.</li>"
        f"<li>The inventory's proposed probe contract was “pending sign-off” and was superseded in practice by the gold holdout. "
        f"Source: <a href=\"{ctx.doc('gold-case-inventory.md')}\">gold-case-inventory.md</a>, "
        f"<a href=\"{ctx.raw('gold-case-inventory.json')}\">gold-case-inventory.json</a>.</li>"
        "</ul>"
    )
    kp = {"gold": cm["gold_studies"], "exposed": exposed}
    return section(sid, claim, body.replace("__SID__", sid), caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Per-label table ({len(labels)} labels, {feas['studies']} studies)"), {f"data-{sid}": blob}, kp


# --------------------------------------------------------------------------- protocol leak


def leak_section(ctx) -> tuple[str, dict, dict]:
    sid = "leak"
    j = D.read_json("protocol-leak-check.json")
    if not j:
        return section(sid, "Protocol leak", notice("protocol-leak-check.json missing.")), {}, {}
    rows = []
    for name, v in j["primary"].items():
        sig = v["permutation_p"] < 0.05
        rows.append({"id": "l-" + name.lower().replace(" ", "-").replace("'", ""), "label": name,
                     "group": "sig" if sig else "ns", "est": v["auroc"], "lo": None, "hi": None,
                     "n_pos": v["n_pos"], "p": v["permutation_p"]})
    n_sig = sum(1 for r in rows if r["group"] == "sig")
    far = max(rows, key=lambda r: abs(r["est"] - 0.5))
    groups = {"ns": {"label": "p ≥ 0.05", "color": "var(--g-anchor)"},
              "sig": {"label": "p < 0.05", "color": "var(--bad)"}}
    sc = j["series_count"]
    claim = (f"Series count predicts none of the {len(rows)} gold labels: {n_sig} of {len(rows)} at p &lt; 0.05, "
             f"furthest from chance {esc(far['label'])} at AUROC {num(far['est'], '.3f')}")
    blob = {"rows": rows, "groups": groups}
    body = f"""{chart('dots', 'AUROC of series count for each gold label (0.5 = no signal)')}
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.ciDotPlot(sec.querySelector('[data-chart=dots]'), d.rows,
    {{groups: d.groups, ref: 0.5, refLabel: 'chance 0.5', fmt: '.3f', axisFmt: '.2f', domain: [0.3, 0.7],
      xLabel: 'AUROC (series count as the score)', store: store, estLabel: 'AUROC',
      tooltipExtra: function (r) {{ return '<div>' + r.n_pos + ' positives of {j['n_gold']} · permutation p ' + r.p.toFixed(3) + '</div>'; }}}});
}})();
</script>"""
    caption = (f"Population: all {j['n_gold']} gold studies (positives count all {j['n_gold']}, unlike the 56-study inventory). "
               f"Series per study: median {sc['median']:.0f}, range {sc['min']}–{sc['max']}; {j['resamples']:,} permutations per label.")
    tbl = table(["Label", "AUROC", "Positives (of 58)", "Permutation p"],
                [[r["label"], num(r["est"], ".3f"), r["n_pos"], num(r["p"], ".3f")] for r in rows],
                num_cols=(1, 2, 3), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    px = j["proxy"]
    notes = (
        "<ul>"
        f"<li>Power is limited: with these label sizes the check can only detect |AUROC − 0.5| &gt; {LEAK_POWER[0]}–{LEAK_POWER[1]}. "
        "It rules out <i>large</i> protocol leakage, not a modest one. No permutation-null band is drawn because "
        f"only two labels have one in <a href=\"{ctx.doc('acquisition-metadata-trust.md')}\">acquisition-metadata-trust.md</a>.</li>"
        f"<li>Full-cohort proxy: Spearman(series count, report length) = {num(px['spearman_series_vs_report_length'], '.3f')} "
        f"(p {px['permutation_p']}, n {px['n']:,}).</li>"
        f"<li>Verdict: proceed, with bounded residual risk. Source: <a href=\"{ctx.raw('protocol-leak-check.json')}\">protocol-leak-check.json</a>.</li>"
        "</ul>"
    )
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary="Per-label table"), {f"data-{sid}": blob}, {"n_sig": n_sig, "n": len(rows)}


# --------------------------------------------------------------------------- inputs: crop + FOV


def inputs_section(ctx) -> tuple[str, dict]:
    sid = "inputs"
    crop = D.read_json("crop-retention.json")
    fov = D.read_json("fov-and-depth-audit.json")
    if not (crop and fov):
        return section(sid, "Crop and field of view", notice("crop-retention or fov-and-depth JSON missing.")), {}
    cand = {int(k): v for k, v in crop["candidates"].items()}
    xs = sorted(cand)
    groups = {"ret": {"label": "Tissue retained", "color": "var(--g-sky)"},
              "mark": {"label": "Crop choices", "color": "var(--text-strong)"}}
    series = [
        {"id": "c-mean", "label": "Mean retained", "group": "ret", "color": "var(--g-sky)",
         "points": [[x, cand[x]["mean_retained"]] for x in xs]},
        {"id": "c-p10", "label": "10th percentile retained", "group": "ret", "color": "var(--g-violet)", "dash": "5 3",
         "points": [[x, cand[x]["p10_retained"]] for x in xs]},
        {"id": "c-95", "label": "Series keeping ≥95%", "group": "ret", "color": "var(--g-teal)", "dash": "2 3",
         "points": [[x, cand[x]["share_of_series_keeping_95pct"]] for x in xs]},
    ]
    cur = int(fov.get("crop_mm_constant", 130))
    if cur in cand:
        series.append({"id": "c-130", "label": f"{cur} mm: current default", "group": "mark", "color": "var(--g-lora)",
                       "points": [[cur, cand[cur]["mean_retained"]]]})
    if 150 in cand:
        series.append({"id": "c-150", "label": "150 mm: tested, NOT ADOPTED", "group": "mark", "color": "var(--bad)",
                       "points": [[150, cand[150]["mean_retained"]]]})
    pooled = D.POOLED["crop150_vs_crop130"]
    per = [{"plane": s["plane"], "fov": s["fov_rows_mm"]} for s in fov["per_series"]]
    claim = (f"A {cur} mm crop keeps {_pct(cand[cur]['mean_retained'], 0)} of the imaged tissue on average; 150 mm keeps "
             f"{_pct(cand[150]['mean_retained'], 0)} but lowered gold AUROC ({num(pooled['value'], '+.4f')} pooled) and was not adopted")
    blob = {"series": series, "groups": groups, "fov": per, "planes": PLANE_GROUPS, "crop": cur,
            "median": fov["fov_rows_mm"]["median"]}
    body = f"""<div class="grid-2">
<div>{chart('crop', f"Tissue retained by crop size ({crop['series']:,} {crop['split']} series)")}<div data-role="legend"></div></div>
<div><div data-role="planes" role="group" aria-label="Filter histogram by plane"></div>{chart('fov', f"True field of view, rows ({fov['series_audited']} series, {fov['studies_sampled']} studies)")}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.lineChart(sec.querySelector('[data-chart=crop]'), d.series, {{groups: d.groups, yLabel: 'Share retained', yFmt: '.0%',
    xLabel: 'Crop (mm)', best: 'last', store: store, xStep: 10}});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {{groups: d.groups, best: 'last', fmt: function () {{ return ''; }}, valuePrefix: ''}});
  // FOV histogram (local renderer: vertical crop line, plane filter via the section store)
  // plane toggle is local to the histogram (the section store's group filter would hide the crop series)
  var plane = 'all', pbar = sec.querySelector('[data-role=planes]');
  pbar.classList.add('filter-bar');
  [['all', 'All planes', null]].concat(Object.keys(d.planes).map(function (k) {{ return [k, d.planes[k].label, d.planes[k].color]; }})).forEach(function (p) {{
    var b = Charts.h('button', {{type: 'button', class: 'pill-btn', 'data-plane': p[0]}});
    if (p[2]) {{ var dot = Charts.h('span', {{class: 'dot'}}); dot.style.background = p[2]; b.appendChild(dot); }}
    b.appendChild(document.createTextNode(p[1])); pbar.appendChild(b);
  }});
  function paintPills() {{ pbar.querySelectorAll('[data-plane]').forEach(function (b) {{ var on = b.getAttribute('data-plane') === plane;
    b.classList.toggle('active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false'); }}); }}
  pbar.addEventListener('click', function (e) {{ var b = e.target.closest('[data-plane]'); if (!b) return;
    plane = b.getAttribute('data-plane'); paintPills(); render(); }});
  paintPills();
  var el = sec.querySelector('[data-chart=fov]'), tip = Charts.tooltip(el), svgEl = null, step = 10;
  var lo = Math.floor(Math.min.apply(null, d.fov.map(function (r) {{ return r.fov; }})) / step) * step;
  var hi = Math.ceil(Math.max.apply(null, d.fov.map(function (r) {{ return r.fov; }})) / step) * step + step;
  function render() {{
    if (svgEl) svgEl.remove();
    var g = plane, rows = d.fov.filter(function (r) {{ return !g || g === 'all' || r.plane === g; }});
    var bins = [], b;
    for (b = lo; b < hi; b += step) bins.push({{x0: b, n: 0}});
    rows.forEach(function (r) {{ var i = Math.min(bins.length - 1, Math.floor((r.fov - lo + 1e-6) / step)); bins[i].n++; }});
    var W = el.clientWidth || 520, H = W < 520 ? 240 : 300, pad = {{l: 44, r: 12, t: 16, b: 42}};
    var x = Charts.scale([lo, hi], [pad.l, W - pad.r]);
    var ymax = Math.max.apply(null, bins.map(function (q) {{ return q.n; }})) || 1;
    var y = Charts.scale([0, ymax * 1.08], [H - pad.b, pad.t]);
    svgEl = Charts.svg('svg', {{viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'field of view histogram'}});
    svgEl.appendChild(Charts.svg('rect', {{class: 'plot-bg', x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b}}));
    var ax = Charts.svg('g', {{class: 'axis'}});
    y.ticks(4).forEach(function (t) {{
      ax.appendChild(Charts.svg('line', {{class: 'gridline', x1: pad.l, x2: W - pad.r, y1: y(t), y2: y(t)}}));
      ax.appendChild(Charts.svg('text', {{x: pad.l - 6, y: y(t) + 4, 'text-anchor': 'end'}}, String(t)));
    }});
    x.ticks(W < 520 ? 4 : 7).forEach(function (t) {{ ax.appendChild(Charts.svg('text', {{x: x(t), y: H - pad.b + 15, 'text-anchor': 'middle'}}, String(t))); }});
    ax.appendChild(Charts.svg('text', {{class: 'axis-title', x: (pad.l + W - pad.r) / 2, y: H - 6, 'text-anchor': 'middle'}}, 'Field of view, rows (mm)'));
    ax.appendChild(Charts.svg('text', {{class: 'axis-title', transform: 'translate(11,' + ((pad.t + H - pad.b) / 2) + ') rotate(-90)', 'text-anchor': 'middle'}}, 'Series'));
    svgEl.appendChild(ax);
    var col = g && g !== 'all' ? d.planes[g].color : 'var(--g-anchor)';
    bins.forEach(function (q) {{
      if (!q.n) return;
      var r = Charts.svg('rect', {{x: x(q.x0) + 1, y: y(q.n), width: Math.max(1, x(q.x0 + step) - x(q.x0) - 2), height: y(0) - y(q.n), rx: 2, style: {{fill: col}}}});
      r.addEventListener('pointermove', function (e) {{ tip.show(e, '<div class="tt-title">' + q.x0 + '–' + (q.x0 + step) + ' mm</div><div>' + q.n + ' series</div>'); }});
      r.addEventListener('pointerleave', function () {{ tip.hide(); }});
      svgEl.appendChild(r);
    }});
    svgEl.appendChild(Charts.svg('line', {{class: 'ref-line', x1: x(d.crop), x2: x(d.crop), y1: pad.t, y2: H - pad.b}}));
    svgEl.appendChild(Charts.svg('text', {{class: 'ref-label', x: x(d.crop) + 4, y: pad.t + 12, 'text-anchor': 'start'}}, 'crop ' + d.crop + ' mm'));
    el.appendChild(svgEl);
  }}
  render();
  if (window.ResizeObserver) {{ var lw = el.clientWidth; new ResizeObserver(function () {{ if (el.clientWidth !== lw) {{ lw = el.clientWidth; render(); }} }}).observe(el); }}
}})();
</script>"""
    caption = (f"Left: centred crop vs Otsu tissue mask (an upper bound: it counts thigh and calf). Right: every audited series "
               f"is wider than the {cur} mm crop (median {fov['fov_rows_mm']['median']:.0f} mm), so the crop is never a no-op; "
               f"three slices leave {_pct(fov['unseen_slice_share'], 0)} of slices unseen.")
    tbl = table(["Crop (mm)", "Mean retained", "Median", "10th pct", "Series ≥95%", "Series ≥99%"],
                [[x, _pct(cand[x]["mean_retained"]), _pct(cand[x]["median_retained"]), _pct(cand[x]["p10_retained"]),
                  _pct(cand[x]["share_of_series_keeping_95pct"]), _pct(cand[x]["share_of_series_keeping_99pct"])] for x in xs],
                num_cols=(0, 1, 2, 3, 4, 5))
    notes = (
        "<ul>"
        f"<li><b>150 mm: NOT ADOPTED.</b> acquisition-metadata-trust.md marked 150 mm “chosen”; the gold probe then scored it "
        f"{num(pooled['value'], '+.4f')} macro AUROC against 130 mm, pooled over 3 seeds "
        f"(<a href=\"{ctx.doc(pooled['doc'])}\">crop150-probe-v1</a>, "
        f"<a href=\"{ctx.doc('crop150-rebaseline-v1.md')}\">crop150-rebaseline-v1</a>). The default stays at {cur} mm.</li>"
        f"<li>Depth: median gap between sampled slices {fov['gap_mm']['median']:.1f} mm (p90 {fov['gap_mm']['p90']:.1f}); "
        f"{fov['slices_per_plane']} slices per plane from the {fov['band'][0]:.0%}–{fov['band'][1]:.0%} band.</li>"
        "<li>MONOCHROME1 is unhandled, and the depth-window must be computed once per series (latent traps listed in "
        f"<a href=\"{ctx.doc('acquisition-metadata-trust.md')}\">acquisition-metadata-trust.md</a>).</li>"
        f"<li>Sources: <a href=\"{ctx.raw('crop-retention.json')}\">crop-retention.json</a>, "
        f"<a href=\"{ctx.raw('fov-and-depth-audit.json')}\">fov-and-depth-audit.json</a>. Tensor shapes after the crop are on the "
        f"<a href=\"{ctx.page('pipeline')}#architecture\">pipeline page</a>.</li>"
        "</ul>"
    )
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Crop table ({len(xs)} sizes)"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- slots


def slots_section(ctx) -> tuple[str, dict]:
    sid = "slots"
    j = D.read_json("acquisition-slots.json")
    if not j:
        return section(sid, "Acquisition slots", notice("acquisition-slots.json missing.")), {}
    hue = {"PD": "var(--g-teal)", "T1": "var(--g-violet)", "T2": "var(--g-sky)", "GRE": "var(--g-amber)", "UNK": "var(--g-anchor)"}
    rows = []
    for name, v in j["slot_fill"].items():
        plane, w, fs = [p.strip() for p in name.split("/")]
        rows.append({"id": "s-" + name.lower().replace(" / ", "-"), "group": w, "value": v,
                     "label": f"{plane} / {w} / {'FS' if fs == 'fatsat' else 'no FS'}"})
    rows.sort(key=lambda r: -r["value"])
    groups = {w: {"label": w, "color": c} for w, c in hue.items() if any(r["group"] == w for r in rows)}
    over = [r for r in rows if r["value"] >= 0.8]
    sel = j["selected_by_pipeline"]
    claim = (f"Only {len(over)} of {len(rows)} plane / weighting / fat-sat slots are filled in ≥80% of studies; the series "
             f"the pipeline picks are {_pct(sel['fluid_sensitive_share'])} fluid-sensitive")
    blob = {"rows": rows, "groups": groups}
    body = f"""<div data-role="filter"></div>
{chart('bars', f"Share of {j['studies']:,} studies with a series in each slot")}
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.barChart(sec.querySelector('[data-chart=bars]'), d.rows, {{groups: d.groups, fmt: '.1%', axisFmt: '.0%', ref: 0.8,
    refLabel: '80%', domain: [0, 1], xLabel: 'Studies with the slot filled', store: store, estLabel: 'Fill', labelWidth: 210, rowHeight: 20}});
}})();
</script>"""
    dec = j["decided_by"]
    caption = (f"{j['series']:,} series; weighting read from the description for {dec['description']:,} and from TR/TE timing "
               f"for {dec['timing']:,} ({_pct(dec['timing'] / j['series'], 0)}). Colour = weighting; FS = fat-suppressed.")
    dis = j["discarded_by_pipeline"]
    ws = sorted(set(sel["weighting"]) | set(dis["weighting"]))
    tbl = table(["Weighting", "Selected series", "Discarded series"],
                [[w, sel["weighting"].get(w, 0), dis["weighting"].get(w, 0)] for w in ws]
                + [["Fat-suppressed share", _pct(sel["fat_suppressed_share"]), _pct(dis["fat_suppressed_share"])],
                   ["Fluid-sensitive share", _pct(sel["fluid_sensitive_share"]), _pct(dis["fluid_sensitive_share"])]],
                num_cols=(1, 2))
    notes = (
        "<ul>"
        "<li>The series CSV's <code>Fluid_Sensitive</code> and <code>Fat_Suppression</code> flags are identical on every row, "
        "so slot selection gets the right answer for the wrong reason; nothing should rely on that coincidence.</li>"
        f"<li>Sources: <a href=\"{ctx.raw('acquisition-slots.json')}\">acquisition-slots.json</a>, "
        f"<a href=\"{ctx.doc('acquisition-metadata-trust.md')}\">acquisition-metadata-trust.md</a> (its 150 mm crop pick was later not adopted).</li>"
        "</ul>"
    )
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary="Selected vs discarded series"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- report length


def length_section(ctx) -> tuple[str, dict, dict]:
    sid = "length"
    j = D.read_json("report-length-statistics.json")
    qc = D.read_json("multilingual-text-qc.json")
    if not j:
        return section(sid, "Report length", notice("report-length-statistics.json missing.")), {}, {}
    limit = j["max_text_length"]
    pcts = [50, 75, 90, 95, 99]
    key = {50: "median"}
    cols = {"original (multilingual_raw)": ("var(--g-teal)", None, "Original (multilingual)"),
            "translated (english)": ("var(--g-violet)", "5 3", "Translated (English)")}
    series = []
    for name, c in j["cohorts"].items():
        color, dash, lab = cols.get(name, ("var(--g-anchor)", None, name))
        pts = [[p, c.get(key.get(p, f"p{p}"))] for p in pcts]
        series.append({"id": "len-" + name.split()[0], "label": f"{lab}, whole report", "group": "rep", "color": color,
                       "dash": dash, "points": pts})
        cs = c.get("conclusion_start_token") or {}
        series.append({"id": "con-" + name.split()[0], "label": f"{lab}, conclusion start", "group": "con",
                       "color": color, "dash": "2 3", "points": [[p, cs.get(key.get(p, f"p{p}"))] for p in pcts]})
    groups = {"rep": {"label": "Report length", "color": "var(--g-teal)"},
              "con": {"label": "Where the conclusion starts", "color": "var(--g-violet)"}}
    orig = j["cohorts"].get("original (multilingual_raw)") or next(iter(j["cohorts"].values()))
    claim = (f"{_pct(orig['over_limit_fraction'])} of original reports exceed E5's {limit}-token limit, and "
             f"{_pct(orig['conclusion_entirely_truncated_fraction'])} lose their whole conclusion")
    blob = {"series": series, "groups": groups, "refs": [{"y": limit, "label": f"{limit}-token limit"}]}
    body = f"""<div data-role="filter"></div>
{chart('pct', f"Token count at each percentile ({j['tokenizer']}, '{j['prefix'].strip()}' prefix)")}
<div data-role="legend"></div>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.lineChart(sec.querySelector('[data-chart=pct]'), d.series, {{groups: d.groups, yLabel: 'Tokens', yFmt: function (v) {{ return Math.round(v).toString(); }},
    xLabel: 'Percentile', xStep: 10, best: 'last', refs: d.refs, store: store}});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {{groups: d.groups, best: 'last',
    fmt: function (v) {{ return Math.round(v).toString(); }}, valuePrefix: 'p99 '}});
}})();
</script>"""
    caption = (f"Original: {orig['n']:,} reports; conclusions detected in {_pct(orig['conclusion_detection_rate'], 0)}, so the "
               f"conclusion lines cover only those. Everything past token {limit} is truncated before the text tower sees it.")
    tbl = ""
    if qc:
        langs = qc["audited_languages"]
        tbl = table(["Language (audited gold)", "Reports", "Median tokens", "p90 tokens", "Over 512", "Unknown tokens"],
                    [[k, v["count"], num(v["tokens_p50"], ".0f"), num(v["tokens_p90"], ".0f"), v["reports_over_512"],
                      v["unknown_tokens"]] for k, v in sorted(langs.items(), key=lambda kv: -kv[1]["count"])],
                    num_cols=(1, 2, 3, 4, 5))
    fc = (qc or {}).get("full_cohort", {})
    ns = [v["count"] for v in ((qc or {}).get("audited_languages") or {}).values()] or [0]
    nmin, nmax = min(ns), max(ns)
    notes = (
        "<ul>"
        f"<li>Tokenizer coverage: {fc.get('unknown_tokens', '?')} unknown tokens over the full cohort "
        f"({fc.get('reports_over_512', '?')} of {fc.get('count', '?')} reports over 512). Coverage does not establish medical "
        f"semantic quality, and only the 58 gold studies have audited source-language labels (table: n = {nmin} to {nmax} per language).</li>"
        f"<li>The {_pct(orig['over_limit_fraction'], 0)} truncation motivates the long-context (1,024-token) text-tower candidate in "
        f"<a href=\"{ctx.doc('next-model-candidates.md')}\">next-model-candidates.md</a>; text-tower results are on the "
        f"<a href=\"{ctx.page('text')}\">text page</a>.</li>"
        f"<li>Sources: <a href=\"{ctx.raw('report-length-statistics.json')}\">report-length-statistics.json</a>, "
        f"<a href=\"{ctx.raw('multilingual-text-qc.json')}\">multilingual-text-qc.json</a>.</li>"
        "</ul>"
    )
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary="Per-language token counts (audited gold reports)"), {f"data-{sid}": blob}, \
        {"over": orig["over_limit_fraction"]}


# --------------------------------------------------------------------------- page


def build(ctx) -> Page:
    i, i_blob = integrity_section(ctx)
    g, g_blob, g_kp = gold_section(ctx)
    lk, lk_blob, lk_kp = leak_section(ctx)
    inp, inp_blob = inputs_section(ctx)
    sl, sl_blob = slots_section(ctx)
    ln, ln_blob, ln_kp = length_section(ctx)
    blobs = {}
    for b in (i_blob, g_blob, lk_blob, inp_blob, sl_blob, ln_blob):
        blobs.update(b)
    man = D.read_json("full-manifest-migration.json") or {}
    pix = D.read_json("duplicate-group-pixel-verification.json") or {}
    kpis = []
    if man:
        sc = man["split_counts"]
        kpis.append({"label": "Studies", "value": f"{man['rows']:,}",
                     "sub": f"{sc['train']:,} / {sc['validation']} / {sc['test']} train / val / test (manifest, before quarantine)"})
    if g_kp:
        kpis.append({"label": "Gold-labelled studies", "value": str(g_kp["gold"]),
                     "sub": f"{g_kp['exposed']} exposed to the original-cohort encoders (2026-09-11); later held out"})
    if pix:
        kpis.append({"label": "Pixel-identical pairs", "value": str(pix.get("groups_confirmed_identical")),
                     "sub": "quarantined; study-level check only"})
    if lk_kp:
        kpis.append({"label": "Protocol leak", "value": f"{lk_kp['n_sig']}/{lk_kp['n']}",
                     "sub": "gold labels predicted by series count (p < 0.05)", "tone": "good"})
    if ln_kp:
        kpis.append({"label": "Reports over 512 tokens", "value": _pct(ln_kp["over"]), "sub": "original multilingual text"})
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=[i, g, lk, inp, sl, ln], data=blobs, kpis=kpis)


def summary() -> dict:
    scr = D.read_json("acquisition-duplicate-screen.json")
    pix = D.read_json("duplicate-group-pixel-verification.json")
    value = label = None
    if scr and pix:
        label = (f"pixel-identical pair{'s' if pix['groups_confirmed_identical'] != 1 else ''} in "
                 f"{scr['studies_fingerprinted']:,} studies, quarantined")
        value = str(pix["groups_confirmed_identical"])
    return {"stage": BY_SLUG[SLUG].number, "headline": "Duplicate screen", "value": value, "value_label": label,
            "caveat": "Study-level only: patient independence is unverifiable", "href": "#integrity"}
