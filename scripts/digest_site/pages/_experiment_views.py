"""The three views every experiment section carries (page-local helper, shared by topic pages).

``experiment_views(sid, arms, groups, labels=..., baselines=...)`` returns ``(body_html, blob, missing)``:

1. Loss curves: training (in-batch) and validation (full pool) loss by epoch.
2. Recall curves: image -> text and text -> image validation Recall@1 by epoch.
3. Per-label heatmap: gold AUROC per label x arm, shown by default as the change against the arm's
   unadapted (pretrained) image tower, with a toggle to raw AUROC (which adds the unadapted columns).

``arms``: [{id, label, short?, group, curve_root, curve_run, per_label: {label: auroc}, baseline: key|None}]
``baselines``: {key: {label, short?, per_label}}; ``short`` is the heatmap column header; ``positives``: {label: n} for the row labels.
``missing`` lists arms with no curve snapshot (for the section's notes).
"""

from __future__ import annotations

from .. import data
from ..shell import chart
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script

CURVE_METRICS = {"train": "train_loss", "val": "validation_loss",
                 "itot": "image_to_text_r@1", "ttoi": "text_to_image_r@1"}


def experiment_views(sid: str, arms: list[dict], groups: dict, *, labels: list[str], baselines: dict,
                     positives: dict | None = None, train_loss_title: str = "Training in-batch loss") -> tuple[str, dict, list[str]]:
    series = {key: [] for key in CURVE_METRICS}
    missing = []
    for a in arms:
        curves = data.load_curves(a["curve_root"])
        if not curves or a["curve_run"] not in curves.get("runs", {}):
            missing.append(a["label"])
            continue
        for key, metric in CURVE_METRICS.items():
            points = data.curve_series(curves, a["curve_run"], metric)
            if points:
                series[key].append({"id": a["id"], "label": a["label"], "group": a["group"], "points": points})

    rows = [f"{l} ({positives[l]})" if positives and l in positives else l for l in labels]
    base_keys = [k for k in baselines if any(a.get("baseline") == k for a in arms)]
    cols = ([b.get("short", b["label"]) for k, b in baselines.items() if k in base_keys]
            + [a.get("short", a["label"]) for a in arms])
    col_groups = ["anchor"] * len(base_keys) + [a["group"] for a in arms]
    level = [[baselines[k]["per_label"].get(l) for k in base_keys] + [a["per_label"].get(l) for a in arms] for l in labels]
    delta = [[None] * len(base_keys)
             + [(a["per_label"][l] - baselines[a["baseline"]]["per_label"][l])
                if a.get("baseline") in baselines and l in a["per_label"] else None for a in arms]
             for l in labels]
    blob = {"groups": groups, **series, "rows": rows, "labels": labels, "cols": cols, "colGroups": col_groups,
            "level": level, "delta": delta}

    plots = "".join(f"<div>{chart(key, title)}</div>" for key, title in (
        ("train", f"{train_loss_title} by epoch"), ("val", "Validation full-pool loss by epoch"),
        ("itot", "Image → text validation Recall@1"), ("ttoi", "Text → image validation Recall@1")))
    body = (HELPERS + '<div data-role="filter"></div>'
            f'<h3>Loss and recall curves</h3><div class="grid-2">{plots}</div><div data-role="legend"></div>'
            '<h3>Per-label gold AUROC</h3><div data-role="mode"></div>'
            # own selection scope: heatmap rows are labels, not arms, so hovering them must not dim the curves
            + '<div data-chart-scope>' + chart("heat", "Per-label gold AUROC by arm") + '</div>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  [['train', 'Training loss', '.2f', 'min'], ['val', 'Validation loss', '.2f', 'min'],
   ['itot', 'I\\u2192T Recall@1', '.1%', 'max'], ['ttoi', 'T\\u2192I Recall@1', '.1%', 'max']].forEach(function (spec) {
    if (d[spec[0]].length) Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.val, store, {groups: d.groups, best: 'min', fmt: '.3f'});
  var mode = 'delta';
  function draw() {
    var g = store.get().group;
    var idx = d.cols.map(function (c, j) { return j; }).filter(function (j) {
      if (mode === 'delta' && d.colGroups[j] === 'anchor') return false;
      return !g || g === 'all' || d.colGroups[j] === g || d.colGroups[j] === 'anchor'; });
    var m = (mode === 'level' ? d.level : d.delta).map(function (row) { return idx.map(function (j) { return row[j]; }); });
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var o = {rows: d.rows, cols: idx.map(function (j) { return d.cols[j]; }), rowIds: d.labels,
             fmt: mode === 'level' ? '.2f' : '+.2f', ariaLabel: 'per-label gold AUROC heatmap'};
    if (mode === 'delta') { o.diverging = true; o.peakPerRow = false; }
    Charts.heatmap(el, m, o);
  }
  AdaptUI.seg(sec.querySelector('[data-role=mode]'), 'Cells',
    [{key: 'delta', label: '\\u0394 vs unadapted tower'}, {key: 'level', label: 'AUROC'}], 'delta', function (k) { mode = k; draw(); });
  draw();
  var last = store.get().group;
  store.on(function (st) { if (st.group !== last) { last = st.group; draw(); } });"""))
    return body, blob, missing
