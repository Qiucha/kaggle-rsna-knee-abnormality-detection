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
                     positives: dict | None = None, train_loss_title: str = "Training in-batch loss", default_mode: str = "delta",
                     lm_test_per_label: dict[str, dict] | None = None, reference_id: str | None = None,
                     lm_test_observed: dict[str, int] | None = None) -> tuple[str, dict, list[str]]:
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

    arm_ids = [a["id"] for a in arms]
    ref = next((a for a in arms if a["id"] == reference_id), None)
    blob["colIds"] = base_keys + arm_ids
    blob["referenceId"] = reference_id
    blob["heatmaps"] = {"gold": {"level": level, "unadapted": delta}}
    blob["modes"] = {"gold": [{"key": "level", "label": "AUROC"}]}
    if baselines:
        blob["modes"]["gold"].insert(0, {"key": "unadapted", "label": "Δ vs unadapted tower"})
    if ref:
        blob["heatmaps"]["gold"]["reference"] = [[None] * len(base_keys) + [
            a["per_label"].get(label) - ref["per_label"][label]
            if a["per_label"].get(label) is not None and ref["per_label"].get(label) is not None else None
            for a in arms] for label in labels]
        blob["modes"]["gold"].insert(0, {"key": "reference", "label": "Δ vs reference"})
    blob["rowLabels"] = {"gold": rows}
    if lm_test_per_label is not None:
        lm_level = [[None] * len(base_keys) + [lm_test_per_label.get(a["id"], {}).get(label) for a in arms]
                    for label in labels]
        blob["heatmaps"]["lm_test"] = {"level": lm_level}
        blob["modes"]["lm_test"] = [{"key": "level", "label": "AUROC"}]
        if ref:
            lm_ref = lm_test_per_label.get(reference_id, {})
            blob["heatmaps"]["lm_test"]["reference"] = [[None] * len(base_keys) + [
                lm_test_per_label.get(a["id"], {}).get(label) - lm_ref[label]
                if lm_test_per_label.get(a["id"], {}).get(label) is not None and lm_ref.get(label) is not None else None
                for a in arms] for label in labels]
            blob["modes"]["lm_test"].insert(0, {"key": "reference", "label": "Δ vs reference"})
        blob["rowLabels"]["lm_test"] = [f"{label} ({lm_test_observed[label]} observed)"
            if lm_test_observed and label in lm_test_observed else label for label in labels]
    default = "unadapted" if default_mode == "delta" else default_mode
    available = [choice["key"] for choice in blob["modes"]["gold"]]
    blob["defaultMode"] = default if default in available else available[0]
    blob["splitChoices"] = [{"key": "gold", "label": "Gold-58 (human)"}]
    if lm_test_per_label is not None:
        blob["splitChoices"].append({"key": "lm_test", "label": "LM-test (432)"})
    plots = "".join(f"<div>{chart(key, title)}</div>" for key, title in (
        ("train", f"{train_loss_title} by epoch"), ("val", "Validation full-pool loss by epoch"),
        ("itot", "Image → text validation Recall@1"), ("ttoi", "Text → image validation Recall@1")))
    body = (HELPERS + '<div data-role="filter"></div>'
            f'<h3>Loss and recall curves</h3><div class="grid-2">{plots}</div><div data-role="legend"></div>'
            '<p class="legend-hint">No pathology-AUROC-by-epoch curve: contrastive runs keep only their selected '
            'checkpoint, so per-epoch downstream AUROC cannot be recovered. Pathology AUROC is measured once, on the '
            'selected checkpoint (heatmap below).</p>'
            '<h3>Per-label pathology AUROC</h3><div data-role="split"></div><div data-role="mode"></div>'
            + '<div data-chart-scope>' + chart("heat", "Per-label AUROC by arm") + '</div>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  [['train', 'Training loss', '.2f', 'min'], ['val', 'Validation loss', '.2f', 'min'],
   ['itot', 'Image to text Recall@1', '.1%', 'max'], ['ttoi', 'Text to image Recall@1', '.1%', 'max']].forEach(function (spec) {
    if (d[spec[0]].length) Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.val, store, {groups: d.groups, best: 'min', fmt: '.3f'});
  var split = 'gold', mode = d.defaultMode;
  function draw() {
    var g = store.get().group;
    var idx = d.cols.map(function (c, j) { return j; }).filter(function (j) {
      if ((split !== 'gold' || mode !== 'level') && d.colGroups[j] === 'anchor') return false;
      if (mode === 'reference' && d.colIds[j] === d.referenceId) return false;
      return !g || g === 'all' || d.colGroups[j] === g || d.colGroups[j] === 'anchor'
             || (mode === 'level' && d.colIds[j] === d.referenceId);
    });
    var matrix = d.heatmaps[split][mode].map(function (row) { return idx.map(function (j) { return row[j]; }); });
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var title = 'Per-label ' + (split === 'gold' ? 'gold-58' : 'LM-test') + ' AUROC';
    if (mode === 'reference') title += ' difference vs reference';
    else if (mode === 'unadapted') title += ' difference vs unadapted tower';
    else title += ' by arm';
    el.previousElementSibling.textContent = title;
    var opts = {rows: d.rowLabels[split], cols: idx.map(function (j) { return d.cols[j]; }), rowIds: d.labels,
                fmt: mode === 'level' ? '.3f' : '+.3f', ariaLabel: title};
    el.setAttribute('aria-label', title);
    if (mode !== 'level') { opts.diverging = true; opts.peakPerRow = false; }
    Charts.heatmap(el, matrix, opts);
  }
  function drawModes() {
    var choices = d.modes[split];
    if (!choices.some(function (choice) { return choice.key === mode; })) mode = choices[0].key;
    var el = sec.querySelector('[data-role=mode]');
    var fresh = el.cloneNode(false); el.parentNode.replaceChild(fresh, el);
    AdaptUI.seg(fresh, 'Cells', choices, mode, function (key) { mode = key; draw(); });
  }
  if (d.splitChoices.length > 1) AdaptUI.seg(sec.querySelector('[data-role=split]'), 'Scored on',
    d.splitChoices, split, function (key) { split = key; drawModes(); draw(); });
  drawModes(); draw();
  var last = store.get().group;
  store.on(function (st) { if (st.group !== last) { last = st.group; draw(); } });"""))
    return body, blob, missing
