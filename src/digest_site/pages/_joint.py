"""One-stage label + contrastive loss, with its matched label-only comparator."""

from __future__ import annotations

from .. import data
from ..shell import chart, esc, num, section, table
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card

SRC = "joint-loss-v1.json"
GROUPS = {"joint": {"label": "One-stage joint loss", "color": "var(--g-rose)"},
          "label": {"label": "Label-only references", "color": "var(--g-sky)"},
          "staged": {"label": "Two-stage CLIP → labels", "color": "var(--g-teal)"}}


def results(ctx, j) -> tuple[str, dict]:
    pairs = [  # noise floor first, then the two-stage reference, then the joint comparisons
        ("run5 seed2027 - run5 (label, u3)", "Run-5 seed 2027 − seed 2026 (noise floor)", "label"),
        ("CLIP->label (u3) - run5 (label, u3)", "CLIP → labels u3 − label-only u3", "staged"),
        (j["primary_pair"], "Joint u9 − label-only u9", "joint"),
        ("joint (u9) - CLIP->label (u3)", "Joint u9 − CLIP → labels u3", "joint"),
    ]
    rows = [{"id": str(i), "label": label, "group": group,
             "lm": j["paired"]["lm_test"][key], "gold": j["paired"]["gold"][key]}
            for i, (key, label, group) in enumerate(pairs)]
    primary = next(r for r in rows if r["group"] == "joint")
    claim = (f"One-stage joint loss: no clear LM-test gain over label-only u9 "
             f"({num(primary['lm']['difference'], '.4f', True)}); gold-58 falls "
             f"{num(primary['gold']['difference'], '.4f', True)}")
    recipe = j["recipe"]
    lead = (f'<p><b>One shared image encoder, two losses in each training step:</b> '
            f'masked label BCE + {recipe["contrastive_weight"]:g} × image-to-report contrastive loss. '
            f'OmniRad-base, {recipe["image_unfreeze_last"]} unfrozen blocks, soft labels, '
            f'K{recipe["windows_per_plane"]} windows per plane at {recipe["image_size"]} px. '
            f'The frozen multilingual-E5-base text bank contains {j["text_bank"]["reports"]:,} distinct training '
            f'reports, so batch size 1 still has negatives; the text projection trains.</p>'
            '<p><b>Result:</b> keep label-only u9 as the reference. The primary pair holds image depth and batch '
            'size fixed; its LM-test interval crosses zero and its gold-58 interval is entirely negative. '
            'The joint-vs-two-stage row changes depth (u9 vs u3), so it cannot isolate the effect of combining losses.</p>')
    scores = table(["Training", "LM-test AUROC", "Gold-58 AUROC", "Selected epoch", "Validation BCE"],
                   [[r["title"], num(j["point"]["lm_test"][r["key"]]),
                     num(j["point"]["gold"][r["key"]]), r["best_epoch"], num(r["validation_masked_bce"])]
                    for r in j["runs"]], num_cols=(1, 2, 3, 4))
    paired = table(["Candidate − reference", "Δ LM-test [95% CI]", "Δ gold-58 [95% CI]"],
                   [[r["label"], ci_text(r["lm"]), ci_text(r["gold"])] for r in rows])
    notes = (f'<p>Paired study bootstrap, {j["bootstrap"]["resamples"]:,} resamples, '
             f'bootstrap seed {j["bootstrap"]["seed"]}. The comparison script verifies saved study order and '
             'gold targets against run 5. Epoch selection uses validation hard-verdict masked BCE, with retrieval '
             'Recall@1 only breaking ties. LM-test is 432 non-gold studies scored against LM verdicts; gold-58 '
             'is exploratory and has been inspected repeatedly. One joint training seed and one loss weight '
             'were tested; these intervals quantify study sampling, not training-seed uncertainty. '
             'This result does not rule out other loss weights or text-bank recipes. No Kaggle score is recorded '
             'for this joint run.</p>'
             f'<p>Sources: <a href="{ctx.raw(SRC)}">snapshot and source hashes</a>, '
             f'<a href="{ctx.doc("joint-loss-v1.md")}">result write-up</a>.</p>')
    return delta_card("joint-loss", claim, rows, GROUPS, x_label="Candidate minus its stated reference",
                      lead=lead + scores, notes=notes, table_html=paired, table_summary="Paired comparison table")


# Runs in reading order: the label-only references, the two-stage run, then the joint run.
ORDER = ["run5 (label, u3)", "run5 seed2027", "u9-b1 (label, u9)", "CLIP->label (u3)", "joint (u9)"]
JOINT, REFERENCE = "joint (u9)", "u9-b1 (label, u9)"


def _group(key: str) -> str:
    return "joint" if key == JOINT else "staged" if key.startswith("CLIP") else "label"


def _runs(j) -> list[dict]:
    by_key = {r["key"]: r for r in j["runs"]}
    return [by_key[k] for k in ORDER if k in by_key] + [r for r in j["runs"] if r["key"] not in ORDER]


def curves(ctx, j) -> tuple[str, dict]:
    sid = "joint-loss-curves"
    runs = _runs(j)
    blob = {"groups": GROUPS, "defaultPinned": [JOINT, REFERENCE]}
    for key, metric in (("train", "train_bce"), ("val", "validation_bce"), ("auc", "validation_macro_auroc"),
                        ("contrastive", "train_contrastive_loss"), ("itot", "image_to_text_r@1"),
                        ("ttoi", "text_to_image_r@1")):
        blob[key] = [{"id": r["key"], "label": r["title"], "group": _group(r["key"]),
                      "points": [[h["epoch"], h[metric]] for h in r["history"] if metric in h]}
                     for r in runs if any(metric in h for h in r["history"])]
    joint = next(r for r in runs if r["key"] == JOINT)
    joint_auc = any("validation_macro_auroc" in h for h in joint["history"])
    body = (HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            + ''.join(f'<div>{chart(key, title)}</div>' for key, title in (
                ("train", "Training label BCE by epoch"),
                ("val", "Validation label BCE by epoch · selects the checkpoint"),
                ("auc", "Validation macro AUROC by epoch"
                        + ("" if joint_auc else " · reference runs (joint run did not log it)")),
                ("contrastive", "Joint training image-to-report loss · unweighted"),
                ("itot", "Joint validation image → text Recall@1 · 426 candidates"),
                ("ttoi", "Joint validation text → image Recall@1 · 426 candidates"))) + '</div>'
            + '<div data-role="legend"></div>'
            + '<p class="legend-hint">Highlighted on load: the joint run and its matched label-only u9. '
              'Click lines or legend entries to highlight runs in every chart.</p>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  store.set({pinned: d.defaultPinned});
  [['train', 'Training BCE', '.3f', 'min'], ['val', 'Validation BCE', '.3f', 'min'],
   ['auc', 'Validation AUROC', '.3f', 'max'], ['contrastive', 'Contrastive loss', '.3f', 'min'],
   ['itot', 'Image to text R@1', '.1%', 'max'], ['ttoi', 'Text to image R@1', '.1%', 'max']].forEach(function (spec) {
    Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.val, store, {groups: d.groups, best: 'min', fmt: '.3f'});"""))
    claim = (f'Joint loss completed {joint["epochs_completed"]}/{j["recipe"]["epochs"]} epochs; '
             f'validation label BCE selects epoch {joint["best_epoch"]} at {num(joint["validation_masked_bce"])}')
    notes = ('<p>Training uses soft targets; checkpoint selection uses hard LM verdicts. '
             + ('' if joint_auc else
                '<strong>The joint run has no validation-AUROC curve:</strong> its trainer logged validation BCE and '
                'retrieval but not AUROC, and only the best and last checkpoints were kept, so the curve cannot be '
                'recovered without retraining (about 6.5 GPU hours). The AUROC chart shows the label-only and '
                'two-stage runs it is compared with. ')
             + 'Contrastive training loss is image-to-report cross-entropy over the full training text bank; '
             'validation retrieval is evaluated on paired images/reports in the 426-study validation split. '
             'Their losses use different candidate pools and should not be compared as train/validation curves. '
             'Recall@1 does not select the checkpoint except to break a BCE tie.</p>')
    return section(sid, claim, body, notes=notes), {f"data-{sid}": blob}


def per_label(ctx, j) -> tuple[str, dict]:
    sid = "joint-loss-labels"
    runs = [r for r in _runs(j) if r["key"] in j["per_label"]["lm_test"]]
    labels = list(j["per_label"]["lm_test"][JOINT])
    cols = [r["key"] for r in runs]
    blob = {"rows": labels, "cols": [r["title"] for r in runs],
            "refCol": cols.index(REFERENCE), "defaultMode": "delta",
            "modeChoices": [{"key": "delta", "label": "Δ vs label-only u9"}, {"key": "level", "label": "AUROC"}]}
    for split in ("gold", "lm_test"):
        pl = j["per_label"][split]
        blob[split] = {"level": [[pl[c][l] for c in cols] for l in labels],
                       "delta": [[pl[c][l] - pl[REFERENCE][l] for c in cols] for l in labels]}
    body = (HELPERS + '<div data-role="split"></div><div data-role="mode"></div>'
            + '<div data-chart-scope>' + chart("heat", "Per-label AUROC by run") + '</div>'
            + section_script(sid, """
  var split = 'gold', mode = d.defaultMode;
  function draw() {
    var keep = d.cols.map(function (c, i) { return i; }).filter(function (i) { return mode === 'level' || i !== d.refCol; });
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var o = {rows: d.rows, cols: keep.map(function (i) { return d.cols[i]; }), rowIds: d.rows,
             fmt: mode === 'level' ? '.3f' : '+.3f',
             ariaLabel: 'per-label ' + (split === 'gold' ? 'gold-58' : 'LM-test') + ' AUROC heatmap'};
    if (mode !== 'level') { o.diverging = true; o.peakPerRow = false; }
    Charts.heatmap(el, d[split][mode].map(function (row) { return keep.map(function (i) { return row[i]; }); }), o);
  }
  AdaptUI.seg(sec.querySelector('[data-role=split]'), 'Scored on',
    [{key: 'gold', label: 'Gold-58 (human)'}, {key: 'lm_test', label: 'LM-test (432)'}], 'gold',
    function (k) { split = k; draw(); });
  AdaptUI.seg(sec.querySelector('[data-role=mode]'), 'Cells', d.modeChoices, d.defaultMode,
    function (k) { mode = k; draw(); });
  draw();"""))
    rows = []
    for label in labels:
        cells = [esc(label)]
        for split in ("lm_test", "gold"):
            a, b = j["per_label"][split][JOINT][label], j["per_label"][split][REFERENCE][label]
            cells += [num(a), num(b), num(a - b, '.4f', True)]
        rows.append(cells)
    caption = ("Columns run label-only (run 5, its seed replicate, u9) → two-stage CLIP → labels → joint. Cells default "
               "to each run minus label-only u9 on the same label; the seed-replicate column shows how far a label "
               "moves from training noise alone. Per-label values are point estimates without intervals.")
    return (section(sid, "Per label: joint loss trades meniscus and OA performance for effusion and synovitis",
                    body, caption=caption,
                    table_html=table(["Label", "Joint LM", "Label-only LM", "Δ LM", "Joint gold",
                                      "Label-only gold", "Δ gold"], rows, num_cols=(1, 2, 3, 4, 5, 6)),
                    table_summary="All 12 labels · joint vs label-only u9"), {f"data-{sid}": blob})


def build(ctx) -> tuple[list[str], dict]:
    j = data.read_json(SRC)
    if not j:
        return [], {}
    result, result_data = results(ctx, j)
    curve, curve_data = curves(ctx, j)
    labels, label_data = per_label(ctx, j)
    return [result, curve, labels], {**result_data, **curve_data, **label_data}
