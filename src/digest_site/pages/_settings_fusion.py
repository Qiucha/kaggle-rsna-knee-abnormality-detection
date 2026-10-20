"""Settings fusion: do the non-fat-suppressed series add label information to u9-b1? (frozen-feature probe)

Sources: digest/settings-fusion-v1.json (scores and paired deltas vs arm A) and
digest/settings-fusion-curves-v1.json (per-epoch head histories, from the AUROC-logging rerun).
"""

from __future__ import annotations

from .. import data
from ..shell import chart, esc, num, section, table
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card

SRC, CURVES, MD = "settings-fusion-v1.json", "settings-fusion-curves-v1.json", "settings-fusion-v1.md"
# The pretrained encoder is the clean test (it saw neither slot), so it reads first.
ENCODERS = [("pretrained", "Pretrained OmniRad-base (clean test)", "var(--g-teal)"),
            ("finetuned", "u9-b1's fine-tuned encoder (favours A)", "var(--g-violet)")]
GROUPS = {key: {"label": label, "color": color} for key, label, color in ENCODERS}
# Arms from the smallest change to the largest; A is the reference every other arm is paired against.
ARMS = [("A_primary_mean", "A · primary planes (reference)"),
        ("B_concat_secondary", "B · + secondary series"),
        ("C_slot_mean", "C · mean over all six slots"),
        ("D_label_attention", "D · per-label attention")]
DASHES = {"A_primary_mean": None, "B_concat_secondary": "7 4", "C_slot_mean": "2 3", "D_label_attention": "10 3 2 3"}


def _delta(paired: dict) -> dict:
    i = paired["seed_averaged_interval"]
    return {"difference": paired["mean_delta"], "ci95": [i["low"], i["high"]]}


def results(ctx, j) -> tuple[str, dict]:
    rows, trs = [], []
    for enc, enc_label, _ in ENCODERS:
        arms = j["encoders"][enc]["arms"]
        for arm, arm_label in ARMS:
            a = arms[arm]
            p = a.get("paired_vs_A")
            if p:
                rows.append({"id": f"{enc}-{arm}", "label": f"{arm_label[:1]} − A · {enc}", "group": enc,
                             "lm": _delta(p["lm_test"]), "gold": _delta(p["gold"])})
            trs.append([enc_label, arm_label, num(a["mean_lm_test"]), ci_text(_delta(p["lm_test"])) if p else "reference",
                        num(a["mean_gold"]), ci_text(_delta(p["gold"])) if p else "reference"])
    clean = next(r for r in rows if r["id"] == "pretrained-B_concat_secondary")
    claim = (f"Settings fusion: the non-fat-suppressed series add no label information "
             f"({num(clean['lm']['difference'], '.4f', True)} LM-test with the pretrained encoder)")
    share = j["encoders"]["pretrained"]["secondary_present_share"]["lm_test"]
    lead = ('<p><b>Question.</b> The pipeline reads one fat-suppressed series per plane and drops the other '
            'weighting (42% of all series). Do those series carry label information the primary slots miss? '
            'Frozen features from six slots per study (primary + opposite-flag secondary per plane) feed small '
            'heads, three seeds each. Secondary coverage on LM-test: sagittal '
            f'{num(share[0], ".0%")}, coronal {num(share[1], ".0%")}, axial {num(share[2], ".0%")}.</p>'
            '<p><b>Result:</b> a null. B under the pretrained encoder, the clean test, is zero; under u9-b1\'s own '
            'encoder every arm sits at or below A. Keep the three-slot input; a six-slot fine-tune is not worth '
            'building.</p>')
    notes = ('<p>Δ is each arm minus arm A on the same encoder; intervals are a 1,000-resample study bootstrap of '
             'the seed-averaged predictions. The fine-tuned encoder was trained on the primary slots of the same '
             'training studies the heads use, which favours A; the pretrained encoder has seen neither slot. '
             'Pretrained D is the only positive row, but both intervals span zero and its per-label gains are not '
             'where the extra series should help (see the heatmap below). Gate: u9-b1\'s own head on arm-A features '
             'reproduced its saved predictions to within 0.0002.</p>'
             f'<p>Sources: <a href="{ctx.raw(SRC)}">scores and paired deltas</a>, '
             f'<a href="{ctx.doc(MD)}">write-up</a>.</p>')
    tbl = table(["Encoder", "Head arm", "LM-test", "Δ LM-test vs A [95% CI]", "Gold-58", "Δ gold vs A [95% CI]"],
                trs, num_cols=(2, 4))
    return delta_card("settings-fusion", claim, rows, GROUPS, x_label="Arm minus arm A (primary planes only)",
                      lead=lead, notes=notes, table_html=tbl, table_summary="All eight arms")


def curves(ctx, c) -> tuple[str, dict]:
    sid = "settings-fusion-curves"
    seeds = sorted({run["seed"] for enc in c["encoders"].values() for arm in enc.values() for run in arm})
    blob = {"groups": GROUPS, "seeds": [str(s) for s in seeds]}
    for key, metric in (("train", "train_masked_bce"), ("val", "validation_masked_bce"),
                        ("auc", "validation_macro_auroc")):
        blob[key] = {str(seed): [{"id": f"{enc}-{arm}", "label": f"{label} · {enc}", "group": enc,
                                  "dash": DASHES[arm],
                                  "points": [[h["epoch"], h[metric]] for h in run["history"]]}
                                 for enc, _, _ in ENCODERS for arm, label in ARMS
                                 for run in c["encoders"][enc][arm] if run["seed"] == seed]
                     for seed in seeds}
    body = (HELPERS + '<div data-role="seed"></div><div data-role="filter"></div><div class="grid-2">'
            + ''.join(f'<div>{chart(k, t)}</div>' for k, t in (
                ("train", "Head training masked BCE by epoch"),
                ("val", "Validation masked BCE by epoch (selects the epoch)"),
                ("auc", "Validation macro AUROC by epoch")))
            + '</div><div data-role="legend"></div>'
            + '<p class="legend-hint">Colour is the encoder, line style the head arm (solid = A). '
              'Switch seeds above; click lines or legend entries to highlight an arm in every chart.</p>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var seed = d.seeds[0];
  function draw() {
    [['train', 'Training BCE', '.3f', 'min'], ['val', 'Validation BCE', '.3f', 'min'],
     ['auc', 'Validation AUROC', '.3f', 'max']].forEach(function (spec) {
      Charts.lineChart(AdaptUI.remount(sec.querySelector('[data-chart=' + spec[0] + ']')), d[spec[0]][seed],
        {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
    });
    Charts.legend(AdaptUI.remount(sec.querySelector('[data-role=legend]')), d.val[seed], store,
      {groups: d.groups, best: 'min', fmt: '.3f'});
  }
  AdaptUI.seg(sec.querySelector('[data-role=seed]'), 'Head seed',
    d.seeds.map(function (s) { return {key: s, label: s}; }), seed, function (k) { seed = k; draw(); });
  draw();"""))
    claim = ("Settings-fusion heads: the pretrained encoder's arms train and validate alike; "
             "the fine-tuned encoder's heads peak within a few epochs")
    notes = ("<p>Heads train on soft targets; validation BCE (hard LM verdicts, 426 studies) picks the epoch, with "
             "patience 15. Curves come from a rerun of the same heads with validation AUROC logged; the rerun's "
             "test scores match the committed table (checked per arm).</p>")
    return section(sid, claim, body, notes=notes), {f"data-{sid}": blob}


def per_label(ctx, j) -> tuple[str, dict]:
    sid = "settings-fusion-labels"
    cols, colnames = [], []
    for enc, _, _ in ENCODERS:
        for arm, label in ARMS[1:]:
            cols.append((enc, arm))
            colnames.append(f"{label[:1]} − A · {enc}")
    labels = list(j["encoders"]["pretrained"]["arms"]["B_concat_secondary"]["paired_vs_A"]["lm_test"]["per_label_delta"])
    blob = {"rows": labels, "cols": colnames}
    for split in ("gold", "lm_test"):
        blob[split] = [[j["encoders"][enc]["arms"][arm]["paired_vs_A"][split]["per_label_delta"].get(l)
                        for enc, arm in cols] for l in labels]
    body = (HELPERS + '<div data-role="split"></div><div data-chart-scope>'
            + chart("heat", "Per-label Δ AUROC vs arm A") + '</div>'
            + section_script(sid, """
  var split = 'gold';
  function draw() {
    Charts.heatmap(AdaptUI.remount(sec.querySelector('[data-chart=heat]')), d[split],
      {rows: d.rows, cols: d.cols, rowIds: d.rows, fmt: '+.3f', diverging: true, peakPerRow: false,
       ariaLabel: 'per-label ' + (split === 'gold' ? 'gold-58' : 'LM-test') + ' delta heatmap'});
  }
  AdaptUI.seg(sec.querySelector('[data-role=split]'), 'Scored on',
    [{key: 'gold', label: 'Gold-58 (human)'}, {key: 'lm_test', label: 'LM-test (432)'}], 'gold',
    function (k) { split = k; draw(); });
  draw();"""))
    caption = ("Columns: pretrained encoder first (B, C, D minus A), then the fine-tuned encoder. The extra series "
               "should help meniscus (sagittal PD) and fracture/marrow (T1); a real signal would concentrate there. "
               "Point estimates of the seed-averaged predictions, no intervals.")
    return (section(sid, "Per label: no arm gains where the non-fat-suppressed series should help", body,
                    caption=caption), {f"data-{sid}": blob})


def build(ctx) -> tuple[list[str], dict]:
    j = data.read_json(SRC)
    if not j:
        return [], {}
    sections, blobs = [], {}
    parts = [results(ctx, j)]
    c = data.read_json(CURVES)
    if c:
        parts.append(curves(ctx, c))
    parts.append(per_label(ctx, j))
    for html, blob in parts:
        sections.append(html)
        blobs.update(blob)
    return sections, blobs
