"""Training-time image augmentation on the 30-epoch lateral oversample arm (A30), vs A30 itself.

Source: digest/finetune-image-augmentation-v1.json, written by scripts/analysis/compare_image_augmentation.py
(worktree t3code-8d0cbe1e, branch feat/label-image-augmentation, uncommitted).
"""

from __future__ import annotations

from .. import data
from ..shell import chart, esc, group_major, num, section, table
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card

SRC = "finetune-image-augmentation-v1.json"
BASE, REF, AUG = "u9-b1", "oversample+noise ep30", "oversample+noise ep30 aug"
SEED, SEED_REF = "run5 seed2027", "run5"
GROUPS = {"noise": {"label": "Training-seed noise floor", "color": "var(--g-sky)"},
          "aug": {"label": "Image augmentation (one factor)", "color": "var(--g-amber)"},
          "context": {"label": "Against u9-b1 (context)", "color": "var(--g-rose)"}}
CURVE_GROUPS = {"ref": {"label": "u9-b1 and A30 (no augmentation)", "color": "var(--g-sky)"},
                "aug": {"label": "A30 + image augmentation", "color": "var(--g-amber)"}}
# Every run shown, in reading order.
ORDER = [SEED_REF, SEED, BASE, REF, AUG]


def _pair(j, split, first, second):
    return j["paired"][split].get(f"{first} - {second}")


def results(ctx, j) -> tuple[str, dict]:
    pairs = [(SEED, SEED_REF, "Run-5 seed 2027 − seed 2026 (noise floor)", "noise"),
             (AUG, REF, "A30 + augmentation − A30", "aug"),
             (AUG, BASE, "A30 + augmentation − u9-b1", "context"),
             (REF, BASE, "A30 − u9-b1", "context")]
    rows = group_major([{"id": str(i), "label": label, "group": group,
                         "lm": _pair(j, "lm_test", a, b), "gold": _pair(j, "gold", a, b)}
                        for i, (a, b, label, group) in enumerate(pairs)], GROUPS)
    by_key = {r["key"]: r for r in j["runs"]}
    aug, ref = by_key[AUG], by_key[REF]
    lm, gold = _pair(j, "lm_test", AUG, REF), _pair(j, "gold", AUG, REF)
    pl = j["per_label"]

    def label_delta(split, label):
        return num(pl[split][AUG][label] - pl[split][REF][label], ".3f", True)

    claim = (f"Image augmentation does not help: {num(j['point']['lm_test'][AUG])} LM-test, "
             f"{num(lm['difference'], '.4f', True)} against the same recipe without it; A30 stays the best single run")
    aug_cfg = aug.get("image_augmentation", {})
    lead = (
        '<p><b>Question:</b> does training-time image augmentation make the best single recipe generalise better? '
        'The trial copies the 30-epoch lateral oversample + noise arm (<b>A30</b>: u9-b1, one extra draw per epoch for '
        'every study positive on Lateral Meniscus, Lateral OA or Synovitis, N(0, 0.1) noise on the lateral soft targets, '
        'cosine over 30 epochs, patience off, divergence stop) and changes only one thing: it augments the training images.</p><ul>'
        f'<li><b>Transforms:</b> rotation ±{num(aug_cfg.get("rotation_degrees", 8), ".0f")}°, zoom-in '
        f'1.00–{num(1 + aug_cfg.get("zoom", 0.08), ".2f")}, shift ±{num(aug_cfg.get("shift", 0.05), ".2f")} in normalised '
        f'grid coordinates, multiplicative intensity gain {num(1 - aug_cfg.get("intensity", 0.1), ".2f")}–'
        f'{num(1 + aug_cfg.get("intensity", 0.1), ".2f")} clamped to [0, 1]. Bilinear resampling with border padding, '
        'no flips. These reuse the contrastive pipeline\'s helper.</li>'
        '<li><b>Scope:</b> one fresh draw per study and plane on every read, oversampled repeats included. The three '
        'channels and all eight windows of a plane share that draw; the three planes draw independently. Validation, '
        'LM-test, gold-58 and exported inference are never augmented.</li>'
        f'<li><b>Randomness:</b> a separate generator (seed {aug_cfg.get("seed", 2028)}, i.e. training seed + 2), so '
        'sampling order, dropout and target noise follow A30\'s random streams.</li></ul>')
    verdict = (
        f'<p><b>Result: no gain, not adopted.</b> Against A30, augmentation moves {ci_text(lm)} on LM-test and '
        f'{ci_text(gold)} on gold-58. Both intervals cross zero, and both point estimates sit inside the run-5 seed '
        'floor (about ±0.01 LM-test, ±0.02 gold-58), with gold leaning negative. Against u9-b1 it is '
        f'{ci_text(_pair(j, "lm_test", AUG, BASE))} LM-test and {ci_text(_pair(j, "gold", AUG, BASE))} gold-58, inside the '
        '−0.024 gold veto, so it is not vetoed against u9-b1. A30 (LM-test '
        f'{num(j["point"]["lm_test"][REF])}, public 0.890) remains the best single model.</p>'
        f'<p><b>Training barely changed.</b> Both runs used all {aug["epochs_completed"]} epochs (the divergence stop '
        f'never fired) and both select epoch {aug["best_epoch"]}. Validation BCE at the selected epoch is '
        f'{num(aug["validation_masked_bce"])} against A30\'s {num(ref["validation_masked_bce"])}. The final training BCE, '
        f'measured on augmented inputs, is {num(aug["history"][-1]["train_masked_bce"])} against A30\'s '
        f'{num(ref["history"][-1]["train_masked_bce"])}. The train–validation gap is the same, so these magnitudes did '
        'not act as a regulariser. Per label, the larger moves are losses: MCL '
        f'({label_delta("lm_test", "MCL")} LM-test, {label_delta("gold", "MCL")} gold) and Effusion on gold '
        f'({label_delta("gold", "Effusion")}); the largest rise is ACL ({label_delta("lm_test", "ACL")} LM-test). The three lateral target labels do not move (Lateral Meniscus '
        f'{ci_text(j["paired_target_labels"]["lm_test"]["Lateral Meniscus"][f"{AUG} - {REF}"])} LM-test).</p>')
    scores = table(["Training", "LM-test AUROC", "Gold-58 AUROC", "Best epoch", "Epochs run", "Validation BCE",
                    "Wall time (h)"],
                   [[by_key[k]["title"], num(j["point"]["lm_test"][k]), num(j["point"]["gold"][k]),
                     by_key[k]["best_epoch"], by_key[k]["epochs_completed"], num(by_key[k]["validation_masked_bce"]),
                     num(by_key[k]["seconds"] / 3600, ".1f") if by_key[k].get("seconds") else "—"]
                    for k in ORDER if k in by_key], num_cols=(1, 2, 3, 4, 5, 6))
    target_rows = []
    for label in j["target_labels"]:
        cells = [esc(label)]
        for a, b in ((SEED, SEED_REF), (AUG, REF), (AUG, BASE)):
            for split in ("lm_test", "gold"):
                cells.append(ci_text(j["paired_target_labels"][split][label].get(f"{a} - {b}")))
        target_rows.append(cells)
    target_table = ('<p><b>The three lateral target labels alone</b> (paired study bootstrap on that label only):</p>'
                    + table(["Label", "Seed noise LM", "Seed noise gold", "Aug − A30 LM", "Aug − A30 gold",
                             "Aug − u9-b1 LM", "Aug − u9-b1 gold"], target_rows, raw_html=True))
    paired = table(["Candidate − reference", "Δ LM-test [95% CI]", "Δ gold-58 [95% CI]"],
                   [[r["label"], ci_text(r["lm"]), ci_text(r["gold"])] for r in rows])
    notes = (f'<p>Paired study bootstrap, {j["bootstrap"]["resamples"]:,} resamples, seed {j["bootstrap"]["seed"]}, the '
             'same draws as the lateral-rebalance comparison (A30 − u9-b1 reproduces its interval). The comparison script '
             'checks every run\'s study order and gold targets against run 5. One training seed per run, and no CUDA '
             'determinism is pinned, so a macro delta inside about ±0.01 LM-test and ±0.02 gold-58 is seed noise. '
             'Per-label seed noise is larger. Gold-58 is a guard (veto at −0.024) and a tiebreaker, not a selector. '
             'Only one augmentation strength was tried; stronger or different transforms (flips, elastic, slice '
             'jitter) are untested. The augmented run took about 26.3 min per epoch against A30\'s 26.0, with 5.4 GiB '
             'peak memory.</p>'
             f'<p>Sources: <a href="{ctx.raw(SRC)}">scores, paired intervals, histories</a>, '
             f'<a href="{ctx.doc("finetune-image-augmentation-v1.md")}">implementation and result write-up</a>.</p>')
    return delta_card("image-augmentation", claim, rows, GROUPS, x_label="Candidate minus its stated reference",
                      lead=verdict + lead + scores + target_table, notes=notes, table_html=paired,
                      table_summary="Paired comparison table")


def curves(ctx, j) -> tuple[str, dict]:
    sid = "image-augmentation-curves"
    by_key = {r["key"]: r for r in j["runs"]}
    runs = [by_key[k] for k in (BASE, REF, AUG) if k in by_key]
    blob = {"groups": CURVE_GROUPS, "defaultPinned": [r["key"] for r in runs]}
    for key, metric in (("train", "train_masked_bce"), ("val", "validation_masked_bce"),
                        ("auc", "validation_macro_auroc")):
        blob[key] = [{"id": r["key"], "label": r["title"], "group": "aug" if r["key"] == AUG else "ref",
                      "points": [[h["epoch"], h[metric]] for h in r["history"] if metric in h]}
                     for r in runs if any(metric in h for h in r["history"])]
    body = (HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            + ''.join(f'<div>{chart(key, title)}</div>' for key, title in (
                ("train", "Training BCE by epoch · unweighted, clean targets"),
                ("val", "Validation BCE by epoch · selects the checkpoint"),
                ("auc", "Validation macro AUROC by epoch"))) + '</div>'
            + '<div data-role="legend"></div>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  store.set({pinned: d.defaultPinned});
  [['train', 'Training BCE', '.3f', 'min'], ['val', 'Validation BCE', '.3f', 'min'],
   ['auc', 'Validation AUROC', '.3f', 'max']].forEach(function (spec) {
    Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.val, store, {groups: d.groups, best: 'min', fmt: '.3f'});"""))
    aug, ref = by_key[AUG], by_key[REF]
    claim = (f"Training curves: augmentation tracks A30 almost epoch for epoch; both select epoch {aug['best_epoch']} "
             f"of {aug['epochs_completed']}")
    notes = ('<p>The augmented run\'s training BCE is measured on augmented inputs, so it is not a like-for-like fit '
             'measure against A30; it sits only slightly higher (about +0.01 mid-run, '
             f'{num(aug["history"][-1]["train_masked_bce"] - ref["history"][-1]["train_masked_bce"], ".4f", True)} at '
             'the end). Both A30 runs average their training BCE over all 4,946 draws, including the masked oversampled '
             'repeats, which is why they sit below u9-b1 from epoch 1 (see the lateral-rebalance curves). u9-b1 ran '
             '17 epochs under a 20-epoch cosine and patience 5; both A30 runs stretch a cosine over 30 epochs.</p>')
    return section(sid, claim, body, notes=notes), {f"data-{sid}": blob}


def per_label(ctx, j) -> tuple[str, dict]:
    sid = "image-augmentation-labels"
    labels = list(j["per_label"]["lm_test"][REF])
    titles = {r["key"]: r["title"] for r in j["runs"]}
    present = [k for k in (BASE, REF, AUG) if k in j["per_label"]["lm_test"]]
    deltas = [("Seed noise (run 5)", SEED, SEED_REF), ("Aug − A30", AUG, REF), ("Aug − u9-b1", AUG, BASE),
              ("A30 − u9-b1", REF, BASE)]
    blob = {"rows": labels, "defaultMode": "delta",
            "modeChoices": [{"key": "delta", "label": "Δ (+ seed noise)"}, {"key": "level", "label": "AUROC"}],
            "cols": {"delta": [d[0] for d in deltas], "level": [titles[k] for k in present]}}
    for split in ("gold", "lm_test"):
        pl = j["per_label"][split]
        blob[split] = {"level": [[pl[k][l] for k in present] for l in labels],
                       "delta": [[pl[a][l] - pl[b][l] for _, a, b in deltas] for l in labels]}
    body = (HELPERS + '<div data-role="split"></div><div data-role="mode"></div>'
            + '<div data-chart-scope>' + chart("heat", "Per-label AUROC") + '</div>'
            + section_script(sid, """
  var split = 'gold', mode = d.defaultMode;
  function draw() {
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var o = {rows: d.rows, cols: d.cols[mode], rowIds: d.rows, fmt: mode === 'level' ? '.3f' : '+.3f',
             ariaLabel: 'per-label ' + (split === 'gold' ? 'gold-58' : 'LM-test') + ' AUROC heatmap'};
    if (mode !== 'level') { o.diverging = true; o.peakPerRow = false; }
    Charts.heatmap(el, d[split][mode], o);
  }
  AdaptUI.seg(sec.querySelector('[data-role=split]'), 'Scored on',
    [{key: 'gold', label: 'Gold-58 (human)'}, {key: 'lm_test', label: 'LM-test (432)'}], 'gold',
    function (k) { split = k; draw(); });
  AdaptUI.seg(sec.querySelector('[data-role=mode]'), 'Cells', d.modeChoices, d.defaultMode,
    function (k) { mode = k; draw(); });
  draw();"""))
    rows = []
    for label in labels:
        cells = [esc(label) + (" ★" if label in j["target_labels"] else "")]
        for split in ("lm_test", "gold"):
            pl = j["per_label"][split]
            cells += [num(pl[REF][label]), num(pl[AUG][label]), num(pl[AUG][label] - pl[REF][label], ".4f", True)]
        rows.append(cells)
    headers = ["Label", "A30 LM", "Aug LM", "Δ LM", "A30 gold", "Aug gold", "Δ gold"]
    caption = ("Cells default to deltas; the first column, the run-5 seed replicate, shows how far a label moves from "
               "training noise alone. ★ marks the lateral target labels in the table. Point estimates; the target "
               "labels' intervals are in the first section.")
    return (section(sid, "Per label: augmentation's largest moves are losses, MCL on both test sets and Effusion on "
                         "gold-58", body, caption=caption,
                    table_html=table(headers, rows, num_cols=tuple(range(1, len(headers)))),
                    table_summary="All 12 labels · augmentation vs A30"), {f"data-{sid}": blob})


def build(ctx) -> tuple[list[str], dict]:
    j = data.read_json(SRC)
    if not j:
        return [], {}
    result, result_data = results(ctx, j)
    curve, curve_data = curves(ctx, j)
    labels, label_data = per_label(ctx, j)
    return [result, curve, labels], {**result_data, **curve_data, **label_data}
