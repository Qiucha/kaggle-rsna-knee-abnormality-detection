"""Wide slice band on the A30 recipe, vs A30; plus A30's inference-only probes (weight averaging, soups, TTA).

Sources: digest/finetune-slice-band-v1.json, written by scripts/compare_slice_band.py (worktree t3code-d7aef637,
branch t3code/fine-tune-weak-labels), and digest/a30-inference-probes-v1.json, exported from
checkpoints/finetune/a30-inference-probes/compare.json (.scratch/leaderboard-next/scratch/a30_inference_*.py).
"""

from __future__ import annotations

from .. import data
from ..shell import chart, esc, group_major, num, section, table
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card

SRC, PROBES = "finetune-slice-band-v1.json", "a30-inference-probes-v1.json"
BASE, REF, ARM = "u9-b1", "oversample+noise ep30", "oversample+noise ep30 band05"
SEED, SEED_REF = "run5 seed2027", "run5"
GROUPS = {"noise": {"label": "Training-seed noise floor", "color": "var(--g-sky)"},
          "band": {"label": "Slice band [0.05, 0.95] (one factor)", "color": "var(--g-amber)"},
          "context": {"label": "Against u9-b1 (context)", "color": "var(--g-rose)"}}
CURVE_GROUPS = {"ref": {"label": "u9-b1 and A30 (band [0.2, 0.8])", "color": "var(--g-sky)"},
                "band": {"label": "A30 with band [0.05, 0.95]", "color": "var(--g-amber)"}}
PROBE_GROUPS = {"avg": {"label": "Checkpoint averaging (A30 alone)", "color": "var(--g-sky)"},
                "soup": {"label": "Weight soups with a sibling run", "color": "var(--g-violet)"},
                "tta": {"label": "Window-jitter test-time augmentation", "color": "var(--g-amber)"}}
ORDER = [SEED_REF, SEED, BASE, REF, ARM]


def _pair(j, split, first, second):
    return j["paired"][split].get(f"{first} - {second}")


def results(ctx, j) -> tuple[str, dict]:
    pairs = [(SEED, SEED_REF, "Run-5 seed 2027 − seed 2026 (noise floor)", "noise"),
             (ARM, REF, "Wide band − A30", "band"),
             (ARM, BASE, "Wide band − u9-b1", "context"),
             (REF, BASE, "A30 − u9-b1", "context")]
    rows = group_major([{"id": str(i), "label": label, "group": group,
                         "lm": _pair(j, "lm_test", a, b), "gold": _pair(j, "gold", a, b)}
                        for i, (a, b, label, group) in enumerate(pairs)], GROUPS)
    by_key = {r["key"]: r for r in j["runs"]}
    arm, ref = by_key[ARM], by_key[REF]
    lm, gold = _pair(j, "lm_test", ARM, REF), _pair(j, "gold", ARM, REF)
    pl = j["per_label"]

    def label_delta(split, label):
        return num(pl[split][ARM][label] - pl[split][REF][label], ".3f", True)

    claim = (f"A wider slice band does not help: {num(j['point']['lm_test'][ARM])} LM-test, "
             f"{num(lm['difference'], '.4f', True)} against the same recipe on [0.2, 0.8]; A30 stays the best single run")
    lead = (
        '<p><b>Question:</b> every run so far keeps only the middle of each stack, slice band [0.2, 0.8]. On sagittal '
        'stacks the outer 40% holds the meniscal bodies and the collateral ligaments. Does widening the band to '
        '[0.05, 0.95] recover label signal there? The trial copies A30 (u9-b1 + lateral oversampling and target '
        'noise, cosine over 30 epochs, patience off, divergence stop) and changes only the band, in training and in '
        'every evaluation. The K = 8 windows still split the band into eight equal strata, so each window now '
        'covers a wider stretch of slices and the outer two sit nearer the edges.</p>')
    verdict = (
        f'<p><b>Result: no gain, not adopted.</b> Against A30 the wide band moves {ci_text(lm)} on LM-test and '
        f'{ci_text(gold)} on gold-58. Both intervals cross zero and both point estimates lean negative. Against u9-b1 '
        f'it is {ci_text(_pair(j, "lm_test", ARM, BASE))} LM-test and {ci_text(_pair(j, "gold", ARM, BASE))} '
        'gold-58, so most of A30\'s edge over u9-b1 is gone.</p>'
        f'<p><b>The edge structures did not improve.</b> The three labels the band was meant to help move '
        f'{label_delta("lm_test", "Lateral Meniscus")} (Lateral Meniscus), {label_delta("lm_test", "Medial Meniscus")} '
        f'(Medial Meniscus) and {label_delta("lm_test", "MCL")} (MCL) on LM-test, and '
        f'{label_delta("gold", "Lateral Meniscus")}, {label_delta("gold", "Medial Meniscus")} and '
        f'{label_delta("gold", "MCL")} on gold-58. Elsewhere the LM-test losses fall on Medial OA '
        f'({label_delta("lm_test", "Medial OA")}), PF OA ({label_delta("lm_test", "PF OA")}) and Contusion '
        f'({label_delta("lm_test", "Contusion")}): spreading eight windows over more slices thins the central '
        'coverage those labels rely on. Training ran all '
        f'{arm["epochs_completed"]} epochs and selected epoch {arm["best_epoch"]} (A30: {ref["best_epoch"]}); '
        f'validation BCE {num(arm["validation_masked_bce"])} against A30\'s {num(ref["validation_masked_bce"])}.</p>')
    scores = table(["Training", "LM-test AUROC", "Gold-58 AUROC", "Best epoch", "Epochs run", "Validation BCE",
                    "Wall time (h)"],
                   [[by_key[k]["title"], num(j["point"]["lm_test"][k]), num(j["point"]["gold"][k]),
                     by_key[k]["best_epoch"], by_key[k]["epochs_completed"], num(by_key[k]["validation_masked_bce"]),
                     num(by_key[k]["seconds"] / 3600, ".1f") if by_key[k].get("seconds") else "—"]
                    for k in ORDER if k in by_key], num_cols=(1, 2, 3, 4, 5, 6))
    target_rows = []
    for label in j["target_labels"]:
        cells = [esc(label)]
        for a, b in ((SEED, SEED_REF), (ARM, REF), (ARM, BASE)):
            for split in ("lm_test", "gold"):
                cells.append(ci_text(j["paired_target_labels"][split][label].get(f"{a} - {b}")))
        target_rows.append(cells)
    target_table = ('<p><b>The three edge-structure labels alone</b> (paired study bootstrap on that label only):</p>'
                    + table(["Label", "Seed noise LM", "Seed noise gold", "Band − A30 LM", "Band − A30 gold",
                             "Band − u9-b1 LM", "Band − u9-b1 gold"], target_rows, raw_html=True))
    paired = table(["Candidate − reference", "Δ LM-test [95% CI]", "Δ gold-58 [95% CI]"],
                   [[r["label"], ci_text(r["lm"]), ci_text(r["gold"])] for r in rows])
    notes = (f'<p>Paired study bootstrap, {j["bootstrap"]["resamples"]:,} resamples, seed {j["bootstrap"]["seed"]}, the '
             'same draws as the lateral-rebalance and image-augmentation comparisons. One training seed per run; a '
             'macro delta inside about ±0.01 LM-test and ±0.02 gold-58 is seed noise. Gold-58 is a guard (veto at '
             '−0.024) and a tiebreaker, not a selector. The wide band needed a new evaluation window cache (the cache '
             'key includes the band); training reads the local DICOM mirror, which holds whole series. About 26.6 min '
             'per epoch, 5.4 GiB peak memory, the same as A30.</p>'
             f'<p>Sources: <a href="{ctx.raw(SRC)}">scores, paired intervals, histories</a>, '
             f'<a href="{ctx.doc("finetune-slice-band-v1.md")}">write-up</a>.</p>')
    return delta_card("slice-band", claim, rows, GROUPS, x_label="Candidate minus its stated reference",
                      lead=verdict + lead + scores + target_table, notes=notes, table_html=paired,
                      table_summary="Paired comparison table")


def curves(ctx, j) -> tuple[str, dict]:
    sid = "slice-band-curves"
    by_key = {r["key"]: r for r in j["runs"]}
    runs = [by_key[k] for k in (BASE, REF, ARM) if k in by_key]
    blob = {"groups": CURVE_GROUPS, "defaultPinned": [r["key"] for r in runs]}
    for key, metric in (("train", "train_masked_bce"), ("val", "validation_masked_bce"),
                        ("auc", "validation_macro_auroc")):
        blob[key] = [{"id": r["key"], "label": r["title"], "group": "band" if r["key"] == ARM else "ref",
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
    arm, ref = by_key[ARM], by_key[REF]
    claim = (f"Training curves: the wide band tracks A30 closely; it selects epoch {arm['best_epoch']} of "
             f"{arm['epochs_completed']} against A30's {ref['best_epoch']}")
    notes = ('<p>The validation set is scored on wide-band windows for the trial and on [0.2, 0.8] windows for A30, '
             'so the validation curves compare two inputs, not one. Both A30 runs average their training BCE over all '
             '4,946 draws, including the masked oversampled repeats, which is why they sit below u9-b1 from epoch 1. '
             f'Final training BCE: {num(arm["history"][-1]["train_masked_bce"])} against A30\'s '
             f'{num(ref["history"][-1]["train_masked_bce"])}.</p>')
    return section(sid, claim, body, notes=notes), {f"data-{sid}": blob}


def per_label(ctx, j) -> tuple[str, dict]:
    sid = "slice-band-labels"
    labels = list(j["per_label"]["lm_test"][REF])
    titles = {r["key"]: r["title"] for r in j["runs"]}
    present = [k for k in (BASE, REF, ARM) if k in j["per_label"]["lm_test"]]
    deltas = [("Seed noise (run 5)", SEED, SEED_REF), ("Band − A30", ARM, REF), ("Band − u9-b1", ARM, BASE),
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
            cells += [num(pl[REF][label]), num(pl[ARM][label]), num(pl[ARM][label] - pl[REF][label], ".4f", True)]
        rows.append(cells)
    headers = ["Label", "A30 LM", "Band LM", "Δ LM", "A30 gold", "Band gold", "Δ gold"]
    caption = ("Cells default to deltas; the first column, the run-5 seed replicate, shows how far a label moves from "
               "training noise alone. ★ marks the edge-structure labels. Point estimates; the starred labels' "
               "intervals are in the first section.")
    return (section(sid, "Per label: the edge structures do not gain; the LM-test losses fall on the OA labels and "
                         "Contusion", body, caption=caption,
                    table_html=table(headers, rows, num_cols=tuple(range(1, len(headers)))),
                    table_summary="All 12 labels · wide band vs A30"), {f"data-{sid}": blob})


def probes(ctx, p) -> tuple[str, dict]:
    rows = group_major([{"id": v["key"], "label": v["label"], "group": v["group"], "lm": v["d_lm"],
                         "gold": v["d_gold"]} for v in p["variants"]], PROBE_GROUPS)
    by = {v["key"]: v for v in p["variants"]}
    tta = by["tta centre + 4 jitter"]["d_lm"]
    claim = ("Inference-only changes to A30 do not help: checkpoint averaging, weight soups and window-jitter TTA "
             "all sit inside noise")
    lead = (
        '<p><b>Question:</b> can A30 gain without retraining? Three probes reuse its saved weights. '
        '<b>Checkpoint averaging</b> averages the selected epoch-22 weights with the final epoch-30 weights. '
        '<b>Weight soups</b> average A30 with a sibling fine-tuned from the same pretrained OmniRad-base: the '
        'augmentation run (same recipe) and u9-b1 (20-epoch recipe). <b>Window-jitter TTA</b> averages the '
        'centre-window prediction with up to four passes that draw each window uniformly from its stratum, the '
        'training-time jitter.</p>'
        f'<p><b>Result: nothing to adopt.</b> Averaging best and last gives '
        f'{ci_text(by["a30-avg-best-last"]["d_lm"])} LM-test. The augmentation soup gives '
        f'{ci_text(by["soup-a30-a30aug"]["d_lm"])}, while the u9-b1 soup loses ({ci_text(by["soup-a30-u9"]["d_lm"])}). '
        f'Centre + 4 jitter passes gives {ci_text(tta)}, and a single jitter pass alone scores below the centres. '
        'Gold-58 moves are within ±0.011. Each jitter pass also adds a full decode, about 5–7 min per 490 '
        'studies here, which a Kaggle bundle would pay per member.</p>')
    tbl = table(["Probe", "LM-test", "Δ LM-test vs A30 [95% CI]", "Gold-58", "Δ gold-58 vs A30 [95% CI]",
                 "Gold Lateral Meniscus"],
                [[v["label"], num(v["lm_test"]), ci_text(v["d_lm"]), num(v["gold"]), ci_text(v["d_gold"]),
                  num(v["gold_latmen"], ".3f")] for v in p["variants"]], num_cols=(1, 3, 5))
    notes = (f'<p>Paired study bootstrap, {p["bootstrap"]["resamples"]:,} resamples. Re-scoring A30\'s best checkpoint '
             f'reproduced its saved scores exactly ({num(p["reference"]["lm_test"])} LM-test, '
             f'{num(p["reference"]["gold"])} gold-58). Soups average only the trainable weights (last nine blocks and '
             'head); the frozen blocks are identical across the runs.</p>'
             f'<p>Source: <a href="{ctx.raw(PROBES)}">probe scores and intervals</a>.</p>')
    return delta_card("a30-inference-probes", claim, rows, PROBE_GROUPS, x_label="Probe minus A30",
                      lead=lead, notes=notes, table_html=tbl, table_summary="Probe table")


def build(ctx) -> tuple[list[str], dict]:
    sections, blob = [], {}
    j = data.read_json(SRC)
    if j:
        for part in (results, curves, per_label):
            html, b = part(ctx, j)
            sections.append(html); blob.update(b)
    p = data.read_json(PROBES)
    if p:
        html, b = probes(ctx, p)
        sections.append(html); blob.update(b)
    return sections, blob
