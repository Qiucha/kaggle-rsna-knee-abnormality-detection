"""Lateral/synovitis rebalancing on u9-b1: (A) oversample + lateral label noise, (B) 2x loss weight.

Source: digest/lateral-rebalance-v1.json, written by scripts/compare_lateral_rebalance.py
(branch t3code/fine-tune-weak-labels) at the end of checkpoints/finetune/lateral-rebalance/chain.sh.
"""

from __future__ import annotations

from .. import data
from ..shell import chart, esc, group_major, num, section, table
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card

SRC = "lateral-rebalance-v1.json"
BASE, ARM_A, ARM_B, ARM_A30 = "u9-b1", "oversample+noise", "loss weight 2x", "oversample+noise ep30"
ARM_A50 = "oversample+noise ep50"
CTRL30, CTRL43 = "u9-b1 ep30", "u9-b1 ep43"
CONTROL_SHORT = {CTRL30: "u9-b1 30 epochs", CTRL43: "u9-b1 43 epochs"}
GROUPS = {"noise": {"label": "Training-seed noise floor", "color": "var(--g-sky)"},
          "control": {"label": "u9-b1 trained longer (controls)", "color": "var(--g-teal)"},
          "arms": {"label": "Rebalancing arms vs u9-b1", "color": "var(--g-rose)"}}
CURVE_GROUPS = {"base": {"label": "u9-b1 and its longer-training controls", "color": "var(--g-sky)"},
                "arms": {"label": "Rebalancing arms", "color": "var(--g-rose)"}}
TITLES = {ARM_A: "A · oversample + lateral noise", ARM_B: "B · 2× loss weight", ARM_A30: "A · 30-epoch budget",
          ARM_A50: "A · extended to 50 (warm restart)"}
# Arms in reading order with their short column names.
ARMS = [(CTRL30, "U30"), (CTRL43, "U43"), (ARM_A, "A"), (ARM_B, "B"), (ARM_A30, "A30"), (ARM_A50, "A50")]
# Every run in reading order: references, controls, then arms.
ORDER = ["run5", "run5 seed2027", BASE, CTRL30, CTRL43, ARM_A, ARM_B, ARM_A30, ARM_A50]


def _pair(j, split, first, second):
    return j["paired"][split].get(f"{first} - {second}")


def _controls_paragraph(j) -> str:
    """Whether the A30 gain survives u9-b1 trained as long (matched epochs) or on as many draws (matched draws)."""
    done = [c for c in (CTRL30, CTRL43) if c in j["point"]["lm_test"]]
    intro = ('<p><b>Separating oversampling from longer training.</b> The 30-epoch A saw about 2.5× u9-b1\'s training '
             'draws (30 × 4,946 against 17 × 3,438). Two u9-b1 controls use A30\'s schedule and stopping rule: '
             '<b>30 epochs</b> matches A30\'s epochs, and <b>43 epochs</b> matches its draws (43 × 3,438 = 147,834).')
    if not done:
        return intro + ' Neither was run: the queue was abandoned at the 2026-10-20 wrap-up, so the gain could still come from the longer schedule alone.</p>'
    parts = []
    for c in done:
        parts.append(f'{CONTROL_SHORT[c]} scores {num(j["point"]["lm_test"][c])} LM-test '
                     f'({ci_text(_pair(j, "lm_test", c, BASE))} vs u9-b1), and A30 minus it is '
                     f'{ci_text(_pair(j, "lm_test", ARM_A30, c))} LM-test, {ci_text(_pair(j, "gold", ARM_A30, c))} gold-58')
    pending = [CONTROL_SHORT[c] for c in (CTRL30, CTRL43) if c not in done]
    return (intro + ' ' + '; '.join(parts) + '.'
            + (f' Not run (queue abandoned 2026-10-20): {", ".join(pending)}.' if pending else '') + '</p>')


def results(ctx, j) -> tuple[str, dict]:
    pairs = [("run5 seed2027", "run5", "Run-5 seed 2027 − seed 2026 (noise floor)", "noise"),
             (ARM_A, BASE, "A · oversample + noise − u9-b1", "arms"),
             (ARM_B, BASE, "B · 2× loss weight − u9-b1", "arms"),
             (ARM_A, ARM_B, "A − B", "arms"),
             (ARM_A30, BASE, "A · 30-epoch budget − u9-b1", "arms"),
             (ARM_A30, ARM_A, "A 30-epoch − A 20-epoch", "arms"),
             (ARM_A50, BASE, "A · extended to 50 − u9-b1", "arms"),
             (ARM_A50, ARM_A30, "A extended to 50 − A 30-epoch", "arms")]
    pairs += [(CTRL30, BASE, "u9-b1 30 ep − u9-b1", "control"),
              (CTRL43, BASE, "u9-b1 43 ep − u9-b1", "control"),
              (ARM_A30, CTRL30, "A30 − u9-b1 30 ep (same epochs)", "arms"),
              (ARM_A30, CTRL43, "A30 − u9-b1 43 ep (same draws)", "arms")]
    rows = [{"id": str(i), "label": label, "group": group,
             "lm": _pair(j, "lm_test", a, b), "gold": _pair(j, "gold", a, b)}
            for i, (a, b, label, group) in enumerate(pairs)]
    rows = group_major([r for r in rows if r["lm"]], GROUPS)
    a_lm, b_lm = _pair(j, "lm_test", ARM_A, BASE), _pair(j, "lm_test", ARM_B, BASE)
    a_gold, b_gold = _pair(j, "gold", ARM_A, BASE), _pair(j, "gold", ARM_B, BASE)
    a30_lm, a30_gold = _pair(j, "lm_test", ARM_A30, BASE), _pair(j, "gold", ARM_A30, BASE)
    claim = (f"Oversampling with a 30-epoch budget gives the best single LM-test score so far, "
             f"{num(j['point']['lm_test'][ARM_A30])} ({num(a30_lm['difference'], '.4f', True)} vs u9-b1, interval "
             f"crosses zero); the 20-epoch arms lose" if a30_lm else "Rebalancing the weak labels does not beat u9-b1: "
             + ("; ".join(f"{name} {num(lm['difference'], '.4f', True)} LM-test, "
                          f"{num(gold['difference'], '.4f', True)} gold-58"
                          for name, lm, gold in (("A", a_lm, a_gold), ("B", b_lm, b_gold)) if lm)
                or "results pending"))
    verdict = ""
    if a_lm and b_lm:
        tl = j["paired_target_labels"]

        def target(split, label, arm):
            return ci_text(tl[split][label].get(f"{arm} - {BASE}"))

        verdict = (
            '<p><b>20-epoch arms: neither lifts the macro score.</b> A (oversample + noise) moves '
            f'{ci_text(a_lm)} on LM-test and {ci_text(a_gold)} on gold-58: both intervals cross zero and lean negative. '
            f'B (2× weight) moves {ci_text(b_lm)} on LM-test and {ci_text(b_gold)} on gold-58, past the −0.024 gold veto. '
            'Of the three targeted labels, only <b>Lateral Meniscus</b> clearly rises on LM-test: '
            f'A {target("lm_test", "Lateral Meniscus", ARM_A)}, B {target("lm_test", "Lateral Meniscus", ARM_B)}. '
            f'On gold-58 the rise holds only for A ({target("gold", "Lateral Meniscus", ARM_A)}, interval crosses zero). '
            'Lateral OA and Synovitis do not move beyond noise. The gain is paid for elsewhere, mostly MCL (gold −0.10 '
            'for A and −0.14 for B), plus Medial Meniscus, ACL and, for A, Effusion (gold −0.10). B also overfits sooner: its validation BCE bottoms at '
            'epoch 7 and early stopping ends it at 12, against u9-b1\'s 12 and 17.</p>')
    a30 = next((r for r in j["runs"] if r["key"] == ARM_A30), None)
    if a30:
        verdict = (
            f'<p><b>Result: arm A was under-trained, not wrong.</b> Retrained with a 30-epoch budget (cosine over 30 '
            f'epochs, patience off), A scores {num(j["point"]["lm_test"][ARM_A30])} LM-test and '
            f'{num(j["point"]["gold"][ARM_A30])} gold-58: {ci_text(a30_lm)} and {ci_text(a30_gold)} vs u9-b1. '
            f'That is the highest single-run LM-test score so far, but its interval still crosses zero. Against the 20-epoch A '
            f'the gain is clear: {ci_text(_pair(j, "lm_test", ARM_A30, ARM_A))} LM-test, '
            f'{ci_text(_pair(j, "gold", ARM_A30, ARM_A))} gold-58. Validation BCE kept falling until epoch '
            f'{a30["best_epoch"]} ({num(a30["validation_masked_bce"])}, below u9-b1\'s 0.4308), then flattened. It never '
            f'rose enough to trigger the divergence stop, so the run used all {a30["epochs_completed"]} epochs. '
            'Lateral Meniscus keeps its LM-test gain '
            f'({ci_text(j["paired_target_labels"]["lm_test"]["Lateral Meniscus"].get(f"{ARM_A30} - {BASE}"))}), and the '
            'costs the 20-epoch arm paid on MCL, Effusion and Fracture are gone: no label falls more than seed noise on '
            'LM-test, and the largest gold-58 dips are ACL (−0.033) and Baker\'s (−0.016).</p>'
            '<p><b>Public leaderboard (2026-10-12): 0.890</b> (submission 57094902), against u9-b1\'s 0.888. That makes it '
            'the best public single model, but only by +0.002. It falls short of both predictions: about 0.896 if LM-test '
            'tracked and about 0.892 if gold-58 did. Public moved about a quarter of the LM-test gain, which is within '
            'single-seed noise, and the extra-draws confound remains open.</p>'
            + _controls_paragraph(j) + verdict)
    else:
        verdict += ('<p><b>Pending:</b> arm A is being retrained with a 30-epoch budget (cosine over 30 epochs, patience '
                    'off, stopping only once validation BCE sits 0.02 above its best for 3 epochs in a row).</p>')
    a50 = next((r for r in j["runs"] if r["key"] == ARM_A50), None)
    if a50:
        verdict = (
            f'<p><b>Extension to 50 epochs (warm restart):</b> {num(j["point"]["lm_test"][ARM_A50])} LM-test and '
            f'{num(j["point"]["gold"][ARM_A50])} gold-58. Against u9-b1: {ci_text(_pair(j, "lm_test", ARM_A50, BASE))} '
            f'LM-test, {ci_text(_pair(j, "gold", ARM_A50, BASE))} gold-58. Against the 30-epoch run: '
            f'{ci_text(_pair(j, "lm_test", ARM_A50, ARM_A30))} LM-test, {ci_text(_pair(j, "gold", ARM_A50, ARM_A30))} '
            f'gold-58. It ran to epoch {a50["epochs_completed"]}, stopped by {a50.get("stopped", "?").replace("_", " ")}, '
            f'and selected epoch {a50["best_epoch"]} (validation BCE {num(a50["validation_masked_bce"])}).</p>' + verdict)
    elif a30:
        verdict = ('<p><b>Not run:</b> a 50-epoch extension of the 30-epoch A was planned after the two u9-b1 controls, resuming from its epoch-30 state '
                   'with a warm restart at half the original learning rates and stopping early once validation BCE sat '
                   '0.02 above its best for 3 epochs in a row. The queue was abandoned at the 2026-10-20 wrap-up.</p>' + verdict)
    lead = ('<p><b>Question:</b> u9 deep unfreezing left <b>Lateral Meniscus</b>, <b>Lateral OA</b> and '
            '<b>Synovitis</b> behind the other labels. Does pushing training toward them lift them, and what does it '
            'cost the macro score? Both arms keep the u9-b1 recipe (OmniRad-base, 9 unfrozen blocks, soft targets, '
            'K8 windows at 336 px, seed 2026) and change one thing.</p><ul>'
            '<li><b>A · oversample + label noise.</b> Every training study with an observed positive verdict on any '
            'of the three labels gets one extra draw per epoch (1,508 extra draws, so 4,946 per epoch instead of '
            '3,438). On the extra draws the other nine labels are masked, so they see exactly u9-b1\'s data. Each draw '
            'adds fresh N(0, 0.1) noise to the two lateral soft targets, clipped to [0, 1], so repeated studies '
            'never present identical targets.</li>'
            '<li><b>B · loss weight.</b> 2× BCE weight on the three labels, renormalised to mean 1 '
            '(1.6 on the targets, 0.8 on the rest), so the total loss and gradient scale stay u9-b1\'s.</li></ul>'
            '<p>Both arms select their checkpoint the same way u9-b1 does: on validation masked BCE against hard verdicts, '
            'with no weighting and no noise. <b>Confound in A:</b> its epochs hold 1.44× as many draws under the same '
            '20-epoch, patience-5 budget, so A also gets about 1.44× u9-b1\'s optimiser steps. A − u9-b1 therefore mixes '
            'the oversampling effect with extra training. Clipping the noise at 0 also nudges NO targets (0.08) up by about '
            '+0.012 on average.</p>')
    by_key = {r["key"]: r for r in j["runs"]}
    order = [k for k in ORDER if k in by_key]
    scores = table(["Training", "LM-test AUROC", "Gold-58 AUROC", "Best epoch", "Epochs run", "Validation BCE"],
                   [[by_key[k]["title"], num(j["point"]["lm_test"][k]), num(j["point"]["gold"][k]),
                     by_key[k]["best_epoch"], by_key[k]["epochs_completed"], num(by_key[k]["validation_masked_bce"])]
                    for k in order], num_cols=(1, 2, 3, 4, 5))
    paired = table(["Candidate − reference", "Δ LM-test [95% CI]", "Δ gold-58 [95% CI]"],
                   [[r["label"], ci_text(r["lm"]), ci_text(r["gold"])] for r in rows])
    target_rows = []
    for label in j["target_labels"]:
        cells = [esc(label)]
        for a, b in (("run5 seed2027", "run5"), (ARM_A, BASE), (ARM_B, BASE), (ARM_A30, BASE), (ARM_A50, BASE)):
            for split in ("lm_test", "gold"):
                cells.append(ci_text(j["paired_target_labels"][split][label].get(f"{a} - {b}")))
        target_rows.append(cells)
    target_table = ('<p><b>The three target labels alone</b> (paired study bootstrap on that label only):</p>'
                    + table(["Label", "Seed noise LM", "Seed noise gold", "A − u9-b1 LM", "A − u9-b1 gold",
                             "B − u9-b1 LM", "B − u9-b1 gold", "A30 − u9-b1 LM", "A30 − u9-b1 gold",
                             "A50 − u9-b1 LM", "A50 − u9-b1 gold"],
                            target_rows, raw_html=True))
    missing = j.get("missing_arms") or []
    notes = (('<p><strong>Pending:</strong> ' + ", ".join(TITLES.get(m, m) for m in missing)
              + ' had not finished when this snapshot was taken.</p>' if missing else '')
             + f'<p>Paired study bootstrap, {j["bootstrap"]["resamples"]:,} resamples, seed {j["bootstrap"]["seed"]}; '
             'the comparison script checks every run\'s study order and gold targets against run 5. One training seed '
             'per arm: by the run-5 seed replicate, a macro delta inside about ±0.01 on LM-test and ±0.02 on gold-58 '
             'cannot be told apart from seed noise. Per-label seed noise is larger. '
             'LM labels agree with gold on Lateral Meniscus only about 0.84 of the time, so a lateral gain may show on gold-58 '
             'and not on LM-test. Gold-58 is a guard (veto at −0.024) and a tiebreaker, not a selector. '
             'Synovitis is observed in only 537 of 3,438 training studies, and 454 of those are positive, so oversampling '
             'its positives mostly repeats the same ~450 studies. Its gap is coverage, not class imbalance.</p>'
             f'<p>Sources: <a href="{ctx.raw(SRC)}">scores, paired intervals, histories</a>, '
             f'<a href="{ctx.doc("lateral-rebalance-v1.md")}">result write-up</a>.</p>')
    return delta_card("lateral-rebalance", claim, rows, GROUPS, x_label="Candidate minus its stated reference",
                      lead=verdict + lead + scores + target_table, notes=notes, table_html=paired,
                      table_summary="Paired comparison table")


def curves(ctx, j) -> tuple[str, dict]:
    sid = "lateral-rebalance-curves"
    by_key = {r["key"]: r for r in j["runs"]}
    runs = [by_key[k] for k in ORDER if k in by_key and not k.startswith("run5")]
    blob = {"groups": CURVE_GROUPS, "defaultPinned": [r["key"] for r in runs]}
    for key, metric in (("train", "train_masked_bce"), ("objective", "train_objective"),
                        ("val", "validation_masked_bce"), ("auc", "validation_macro_auroc")):
        blob[key] = [{"id": r["key"], "label": r["title"], "group": "base" if r["key"].startswith(BASE) else "arms",
                      "points": [[h["epoch"], h[metric]] for h in r["history"] if metric in h]}
                     for r in runs if any(metric in h for h in r["history"])]
    body = (HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            + ''.join(f'<div>{chart(key, title)}</div>' for key, title in (
                ("train", "Training BCE by epoch · unweighted, clean targets"),
                ("val", "Validation BCE by epoch · selects the checkpoint"),
                ("auc", "Validation macro AUROC by epoch"),
                ("objective", "Optimised training loss · arms only (weighted / noisy)"))) + '</div>'
            + '<div data-role="legend"></div>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  store.set({pinned: d.defaultPinned});
  [['train', 'Training BCE', '.3f', 'min'], ['val', 'Validation BCE', '.3f', 'min'],
   ['auc', 'Validation AUROC', '.3f', 'max'], ['objective', 'Training objective', '.3f', 'min']].forEach(function (spec) {
    Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.val, store, {groups: d.groups, best: 'min', fmt: '.3f'});"""))
    claim = "Training curves: " + ("; ".join(
        f'{TITLES.get(r["key"], r["title"])} selects epoch {r["best_epoch"]} of {r["epochs_completed"]}'
        for r in runs if r["key"] != BASE) or "arms pending")
    notes = ('<p>The training BCE is unweighted and uses the clean soft targets in every run, so the lines can be compared. '
             'For arm A it averages over all 4,946 draws, including the masked repeats. Those repeats score only their 1–3 target labels, and the loss is normalised by u9-b1\'s observation rates, so A\'s curve sits lower from epoch 1. That offset comes from the bookkeeping, not from a better fit. The optimised loss '
             '(weighted for B, noisy targets and repeats for A) is plotted separately. '
             'An arm-A epoch takes about 1.44× as long as u9-b1\'s, and its learning-rate schedule stretches with it. '
             'The 30-epoch A rerun stretches its cosine schedule over 30 epochs, so it is a separate training run, not a '
             'continuation: its first 17 epochs see a higher learning rate than the 20-epoch A did. '
             'The 50-epoch extension is a true continuation: its epochs 1–30 are the 30-epoch run\'s history, so the two '
             'lines overlap there. From epoch 31 it resumes that run\'s weights and optimizer state, with a fresh warmup '
             'and cosine at half the original learning rates.</p>')
    return section(sid, claim, body, notes=notes), {f"data-{sid}": blob}


def per_label(ctx, j) -> tuple[str, dict]:
    sid = "lateral-rebalance-labels"
    labels = list(j["per_label"]["lm_test"][BASE])
    present = [k for k in ORDER if k in j["per_label"]["lm_test"]]
    titles = {r["key"]: r["title"] for r in j["runs"]}
    deltas = [("Seed noise (run 5)", "run5 seed2027", "run5")] + [
        (f"{short} − u9-b1", key, BASE) for key, short in ARMS if key in present] + [
        (f"A30 − {CONTROL_SHORT[c]}", ARM_A30, c) for c in (CTRL30, CTRL43) if c in present and ARM_A30 in present]
    blob = {"rows": labels, "defaultMode": "delta",
            "modeChoices": [{"key": "delta", "label": "Δ vs u9-b1 (+ seed noise)"}, {"key": "level", "label": "AUROC"}],
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
            cells.append(num(pl[BASE][label]))
            cells += [num(pl[k][label] - pl[BASE][label], '.4f', True) for k, _ in ARMS if k in pl]
        rows.append(cells)
    arm_heads = [f"{short} Δ" for k, short in ARMS if k in present]
    headers = (["Label"] + ["u9-b1 LM"] + [f"{h} LM" for h in arm_heads]
               + ["u9-b1 gold"] + [f"{h} gold" for h in arm_heads])
    caption = ("Cells default to each arm minus u9-b1 on the same label. The first column, the run-5 seed replicate, "
               "shows how far a label moves from training noise alone. ★ marks the targeted labels in the table. "
               "Point estimates; the targeted labels' intervals are in the first section.")
    return (section(sid, "Per label: Lateral Meniscus rises on LM-test in every arm; only the 30-epoch A avoids the MCL and Effusion losses",
                    body, caption=caption,
                    table_html=table(headers, rows, num_cols=tuple(range(1, len(headers)))),
                    table_summary="All 12 labels · arms vs u9-b1"), {f"data-{sid}": blob})


def build(ctx) -> tuple[list[str], dict]:
    j = data.read_json(SRC)
    if not j:
        return [], {}
    result, result_data = results(ctx, j)
    curve, curve_data = curves(ctx, j)
    labels, label_data = per_label(ctx, j)
    return [result, curve, labels], {**result_data, **curve_data, **label_data}
