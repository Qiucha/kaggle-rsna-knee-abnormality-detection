"""CLIP-init u9: the A30 recipe started from the CLIP-adapted OmniRad-base, as a single run and as an ens4 member.

Source: digest/finetune-clip-init-u9-v1.json, written by scripts/compare_clip_init_u9.py (worktree integrate-2026-10,
branch exp/clip-init-u9). Ensembles are per-label rank means of the members' saved predictions.
"""

from __future__ import annotations

from .. import data
from ..shell import chart, esc, group_major, num, section, table
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card

SRC = "finetune-clip-init-u9-v1.json"
BASE, REF, ARM, SHALLOW = "u9-b1", "A30", "CLIP-init u9", "from-clip u3"
SEED, SEED_REF = "run5 seed2027", "run5"
ENS4, SWAP, DEEP = "ens4 (submitted)", "ens4 swap", "deep three"
GROUPS = {"noise": {"label": "Training-seed noise floor", "color": "var(--g-sky)"},
          "init": {"label": "CLIP init (one factor, vs A30)", "color": "var(--g-amber)"},
          "context": {"label": "Context", "color": "var(--g-rose)"}}
ENS_GROUPS = {"ens": {"label": "Ensembles vs the submitted ens4", "color": "var(--g-violet)"},
              "context": {"label": "Submitted ens4 vs A30 (context)", "color": "var(--g-rose)"}}
CURVE_GROUPS = {"ref": {"label": "Pretrained init (u9-b1, A30)", "color": "var(--g-sky)"},
                "clip": {"label": "CLIP init (u3 run-5 recipe, u9 A30 recipe)", "color": "var(--g-amber)"}}
ORDER = [SEED_REF, SEED, BASE, REF, SHALLOW, ARM, "dS336"]


def _pair(j, split, first, second):
    return j["paired"][split].get(f"{first} - {second}")


def results(ctx, j) -> tuple[str, dict]:
    pairs = [(SEED, SEED_REF, "Run-5 seed 2027 − seed 2026 (noise floor)", "noise"),
             (ARM, REF, "CLIP-init u9 − A30", "init"),
             (ARM, SHALLOW, "CLIP-init u9 − from-clip u3 (the member it replaces)", "context"),
             (REF, BASE, "A30 − u9-b1", "context")]
    rows = group_major([{"id": str(i), "label": label, "group": group,
                         "lm": _pair(j, "lm_test", a, b), "gold": _pair(j, "gold", a, b)}
                        for i, (a, b, label, group) in enumerate(pairs)], GROUPS)
    by_key = {r["key"]: r for r in j["runs"]}
    arm, ref = by_key[ARM], by_key[REF]
    lm, gold = _pair(j, "lm_test", ARM, REF), _pair(j, "gold", ARM, REF)
    pl = j["per_label"]
    claim = (f"Starting A30 from the CLIP-adapted OmniRad-base does not help: {num(j['point']['lm_test'][ARM])} "
             f"LM-test, {num(lm['difference'], '.4f', True)} against A30; A30 stays the best single run")
    lead = (
        '<p><b>Question:</b> the from-clip member of the submitted ensemble starts from OmniRad-base after the '
        'image–report contrastive stage (omnirad-clip-run5-v1, u6 best) but is fine-tuned on the shallow run-5 '
        'recipe, 3 blocks unfrozen. Deep unfreezing lifts gold Lateral Meniscus, and the shallow members dilute it '
        'in ensembles. Does the CLIP init still add something on the deep A30 recipe (u9-b1 + lateral oversampling '
        'and target noise, cosine over 30 epochs, divergence stop)? The run copies A30 and changes only the '
        'starting weights.</p>')
    verdict = (
        f'<p><b>Result: no gain as a single model.</b> Against A30 the CLIP init moves {ci_text(lm)} on LM-test and '
        f'{ci_text(gold)} on gold-58; both intervals cross zero. Against the shallow from-clip run it is '
        f'{ci_text(_pair(j, "lm_test", ARM, SHALLOW))} LM-test and {ci_text(_pair(j, "gold", ARM, SHALLOW))} '
        f'gold-58, and gold Lateral Meniscus rises from {num(pl["gold"][SHALLOW]["Lateral Meniscus"], ".3f")} to '
        f'{num(pl["gold"][ARM]["Lateral Meniscus"], ".3f")} (A30: {num(pl["gold"][REF]["Lateral Meniscus"], ".3f")}). '
        'So the deep recipe, not the init, is what the CLIP member was missing. Training ran all '
        f'{arm["epochs_completed"]} epochs and selected epoch {arm["best_epoch"]} (A30: {ref["best_epoch"]}); its '
        f'validation BCE, {num(arm["validation_masked_bce"])}, is lower than A30\'s '
        f'{num(ref["validation_masked_bce"])}, but that did not carry over to either test set.</p>')
    scores = table(["Training", "LM-test AUROC", "Gold-58 AUROC", "Best epoch", "Epochs run", "Validation BCE",
                    "Wall time (h)"],
                   [[by_key[k]["title"], num(j["point"]["lm_test"][k]), num(j["point"]["gold"][k]),
                     by_key[k]["best_epoch"], by_key[k]["epochs_completed"], num(by_key[k]["validation_masked_bce"]),
                     num(by_key[k]["seconds"] / 3600, ".1f") if by_key[k].get("seconds") else "—"]
                    for k in ORDER if k in by_key], num_cols=(1, 2, 3, 4, 5, 6))
    target_rows = []
    for label in j["target_labels"]:
        cells = [esc(label)]
        for a, b in ((SEED, SEED_REF), (ARM, REF), (ARM, SHALLOW)):
            for split in ("lm_test", "gold"):
                cells.append(ci_text(j["paired_target_labels"][split][label].get(f"{a} - {b}")))
        target_rows.append(cells)
    target_table = ("<p><b>A30's three oversampled labels alone</b> (paired study bootstrap on that label only):</p>"
                    + table(["Label", "Seed noise LM", "Seed noise gold", "CLIP − A30 LM", "CLIP − A30 gold",
                             "u9 − u3 (CLIP) LM", "u9 − u3 (CLIP) gold"], target_rows, raw_html=True))
    paired = table(["Candidate − reference", "Δ LM-test [95% CI]", "Δ gold-58 [95% CI]"],
                   [[r["label"], ci_text(r["lm"]), ci_text(r["gold"])] for r in rows])
    notes = (f'<p>Paired study bootstrap, {j["bootstrap"]["resamples"]:,} resamples, seed {j["bootstrap"]["seed"]}, the '
             'same draws as the slice-band and image-augmentation comparisons. One training seed per run; a macro '
             'delta inside about ±0.01 LM-test and ±0.02 gold-58 is seed noise. Gold-58 is a guard (veto at −0.024) '
             'and a tiebreaker, not a selector. The from-clip u3 row differs from the CLIP-init u9 in depth and in '
             'recipe (run 5: 20 epochs, no oversampling), so it is context, not one factor. About 26.1 min per epoch, '
             '5.4 GiB peak memory, the same as A30.</p>'
             f'<p>Sources: <a href="{ctx.raw(SRC)}">scores, paired intervals, histories</a>, '
             f'<a href="{ctx.doc("finetune-clip-init-u9-v1.md")}">completed-results write-up</a>.</p>')
    return delta_card("clip-init-u9", claim, rows, GROUPS, x_label="Candidate minus its stated reference",
                      lead=verdict + lead + scores + target_table, notes=notes, table_html=paired,
                      table_summary="Paired comparison table")


def ensembles(ctx, j) -> tuple[str, dict]:
    titles = {e["key"]: e["title"] for e in j["ensembles"]}
    pairs = [(SWAP, ENS4, "Swap: CLIP-init u9 for from-clip u3", "ens"),
             (DEEP, ENS4, "Deep three: drop both shallow members", "ens"),
             (ENS4, REF, "Submitted ens4 − A30", "context")]
    rows = group_major([{"id": f"ens-{i}", "label": label, "group": group,
                         "lm": _pair(j, "lm_test", a, b), "gold": _pair(j, "gold", a, b)}
                        for i, (a, b, label, group) in enumerate(pairs)], ENS_GROUPS)
    pt, pl = j["point"], j["per_label"]
    swap_lm, swap_gold = _pair(j, "lm_test", SWAP, ENS4), _pair(j, "gold", SWAP, ENS4)
    deep_lm, deep_gold = _pair(j, "lm_test", DEEP, ENS4), _pair(j, "gold", DEEP, ENS4)

    def latmen(key):
        return num(pl["gold"][key]["Lateral Meniscus"], ".3f")

    history = data.read_json("leaderboard-history-v1.json") or {}
    submitted = {r["bundle"]: r for r in history.get("submissions", [])}
    bundles = {ENS4: "a30-ens4-rank", SWAP: "a30-ens4-swap-rank", DEEP: "a30-deep3-rank",
               REF: "r4a-k8-omb-soft-336-u9-a30"}

    def public(key):
        r = submitted.get(bundles[key], {})
        return num(r.get("public"), ".3f")

    claim = (f"Deep three leads public at {public(DEEP)}; the CLIP-init swap scored {public(SWAP)} "
             f"after the original ensemble's {public(ENS4)}")
    lead = (
        '<p><b>Question:</b> the run was built to replace the shallow from-clip member of the submitted 4-member '
        'ensemble (0.894 public). Does the swap keep the ensemble\'s LM-test gain while holding gold Lateral '
        'Meniscus nearer A30?</p>'
        f'<p><b>Result: a tie on the selector, a gain on the guard.</b> The swap moves {ci_text(swap_lm)} on LM-test '
        f'and {ci_text(swap_gold)} on gold-58; gold Lateral Meniscus goes from {latmen(ENS4)} to {latmen(SWAP)}. '
        'DINOv2-small, the other shallow member, still dilutes it. Dropping it as well (deep three) restores gold '
        f'Lateral Meniscus to {latmen(DEEP)} (A30: {latmen(REF)}) and moves gold-58 {ci_text(deep_gold)}, at '
        f'{ci_text(deep_lm)} LM-test. LM-test is the selector, so neither is a clear upgrade on paper; the swap was '
        'submitted to test whether the gold-58 gain transfers to the public board at equal LM-test.</p>'
        '<p><b>Public results: the swap scored 0.899</b> (submission 57291794), +0.005 over ens4\'s 0.894, where '
        'LM-test predicted a tie. <b>The deep three then scored 0.902</b> (57296356), the best so far, although its '
        'LM-test is 0.003 lower than the swap\'s; its gold-58 is 0.004 higher. Both times gold-58 called the '
        'direction and LM-test did not. Earlier, gold-58 pointed the wrong way for ens4 itself (−0.009 vs A30, then '
        '+0.004 public), and every gap here is inside LM-test\'s interval, so read this as: dropping the shallow '
        'members helps on the public split, not as gold-58 now being a selector. One submission each; the public '
        'split\'s noise is not estimated.</p>')
    tbl = table(["Ensemble", "LM-test", "Gold-58", "Public", "Gold Lateral Meniscus", "Worst LM-test label"],
                [[titles.get(k, k), num(pt["lm_test"][k]), num(pt["gold"][k]), public(k), latmen(k),
                  num(min(v for v in pl["lm_test"][k].values() if v is not None), ".3f")]
                 for k in (ENS4, SWAP, DEEP)] +
                [["A30 alone", num(pt["lm_test"][REF]), num(pt["gold"][REF]), public(REF), latmen(REF),
                  num(min(v for v in pl["lm_test"][REF].values() if v is not None), ".3f")]],
                num_cols=(1, 2, 3, 4, 5))
    notes = (f'<p>Per-label rank mean of the members\' saved predictions, the rule the Kaggle bundle uses. Paired '
             f'study bootstrap, {j["bootstrap"]["resamples"]:,} resamples, on the same draws as the single-run card. '
             'The member sets were picked on this LM-test, so expect mild optimism. CoAtNet-384 is left out: the '
             'bundle exporter cannot package timm members.</p>'
             f'<p>Source: <a href="{ctx.raw(SRC)}">scores and paired intervals</a>.</p>')
    return delta_card("clip-init-u9-ensembles", claim, rows, ENS_GROUPS, x_label="Ensemble minus its stated reference",
                      lead=lead + tbl, notes=notes)


def curves(ctx, j) -> tuple[str, dict]:
    sid = "clip-init-u9-curves"
    by_key = {r["key"]: r for r in j["runs"]}
    runs = [by_key[k] for k in (BASE, REF, SHALLOW, ARM) if k in by_key]
    blob = {"groups": CURVE_GROUPS, "defaultPinned": [r["key"] for r in runs]}
    for key, metric in (("train", "train_masked_bce"), ("val", "validation_masked_bce"),
                        ("auc", "validation_macro_auroc")):
        blob[key] = [{"id": r["key"], "label": r["title"], "group": "clip" if r["key"] in (ARM, SHALLOW) else "ref",
                      "points": [[h["epoch"], h[metric]] for h in r["history"] if metric in h]}
                     for r in runs if any(metric in h for h in r["history"])]
    body = (HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            + ''.join(f'<div>{chart(key, title)}</div>' for key, title in (
                ("train", "Training BCE by epoch"),
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
    claim = (f"Training curves: the CLIP init reaches a lower validation BCE than A30 "
             f"({num(arm['validation_masked_bce'])} against {num(ref['validation_masked_bce'])}, epoch "
             f"{arm['best_epoch']}), which the test sets do not reflect")
    notes = ('<p>Both A30-recipe runs average their training BCE over every draw, including the oversampled '
             'repeats, so they sit below u9-b1 and from-clip u3 from epoch 1. The from-clip u3 run uses the run-5 '
             'recipe (20 epochs). No I→T / T→I recall curves: these are label-only fine-tuning runs with no '
             'retrieval objective; the contrastive stage behind the CLIP init has its own on the Phase-1 pages.</p>')
    return section(sid, claim, body, notes=notes), {f"data-{sid}": blob}


def per_label(ctx, j) -> tuple[str, dict]:
    sid = "clip-init-u9-labels"
    labels = list(j["per_label"]["lm_test"][REF])
    titles = {r["key"]: r["title"] for r in j["runs"]}
    titles.update({e["key"]: e["key"] for e in j["ensembles"]})
    present = [k for k in (REF, SHALLOW, ARM, ENS4, SWAP, DEEP) if k in j["per_label"]["lm_test"]]
    deltas = [("Seed noise (run 5)", SEED, SEED_REF), ("CLIP u9 − A30", ARM, REF),
              ("CLIP u9 − CLIP u3", ARM, SHALLOW), ("Swap − ens4", SWAP, ENS4), ("Deep three − ens4", DEEP, ENS4)]
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
            cells += [num(pl[ARM][label] - pl[REF][label], ".3f", True), num(pl[SWAP][label] - pl[ENS4][label], ".3f", True)]
        rows.append(cells)
    headers = ["Label", "CLIP − A30 LM", "Swap − ens4 LM", "CLIP − A30 gold", "Swap − ens4 gold"]
    caption = ("Cells default to deltas; the first column, the run-5 seed replicate, shows how far a label moves from "
               "training noise alone. ★ marks A30's oversampled labels. Point estimates; the starred labels' "
               "intervals are in the first section.")
    return (section(sid, "Per label: the swap trades a little Contusion for Lateral OA and Medial Meniscus on gold; "
                         "no LM-test label drops by more than 0.01", body, caption=caption,
                    table_html=table(headers, rows, num_cols=tuple(range(1, len(headers)))),
                    table_summary="All 12 labels · CLIP-init u9 vs A30, swap vs ens4"), {f"data-{sid}": blob})


def build(ctx) -> tuple[list[str], dict]:
    sections, blob = [], {}
    j = data.read_json(SRC)
    if j:
        for part in (results, ensembles, curves, per_label):
            html, b = part(ctx, j)
            sections.append(html); blob.update(b)
    return sections, blob
