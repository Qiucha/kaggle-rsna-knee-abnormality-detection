"""Phase 2 · ensembles: which fine-tuned runs to rank-average, and how to choose them.

Four sections from digest/ensemble-selection-v1.json (scripts/build_ensemble_selection_digest.py):
combinations vs u9 alone, direct design comparisons, selection rules judged out of sample, per-label views.
"""

from __future__ import annotations

from .. import data
from ..shell import chart, esc, group_major, num, section, table
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card, excludes_zero

SRC = "ensemble-selection-v1.json"
DESIGNS = {"sibling stack": {"label": "OmniRad siblings stacked", "color": "var(--g-sky)"},
           "backbone mix": {"label": "u9 + other backbones only", "color": "var(--g-violet)"},
           "strong core + other backbone": {"label": "Strong core + other backbones", "color": "var(--g-lime)"},
           "input-size mix": {"label": "Input sizes mixed", "color": "var(--g-amber)"},
           "label-type mix": {"label": "Hard + soft labels", "color": "var(--g-rose)"},
           "seed average": {"label": "Seed average", "color": "var(--g-teal)"},
           "previous candidate": {"label": "Earlier candidates", "color": "var(--g-anchor)"}}
STRATEGIES = {"top-k (best macro)": "var(--g-anchor)", "cover: mean of per-label best": "var(--g-lime)",
              "cover: worst-label best": "var(--g-amber)", "greedy ensemble": "var(--g-violet)",
              "route": "var(--g-rose)"}


def _d(x: dict) -> dict:
    return {"difference": x["difference"], "ci95": x["ci95"]}


def combos_section(ctx, j) -> tuple[str, dict]:
    combos = [c for c in j["combos"] if c["members"] != [j["reference"]]]
    rows = [{"id": c["name"], "label": f"{c['name']} ({len(c['members'])})", "group": c["group"],
             "lm": _d(c["vs_reference"]["lm_test"]["bootstrap_seed_101"]), "gold": _d(c["vs_reference"]["gold"])}
            for c in combos]
    best = max(combos, key=lambda c: c["lm_test"]["macro"])
    by = {c["name"]: c for c in combos}
    sib = max((c for c in combos if c["group"] == "sibling stack"), key=lambda c: c["lm_test"]["macro"])
    n_pass = sum(excludes_zero(r["lm"]) for r in rows)
    claim = (f"{n_pass} of {len(rows)} ensembles beat u9 alone on LM-test; the best ones mix a strong OmniRad core "
             f"with other backbones ({esc(best['name'])}: {num(best['lm_test']['macro'])}, "
             f"{ci_text(best['vs_reference']['lm_test']['bootstrap_seed_101'], '.3f')}), while a sibling stack tops out "
             f"at {num(sib['lm_test']['macro'])}")
    veto = [c["name"] for c in combos if c["vs_reference"]["gold"]["difference"] <= -0.024]
    lead = (f"<p>Each row is one hand-picked combination, built on one design principle, minus u9 alone "
            f"(unfreeze 9 blocks, LM-test {num(j['combos'][0]['lm_test']['macro'])}, public 0.888; A30 later scored 0.890). "
            f"LM-test interval excludes zero for {n_pass} of {len(rows)} combinations. Gold-58 sits below u9 for every "
            f"ensemble because u9's own gold-58 score is high; it vetoes (≤ −0.024) only "
            f"{', '.join(esc(v) for v in veto) or 'none'}.</p>")
    trs = []
    for c in j["combos"]:
        v = c["vs_reference"]["lm_test"]
        rng = c.get("member_seed_range")
        trs.append([DESIGNS[c["group"]]["label"] if c["group"] in DESIGNS else c["group"], c["name"], str(len(c["members"])),
                    num(c["lm_test"]["macro"]), ci_text(v["bootstrap_seed_101"]), ci_text(v["bootstrap_seed_202"]),
                    ci_text(v["bootstrap_seed_303"]), ci_text(c["vs_built_trio_lm_test"]), num(c["gold"]["macro"]),
                    ci_text(c["vs_reference"]["gold"]), num(c["lm_test"]["worst_label"]),
                    f"{min(rng):.4f}–{max(rng):.4f}" if rng else "—"])
    tbl = table(["Design", "Combination", "n", "LM-test", "Δ vs u9 (boot seed 101)", "seed 202", "seed 303",
                 "Δ vs built trio", "Gold-58", "Gold Δ vs u9", "Worst label", "Member-seed range"], trs)
    names = ", ".join(f"<b>{esc(k)}</b> = <code>{esc(v)}</code>" for k, v in j["short_names"].items())
    notes = (f"<p>Rank-mean ensembles; paired study bootstrap, 2,000 resamples under three bootstrap seeds (they agree "
             f"to ±0.0003). <i>Built trio</i> = the 3-member bundle exported on 2026-10-02 and never submitted. "
             f"<i>Member-seed range</i> = LM-test when run 5 or OmniRad-base 224 is swapped for its seed-2027 retrain. "
             f"Combinations were picked after the single-run scores were known; the out-of-sample study below suggests "
             f"about 0.004 shrinkage.</p><p>Members: {names}.</p>"
             f'<p>Sources: <a href="{ctx.raw(SRC)}">snapshot</a>, <a href="{ctx.doc("ensemble-selection-v1.md")}">summary</a>.</p>')
    used = {r["group"] for r in rows}
    return delta_card("ensembles", claim, rows, {k: v for k, v in DESIGNS.items() if k in used},
                      x_label="Ensemble minus u9 alone", lead=lead, notes=notes, table_html=tbl,
                      table_summary="All combinations, three bootstrap seeds")


def pairs_section(ctx, j) -> tuple[str, dict]:
    groups = {"design": {"label": "Other backbones vs more siblings", "color": "var(--g-lime)"},
              "within": {"label": "Among diverse sets", "color": "var(--g-anchor)"}}
    rows = []
    for i, p in enumerate(j["pairs"]):
        rows.append({"id": f"pair{i}", "label": p["question"], "group": "design" if "sibling" in p["question"] else "within",
                     "lm": _d(p["lm_test"]), "gold": _d(p["gold"])})
    main = j["pairs"][0]
    claim = (f"At equal size, swapping OmniRad siblings for other backbones gains "
             f"{ci_text(main['lm_test'], '.4f')} LM-test (A beats B in {main['lm_test']['p_positive']:.0%} of resamples); "
             f"which diverse set you pick is noise")
    trs = [[p["question"], " + ".join(p["a"]), " + ".join(p["b"]), ci_text(p["lm_test"]),
            f"{p['lm_test']['p_positive']:.2f}", ci_text(p["gold"])] for p in j["pairs"]]
    tbl = table(["Question", "A", "B", "Δ LM-test (A − B)", "P(A > B)", "Δ gold-58"], trs)
    notes = "<p>Paired study bootstrap, 2,000 resamples (seed 101; seed 202 agrees to ±0.0003).</p>"
    return delta_card("ensembles-design", claim, rows, groups, x_label="Ensemble A minus ensemble B",
                      notes=notes, table_html=tbl, table_summary="Direct comparisons")


def selection_section(ctx, j) -> tuple[str, dict]:
    sel = j["selection"]
    sid = "ensembles-selection"
    groups = {}
    rows = []
    for s in sel["strategies"]:
        if s["strategy"].startswith("top-k") or s["k"] < 2:
            continue
        key = "route" if s["strategy"].startswith("route") else s["strategy"]
        groups.setdefault(key, {"label": "per-label routing" if key == "route" else key, "color": STRATEGIES[key]})
        rows.append({"id": f"{s['strategy']}|{s['k']}", "label": f"k={s['k']} · {s['strategy']}", "group": key,
                     "est": s["delta_vs_top_k"]["difference"], "lo": s["delta_vs_top_k"]["ci95"][0],
                     "hi": s["delta_vs_top_k"]["ci95"][1]})
    pooled = next(r for r in sel["k5_seeds"] if r["split_seed"] == "pooled")
    for key, name in (("coverage_minus_top5", "cover: mean of per-label best"), ("greedy_minus_top5", "greedy ensemble")):
        d = pooled[key]
        rows.append({"id": f"k5-{key}", "label": f"k=5 · {name} (5 split seeds)", "group": name,
                     "est": d["difference"], "lo": d["ci95"][0], "hi": d["ci95"][1]})
    rows.sort(key=lambda r: (r["label"].split(" · ")[0], r["group"]))
    p = sel["per_label_winner_persistence"]
    cg = pooled["coverage_minus_greedy"]
    claim = (f"Picking members to cover each label is no better than greedy: at k = 5, coverage − greedy = "
             f"{ci_text(cg, '.4f')} held out; per-label 'specialists' do not replicate")
    lead = (f"<p>Selection rules judged out of sample: LM-test is split in half {sel['half_splits']} times; each rule "
            f"picks members on one half and is scored on the other. A run that beats the best-macro run on one label "
            f"by {num(p['selection_edge'], '.3f', True)} on the selection half keeps {num(p['held_out_edge'], '.3f', True)} "
            f"on the held-out half (positive in {p['held_out_positive']:.0%} of {p['cases']:,} cases, correlation "
            f"{num(p['correlation'], '.2f', True)}): per-label AUROC on ~216 studies is too noisy to name specialists.</p>")
    body = (lead + HELPERS + '<div data-role="filter"></div>' + chart("sel", "Held-out macro AUROC minus top-k by macro, 2.5–97.5% over splits")
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.ciDotPlot(sec.querySelector('[data-chart=sel]'), d.rows,
    {groups: d.groups, ref: 0, refLabel: 'same as top-k', fmt: '+.4f', axisFmt: '+.3f', sort: false,
     store: store, labelWidth: 300, xLabel: 'Rule minus top-k (same k), held-out half'});"""))
    k5 = [[str(r["split_seed"]), str(r["splits"]), num(r["top5"]), num(r["coverage"]), num(r["greedy"]),
           ci_text(r["coverage_minus_greedy"]), f"{r['coverage_minus_greedy']['p_positive']:.2f}",
           ci_text(r["coverage_minus_top5"]), ci_text(r["greedy_minus_top5"])] for r in sel["k5_seeds"]]
    strat = [[s["strategy"], str(s["k"]), num(s["held_out_macro"]), ci_text(s["delta_vs_top_k"]), f"{s['p_beats_top_k']:.2f}",
              num(s["held_out_worst_label"]), f"{s['distinct_runs']:.1f}"]
             for s in sorted(sel["strategies"], key=lambda s: (s["k"], s["strategy"]))]
    pr = sel["k5_pick_rate"]
    picks = sorted(set(pr["top5"]) | set(pr["coverage"]) | set(pr["greedy"]),
                   key=lambda r: -(pr["coverage"].get(r, 0) + pr["greedy"].get(r, 0)))[:12]
    tbl = (table(["Split seed", "Splits", "Top-5", "Coverage", "Greedy", "Coverage − greedy", "P(coverage > greedy)",
                  "Coverage − top-5", "Greedy − top-5"], k5)
           + table(["Rule", "k", "Held-out macro", "Δ vs top-k", "P(beats top-k)", "Held-out worst label", "Distinct runs"], strat)
           + table(["Run", "Top-5 pick rate", "Coverage", "Greedy"],
                   [[r, f"{pr['top5'].get(r, 0):.2f}", f"{pr['coverage'].get(r, 0):.2f}", f"{pr['greedy'].get(r, 0):.2f}"] for r in picks]))
    notes = ("<ul><li><i>Coverage</i> greedily adds the run that most raises the mean over labels of the best member's "
             "AUROC; <i>worst-label</i> raises the weakest label instead; <i>greedy</i> adds the run that most raises "
             "the ensemble's own macro AUROC; <i>routing</i> scores each label with only its own top-k runs (9–22 "
             "distinct runs, not deployable).</li><li>Split seeds re-split the same 432 studies, so they show "
             "Monte-Carlo stability, not new data. Both rules beat top-5 slightly because they pull in other backbones "
             "(CoAtNet, DINOv2-small) where top-5 stacks OmniRad siblings — see the pick rates.</li></ul>")
    return section(sid, claim, body, notes=notes, table_html=tbl,
                   table_summary="k = 5 per split seed, all rules, pick rates"), {f"data-{sid}": {"rows": group_major(rows, groups), "groups": groups}}


def per_label_section(ctx, j) -> tuple[str, dict]:
    sid = "ensembles-per-label"
    labels = j["labels"]
    combos = [c for c in j["combos"] if c["members"] != [j["reference"]]]
    singles = j["singles"]
    short = {v: k for k, v in j["short_names"].items()}
    blob = {"rows": labels,
            "combos": {"cols": [c["name"] for c in combos],
                       **{split: {"delta": [[c[split]["delta_per_label"][l] for c in combos] for l in labels],
                                  "level": [[c[split]["per_label"][l] for c in combos] for l in labels]}
                          for split in ("lm_test", "gold")}},
            "singles": {"cols": [short.get(s["run"], s["run"].replace("r4a-k8-", "").replace("omnirad-base", "omb"))
                                 for s in singles],
                        **{split: {"level": [[s[split]["per_label"][l] for s in singles] for l in labels]}
                           for split in ("lm_test", "gold")}}}
    claim = ("Per label: ensembles lift most labels over u9 on LM-test; the single-run map shows each label's best run "
             "is scattered across many runs, the pattern that does not replicate out of sample")
    body = (HELPERS + '<div data-role="cols"></div><div data-role="split"></div><div data-role="mode"></div>'
            + '<div data-chart-scope>' + chart("heat", "Per-label AUROC") + '</div>'
            + section_script(sid, """
  var cols = 'combos', split = 'lm_test', mode = 'delta';
  function draw() {
    var src = d[cols], m = cols === 'singles' ? src[split].level : src[split][mode];
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var o = {rows: d.rows, cols: src.cols, rowIds: d.rows, maxColChars: 34,
             fmt: (cols === 'combos' && mode === 'delta') ? '+.3f' : '.3f',
             ariaLabel: 'per-label ' + (split === 'gold' ? 'gold-58' : 'LM-test') + ' heatmap'};
    if (cols === 'combos' && mode === 'delta') { o.diverging = true; o.peakPerRow = false; }
    Charts.heatmap(el, m, o);
  }
  AdaptUI.seg(sec.querySelector('[data-role=cols]'), 'Columns',
    [{key: 'combos', label: 'Ensembles'}, {key: 'singles', label: 'Single runs (sorted by macro)'}], 'combos',
    function (k) { cols = k; draw(); });
  AdaptUI.seg(sec.querySelector('[data-role=split]'), 'Scored on',
    [{key: 'lm_test', label: 'LM-test (432)'}, {key: 'gold', label: 'Gold-58 (human)'}], 'lm_test',
    function (k) { split = k; draw(); });
  AdaptUI.seg(sec.querySelector('[data-role=mode]'), 'Ensemble cells',
    [{key: 'delta', label: 'Δ vs u9'}, {key: 'level', label: 'AUROC'}], 'delta',
    function (k) { mode = k; draw(); });
  draw();"""))
    caption = ("Rows are labels. <i>Ensembles</i>: each combination's per-label AUROC minus u9's (or raw AUROC). "
               "<i>Single runs</i>: raw AUROC for all fine-tuned runs, highest per label marked. Per-label values are "
               "point estimates without intervals.")
    return section(sid, claim, body, caption=caption), {f"data-{sid}": blob}


def a30_section(ctx, *, clip_init=False) -> tuple[str, dict]:
    j = data.read_json("ensemble-a30-v1.json")
    if not j:
        return "", {}
    combos = j["clip_init_sweep"]["combos"] if clip_init else j["combos"]
    colors = ["var(--g-anchor)", "var(--g-sky)", "var(--g-teal)", "var(--g-violet)",
              "var(--g-lime)", "var(--g-amber)", "var(--g-rose)"]
    names = list(dict.fromkeys(c["group"] for c in combos if c["group"] != "single"))
    groups = {name: {"label": name.capitalize(), "color": colors[i]} for i, name in enumerate(names)}

    def interval(values):
        return {"difference": values[0], "ci95": values[1:]}

    rows = group_major([{"id": c["name"], "label": c["name"], "group": c["group"],
                         "lm": interval(c["d101"]), "gold": interval(c["d_gold"])}
                        for c in combos if c["group"] != "single"], groups)
    tbl = table(["Design", "Combination", "Members", "LM-test", "Δ vs A30 [95% CI], seed 101",
                 "Seed 202", "Gold-58", "Gold Δ vs A30 [95% CI]", "Gold Lateral Meniscus", "Worst LM label"],
                [[c["group"], c["name"], len(c["members"]), num(c["lm"]), ci_text(interval(c["d101"])),
                  ci_text(interval(c["d202"])), num(c["gold"]), ci_text(interval(c["d_gold"])),
                  num(c["gold_latmen"], ".3f"), num(c["worst"], ".3f")] for c in combos])
    lead = ('<p>The completed 2026-10-14 sweep compares 17 combinations around A30, including the single-model '
            'reference and the earlier best ensemble. A30 + u9 + shallow CLIP + CoAtNet + DINOv2-S reaches '
            '0.9063 LM-test, +0.0146 [+0.0079, +0.0218] over A30, with gold-58 0.8760. The highest LM-test '
            'point estimate is 0.9065 after replacing u9 with soft R4a; this does not establish it as better '
            'than the other strong-core sets.</p>'
            '<p>CoAtNet could not be packaged by the exporter. The four-member set without it was submitted '
            'and scored 0.894 public. <a href="#clip-init-u9-ensembles">The later CLIP-init u9 swap and deep '
            'three scored 0.899 and 0.902</a>; those newer results supersede the sweep as the submission choice. '
            'A high local sweep score is not a public score.</p>')
    notes = (f'<p>Source: <a href="{ctx.raw("ensemble-a30-v1.json")}">full sweep snapshot</a>. '
             'Paired study bootstrap, 2,000 resamples; two bootstrap seeds use the same studies. Member sets '
             'were picked using these LM-test results. Gold-58 is exploratory, with a veto at Δ ≤ −0.024 '
             'against A30. <code>clip</code> means the older from-CLIP u3 member; <code>coat</code> is CoAtNet-384, '
             '<code>dS336</code> is DINOv2-small soft 336, <code>384</code> is OmniRad-base soft 384, '
             '<code>softR4a</code> is DINOv2-small soft 224, and <code>A30aug</code> is A30 with image augmentation. '
             'Historical sweep deltas are bootstrap means; AUROC levels use the full population. '
             'Ensembles reuse member histories; their training curves are in the corresponding run sections.</p>')
    sid = "a30-ensemble-sweep"
    claim = "A30 ensemble sweep: strong cores improve LM-test; later public tests favor deep members"
    if clip_init:
        sid = "clip-init-ensemble-sweep"
        claim = "CLIP-init ensemble sweep: all 12 combinations, including swaps, additions and deep-only members"
        lead = ('<p>The completed 2026-10-16 sweep adds the new CLIP-init u9 member (<code>clipU9</code>). '
                'Every interval here is against A30 alone, using bootstrap seeds 101 and 202 for LM-test and '
                '7 for gold. The <a href="#clip-init-u9-ensembles">matched submission comparison above</a> '
                'uses the original ens4 as its reference and bootstrap seed 2030; keep those baselines distinct.</p>'
                '<p>Adding CLIP-init u9 alongside the old shallow CLIP member reaches 0.9063 LM-test. '
                'That local score is not a public result: only the original ens4, the four-member swap '
                'and the deep three were submitted. Their public scores are 0.894, 0.899 and 0.902.</p>')
    else:
        check = j["exportability_check"]
        tbl += ('<h3>Completed exportability check · 2026-10-15</h3>'
                '<p>Eight recorded combinations compare the five-member CoAtNet pick with variants the exporter '
                'could package. Values retain the precision of the original run record.</p>'
                + table(check["headers"], check["rows"]))
    return delta_card(sid, claim,
                      rows, groups, x_label="Ensemble minus A30 alone", lead=lead, notes=notes,
                      table_html=tbl, table_summary=f"All {len(combos)} combinations and paired intervals")


def build(ctx) -> tuple[list[str], dict]:
    j = data.read_json(SRC)
    if not j:
        return [], {}
    out, blobs = [], {}
    for html, blob in (a30_section(ctx), a30_section(ctx, clip_init=True), combos_section(ctx, j),
                       pairs_section(ctx, j), selection_section(ctx, j), per_label_section(ctx, j)):
        out.append(html)
        blobs.update(blob)
    return out, blobs
