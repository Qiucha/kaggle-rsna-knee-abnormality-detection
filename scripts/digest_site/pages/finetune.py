"""Supervised fine-tuning ladder: R0 → R1–R3 → R1c → random slices → R4 multi-slice → K → backbones.

Every step is shown against the step it was built from (its "parent"), not the run before it, and
again against the fixed R0 baseline.
Sections: the step-by-step deltas, training curves, per-label heatmaps (gold-58 and LM-test,
Δ vs R0 or raw AUROC), and R4's window masking. Source: digest/finetune-ladder-v1.json, written by
scripts/build_finetune_ladder_digest.py from checkpoints/finetune/.
"""

from __future__ import annotations

from .. import data
from ..shell import Page, caveat_badge, chart, esc, notice, num, section, table
from ..sitemap import BY_SLUG
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script

SLUG = "finetune"
SRC = "finetune-ladder-v1.json"
DESIGN_MD = "finetune-r4-multislice-design.md"
BACKBONES_MD = "research/mri-pretrained-backbones.md"

GROUPS = {"baseline": {"label": "R0 baseline (frozen encoder)", "color": "var(--g-anchor)"},
          "ladder": {"label": "R1–R3, R1c: fine-tuning recipe", "color": "var(--g-violet)"},
          "slices": {"label": "Slice sampling and multi-slice (R4)", "color": "var(--g-teal)"},
          "backbones": {"label": "Backbone swaps (R4a K8, pretrained start)", "color": "var(--g-amber)"},
          "labels": {"label": "Soft labels, seed replicates and 336 px (R4a K8)", "color": "var(--g-rose)"},
          "combined": {"label": "Combined: OmniRad-base + soft + 336 px", "color": "var(--g-lime)"},
          "diagnostic": {"label": "Backbone diagnostic: CoAtNet-384 (ticket 16)", "color": "var(--g-sky)"}}


def _ci(lo: float, hi: float) -> str:
    return f"[{num(lo, '.3f', True)}, {num(hi, '.3f', True)}]"


def _excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def _verdict(step: dict, bar_guard: float = -0.024) -> str:
    """The ticket-03 bar applied to this step against its parent."""
    v = step.get("vs_parent")
    if not v:
        return "baseline"
    lm, gold = v["lm_test"], v["gold"]
    guard = gold["difference"] >= bar_guard
    if _excludes_zero(*lm["ci95"]) and lm["difference"] > 0:
        return "passes" if guard else "fails gold guard"
    if _excludes_zero(*lm["ci95"]):
        return "worse"
    return "no clear change"


def _load():
    j = data.read_json(SRC)
    return j, ({s["id"]: s for s in j["steps"]} if j else {})


# --------------------------------------------------------------------------- 1. steps


def _combined_note(by_id) -> str:
    """The stacked OmniRad-base + soft + 336 px run, which changes two things at once."""
    s = by_id.get("OmniRad-B-soft-336")
    if not s:
        return ""
    lm, gold = s["vs_parent"]["lm_test"], s["vs_parent"]["gold"]
    return ("<li><strong>OmniRad-B soft 336 (2026-10-02)</strong> stacks soft labels and 336 px on OmniRad-base. "
            f"It has the best LM-test ({num(s['lm_test']['macro'])}) and gold-58 ({num(s['gold']['macro'])}) of any "
            f"single run, but against OmniRad-base the LM-test gain is {num(lm['difference'], '.4f', True)} "
            f"{_ci(*lm['ci95'])}, which does not pass the bar on its own; gold {num(gold['difference'], '.3f', True)} "
            f"{_ci(*gold['ci95'])}. Its value is as an ensemble member: see ticket 14 in the leaderboard map.</li>")


def _coatnet_note(by_id) -> str:
    """Ticket 16: does swapping in CoAtNet-384 close the gap to the 0.924 public single model?"""
    s = by_id.get("CoAtNet-384")
    if not s:
        return ""
    lm, gold = s["vs_parent"]["lm_test"], s["vs_parent"]["gold"]
    r0 = s["vs_r0"]["lm_test"]
    epoch = s["best_epoch"]
    train = {k: next(h["train_masked_bce"] for h in by_id[k]["history"] if h["epoch"] == epoch)
             for k in ("CoAtNet-384", "OmniRad-B-soft-336")}
    return ("<li><strong>CoAtNet 384 (2026-10-02, ticket 16)</strong> asked whether the backbone explains the gap to "
            "a reported 0.924 public CoAtNet single model; a gain of about +0.05 LM-test would have said yes. It lands "
            f"the other way: {num(lm['difference'], '.3f', True)} {_ci(*lm['ci95'])} LM-test and "
            f"{num(gold['difference'], '.3f', True)} {_ci(*gold['ci95'])} gold against run 5 (OmniRad-B soft 336), "
            f"and {num(r0['difference'], '.3f', True)} {_ci(*r0['ci95'])} LM-test against the frozen R0. "
            "That rules out the plain swap, not the backbone. On the same soft targets its training BCE at epoch "
            f"{epoch} is {num(train['CoAtNet-384'], '.3f')} against run 5's {num(train['OmniRad-B-soft-336'], '.3f')}, "
            "so it underfits: only 3 of 24 blocks train, and stages 0–2 keep ImageNet features. The 0.924 reference "
            "fine-tuned every block for 16 epochs with top-3 weight averaging, on private labels. A deeper-unfreeze "
            "arm is queued.</li>")


def _label_group_note(j, by_id) -> str:
    """One bullet that reads the soft-label, seed and 336 px steps against their parents."""
    def d(sid, split, fmt=".3f"):
        v = by_id[sid]["vs_parent"][split]
        lo, hi = v["ci95"]
        return f"{num(v['difference'], fmt, True)} [{num(lo, fmt, True)}, {num(hi, fmt, True)}]"
    needed = ("R4a-soft", "R4a-K8-s27", "OmniRad-B-s27", "DINOv2-S-336")
    if not all(k in by_id for k in needed):
        return ""
    return ("<li><strong>Soft labels, seeds and 336 px (2026-10-01).</strong> "
            f"Soft labels leave LM-test flat ({d('R4a-soft', 'lm_test')}) but lift gold-58 {d('R4a-soft', 'gold')}, "
            f"and move the best epoch from {by_id['R4a-K8']['best_epoch']} to {by_id['R4a-soft']['best_epoch']}. "
            f"The two seed-2027 replicates differ from seed 2026 by {d('R4a-K8-s27', 'lm_test')} (R4a K8) and "
            f"{d('OmniRad-B-s27', 'lm_test')} (OmniRad-B) on LM-test, so seed spread on this recipe is small. "
            f"336 px passes the bar by a hair: LM-test {d('DINOv2-S-336', 'lm_test', '.4f')}, gold {d('DINOv2-S-336', 'gold')}, "
            f"at {by_id['DINOv2-S-336']['minutes']:.0f} training minutes against {by_id['DINOv2-S']['minutes']:.0f}.</li>")


def steps_section(ctx, j, by_id) -> tuple[str, dict]:
    sid = "steps"
    rows = [s for s in j["steps"] if s.get("vs_parent")]
    blob = {"groups": GROUPS,
            "lm": [{"id": s["id"], "label": f"{s['short']} vs {by_id[s['parent']]['short']}", "group": s["group"],
                    "est": s["vs_parent"]["lm_test"]["difference"], "lo": s["vs_parent"]["lm_test"]["ci95"][0],
                    "hi": s["vs_parent"]["lm_test"]["ci95"][1], "level": s["lm_test"]["macro"]} for s in rows],
            "gold": [{"id": s["id"], "label": f"{s['short']} vs {by_id[s['parent']]['short']}", "group": s["group"],
                      "est": s["vs_parent"]["gold"]["difference"], "lo": s["vs_parent"]["gold"]["ci95"][0],
                      "hi": s["vs_parent"]["gold"]["ci95"][1], "level": s["gold"]["macro"]} for s in rows]}
    # The same steps against one fixed baseline (R0), so their gains stack up on a common zero.
    for split, key in (("lm_test", "lm_r0"), ("gold", "gold_r0")):
        blob[key] = [{"id": s["id"], "label": s["short"], "group": s["group"],
                      "est": s["vs_r0"][split]["difference"], "lo": s["vs_r0"][split]["ci95"][0],
                      "hi": s["vs_r0"][split]["ci95"][1], "level": s[split]["macro"]} for s in rows]
    body = (HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            f'<div>{chart("lm", "Δ LM-test macro AUROC vs the step it changes")}</div>'
            f'<div>{chart("gold", "Δ gold-58 macro AUROC vs the step it changes")}</div></div>'
            '<div class="grid-2">'
            f'<div>{chart("lm_r0", "Δ LM-test macro AUROC vs R0 (fixed frozen baseline)")}</div>'
            f'<div>{chart("gold_r0", "Δ gold-58 macro AUROC vs R0 (fixed frozen baseline)")}</div></div>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  [['lm', 'LM-test', 'the step it changes'], ['gold', 'gold-58', 'the step it changes'],
   ['lm_r0', 'LM-test', 'R0'], ['gold_r0', 'gold-58', 'R0']].forEach(function (spec) {
    Charts.ciDotPlot(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, ref: 0, sort: false, fmt: '+.3f', axisFmt: '+.2f',
       xLabel: '\\u0394 ' + spec[1] + ' macro AUROC vs ' + spec[2] + ' (95% CI)', store: store,
       tooltipExtra: function (r) { return '<div>Level ' + Charts.fmt('.4f')(r.level) + '</div>'; }});
  });"""))

    trs, attrs = [], []
    for s in j["steps"]:
        v = s.get("vs_parent") or {}
        lm, gold = v.get("lm_test"), v.get("gold")
        trs.append([f"<strong>{esc(s['short'])}</strong>", esc(s["change"]),
                    esc(by_id[s["parent"]]["short"]) if s.get("parent") else "—",
                    num(s["lm_test"]["macro"]),
                    f"{num(lm['difference'], '.4f', True)} {_ci(*lm['ci95'])}" if lm else "—",
                    num(s["gold"]["macro"]),
                    f"{num(gold['difference'], '.4f', True)} {_ci(*gold['ci95'])}" if gold else "—",
                    esc(_verdict(s)), str(s.get("best_epoch", "—")),
                    num(s["public"]["score"], ".3f") if s.get("public") else "—"])
        attrs.append({"data-series-id": s["id"], "data-group": s["group"]})
    tbl = table(["Step", "What it changes from its parent", "Parent", "LM-test", "Δ LM-test [95% CI]",
                 "Gold-58", "Δ gold [95% CI]", "Ticket-03 bar", "Best epoch",
                 "Public LB"],
                trs, raw_html=True, num_cols=(3, 5, 8, 9), row_attrs=attrs)

    best = max(j["steps"], key=lambda s: s["lm_test"]["macro"])
    r4 = by_id.get("R4a-K8")
    claim = (f"Multi-slice input (R4a) is the largest single gain, "
             f"{num(r4['vs_parent']['lm_test']['difference'], '.3f', True)} LM-test over random-slice R1c; "
             f"fine-tuning alone (R1–R3) lost to the frozen baseline"
             + " " + caveat_badge("single seed per step"))
    caption = (f"Top row: each dot is one step minus its parent (the run it was built from), so each row has its own zero. "
               f"Bottom row: the same steps minus R0, the frozen-encoder baseline, so every dot shares one zero and "
               f"the gains read cumulatively. Both on {j['lm_test_studies']} LM-test "
               f"and {j['gold_studies']} gold studies. {esc(j['bootstrap'])} The full table says what each step changes. "
               f"Best LM-test so far: {esc(best['short'])} ({num(best['lm_test']['macro'])}).")
    also = [s for s in j["steps"] if s.get("vs_also")]
    notes = ("<ul>"
             f"<li><strong>Bar.</strong> {esc(j['bar'])} A step only passes against its own parent, so a backbone swap "
             "is judged against the DINOv2-small arm with the same pretrained start, not against R4a.</li>"
             "<li><strong>Parents are not run order.</strong> R1c changes R1 (start from R0's encoder), not R3; "
             "the three backbone swaps and the 336 px run each change the pretrained-DINOv2 arm.</li>"
             + "".join(f"<li><strong>{esc(s['short'])} vs {esc(by_id[s['also']]['short'])}:</strong> LM-test "
                       f"{num(s['vs_also']['lm_test']['difference'], '.4f', True)} {_ci(*s['vs_also']['lm_test']['ci95'])}, "
                       f"gold {num(s['vs_also']['gold']['difference'], '.4f', True)} {_ci(*s['vs_also']['gold']['ci95'])}.</li>"
                       for s in also)
             + _label_group_note(j, by_id)
             + _combined_note(by_id)
             + _coatnet_note(by_id)
             + "<li>Gold-58 has been examined many times and rests on few positives per label: treat it as exploratory. "
               "LM-test scores agreement with LM-generated labels, not radiologists, and the two disagree on OmniRad-small.</li>"
             "<li>Every step is one training seed (2026), except the seed-2027 replicates, which repeat their parent "
             "exactly. A gap under ~0.05 needs a 3-seed rerun before it is trusted.</li>"
             "<li><strong>Public LB</strong> is the Kaggle public macro ROC-AUC for the steps that were submitted. "
             "OmniRad-base tied R4a on LM-test but led on gold-58, and it also led on the public board.</li>"
             f"<li>Sources: <a href=\"{ctx.raw(SRC)}\">{SRC}</a>, "
             f"<a href=\"{ctx.doc(DESIGN_MD)}\">R4 design</a>, <a href=\"{ctx.doc(BACKBONES_MD)}\">backbone survey</a>.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"All {len(j['steps'])} steps: what changes, scores and deltas"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 2. curves


def curves_section(ctx, j) -> tuple[str, dict]:
    sid = "curves"
    series = {"train": [], "val": [], "auc": []}
    for s in j["steps"]:
        h = s.get("history")
        if not h:
            continue
        for key, metric in (("train", "train_masked_bce"), ("val", "validation_masked_bce"),
                            ("auc", "validation_macro_auroc")):
            series[key].append({"id": s["id"], "label": s["label"], "group": s["group"],
                                "points": [[r["epoch"], r[metric]] for r in h]})
    blob = {"groups": {k: v for k, v in GROUPS.items() if k != "baseline"}, **series}
    plots = "".join(f"<div>{chart(k, t)}</div>" for k, t in (
        ("train", "Training masked BCE by epoch"), ("val", "Validation masked BCE by epoch (selects the epoch)"),
        ("auc", "Validation macro AUROC by epoch")))
    body = (HELPERS + '<div data-role="filter"></div>'
            f'<div class="grid-2">{plots}</div><div data-role="legend"></div>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  [['train', 'Training BCE', '.3f', 'min'], ['val', 'Validation BCE', '.3f', 'min'],
   ['auc', 'Validation AUROC', '.3f', 'max']].forEach(function (spec) {
    Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.val, store, {groups: d.groups, best: 'min', fmt: '.3f'});"""))
    mins = {}
    for s in j["steps"]:
        if s.get("history"):
            b = min(s["history"], key=lambda r: r["validation_masked_bce"])
            mins[s["id"]] = (b["epoch"], b["validation_masked_bce"])
    chain = [k for k in ("R1c", "R1c-rand", "R4a-K8") if k in mins]
    claim = ("Every rung overfits after its validation minimum; along R1c → random slices → R4a the minimum moves later "
             "and lower (" + ", ".join(f"epoch {mins[k][0]} at {num(mins[k][1], '.3f')}" for k in chain)
             + "), but it does not go away")
    caption = ("Validation BCE is against hard LM verdicts for every rung (soft-label rungs included), and its minimum "
               "picks the epoch; training stops 5 epochs after it. R0 has no fine-tuning curve.")
    notes = ("<ul><li>Training BCE is the masked macro BCE on each rung's own target (soft values for R2–R3), so compare "
             "it within a group; validation BCE is on one yardstick for all.</li>"
             "<li>R4 epochs read 8–16× more slices per study; the x axis is epochs, not time.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 3. per-label heatmaps


def per_label_section(ctx, j, by_id) -> tuple[str, dict]:
    sid = "per-label"
    labels = j["labels"]
    r0 = by_id["R0"]
    steps = [s for s in j["steps"] if s["id"] != "R0"]
    blob = {"rows": labels, "cols": [s["short"] for s in steps], "colIds": [s["id"] for s in steps],
            "colGroups": [s["group"] for s in steps], "baseCol": r0["short"], "groups": GROUPS}
    for split in ("gold", "lm_test"):
        blob[split] = {
            "level": [[r0[split]["per_label"].get(l)] + [s[split]["per_label"].get(l) for s in steps] for l in labels],
            "delta": [[None] + [(s[split]["per_label"][l] - r0[split]["per_label"][l])
                                if l in s[split]["per_label"] and l in r0[split]["per_label"] else None
                                for s in steps] for l in labels]}
    body = (HELPERS + '<div data-role="split"></div><div data-role="mode"></div>'
            + '<div data-chart-scope>' + chart("heat", "Per-label AUROC by step") + '</div>'
            + section_script(sid, """
  var split = 'gold', mode = 'delta';
  function draw() {
    var cols = [d.baseCol].concat(d.cols), m = d[split][mode];
    var keep = cols.map(function (c, j) { return j; }).filter(function (j) { return mode === 'level' || j > 0; });
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var o = {rows: d.rows, cols: keep.map(function (j) { return cols[j]; }), rowIds: d.rows,
             fmt: mode === 'level' ? '.2f' : '+.2f',
             ariaLabel: 'per-label ' + (split === 'gold' ? 'gold-58' : 'LM-test') + ' AUROC heatmap'};
    if (mode === 'delta') { o.diverging = true; o.peakPerRow = false; }
    Charts.heatmap(el, m.map(function (row) { return keep.map(function (j) { return row[j]; }); }), o);
  }
  AdaptUI.seg(sec.querySelector('[data-role=split]'), 'Scored on',
    [{key: 'gold', label: 'Gold-58 (human)'}, {key: 'lm_test', label: 'LM-test (432)'}], 'gold',
    function (k) { split = k; draw(); });
  AdaptUI.seg(sec.querySelector('[data-role=mode]'), 'Cells',
    [{key: 'delta', label: '\\u0394 vs R0'}, {key: 'level', label: 'AUROC'}], 'delta', function (k) { mode = k; draw(); });
  draw();"""))

    def gain(split, step_id):
        s = by_id.get(step_id)
        if not s:
            return None
        ds = [s[split]["per_label"][l] - r0[split]["per_label"][l] for l in labels
              if l in s[split]["per_label"] and l in r0[split]["per_label"]]
        return sum(d > 0 for d in ds), len(ds)

    up = gain("lm_test", "R4a-K8")
    claim = (f"R4a lifts LM-test AUROC over R0 on {up[0]} of {up[1]} labels; per-label gold-58 moves are large and noisy"
             if up else "Per-label AUROC by step") + " " + caveat_badge("few gold positives per label")
    caption = ("Rows are labels; columns are steps in ladder order. Cells default to the step's AUROC minus R0's on the same "
               "label (the frozen-encoder baseline); switch to raw AUROC to see R0's column. Toggle between gold-58 and LM-test.")
    notes = ("<ul><li>R0's per-label scores come from its 3-seed MLP heads: gold from <code>benchmark.json</code>, LM-test "
             "from <code>lm-proxy.json</code>, the same predictions the paired comparisons use.</li>"
             "<li>Per-label LM-test AUROC rests on each label's observed (non-UNK) verdicts, which vary widely by label.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 4. windows


def windows_section(ctx, j) -> tuple[str, dict]:
    sid = "windows"
    trs = []
    for s in j["steps"]:
        wc = s.get("window_counts")
        if not wc:
            continue
        for split in ("train", "validation", "lm_test", "gold"):
            c = wc.get(split)
            if c:
                planes = c["studies"] * 3
                trs.append([s["short"], split, str(c["studies"]), f"{c['masked_windows']:,} / {c['windows']:,}",
                            num(c["masked_windows"] / c["windows"], ".1%"),
                            f"{c['short_planes']} / {planes}", str(c["masked_planes"])])
    tbl = table(["Step", "Split", "Studies", "Masked windows", "Share", "Short planes", "Missing planes"],
                trs, num_cols=(2, 4, 6))
    k16 = next((s for s in j["steps"] if s["id"] == "R4a-K16"), None)
    share = (k16["window_counts"]["lm_test"]["masked_windows"] / k16["window_counts"]["lm_test"]["windows"]
             if k16 and k16.get("window_counts") else None)
    claim = ("At 8 windows no plane needs masking; at 16 about a fifth of planes are short stacks"
             + (f" and {num(share, '.1%')} of LM-test windows are masked" if share is not None else "")
             + "; no plane is ever fully missing")
    caption = ("A window is masked when its depth stratum holds no slice. A masked window gets zero weight in pooling "
               "and attention, instead of repeating a neighbour.")
    return section(sid, claim, tbl, caption=caption), {}


# --------------------------------------------------------------------------- page


def build(ctx) -> Page:
    j, by_id = _load()
    info = BY_SLUG[SLUG]
    if not j:
        return Page(title=info.title, subtitle="",
                    sections=[section("missing", "Fine-tuning ladder", notice(f"<code>{SRC}</code> is missing."))])
    sections, blobs = [], {}
    for s, b in (steps_section(ctx, j, by_id), curves_section(ctx, j), per_label_section(ctx, j, by_id),
                 windows_section(ctx, j)):
        sections.append(s)
        blobs.update(b)
    r0, best = by_id["R0"], max(j["steps"], key=lambda s: s["lm_test"]["macro"])
    best_gold = max(j["steps"], key=lambda s: s["gold"]["macro"])
    r4 = by_id.get("R4a-K8")
    kpis = [{"label": "Best LM-test", "value": num(best["lm_test"]["macro"]),
             "sub": f"{esc(best['short'])} · R0 {num(r0['lm_test']['macro'])}"},
            {"label": "Best gold-58", "value": num(best_gold["gold"]["macro"]),
             "sub": f"{esc(best_gold['short'])} · exploratory"}]
    if r4:
        d = r4["vs_parent"]["lm_test"]
        kpis.append({"label": "Multi-slice step (R4a)", "value": num(d["difference"], ".3f", True),
                     "sub": f"LM-test vs random-slice R1c {_ci(*d['ci95'])}", "tone": "good"})
    return Page(title=info.title,
                subtitle="Fine-tuning the image encoder on LM labels, one change per step, each against the step it builds on.",
                sections=sections, data=blobs, kpis=kpis)


def summary() -> dict:
    j, by_id = _load()
    info = BY_SLUG[SLUG]
    if not j:
        return {"stage": info.number, "headline": None, "value": None, "value_label": None, "caveat": None, "href": None}
    best = max(j["steps"], key=lambda s: s["lm_test"]["macro"])
    return {"stage": info.number, "headline": f"Multi-slice fine-tuning ({best['short']}) leads",
            "value": num(best["lm_test"]["macro"]), "value_label": "LM-test macro AUROC",
            "caveat": "Single seed per step; OmniRad-base R4a scores 0.839 public", "href": "#steps"}
