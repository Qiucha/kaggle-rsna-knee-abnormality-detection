"""Phase 2 · label fine-tuning: the mind map's Phase-2 factors first, then the full fine-tuning record.

Lead card: the factors the mind map lists for Phase 2 (multi-slice, resolution, soft labels from the LM, the
OmniRad-base backbone, and the combined run 5), each read as one ladder step against the step it was built from.
Below it, the whole legacy fine-tuning page (run-5 follow-ups, ladder, curves, per-label, windows).
Source: digest/finetune-ladder-v1.json and digest/run5-followups-v1.json.
"""

from __future__ import annotations

import re

from .. import data
from ..shell import Page, esc, num, table
from ..sitemap import BY_SLUG
from . import _ensembles, _image_augmentation, _joint, _lateral_rebalance, _settings_fusion, finetune
from ._mindmap import ci_text, delta_card, excludes_zero

SLUG = "phase2"
SUBTITLE = ("Fine-tune the image encoder on LM labels. Each Phase-2 factor is one ladder step against the step it "
            "builds on; run 5 (OmniRad-base, K8, soft labels, 336 px) is the reference for every follow-up, unfreezing "
            "9 blocks (u9) is the best single run, and rank ensembles of u9 with other backbones add about +0.02. "
            "The tested one-stage label + contrastive loss shows no clear LM-test gain over label-only u9, and the "
            "non-fat-suppressed series add no label information (settings fusion). Training-time image augmentation "
            "on the A30 recipe (u9-b1 + lateral oversampling, 30 epochs) gives no gain.")

GROUPS = {"multi-slice": {"label": "① Multi-slice", "color": "var(--g-teal)"},
          "resolution": {"label": "② Resolution", "color": "var(--g-amber)"},
          "soft": {"label": "③ Soft labels from the LM", "color": "var(--g-rose)"},
          "model": {"label": "Model swap", "color": "var(--g-violet)"},
          "unfreeze": {"label": "Unfrozen image blocks (vs run 5's 3)", "color": "var(--g-sky)"},
          "combined": {"label": "Combined (run 5) vs R0", "color": "var(--g-lime)"}}

# (ladder step id, factor group, what changed). The parent comes from the ladder JSON itself.
FACTORS = [
    ("R4a-K8", "multi-slice", "Multi-slice: 3 planes × 8 windows × 3 slices, uniform mean"),
    ("R4a-K16", "multi-slice", "Multi-slice: 8 → 16 windows per plane"),
    ("DINOv2-S-336", "resolution", "Resolution: 224 → 336 px (DINOv2-small)"),
    ("R4a-soft", "soft", "Soft LM labels in [0, 1] instead of hard 0/1"),
    ("R4a-soft-unk", "soft", "Soft labels, UNK as 0.2 instead of masked"),
    ("R4a-soft-grouped", "soft", "Soft labels, grouped lateral labels"),
    ("OmniRad-B", "model", "Backbone: DINOv2-small → OmniRad-base"),
]


def factor_section(ctx) -> tuple[str, dict]:
    j, by_id = finetune._load()
    if not j:
        return "", {}
    rows, table_rows = [], []
    run5 = data.read_json("run5-followups-v1.json") or {}
    for step_id, group, what in FACTORS:
        step = by_id.get(step_id)
        if not step or not step.get("vs_parent"):
            continue
        parent = by_id.get(step.get("parent"), {})
        v = step["vs_parent"]
        rows.append({"id": step_id, "label": f"{step['short']} vs {parent.get('short', step.get('parent'))}",
                     "group": group, "lm": v["lm_test"], "gold": v["gold"]})
        table_rows.append([GROUPS[group]["label"], what, step["short"], parent.get("short", ""),
                           ci_text(v["lm_test"]), ci_text(v["gold"])])
    paired, base = run5.get("paired") or {}, run5.get("baseline")
    sweep = sorted(((int(m.group(1)), c) for c in run5.get("completed", [])
                    if (m := re.fullmatch(re.escape(base or "") + r"-u(\d+)", c["name"]))), key=lambda t: t[0])
    for blocks, c in sweep:
        lm, gold = paired.get("lm_test", {}).get(c["name"]), paired.get("gold", {}).get(c["name"])
        if lm and gold:
            rows.append({"id": c["name"], "label": f"Unfreeze {blocks} vs 3 blocks (run 5)", "group": "unfreeze",
                         "lm": lm, "gold": gold})
            table_rows.append([GROUPS["unfreeze"]["label"], f"Last {blocks} image blocks trainable instead of 3",
                               c["title"], "Run 5", ci_text(lm), ci_text(gold)])
    # The whole stack against the frozen baseline closes the card, after every single-factor step.
    vs_r0 = run5.get("paired_vs_r0") or {}
    if base and base in vs_r0.get("lm_test", {}):
        rows.append({"id": "run5-vs-r0", "label": "Run 5 vs R0 (frozen encoder)", "group": "combined",
                     "lm": vs_r0["lm_test"][base], "gold": vs_r0["gold"][base]})
        table_rows.append([GROUPS["combined"]["label"], "OmniRad-base + K8 + soft labels + 336 px", "Run 5", "R0",
                           ci_text(vs_r0["lm_test"][base]), ci_text(vs_r0["gold"][base])])
    by = {r["id"]: r for r in rows}

    def short(step_id):
        r = by.get(step_id)
        return f"{num(r['gold']['difference'], '.3f', True)} gold / {num(r['lm']['difference'], '.3f', True)} LM" if r else "pending"

    claim = (f"Phase 2 factors: multi-slice {short('R4a-K8')}, 336 px {short('DINOv2-S-336')}, "
             f"soft labels {short('R4a-soft')}; together, run 5 sits {short('run5-vs-r0')} above the frozen R0")
    lead = ('<p>Read each row as <em>this step minus the step it was built from</em>; the last row is the whole '
            'stack against the frozen-encoder baseline R0. LM-test (432 non-gold studies, LM verdicts) is the '
            'selection criterion; gold-58 is exploratory and noisy. '
            f'Rows whose LM-test interval excludes zero: {sum(excludes_zero(r["lm"]) for r in rows)} of {len(rows)}.</p>')
    tbl = table(["Factor", "Change", "Step", "Built on", "Δ LM-test [95% CI]", "Δ gold-58 [95% CI]"], table_rows)
    notes = ("<p>Paired study bootstrap, 2,000 resamples. One training seed per step; the run-5 seed replicate "
             "differs by about 0.003 LM-test, so single-step differences below that are noise. The resolution step "
             "is measured on DINOv2-small because that is where the ladder isolated it; run 5 applies 336 px to "
             "OmniRad-base together with soft labels.</p>"
             f'<p>Sources: <a href="{ctx.raw("finetune-ladder-v1.json")}">ladder snapshot</a>, '
             f'<a href="{ctx.raw("run5-followups-v1.json")}">run-5 follow-ups</a>.</p>')
    return delta_card("phase2-factors", claim, rows, GROUPS, x_label="Step minus the step it builds on",
                      lead=lead, notes=notes, table_html=tbl, table_summary="Factor table")


def build(ctx) -> Page:
    lead, blob = factor_section(ctx)
    page = finetune.build(ctx)
    ens_sections, ens_blob = _ensembles.build(ctx)
    joint_sections, joint_blob = _joint.build(ctx)
    fusion_sections, fusion_blob = _settings_fusion.build(ctx)
    lateral_sections, lateral_blob = _lateral_rebalance.build(ctx)
    aug_sections, aug_blob = _image_augmentation.build(ctx)
    # u9-b1 follow-ups (joint loss, settings fusion, weak-label rebalancing, image augmentation) and the ensembles
    # follow the run-5 scores they build on.
    sections = (page.sections[:1] + joint_sections + fusion_sections + lateral_sections + aug_sections + ens_sections
                + page.sections[1:])
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE,
                sections=([lead] if lead else []) + sections,
                data={**blob, **page.data, **joint_blob, **fusion_blob, **lateral_blob, **aug_blob, **ens_blob}, kpis=page.kpis)


def summary() -> dict:
    s = dict(finetune.summary())
    s["stage"] = BY_SLUG[SLUG].number
    s["headline"] = ("Label-only u9 leads singles; joint loss, weak-label rebalancing and image augmentation do not beat it; "
                     "mixed-backbone ensembles add ~+0.02")
    return s
