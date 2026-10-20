"""Picking · which Phase-1 model to transfer: does the Phase-1 score predict Phase-2 value?

Three readings of each Phase-1 candidate side by side: the retrieval selector (mean R@1), a frozen-tower head,
and the AUROC after Phase-2 fine-tuning from that checkpoint. Then the paired Phase-2 transfer test and the
OmniRad frozen-tower probe. Sources: run5-followups-v1.json, combined-lora-k8-336-v1.json,
omnirad-clip-frozen-probe-v1.json, finetune-ladder-v1.json.
"""

from __future__ import annotations

from .. import data
from ..shell import Page, chart, esc, num, section, table
from ..sitemap import BY_SLUG
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script
from ._mindmap import ci_text, delta_card, excludes_zero

SLUG = "picking"
SUBTITLE = ("Phase 1 selects checkpoints by retrieval (mean R@1), but prior work finds representation metrics do not "
            "reliably pick the best model for downstream fine-tuning. This page checks that on our own runs: retrieval, "
            "frozen-head AUROC and post-fine-tuning AUROC for the same candidates.")

GROUPS = {"unadapted": {"label": "No Phase 1 (pretrained tower)", "color": "var(--g-anchor)"},
          "clip": {"label": "Plain CLIP, last 6 blocks", "color": "var(--g-teal)"},
          "lora": {"label": "CLIP + LoRA r64", "color": "var(--g-rose)"},
          "noise": {"label": "Seed replicate (noise)", "color": "var(--g-control)"}}


def _completed(run5: dict) -> dict:
    return {r["name"]: r for r in run5.get("completed", [])}


def candidates() -> list[dict]:
    run5 = data.read_json("run5-followups-v1.json") or {}
    combined = {a["name"]: a for a in (data.read_json("combined-lora-k8-336-v1.json") or {}).get("arms", [])}
    probe = data.read_json("omnirad-clip-frozen-probe-v1.json") or {}
    done = _completed(run5)
    mlp = (probe.get("readout") or {}).get("mlp") or {}
    frozen = lambda split, arm: ((mlp.get(split) or {}).get("macro_auroc") or {}).get(arm)  # noqa: E731
    base = run5.get("baseline", "r4a-k8-omnirad-base-soft-336")
    ft = lambda name: done.get(name) or {}  # noqa: E731
    lora_omni, lora_dino = combined.get("omnirad-base-lora-k8-336", {}), combined.get("dinov2-small-lora-k8-336", {})
    pre = run5.get("contrastive_pretraining") or {}
    return [
        {"id": "omnirad-unadapted", "group": "unadapted", "label": "OmniRad-base, pretrained only",
         "r1": None, "frozen_lm": frozen("test", "base"), "frozen_gold": frozen("gold", "base"), "frozen_src": "probe",
         "ft_lm": ft(base).get("lm_test"), "ft_gold": ft(base).get("gold"), "ft_run": "run 5"},
        {"id": "omnirad-clip", "group": "clip", "label": "OmniRad-base, plain CLIP (u6)",
         "r1": pre.get("best_mean_recall_at_1"), "frozen_lm": frozen("test", "clip"), "frozen_gold": frozen("gold", "clip"),
         "frozen_src": "probe", "ft_lm": ft(base + "-from-clip").get("lm_test"), "ft_gold": ft(base + "-from-clip").get("gold"),
         "ft_run": "run 5 from CLIP"},
        {"id": "omnirad-lora", "group": "lora", "label": "OmniRad-base, CLIP + LoRA",
         "r1": lora_omni.get("best_mean_recall_at_1"), "frozen_lm": lora_omni.get("lm_test_mlp"),
         "frozen_gold": lora_omni.get("gold_mlp"), "frozen_src": "lm-mlp",
         "ft_lm": ft(base + "-from-clip-lora").get("lm_test"), "ft_gold": ft(base + "-from-clip-lora").get("gold"),
         "ft_run": "run 5 from CLIP + LoRA"},
        {"id": "dinov2-unadapted", "group": "unadapted", "label": "DINOv2-small, pretrained only",
         "r1": None, "frozen_lm": None, "frozen_gold": None, "frozen_src": "",
         "ft_lm": ft("r4a-k8-dinov2-small-soft-336").get("lm_test"), "ft_gold": ft("r4a-k8-dinov2-small-soft-336").get("gold"),
         "ft_run": "DINOv2-small soft 336"},
        {"id": "dinov2-lora", "group": "lora", "label": "DINOv2-small, CLIP + LoRA",
         "r1": lora_dino.get("best_mean_recall_at_1"), "frozen_lm": lora_dino.get("lm_test_mlp"),
         "frozen_gold": lora_dino.get("gold_mlp"), "frozen_src": "lm-mlp", "ft_lm": None, "ft_gold": None, "ft_run": ""},
    ]


def _order(cands: list[dict], key: str) -> str:
    ranked = sorted((c for c in cands if c.get(key) is not None), key=lambda c: -c[key])
    return " > ".join(c["label"] for c in ranked) or "pending"


def table_section(ctx) -> str:
    cands = candidates()
    rows = [[esc(c["label"]), num(c["r1"]), num(c["frozen_lm"]), num(c["frozen_gold"]),
             esc({"probe": "run-5 head recipe", "lm-mlp": "lm-mlp config", "": ""}[c["frozen_src"]]),
             num(c["ft_lm"]), num(c["ft_gold"]), esc(c["ft_run"] or "not run")] for c in cands]
    omni = [c for c in cands if c["id"].startswith("omnirad")]
    body = ("<p>Read across a row: the same image tower scored three ways. Only the OmniRad-base rows have all three. "
            "Rankings among the OmniRad candidates:</p><ul>"
            f"<li><strong>Phase-1 retrieval (mean R@1):</strong> {esc(_order(omni, 'r1'))}</li>"
            f"<li><strong>Frozen-head LM-test:</strong> {esc(_order(omni, 'frozen_lm'))}</li>"
            f"<li><strong>After Phase-2 fine-tuning, LM-test:</strong> {esc(_order(omni, 'ft_lm'))}</li>"
            f"<li><strong>After Phase-2 fine-tuning, gold-58:</strong> {esc(_order(omni, 'ft_gold'))}</li></ul>"
            + table(["Phase-1 candidate", "Mean R@1", "Frozen LM-test", "Frozen gold-58", "Frozen head protocol",
                     "Fine-tuned LM-test", "Fine-tuned gold-58", "Phase-2 run"], rows, raw_html=True,
                    num_cols=(1, 2, 3, 5, 6)))
    notes = ("<p>Fine-tuned columns use the run-5 recipe (K8, soft labels, 336 px) started from each checkpoint. "
             "Frozen heads: the OmniRad pretrained and plain-CLIP rows come from the frozen probe (run-5 LM-head "
             "recipe); the LoRA rows from the lm-mlp head configs. The two head protocols are close but not identical, "
             "so compare frozen numbers within a protocol. Differences in the ranking smaller than the run-5 seed "
             "replicate (about 0.003 LM-test) are not resolved; paired intervals are in the next card.</p>")
    run5 = data.read_json("run5-followups-v1.json") or {}
    base = run5.get("baseline", "r4a-k8-omnirad-base-soft-336")
    paired = (run5.get("paired") or {}).get("lm_test", {})
    starts = [paired.get(base + suffix) for suffix in ("-from-clip", "-from-clip-lora")]
    resolved = [d for d in starts if excludes_zero(d)]
    mlp = ((data.read_json("omnirad-clip-frozen-probe-v1.json") or {}).get("readout") or {}).get("mlp") or {}
    frozen_gap = (mlp.get("test") or {}).get("delta_clip_minus_base")
    if all(starts) and not resolved and frozen_gap is not None:
        claim = (f"Frozen heads separate CLIP from the pretrained OmniRad tower ({num(frozen_gap, '.3f', True)} LM-test), "
                 "but after Phase-2 fine-tuning every contrastive start is within noise of the pretrained start")
    else:
        claim = "Phase-1 retrieval, frozen heads and fine-tuning for the same OmniRad candidates"
    return section("transfer-table", claim, body, notes=notes)


def transfer_section(ctx) -> tuple[str, dict]:
    run5 = data.read_json("run5-followups-v1.json") or {}
    paired = run5.get("paired") or {}
    base = run5.get("baseline", "r4a-k8-omnirad-base-soft-336")
    specs = [(base + "-from-clip", "clip", "Start from plain CLIP"),
             (base + "-from-clip-lora", "lora", "Start from CLIP + LoRA"),
             (base + "-seed2027", "noise", "Run 5, seed 2027 (noise yardstick)")]
    rows, trows = [], []
    for name, group, label in specs:
        lm, gold = paired.get("lm_test", {}).get(name), paired.get("gold", {}).get(name)
        if lm and gold:
            rows.append({"id": name, "label": label, "group": group, "lm": lm, "gold": gold})
            trows.append([label, ci_text(lm), ci_text(gold)])
    conf = run5.get("contrastive_confirmation") or {}
    verdict = ("" if not conf else
               f"<p>Pre-registered pass rule for the plain-CLIP start: LM-test interval above zero, gain larger than the "
               f"seed difference, gold drop under 0.024. Result: <strong>{'passed' if conf.get('passed') else 'not passed'}</strong>.</p>")
    lead = ("<p>Same Phase-2 recipe as run 5; only the starting image tower changes. Differences are against run 5, "
            "which starts from the pretrained OmniRad-base.</p>" + verdict)
    starts = [r for r in rows if r["group"] != "noise"]
    wins = [r for r in starts if r["lm"]["ci95"][0] > 0]
    claim = (("Starting Phase 2 from a contrastive checkpoint does not beat starting from the pretrained tower: "
              if not wins else "A contrastive start beats the pretrained start on LM-test: ")
             + "; ".join(f"{r['label'][0].lower() + r['label'][1:]} {ci_text(r['lm'], '.3f')} LM-test" for r in starts))
    notes = (f'<p>Paired study bootstrap, 2,000 resamples, seed 2030. Gold-58 is exploratory. Training curves and '
             f'per-label views for these runs are on <a href="{ctx.page("phase2")}#run5-followups">Phase 2</a>; the '
             f'contrastive pre-training curves are in <a href="{ctx.page("phase2")}#run5-phase1">its Phase-1 card</a>.</p>')
    return delta_card("transfer-test", claim, rows, GROUPS, x_label="Candidate minus run 5", lead=lead, notes=notes,
                      table_html=table(["Start", "Δ LM-test vs run 5 [95% CI]", "Δ gold-58 vs run 5 [95% CI]"], trows))


def probe_section(ctx) -> tuple[str, dict]:
    probe = data.read_json("omnirad-clip-frozen-probe-v1.json")
    if not probe:
        return "", {}
    sid = "frozen-probe"
    vs = probe["versus_finetune"]
    # Frozen readouts first (pretrained, then CLIP), then the fine-tunes (run 5, its seed replicate, CLIP then FT).
    names = {"base-head": ("Pretrained, frozen + head", "unadapted"), "clip-head": ("Plain CLIP, frozen + head", "clip"),
             "run5-ft": ("Run 5 fine-tune", "unadapted"), "run5-ft-seed2027": ("Run 5 fine-tune, seed 2027", "noise"),
             "clip-then-ft": ("Plain CLIP then fine-tune", "clip")}
    blob = {"groups": GROUPS, "levels": {split: [{"id": k, "label": names[k][0], "group": names[k][1],
                                                  "value": vs[split]["macro_auroc"][k]}
                                                 for k in names if k in vs[split]["macro_auroc"]]
                                         for split in ("lm_test", "gold")}}
    mlp, test = probe["readout"]["mlp"], probe["grouping"]["test"]
    acq = test["acquisition_knn"]
    clip_ft = vs["gold"]["paired"]["clip-head - run5-ft"]
    body = (HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            + "<div>" + chart("lm", "LM-test macro AUROC (432 studies)") + "</div>"
            + "<div>" + chart("gold", "Gold-58 macro AUROC (exploratory)") + "</div></div>"
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  [['lm', 'lm_test'], ['gold', 'gold']].forEach(function (spec) {
    Charts.barChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d.levels[spec[1]],
      {groups: d.groups, fmt: '.4f', axisFmt: '.2f', store: store, labelWidth: 210});
  });""")
            + f"<p>Acquisition control: CLIP also made the scanner easier to read. Manufacturer kNN accuracy "
            f"{num(acq['manufacturer']['base']['accuracy'], '.2f')} → {num(acq['manufacturer']['clip']['accuracy'], '.2f')} "
            f"(majority class {num(acq['manufacturer']['base']['majority_baseline'], '.2f')}); field strength "
            f"{num(acq['field_strength']['base']['accuracy'], '.2f')} → {num(acq['field_strength']['clip']['accuracy'], '.2f')}.</p>")
    rows = [[k, ci_text(vs["lm_test"]["paired"][k]), ci_text(vs["gold"]["paired"][k])] for k in vs["lm_test"]["paired"]]
    claim = (f"Plain CLIP lifts the frozen OmniRad tower {ci_text({'difference': mlp['test']['delta_clip_minus_base'], 'ci95': mlp['test']['delta_ci95']}, '.3f')} "
             f"on LM-test, but on gold-58 a frozen head still trails fine-tuning by {num(-clip_ft['difference'], '.3f')}")
    notes = ("<p>Unadapted OmniRad-base vs the epoch-18 image tower of omnirad-clip-run5-v1, same K8 336 inputs, run-5 "
             "LM-head recipe with three head seeds. LM labels come from the reports that CLIP aligns to, so part of the "
             "LM-test gain may be report-style or site signal (the scanner control above rises too); gold-58 is the "
             "independent check. One CLIP training seed; intervals cover head-seed and study resampling only.</p>"
             f'<p>Sources: <a href="{ctx.doc("omnirad-clip-frozen-probe-v1.md")}">results</a>, '
             f'<a href="{ctx.raw("omnirad-clip-frozen-probe-v1.json")}">snapshot</a>.</p>')
    return section(sid, claim, body, notes=notes, table_html=table(["Pair", "Δ LM-test [95% CI]", "Δ gold-58 [95% CI]"], rows),
                   table_summary="All paired comparisons"), {f"data-{sid}": blob}


def build(ctx) -> Page:
    transfer, b1 = transfer_section(ctx)
    probe, b2 = probe_section(ctx)
    cands = candidates()
    best_ft = max((c for c in cands if c["ft_lm"] is not None), key=lambda c: c["ft_lm"], default=None)
    best_frozen = max((c for c in cands if c["frozen_lm"] is not None), key=lambda c: c["frozen_lm"], default=None)
    kpis = []
    if best_frozen:
        kpis.append({"label": "Best frozen tower (LM-test)", "value": num(best_frozen["frozen_lm"]), "sub": esc(best_frozen["label"])})
    if best_ft:
        kpis.append({"label": "Best after fine-tuning (LM-test)", "value": num(best_ft["ft_lm"]), "sub": esc(best_ft["label"])})
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE,
                sections=[table_section(ctx), transfer, probe], data={**b1, **b2}, kpis=kpis or None)


def summary() -> dict:
    cands = candidates()
    best_ft = max((c for c in cands if c["ft_lm"] is not None), key=lambda c: c["ft_lm"], default=None)
    return {"stage": BY_SLUG[SLUG].number, "headline": "Does Phase 1 help Phase 2?",
            "value": num(best_ft["ft_lm"]) if best_ft else None,
            "value_label": f"best fine-tuned LM-test ({best_ft['label']})" if best_ft else None,
            "caveat": "Gaps between starts are near the 0.003 seed difference; see paired CIs", "href": "#transfer-test"}
