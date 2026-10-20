"""Shared pieces for the mind-map pages (phase1, picking, phase2): a two-panel paired-Δ card.

``delta_card(sid, claim, rows, groups, ...)`` draws LM-test and gold-58 paired differences side by side.
rows: [{id, label, group, lm: {difference, ci95}|None, gold: {difference, ci95}|None}].
"""

from __future__ import annotations

from ..shell import chart, num, section
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script


def ci_text(d: dict | None, spec: str = ".4f") -> str:
    if not d:
        return "pending"
    lo, hi = d["ci95"]
    return f"{num(d['difference'], spec, True)} [{num(lo, spec, True)}, {num(hi, spec, True)}]"


def excludes_zero(d: dict | None) -> bool:
    return bool(d) and (d["ci95"][0] > 0 or d["ci95"][1] < 0)


def delta_card(sid: str, claim: str, rows: list[dict], groups: dict, *, x_label: str, caption: str = "",
               notes: str = "", table_html: str = "", table_summary: str = "Full table",
               lead: str = "") -> tuple[str, dict]:
    def series(key):
        return [{"id": r["id"], "label": r["label"], "group": r["group"], "est": r[key]["difference"],
                 "lo": r[key]["ci95"][0], "hi": r[key]["ci95"][1]} for r in rows if r.get(key)]

    blob = {"groups": groups, "lm": series("lm"), "gold": series("gold"), "xLabel": x_label}
    body = (lead + HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            + "<div>" + chart("lm", "Δ LM-test macro AUROC (432 studies), paired 95% CI") + "</div>"
            + "<div>" + chart("gold", "Δ gold-58 macro AUROC (exploratory), paired 95% CI") + "</div></div>"
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  ['lm', 'gold'].forEach(function (key) {
    Charts.ciDotPlot(sec.querySelector('[data-chart=' + key + ']'), d[key],
      {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.3f', sort: false,
       store: store, labelWidth: 250, xLabel: d.xLabel});
  });"""))
    return section(sid, claim, body, caption=caption, notes=notes, table_html=table_html,
                   table_summary=table_summary), {f"data-{sid}": blob}
