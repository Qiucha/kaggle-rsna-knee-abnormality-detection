"""Small helpers shared by the encoders and submission pages (page-local, not a shared module).

- ``md_table(text, heading)``: the first markdown pipe table after a heading line, as a list of dicts.
- ``md_number`` / ``md_interval``: parse "+0.0323", "−0.0100", "[−0.0079, +0.0324]" style cells.
- ``curve_ref(checkpoint_path)``: "…/checkpoints/contrastive/<root>/<run>/best.pt" -> (root, run).
- ``section_script(sid, js)``: the inline script wrapper every section uses (one selection store per section).
"""

from __future__ import annotations

import re

_MINUS = str.maketrans({"−": "-", "–": "-", "−": "-"})


def md_number(cell: str) -> float | None:
    s = cell.replace("*", "").replace(",", "").strip().translate(_MINUS)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def md_interval(cell: str) -> tuple[float, float] | None:
    s = cell.translate(_MINUS)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def md_table(text: str | None, heading: str) -> list[dict]:
    """Rows of the first pipe table after the line that starts with ``heading``; [] when absent."""
    if not text:
        return []
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(heading)), None)
    if start is None:
        return []
    rows: list[list[str]] = []
    first = start if lines[start].strip().startswith("|") else start + 1
    for ln in lines[first:]:
        s = ln.strip()
        if not s.startswith("|"):
            if rows:
                break
            if s.startswith("#"):
                return []
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return []
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:]]


def curve_ref(checkpoint: str | None) -> tuple[str, str] | None:
    if not checkpoint:
        return None
    m = re.search(r"checkpoints/contrastive/([^/]+)/(.+)/[^/]+\.pt$", checkpoint)
    return (m.group(1), m.group(2)) if m else None


def section_script(sid: str, js: str) -> str:
    """Inline script: ``sec``, ``d`` (the section's data blob) and ``store`` are in scope for ``js``."""
    return ("<script>\n(function () {\n"
            f"  var sec = document.getElementById('{sid}');\n"
            f"  var d = Charts.readJSON('data-{sid}');\n"
            "  var store = Charts.selection(sec);\n"
            + js + "\n})();\n</script>")
