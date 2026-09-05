from __future__ import annotations

import re
import unicodedata


REPORT_VIEWS = ("full", "impression", "findings_impression")

_FINDINGS_HEADINGS = {
    "findings",
    "hallazgos",
    "befund",
    "bevindingen",
    "bulgular",
    "nalaz",
    "mr nalaz",
    "мр находка",
    "находка",
    "ευρηματα",
}
_IMPRESSION_HEADINGS = {
    "impression",
    "conclusion",
    "conclusions",
    "impresion",
    "conclusion diagnostica",
    "beurteilung",
    "conclusie",
    "sonuc",
    "zakljucak",
    "заключение",
    "συμπερασμα",
}
_HEADING_PATTERN = re.compile(r"(?m)^[ \t]*([^:\n]{2,60}):[ \t]*")


def _normalize_heading(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(without_marks.split()).strip(" .")


def select_report_view(report: str, view: str = "full") -> str:
    """Select a full or section-aware report representation with full-text fallback."""
    report = report.strip()
    if view not in REPORT_VIEWS:
        raise ValueError(f"report view must be one of {REPORT_VIEWS}")
    if view == "full":
        return report
    sections: list[tuple[str, int, int]] = []
    for match in _HEADING_PATTERN.finditer(report):
        heading = _normalize_heading(match.group(1))
        kind = "findings" if heading in _FINDINGS_HEADINGS else "impression" if heading in _IMPRESSION_HEADINGS else None
        if kind is not None:
            sections.append((kind, match.start(), match.end()))
    findings = next((section for section in sections if section[0] == "findings"), None)
    impression = next((section for section in sections if section[0] == "impression"), None)
    if view == "impression":
        return report[impression[2] :].strip() if impression is not None else report
    if findings is None and impression is None:
        return report
    parts = []
    if findings is not None:
        end = impression[1] if impression is not None and impression[1] > findings[1] else len(report)
        value = report[findings[2] : end].strip()
        if value:
            parts.append(f"FINDINGS: {value}")
    if impression is not None:
        value = report[impression[2] :].strip()
        if value:
            parts.append(f"IMPRESSION: {value}")
    return "\n".join(parts) or report
