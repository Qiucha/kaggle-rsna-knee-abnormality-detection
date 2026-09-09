"""Choose which part of a radiology report the text tower reads.

A report typically has a findings section (detailed observations) and an impression
or conclusion (the radiologist's summary). Those carry different signal, and which
one a contrastive model sees is an experiment variable, so this module turns one
raw report into one of four *views*.

Two rules hold for every view, and matter more than the parsing details:

* **No content is invented or dropped silently.** ``impression`` narrows the text;
  the other views only reorder or relabel. Nothing is paraphrased.
* **Ambiguity falls back to the full report.** Section headings in this cohort are
  free text, in several languages, and sometimes repeated. Where a heading cannot
  be located unambiguously the full report is returned rather than a guess, so a
  view can never be quietly built from a misparse.
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple


REPORT_VIEWS = ("full", "impression", "findings_impression", "impression_first")

# Heading vocabularies, accent- and case-normalized by _normalize_heading. The
# cohort's reports are multilingual, so both the English terms and the observed
# translations are listed; an unlisted heading simply does not match, which the
# full-report fallback then handles.
_FINDINGS_HEADINGS = frozenset(
    {
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
)
_IMPRESSION_HEADINGS = frozenset(
    {
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
)

# A heading is a short label at the start of a line, terminated by a colon.
_HEADING_PATTERN = re.compile(r"(?m)^[ \t]*([^:\n]{2,60}):[ \t]*")

# The impression_first view targets translated English reports, where the heading
# may also begin mid-line after sentence-ending punctuation.
_ENGLISH_IMPRESSION = re.compile(
    r"(?:^|\n|(?<=[.!?])[ \t]+)[ \t]*(?:impressions?|conclusions?)\s*:",
    re.IGNORECASE,
)


class _Section(NamedTuple):
    """A located heading: its kind, and the span of the heading text itself.

    ``start`` is where the heading begins (used when cutting the preceding section
    short) and ``end`` is where its body begins, just past the colon.
    """

    kind: str
    start: int
    end: int


def _normalize_heading(value: str) -> str:
    """Casefold, strip accents and surrounding punctuation, and collapse spaces.

    Accent stripping is what lets one entry match "Sonuç" and "Sonuc", or
    "Conclusión" and "Conclusion", without listing each spelling.
    """
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(without_marks.split()).strip(" .")


def _find_sections(report: str) -> tuple[_Section | None, _Section | None]:
    """Locate the first recognized findings and impression headings, if present."""
    findings = impression = None
    for match in _HEADING_PATTERN.finditer(report):
        heading = _normalize_heading(match.group(1))
        if findings is None and heading in _FINDINGS_HEADINGS:
            findings = _Section("findings", match.start(), match.end())
        elif impression is None and heading in _IMPRESSION_HEADINGS:
            impression = _Section("impression", match.start(), match.end())
    return findings, impression


def _impression_first(report: str) -> str:
    """Move the impression ahead of everything before it, keeping all words.

    Requires exactly one impression heading: with none there is nothing to move,
    and with several the intended split is genuinely ambiguous.
    """
    headings = list(_ENGLISH_IMPRESSION.finditer(report))
    if len(headings) != 1:
        return report
    start = headings[0].start()
    before, after = report[:start].strip(), report[start:].strip()
    return f"{after}\n\n{before}" if before else report


def select_report_view(report: str, view: str = "full") -> str:
    """Render ``report`` as the named view, falling back to the full report.

    ``full``
        The report unchanged, apart from surrounding whitespace.
    ``impression``
        Only the impression body -- the shortest, most summary-like text.
    ``findings_impression``
        Both sections, relabeled with canonical English headings so that reports in
        different languages present the same structure to the text encoder.
    ``impression_first``
        The whole report reordered to lead with the impression, which puts the
        summary inside the encoder's token budget when a report is truncated.
    """
    report = report.strip()
    if view not in REPORT_VIEWS:
        raise ValueError(f"report view must be one of {REPORT_VIEWS}")
    if view == "full":
        return report
    if view == "impression_first":
        return _impression_first(report)

    findings, impression = _find_sections(report)
    if view == "impression":
        return report[impression.end :].strip() if impression is not None else report
    if findings is None and impression is None:
        return report
    parts = []
    if findings is not None:
        # The findings body ends where the impression begins, when the impression
        # follows it; a preceding impression means findings runs to the end.
        end = impression.start if impression is not None and impression.start > findings.start else len(report)
        if body := report[findings.end : end].strip():
            parts.append(f"FINDINGS: {body}")
    if impression is not None:
        if body := report[impression.end :].strip():
            parts.append(f"IMPRESSION: {body}")
    return "\n".join(parts) or report
