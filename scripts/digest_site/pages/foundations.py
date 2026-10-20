"""Background · pipeline, data & report text: the legacy pipeline, data and text pages on one page."""

from __future__ import annotations

from ..shell import Page
from ..sitemap import BY_SLUG
from . import compose, pipeline

SLUG = "foundations"
SUBTITLE = ("What the two phases stand on: the training pipeline and first baselines, the cohort and gold set with "
            "their integrity checks, and the report text (translation and text tower).")


def build(ctx) -> Page:
    sections, blobs, kpis = compose(ctx, [
        ("pipeline", "Architecture, parameter budget, timeline and the first reproducible baseline."),
        ("data", "Duplicates, the gold set, protocol leakage, input crop, series slots and report length."),
        ("text", "Translation corpora, text-tower swaps and what the report side contributes."),
    ])
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=sections, data=blobs, kpis=kpis or None)


def summary() -> dict:
    s = dict(pipeline.summary())
    s["stage"] = BY_SLUG[SLUG].number
    return s
