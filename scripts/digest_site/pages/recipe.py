"""Background · how the Phase-1 recipe was chosen: adaptation depth/LoRA, earlier encoder choices, probes.

Composes the legacy adaptation and probes pages and the pre-factor-isolation encoder sections (tower-size
factorial, size-matched swap, label similarity, DeCLIP). Phase-1 work since factor isolation is on phase1.
"""

from __future__ import annotations

from ..shell import Page
from ..sitemap import BY_SLUG
from . import adaptation, compose, encoders, part_divider, probes

SLUG = "recipe"
SUBTITLE = ("Why Phase 1 uses LoRA on the last six blocks of both towers, DINOv2/E5 towers and a frozen-tower probe: "
            "the adaptation sweeps, tower-size and loss experiments, and the gold-holdout probe that measured them.")


def build(ctx) -> Page:
    sections, blobs, kpis = compose(ctx, [
        ("adaptation", "Which blocks to adapt, and how: depth, both towers, LoRA rank and memory, plane pooling."),
    ])
    sections.append(part_divider("encoders", "Tower sizes and loss variants tried before factor isolation."))
    for fn in (encoders.factorial_section, encoders.tower_curves_section, encoders.size_matched_section,
               encoders.label_similarity_section, encoders.declip_section):
        s, b, *_ = fn(ctx)
        sections.append(s)
        blobs.update(b)
    more, more_blobs, _ = compose(ctx, [("probes", "How a frozen tower is read: the gold probe, exposure and the holdout estimate.")])
    sections += more
    blobs.update(more_blobs)
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=sections, data=blobs, kpis=kpis or None)


def summary() -> dict:
    s = dict(adaptation.summary())
    s["stage"] = BY_SLUG[SLUG].number
    return s
