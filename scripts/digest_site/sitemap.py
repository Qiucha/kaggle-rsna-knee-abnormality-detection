"""The site pages, in mind-map order. Shared by the shell, catalog and home page.

The main track follows the project's question, "which model to transfer?": Phase 1 trains contrastive
image encoders, the picking page asks which Phase-1 model is worth carrying forward, Phase 2 fine-tunes
on labels, and the submission ships it. Background pages hold the earlier work that fixed the recipe.

Legacy slugs (the pre-2026-10-07 stage pages) are now *parts* composed into the pages above. They keep a
PageInfo so their modules still resolve ``BY_SLUG[SLUG]``, and build.py writes a redirect stub for each.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageInfo:
    slug: str
    number: int  # 0 = home, 1..7 = topic pages, 8 = library
    title: str
    short: str  # sidebar label
    group: str = ""  # sidebar group heading ("" = no heading)


PAGES: list[PageInfo] = [
    PageInfo("index", 0, "Which model to transfer?", "Home"),
    PageInfo("phase1", 1, "Phase 1 · Contrastive pre-training", "Phase 1 · Contrastive", "Main track"),
    PageInfo("picking", 2, "Picking · which Phase-1 model to transfer", "Picking a model", "Main track"),
    PageInfo("phase2", 3, "Phase 2 · Label fine-tuning", "Phase 2 · Label FT", "Main track"),
    PageInfo("submission", 4, "Heads & Kaggle submission", "Kaggle submission", "Main track"),
    PageInfo("recipe", 5, "How the Phase-1 recipe was chosen", "Phase-1 recipe history", "Background"),
    PageInfo("overfitting", 6, "Overfitting, regularisation & closed gates", "Overfitting & gates", "Background"),
    PageInfo("foundations", 7, "Pipeline, data & report text", "Pipeline, data & text", "Background"),
    PageInfo("library", 8, "Evidence library", "Evidence library", "Reference"),
]

# Old stage pages, now composed into the pages above: slug -> (new page, part heading).
LEGACY: dict[str, tuple[str, str]] = {
    "pipeline": ("foundations", "How it works & first baselines"),
    "data": ("foundations", "Data, gold set & integrity"),
    "text": ("foundations", "Report text: translation & text tower"),
    "adaptation": ("recipe", "Image-tower adaptation: depth, both towers, LoRA, pooling"),
    "probes": ("recipe", "Downstream probes & generalisation"),
    "encoders": ("phase1", "Encoder choices"),
    "finetune": ("phase2", "Supervised fine-tuning & multi-slice input"),
}
_LEGACY_INFO = [PageInfo(slug, -1, title, title) for slug, (_, title) in LEGACY.items()]

BY_SLUG = {p.slug: p for p in PAGES + _LEGACY_INFO}
TOPIC_SLUGS = [p.slug for p in PAGES if 1 <= p.number <= 7]
PAGE_SLUGS = [p.slug for p in PAGES if p.slug != "index"]  # slugs with a pages/<slug>.py module


def home_of(slug: str) -> str:
    """The page a (possibly legacy) slug now lives on."""
    return LEGACY[slug][0] if slug in LEGACY else slug
