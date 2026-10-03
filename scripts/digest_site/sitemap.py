"""The eleven site pages, in process order. Shared by the shell, catalog and home page."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageInfo:
    slug: str
    number: int  # 0 = home, 1..9 = process stages, 10 = library
    title: str
    short: str  # sidebar label


PAGES: list[PageInfo] = [
    PageInfo("index", 0, "Project at a glance", "Home"),
    PageInfo("pipeline", 1, "How it works & first baselines", "How it works & first baselines"),
    PageInfo("data", 2, "Data, gold set & integrity", "Data, gold set & integrity"),
    PageInfo("text", 3, "Report text: translation & text tower", "Report text & text tower"),
    PageInfo("adaptation", 4, "Image-tower adaptation: depth, both towers, LoRA, pooling", "Image-tower adaptation"),
    PageInfo("overfitting", 5, "Overfitting, regularisation & closed gates", "Overfitting & closed gates"),
    PageInfo("probes", 6, "Downstream probes & generalisation", "Probes & generalisation"),
    PageInfo("encoders", 7, "Encoder choices: tower size, size-matched, label similarity, DeCLIP", "Encoder choices"),
    PageInfo("finetune", 8, "Supervised fine-tuning & multi-slice input", "Fine-tuning & multi-slice"),
    PageInfo("submission", 9, "Heads & Kaggle submission", "Heads & Kaggle submission"),
    PageInfo("library", 10, "Evidence library", "Evidence library"),
]

BY_SLUG = {p.slug: p for p in PAGES}
TOPIC_SLUGS = [p.slug for p in PAGES if 1 <= p.number <= 9]
PAGE_SLUGS = [p.slug for p in PAGES if p.slug != "index"]  # slugs with a pages/<slug>.py module
