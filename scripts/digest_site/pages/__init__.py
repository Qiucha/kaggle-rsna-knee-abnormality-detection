"""One module per site page. Each exposes:

    build(ctx) -> shell.Page        # sections + inline data blobs
    summary()  -> dict              # for the Home process flow:
        {stage: int, headline: str|None, value: str|None, value_label: str|None,
         caveat: str|None, href: str|None}

Home is built last from every topic page's summary(); a summary with value None
renders as "being built". See pages/adaptation.py for the reference section pattern.
"""

from __future__ import annotations

from ..shell import Page, esc, notice, section
from ..sitemap import BY_SLUG


def placeholder_page(slug: str, subtitle: str) -> Page:
    info = BY_SLUG[slug]
    body = notice(
        f"This page is being rebuilt. Until its charts land, every file behind "
        f"<strong>{esc(info.title)}</strong> is listed under <a href=\"#evidence\">Evidence</a> below, "
        f"and the <a href=\"library.html\">Evidence library</a> lists all digest files."
    )
    return Page(title=info.title, subtitle=subtitle, sections=[section("being-built", "Being built", body)])


def placeholder_summary(slug: str) -> dict:
    info = BY_SLUG[slug]
    return {"stage": info.number, "headline": None, "value": None, "value_label": None, "caveat": None, "href": None}
