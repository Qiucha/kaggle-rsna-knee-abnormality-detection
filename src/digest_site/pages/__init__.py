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


def part_divider(slug: str, lead: str = "") -> str:
    """Heading that opens one legacy stage page inside a composed page (anchor ``part-<slug>``)."""
    title = BY_SLUG[slug].title
    p = f"<p>{lead}</p>" if lead else ""
    return (f'<div class="part-divider" id="part-{esc(slug)}"><div class="site-kicker">Part</div>'
            f"<h2>{esc(title)}</h2>{p}</div>")


def compose(ctx, parts: list[tuple[str, str]], lead_sections: list[str] | None = None,
            lead_data: dict | None = None) -> tuple[list[str], dict, list]:
    """Concatenate whole legacy pages, each under a part divider. Returns (sections, data, kpis of the first part)."""
    import importlib

    sections, blobs, kpis = list(lead_sections or []), dict(lead_data or {}), []
    for slug, lead in parts:
        page = importlib.import_module(f"digest_site.pages.{slug}").build(ctx)
        sections.append(part_divider(slug, lead))
        sections.extend(page.sections)
        clash = [k for k in page.data if k in blobs and blobs[k] != page.data[k]]
        if clash:
            raise SystemExit(f"composed page: {slug} reuses data ids {clash}")
        blobs.update(page.data)
        if not kpis and page.kpis:
            kpis = page.kpis
    return sections, blobs, kpis
