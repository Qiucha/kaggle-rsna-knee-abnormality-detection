"""Site shell: page template, navigation, evidence drawer, and small HTML helpers.

Page modules build section HTML with the helpers here (``section``, ``chart``,
``json_script``, ``table``, ``details``, ``num``) and return a ``Page``. The
shell wraps it in the header, sidebar, on-page TOC, prev/next buttons and the
Evidence drawer. Page agents treat this module as read-only.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import catalog
from .sitemap import BY_SLUG, PAGES, PageInfo

esc = html.escape

# --------------------------------------------------------------------------- model


@dataclass
class Page:
    """What a page module's ``build(ctx)`` returns.

    sections: HTML strings, each normally produced by ``section(...)``.
    data: ``{element_id: json_able}``; each becomes an inline
          ``<script type="application/json" id=element_id>`` placed before the sections,
          so inline section scripts can ``Charts.readJSON(element_id)``.
    kpis: optional server-rendered KPI row ``[{label, value, sub?, tone?}]`` (value is text).
    """

    title: str
    subtitle: str = ""
    sections: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    kpis: list[dict] | None = None


class Links:
    """Relative hrefs from one output location: 'home' (site/), 'pages' (site/pages/), 'docs' (site/docs/)."""

    def __init__(self, where: str):
        assert where in ("home", "pages", "docs")
        self.where = where
        self.up = "" if where == "home" else "../"

    def page(self, slug: str) -> str:
        if slug == "index":
            return f"{self.up}index.html"
        return f"pages/{slug}.html" if self.where == "home" else (f"{slug}.html" if self.where == "pages" else f"../pages/{slug}.html")

    def asset(self, name: str) -> str:
        return f"{self.up}assets/{name}"

    def doc(self, md_name: str) -> str:
        stem = md_name[:-3] if md_name.endswith(".md") else md_name
        return f"{stem}.html" if self.where == "docs" else f"{self.up}docs/{stem}.html"

    def raw(self, digest_path: str) -> str:
        """A file at the digest/ root."""
        return f"../{digest_path}" if self.where == "home" else f"../../{digest_path}"

    def repo(self, repo_path: str) -> str:
        """A file anywhere in the repository (relative to the repo root)."""
        return f"../../{repo_path}" if self.where == "home" else f"../../../{repo_path}"


# --------------------------------------------------------------------------- helpers


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def json_script(element_id: str, obj: Any) -> str:
    """Inline JSON that is safe inside HTML and parseable by JSON.parse (no NaN/Infinity)."""
    text = json.dumps(_sanitize(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    text = text.replace("</", "<\\/").replace("<!--", "<\\!--")
    return f'<script type="application/json" id="{esc(element_id)}">{text}</script>'


def num(value: Any, spec: str = ".4f", signed: bool = False) -> str:
    """Format a number for prose/tables with a real minus sign. None -> en dash."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if spec.endswith("%"):
        text = format(value * 100, ("+" if signed else "") + spec[:-1] + "f") + "%"
    else:
        text = format(value, ("+" if signed else "") + spec)
    return text.replace("-", "−")


def details(summary: str, body: str, cls: str = "notes", open_: bool = False) -> str:
    return f'<details class="{cls}"{" open" if open_ else ""}><summary>{summary}</summary>{body}</details>'


def section(sid: str, claim: str, body: str, caption: str = "", notes: str = "", table_html: str = "",
            table_summary: str = "Full table") -> str:
    """One claim-led section: h2 claim sentence, chart/body, <=2-line caption, collapsed table and notes."""
    parts = [f'<section class="card" id="{esc(sid)}">', f"<h2>{claim}</h2>", body]
    if caption:
        parts.append(f'<p class="caption">{caption}</p>')
    if table_html:
        parts.append(details(table_summary, table_html, cls="table"))
    if notes:
        parts.append(details("Method and caveats", notes))
    parts.append("</section>")
    return "\n".join(parts)


def chart(name: str, title: str = "", aria: str = "") -> str:
    """A chart mount point: ``<div class="chart" data-chart=name>``; the inline script renders into it."""
    head = f'<div class="chart-title">{title}</div>' if title else ""
    return f'{head}<div class="chart" data-chart="{esc(name)}" aria-label="{esc(aria or title)}"></div>'


def notice(text: str) -> str:
    return f'<div class="notice">{text}</div>'


def table(headers: list[str], rows: Iterable[list[Any]], *, search: str | None = None,
          num_cols: Iterable[int] = (), row_attrs: list[dict] | None = None, raw_html: bool = False,
          cls: str = "") -> str:
    """Simple table. Cells are escaped unless raw_html=True. row_attrs[i] adds attributes to row i
    (e.g. {"data-series-id": run, "data-group": g} to follow the section's selection store)."""
    num_cols = set(num_cols)
    head = "".join(f'<th{" class=num" if i in num_cols else ""} scope="col">{h}</th>' for i, h in enumerate(headers))
    body = []
    for r, row in enumerate(rows):
        attrs = ""
        if row_attrs and r < len(row_attrs) and row_attrs[r]:
            attrs = "".join(f' {k}="{esc(str(v))}"' for k, v in row_attrs[r].items())
        cells = "".join(
            f'<td{" class=num" if i in num_cols else ""}>{c if raw_html else esc(str(c))}</td>' for i, c in enumerate(row)
        )
        body.append(f"<tr{attrs}>{cells}</tr>")
    search_attr = f' data-search="{esc(search)}"' if search else ""
    cls_attr = f' class="{cls}"' if cls else ""
    return (f'<div class="table-wrap"><table{cls_attr}{search_attr}><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def status_badge(status: str) -> str:
    return f'<span class="badge badge-{esc(status)}">{esc(status.replace("-", " "))}</span>'


def caveat_badge(text: str) -> str:
    return f'<span class="badge badge-caveat">{esc(text)}</span>' if text else ""


def kind_label(kind: str) -> str:
    return catalog.KINDS.get(kind, kind)


def kpi_row(tiles: list[dict]) -> str:
    out = []
    for t in tiles:
        tone = f" {t['tone']}" if t.get("tone") else ""
        sub = f'<div class="kpi-sub">{t["sub"]}</div>' if t.get("sub") else ""
        out.append(f'<div class="kpi{tone}"><div class="kpi-label">{t["label"]}</div>'
                   f'<div class="kpi-value">{t["value"]}</div>{sub}</div>')
    return f'<div class="kpi-row">{"".join(out)}</div>'


# --------------------------------------------------------------------------- evidence rows


def entry_links(e: catalog.Entry, links: Links) -> str:
    """Doc / raw / original-report links for one catalog row."""
    if e.status == "excluded":
        return '<span class="muted">not linked</span>'
    out = []
    if e.is_markdown:
        out.append(f'<a href="{links.doc(e.path)}">read</a>')
    if e.is_legacy_html:
        out.append(f'<a href="{links.raw(e.path)}">original report</a>')
    else:
        out.append(f'<a href="{links.raw(e.path)}">raw</a>')
    if e.original_report:
        out.append(f'<a href="{links.raw(e.original_report)}">original report</a>')
    return '<span class="links">' + "".join(out) + "</span>"


def entry_primary_href(e: catalog.Entry, links: Links) -> str | None:
    if e.status == "excluded":
        return None
    return links.doc(e.path) if e.is_markdown else links.raw(e.path)


def entry_name_cell(e: catalog.Entry, links: Links) -> str:
    href = entry_primary_href(e, links)
    name = esc(e.name)
    return f'<a class="path" href="{href}">{name}</a>' if href else f'<span class="path">{name}</span>'


def evidence_section(slug: str, links: Links) -> str:
    rows = catalog.for_page(slug)
    rank = {s: i for i, s in enumerate(catalog.STATUSES)}
    rows.sort(key=lambda e: (e.page != slug, rank.get(e.status, 99), e.path))
    primary = sum(1 for e in rows if e.page == slug)
    body = []
    for e in rows:
        shared = "" if e.page == slug else f' <span class="muted">(home: {esc(BY_SLUG[e.page].short)})</span>'
        body.append([
            entry_name_cell(e, links) + shared,
            esc(kind_label(e.kind)),
            status_badge(e.status) + (" " + caveat_badge(e.caveat) if e.caveat else ""),
            esc(e.purpose),
            entry_links(e, links),
        ])
    tbl = table(["File", "Kind", "Status", "Purpose", "Open"], body, raw_html=True,
                search="Find files" if len(rows) >= 8 else None)
    summary = (f"{len(rows)} files: {primary} owned by this page, {len(rows) - primary} shared with other pages. "
               f"Markdown opens rendered; data files open raw.")
    return (f'<section class="card evidence-card" id="evidence"><h2>Evidence behind this page</h2>'
            f'<details class="evidence"><summary>{summary}</summary>{tbl}</details></section>')


# --------------------------------------------------------------------------- layout


THEME_INIT = ("<script>try{var t=localStorage.getItem('novo-digest-theme');"
              "if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t)}catch(e){}</script>")


def _sidebar(current: str, links: Links) -> str:
    items = []
    for p in PAGES:
        cur = ' aria-current="page"' if p.slug == current else ""
        items.append(f'<a href="{links.page(p.slug)}"{cur}><span class="no">{p.number}</span><span>{esc(p.short)}</span></a>')
    return (f'<aside class="site-sidebar" aria-label="Digest navigation"><div class="site-sidebar-label">Research process</div>'
            f'<nav class="site-nav">{"".join(items)}</nav><div class="site-toc" aria-label="On this page"></div></aside>')


def prev_next(current: str, links: Links) -> str:
    order = [p for p in PAGES]
    idx = next((i for i, p in enumerate(order) if p.slug == current), None)
    if idx is None:
        return ""
    parts = []
    if idx > 0:
        p = order[idx - 1]
        parts.append(f'<a class="pn prev" href="{links.page(p.slug)}"><small>← Previous: {p.number}</small>{esc(p.title)}</a>')
    else:
        parts.append("<span></span>")
    if idx < len(order) - 1:
        p = order[idx + 1]
        parts.append(f'<a class="pn next" href="{links.page(p.slug)}"><small>Next: {p.number} →</small>{esc(p.title)}</a>')
    return f'<nav class="prev-next" aria-label="Previous and next stage">{"".join(parts)}</nav>'


def document(*, title: str, current: str, links: Links, body: str, heading: str = "", extra_head: str = "",
             description: str = "") -> str:
    """Full HTML document in the site shell."""
    desc = f'<meta name="description" content="{esc(description)}">' if description else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · Novo research digest</title>
{desc}<link rel="icon" href="data:,">
{THEME_INIT}
<link rel="stylesheet" href="{links.asset('site.css')}">
<script src="{links.asset('charts.js')}"></script>
{extra_head}</head>
<body class="site-page page-{esc(current)}">
<a class="site-skip" href="#content">Skip to content</a>
<header class="site-header"><a class="site-brand" href="{links.page('index')}">NOVO <span>/</span> RESEARCH</a>
<a class="site-header-link" href="{links.page('library')}">Evidence library</a>
<button type="button" class="theme-toggle" aria-pressed="false">Dark mode</button></header>
<div class="site-layout">
{_sidebar(current, links)}
<div class="site-content" id="content">
{heading}
{body}
</div>
</div>
<script src="{links.asset('site.js')}"></script>
</body>
</html>
"""


def page_heading(info: PageInfo, subtitle: str, kicker: str = "") -> str:
    kick = kicker or (f"Stage {info.number} of 8" if 1 <= info.number <= 8 else info.short)
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return f'<div class="page-heading"><div class="site-kicker">{esc(kick)}</div><h1>{esc(info.title)}</h1>{sub}</div>'


def render_page(slug: str, page: Page, links: Links, with_evidence: bool = True) -> str:
    info = BY_SLUG[slug]
    data = "\n".join(json_script(k, v) for k, v in page.data.items())
    kpis = kpi_row(page.kpis) if page.kpis else ""
    evidence = evidence_section(slug, links) if with_evidence else ""
    body = f"<main>\n{data}\n{kpis}\n" + "\n".join(page.sections) + f"\n{evidence}\n</main>\n{prev_next(slug, links)}"
    return document(title=page.title or info.title, current=slug, links=links, body=body,
                    heading=page_heading(info, page.subtitle), description=page.subtitle)
