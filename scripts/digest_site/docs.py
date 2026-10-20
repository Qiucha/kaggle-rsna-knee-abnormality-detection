"""Render every digest/*.md into digest/site/docs/<name>.html inside the site shell.

Link rewriting (the markdown lives in digest/, the output in digest/site/docs/):
  * ``x.md[#a]`` that resolves to a catalogued digest markdown -> ``x.html[#a]`` (sibling doc)
  * any other relative link is re-based, e.g. ``../configs/a.json`` -> ``../../../configs/a.json``,
    ``foo.json`` -> ``../../foo.json``
  * a relative link whose target does not exist is rendered as plain text marked
    ``.dead-link`` (so the site has no broken links), and counted in the build log.
"""

from __future__ import annotations

import html
import posixpath
import re
from pathlib import Path

import markdown

from . import catalog
from .shell import Links, caveat_badge, document, entry_links, esc, kind_label, status_badge
from .sitemap import BY_SLUG

ATTR = re.compile(r'\b(href|src)="([^"]*)"')
ANCHOR = re.compile(r'<a\b([^>]*)\bhref="([^"]*)"([^>]*)>(.*?)</a>', re.S)
TABLE = re.compile(r"<table>(.*?)</table>", re.S)
SCHEMES = ("http:", "https:", "mailto:", "data:", "javascript:", "#", "/", "file:")


def _rebase(target: str, repo_root: Path, out_dir: str = "digest/site/docs") -> tuple[str | None, bool]:
    """Return (new_target, exists) for a link written relative to digest/, re-based to out_dir."""
    if not target or target.startswith(SCHEMES):
        return target, True
    path, _, frag = target.partition("#")
    path_q, _, query = path.partition("?")
    if not path_q:
        return target, True
    norm = posixpath.normpath(posixpath.join("digest", path_q))
    suffix = (f"?{query}" if query else "") + (f"#{frag}" if frag else "")
    exists = (repo_root / norm).exists()
    if norm.startswith("digest/") and norm.count("/") == 1 and norm.endswith(".md") and norm[7:] in catalog.BY_PATH:
        return posixpath.relpath("digest/site/docs/" + norm[7:-3] + ".html", out_dir) + suffix, exists
    rel = posixpath.relpath(norm, out_dir)
    if path_q.endswith("/") and not rel.endswith("/"):
        rel += "/"
    return rel + suffix, exists


LOCAL_ONLY = ("digest/handoff",)


def _is_local_only(target: str) -> bool:
    """Links into gitignored local notes are never published as links."""
    if not target or target.startswith(SCHEMES):
        return False
    norm = posixpath.normpath(posixpath.join("digest", target.partition("#")[0]))
    return any(norm == p or norm.startswith(p + "/") for p in LOCAL_ONLY)


def rewrite_links(body: str, repo_root: Path, dead: list[str], out_dir: str = "digest/site/docs") -> str:
    def anchor(m: re.Match) -> str:
        pre, href, post, text = m.groups()
        href = html.unescape(href)
        if _is_local_only(href):
            return f'<span class="dead-link" title="Local working notes (digest/handoff/), not published: {esc(href)}">{text}</span>'
        new, ok = _rebase(href, repo_root, out_dir)
        if not ok:
            dead.append(href)
            return f'<span class="dead-link" title="Link target not found in this checkout: {esc(href)}">{text}</span>'
        return f'<a{pre}href="{esc(new, quote=True)}"{post}>{text}</a>'

    body = ANCHOR.sub(anchor, body)

    def src(m: re.Match) -> str:
        attr, target = m.groups()
        if attr != "src":
            return m.group(0)
        target = html.unescape(target)
        new, ok = _rebase(target, repo_root, out_dir)
        if not ok:
            dead.append(target)
        return f'{attr}="{esc(new, quote=True)}"'

    return ATTR.sub(src, body)


def wrap_tables(body: str) -> str:
    def wrap(m: re.Match) -> str:
        inner = m.group(1)
        rows = inner.count("<tr")
        search = ' data-search="Find rows"' if rows >= 13 else ""
        return f'<div class="table-wrap"><table{search}>{inner}</table></div>'

    return TABLE.sub(wrap, body)


def render_markdown(text: str) -> str:
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"permalink": False}})
    return md.convert(text)


def render_all(digest: Path, out_dir: Path) -> dict[str, int]:
    """Render every catalogued markdown; returns {doc name: dead-link count}."""
    repo_root = digest.parent
    links = Links("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, int] = {}
    for e in catalog.ENTRIES:
        if not e.is_markdown:
            continue
        src = digest / e.path
        if not src.exists():
            continue
        dead: list[str] = []
        body = render_markdown(src.read_text())
        body = rewrite_links(body, repo_root, dead)
        body = wrap_tables(body)
        body = body.replace("<pre>", '<pre tabindex="0">')
        home = BY_SLUG[e.page]
        also = ", ".join(f'<a href="{links.page(s)}">{esc(BY_SLUG[s].short)}</a>' for s in e.also_on)
        meta = (f'<div class="doc-meta">{status_badge(e.status)} {caveat_badge(e.caveat)}'
                f'<span class="badge badge-kind">{esc(kind_label(e.kind))}</span>'
                f'{entry_links(e, links)}'
                + (f'<span class="muted">Also on: {also}</span>' if also else "")
                + "</div>")
        back = f'<a class="back-link" href="{links.page(e.page)}">← Back to {esc(home.title)}</a>'
        heading = (f'<div class="page-heading">{back}<div class="site-kicker">Document · digest/{esc(e.path)}</div>'
                   f'<p>{esc(e.purpose)}</p></div>')
        html_body = f'<main class="card doc">{meta}{body}</main>'
        title = _title(src.read_text()) or e.path
        page = document(title=title, current=e.page, links=links, body=html_body, heading=heading,
                        description=e.purpose)
        (out_dir / (e.path[:-3] + ".html")).write_text(page)
        report[e.path] = len(dead)
    return report


def _title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return re.sub(r"[*`]", "", line[2:]).strip()
    return None
