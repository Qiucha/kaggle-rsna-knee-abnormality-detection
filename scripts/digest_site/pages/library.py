"""Evidence library: every catalogued digest file, filterable, plus the reading room and house conventions."""

from __future__ import annotations

import re

from .. import catalog, data, docs
from ..home import headline_numbers
from ..shell import (Page, caveat_badge, entry_links, entry_name_cell, esc, kind_label, kpi_row, num, section,
                     status_badge, table)
from ..sitemap import BY_SLUG, PAGES

SLUG = "library"
SUBTITLE = ("Every file in digest/, with the page that explains it, what kind of evidence it is, and whether it "
            "still stands. Markdown opens rendered; data opens raw.")


def _readme_section(heading: str) -> str | None:
    text = data.read_text("README.md")
    if not text:
        return None
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else None


def _render_md(md: str) -> str:
    dead: list[str] = []
    body = docs.render_markdown(md)
    return docs.rewrite_links(body, data.ROOT, dead, out_dir="digest/site/pages")


def _reading_room(ctx) -> str:
    rows, attrs = [], []
    for name in catalog.READING_ROOM:
        e = catalog.BY_PATH[name]
        rows.append([entry_name_cell(e, ctx.links), status_badge(e.status) + " " + caveat_badge(e.caveat),
                     esc(e.purpose)])
    body = table(["Document", "Status", "What it answers"], rows, raw_html=True)
    return section("reading-room", "Reviews and research notes: read these for the reasoning, not the results",
                   body, caption="External feedback and the response, encoder and translation research, and the "
                                 "error-review reading packet. They explain choices; results live on the stage pages.")


def _conventions(ctx) -> str:
    n = headline_numbers()
    gap = data.POOLED["gold_holdout_exposure_gap"]
    tiles = kpi_row([
        {"label": "Development level", "value": num(n["dev"]), "sub": "Exposed gold, pooled seeds"},
        {"label": "Generalisation level", "value": num(n["gen"]), "sub": "Gold held out of training"},
        {"label": "Exposure gap (pooled)", "value": num(gap["value"]), "sub": f"md-only: {esc(gap['source'])}"},
        {"label": "Unadapted floor", "value": num(n["floor"]), "sub": "Frozen DINOv2, same probe"},
    ])
    rules = _readme_section("Conventions that keep these documents honest")
    rules_html = _render_md(rules) if rules else ""
    body = (tiles + '<p class="lede">Quote the two gold-probe levels together, never one instead of the other. '
            "Arms in a paired contrast share the same exposure, so never subtract the exposure gap from a paired "
            f'difference. Source: <a href="{ctx.doc("gold-holdout-v1.md")}">gold-holdout-v1</a>.</p>'
            f"<h3>Metric-scope rules (from the digest README)</h3>{rules_html}")
    return section("conventions", f"Two numbers to carry everywhere: {num(n['dev'])} with {num(n['gen'])}", body)


def _all_files(ctx) -> str:
    rank = {s: i for i, s in enumerate(catalog.STATUSES)}
    order = {p.slug: p.number for p in PAGES}
    entries = sorted(catalog.ENTRIES, key=lambda e: (order[e.page], rank.get(e.status, 99), e.path))
    rows, attrs = [], []
    for e in entries:
        pages = [f'<a href="{ctx.page(e.page)}">{BY_SLUG[e.page].number} {esc(BY_SLUG[e.page].short)}</a>']
        pages += [f'<a class="muted" href="{ctx.page(s)}">{BY_SLUG[s].number} {esc(BY_SLUG[s].short)}</a>' for s in e.also_on]
        rows.append([
            entry_name_cell(e, ctx.links),
            "<br>".join(pages),
            esc(kind_label(e.kind)),
            status_badge(e.status) + (" " + caveat_badge(e.caveat) if e.caveat else ""),
            esc(e.purpose),
            entry_links(e, ctx.links),
        ])
        attrs.append({"data-pages": " ".join(e.pages), "data-kind": e.kind, "data-status": e.status})
    tbl = table(["File", "Page(s)", "Kind", "Status", "Purpose", "Open"], rows, raw_html=True, row_attrs=attrs,
                cls="library-table")
    used_kinds = sorted({e.kind for e in catalog.ENTRIES}, key=lambda k: kind_label(k))
    used_status = [s for s in catalog.STATUSES if any(e.status == s for e in catalog.ENTRIES)]

    def opts(pairs):
        return "".join(f'<option value="{esc(v)}">{esc(t)}</option>' for v, t in pairs)

    controls = (
        '<div class="site-table-controls library-filters">'
        '<label for="lib-q">Find</label><input id="lib-q" type="search" placeholder="file name or purpose">'
        '<label for="lib-page">Page</label><select id="lib-page"><option value="">All pages</option>'
        + opts([(p.slug, f"{p.number} {p.short}") for p in PAGES if p.slug != "index"]) + "</select>"
        '<label for="lib-kind">Kind</label><select id="lib-kind"><option value="">All kinds</option>'
        + opts([(k, kind_label(k)) for k in used_kinds]) + "</select>"
        '<label for="lib-status">Status</label><select id="lib-status"><option value="">All statuses</option>'
        + opts([(s, s.replace("-", " ")) for s in used_status]) + "</select>"
        '<span id="lib-count" aria-live="polite"></span></div>'
    )
    script = """<script>
(function(){
  var q=document.getElementById('lib-q'), pg=document.getElementById('lib-page'),
      kd=document.getElementById('lib-kind'), st=document.getElementById('lib-status'),
      count=document.getElementById('lib-count');
  var rows=Array.prototype.slice.call(document.querySelectorAll('.library-table tbody tr'));
  function apply(){
    var t=q.value.trim().toLowerCase(), shown=0;
    rows.forEach(function(r){
      var ok=(!t||r.textContent.toLowerCase().indexOf(t)!==-1)
        &&(!pg.value||(' '+r.getAttribute('data-pages')+' ').indexOf(' '+pg.value+' ')!==-1)
        &&(!kd.value||r.getAttribute('data-kind')===kd.value)
        &&(!st.value||r.getAttribute('data-status')===st.value);
      r.hidden=!ok; if(ok) shown++;
    });
    count.textContent=shown+' / '+rows.length+' files';
  }
  [q,pg,kd,st].forEach(function(el){el.addEventListener('input',apply);el.addEventListener('change',apply);});
  var h=(location.hash||'').match(/^#page=([a-z]+)$/); if(h){pg.value=h[1];}
  apply();
})();
</script>"""
    counts = {s: sum(1 for e in catalog.ENTRIES if e.status == s) for s in used_status}
    caption = ("Status counts: " + ", ".join(f"{c} {s.replace('-', ' ')}" for s, c in counts.items())
               + ". digest/handoff/ is local working notes with machine state and is never linked.")
    return section("all-files", f"All {len(catalog.ENTRIES)} digest entries, filterable by page, kind and status",
                   controls + tbl + script, caption=caption)


def build(ctx) -> Page:
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE,
                sections=[_reading_room(ctx), _conventions(ctx), _all_files(ctx)])


def summary() -> dict:
    return {"stage": 9, "headline": f"{len(catalog.ENTRIES)} catalogued files", "value": None,
            "value_label": None, "caveat": None, "href": None}
