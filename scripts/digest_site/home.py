"""Home page (digest/site/index.html), built last from each topic page's summary().

Numbers here come from digest JSON (or the cited md-only constants in data.py);
nothing is typed by hand. Placeholder summaries render as "being built".
"""

from __future__ import annotations

import re

import markdown

from . import data
from .shell import Links, document, esc, kpi_row, num
from .sitemap import BY_SLUG, TOPIC_SLUGS

L = Links("home")


def _seed_mean(j: dict | None, slot_prefix: str) -> float | None:
    runs = [a for a in data.probe_runs(j) if a.slot == slot_prefix and a.feature == "uniform_mean_768"]
    return data.seed_mean([a.macro_auroc for a in runs])


def headline_numbers() -> dict:
    """The numbers the home page and library quote, each with its source."""
    unexposed = data.read_json("downstream-probe-unexposed-v2.json")
    lora = data.read_json("downstream-probe-lora-holdout-v1.json")
    lm = data.read_json("lm-head-tower-size-58-v1.json")
    out = {
        "dev": _seed_mean(unexposed, "exposed"),
        "gen": _seed_mean(unexposed, "unexposed"),
        "floor": data.unadapted_level(unexposed),
        "oof": None,
        "lm58": None,
        "lb": data.MD_ONLY["public_lb"]["value"],
        "gold": ((unexposed or {}).get("population") or {}).get("gold_studies"),
        "pop": ((unexposed or {}).get("population") or {}).get("population_size"),
        "folds": (((unexposed or {}).get("config") or {}).get("probe") or {}).get("folds"),
        "labels": len(data.probe_labels(unexposed or {})),
        "seeds": len([a for a in data.probe_runs(unexposed) if a.slot == "exposed" and a.feature == "uniform_mean_768"]),
    }
    for a in data.probe_runs(lora):
        if a.run == "holdout-both-lora-depth6-r64-seed2026" and a.feature == "uniform_mean_768":
            out["oof"] = a.macro_auroc
    if lm:
        arms = lm.get("arms") or {}
        if "small_image_base_text" in arms:
            out["lm58"] = arms["small_image_base_text"].get("mlp_macro_auroc")
    return out


def _flow(summaries: dict) -> str:
    nodes = []
    for slug in TOPIC_SLUGS:
        info = BY_SLUG[slug]
        s = summaries.get(slug) or {}
        href = L.page(slug) + (s["href"] if (s.get("href") or "").startswith("#") else "")
        if s.get("value"):
            value = f'<div class="value">{esc(str(s["value"]))}</div>'
            vlabel = f'<div class="vlabel">{esc(s.get("value_label") or "")}</div>'
        else:
            value = '<div class="value pending">Page being built</div>'
            vlabel = ""
        headline = f'<span class="headline">{esc(s["headline"])}</span>' if s.get("headline") else ""
        caveat = f'<div class="caveat">⚠ {esc(s["caveat"])}</div>' if s.get("caveat") else '<div class="caveat"></div>'
        # four fixed rows (title block, value, value label, caveat) so values line up across a row of cards
        nodes.append(f'<a class="flow-node" href="{href}" role="listitem"><div class="top"><span class="stage">{esc(info.group)} · {info.number}</span>'
                     f'<span class="name">{esc(info.title)}</span>{headline}</div>{value}{vlabel or "<div class=vlabel></div>"}{caveat}</a>')
    return f'<div class="flow" role="list">{"".join(nodes)}</div>'


def _strip(n: dict) -> str:
    """'The project in one picture': an inline-SVG pipeline strip, each node linking to its stage page."""
    prof = data.read_json("lora-memory-profile.json") or []
    r64 = next((r for r in prof if r.get("rank") == 64 and r.get("layer_count") == 6 and r.get("batch_size") == 32), None)
    lora = f"LoRA r{r64['rank']}, last {r64['layer_count']} blocks" if r64 else "LoRA adapters"
    labels, pop = n.get("labels") or "?", n.get("pop") or "?"
    W, H = 1120, 226
    # (key, x, y, w, h, title, subtitle, href, kind)
    nodes = [
        ("mri", 8, 22, 140, 64, "Knee MRI", "3 planes", "sagittal · coronal · axial", L.page("data"), "data"),
        ("rep", 8, 128, 140, 64, "Radiology report", "full report text", "", L.page("text"), "data"),
        ("img", 182, 22, 180, 64, "DINOv2-small", "image tower", lora, L.page("adaptation"), "lora"),
        ("txt", 182, 128, 180, 64, "multilingual-E5-base", "text tower", lora, L.page("encoders"), "lora"),
        ("loss", 400, 81, 150, 58, "Contrastive loss", "symmetric InfoNCE", "", L.page("pipeline"), "loss"),
        ("feat", 612, 22, 150, 64, "Frozen encoder", "image side only", "CLS+patch, plane mean", L.page("probes"), "step"),
        ("heads", 792, 22, 150, 64, f"{labels}-label heads", "logistic regression", f"fit on {pop} gold studies", L.page("submission") + "#model-card", "step"),
        ("kag", 972, 22, 140, 64, "Kaggle", "offline notebook", "code submission", L.page("submission") + "#bundle", "out"),
    ]
    box = {k: (x, y, w, h) for k, x, y, w, h, *_ in nodes}

    def right(k):
        x, y, w, h = box[k]; return x + w, y + h / 2

    def left(k):
        x, y, w, h = box[k]; return x, y + h / 2

    def arrow(a, b, d=None, dashed=False, label=""):
        (x1, y1), (x2, y2) = right(a), left(b)
        path = d or (f"M{x1},{y1} L{x2 - 3},{y2}" if y1 == y2 else
                     f"M{x1},{y1} C{x1 + 24},{y1} {x2 - 30},{y2} {x2 - 3},{y2}")
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out = f'<path class="ps-arrow" d="{path}"{dash} marker-end="url(#ps-head)"/>'
        if label:
            out += f'<text class="ps-note" x="{(x1 + x2) / 2:.0f}" y="{y1 - 7:.0f}" text-anchor="middle">{esc(label)}</text>'
        return out

    parts = [f'<svg class="ps-svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="ps-title">'
             '<title id="ps-title">Pipeline: knee MRI and reports train two towers with a contrastive loss; the frozen image '
             'encoder then feeds per-label heads for an offline Kaggle submission.</title>'
             '<defs><marker id="ps-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path class="ps-head" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
             f'<rect class="ps-phase" x="2" y="4" width="{box["loss"][0] + box["loss"][2] + 14}" height="{H - 8}" rx="12"/>'
             f'<text class="ps-phase-label" x="12" y="{H - 13}">TRAINING</text>'
             f'<text class="ps-phase-label" x="{box["feat"][0]}" y="{H - 13}">INFERENCE (image side only)</text>']
    parts += [arrow("mri", "img"), arrow("rep", "txt"), arrow("img", "loss"), arrow("txt", "loss"),
              arrow("img", "feat", label="after training: freeze image tower"), arrow("feat", "heads"), arrow("heads", "kag")]
    for k, x, y, w, h, title, sub, sub2, href, kind in nodes:
        top = y + (20 if sub2 else 25)
        line2 = (f'<text class="ps-s" x="{x + w / 2}" y="{top + 32}" text-anchor="middle">{esc(sub2)}</text>' if sub2 else "")
        parts.append(f'<a class="ps-node ps-{kind}" href="{href}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/>'
                     f'<text class="ps-t" x="{x + w / 2}" y="{top}" text-anchor="middle">{esc(title)}</text>'
                     f'<text class="ps-s" x="{x + w / 2}" y="{top + 17}" text-anchor="middle">{esc(sub)}</text>{line2}</a>')
    parts.append("</svg>")
    return ('<section class="card" id="picture"><h2>The project in one picture</h2>'
            '<p class="caption">Click a box to open its stage. The report tower only shapes training; the submission ships the image side.</p>'
            f'<div class="pipeline-strip">{"".join(parts)}</div>'
            '<p class="ps-hint">Scroll sideways for the whole pipeline →</p></section>')


def _phase_map(summaries: dict) -> str:
    """The mind map: Phase 1 → picking → Phase 2 → Kaggle, each box with its page's headline number."""
    def node(slug: str, kicker: str, title: str, bullets: list[str]) -> str:
        s = summaries.get(slug) or {}
        href = L.page(slug) + (s["href"] if (s.get("href") or "").startswith("#") else "")
        value = (f'<div class="pm-value">{esc(str(s["value"]))}</div><div class="pm-label">{esc(s.get("value_label") or "")}</div>'
                 if s.get("value") else "")
        items = "".join(f"<li>{b}</li>" for b in bullets)
        return (f'<a class="pm-node" href="{href}"><div class="pm-kick">{esc(kicker)}</div>'
                f'<div class="pm-title">{esc(title)}</div>{value}<ul>{items}</ul></a>')

    arrow = '<div class="pm-arrow" aria-hidden="true">→</div>'
    return ('<section class="card" id="phase-map"><h2>The project in one picture</h2>'
            '<p class="caption">The current numbers are read from the digest JSON at build time.</p>'
            '<div class="phase-map" role="list">'
            + node("phase1", "Phase 1", "Contrastive pre-training", [
                "CLIP loss, 32 × 32 in-batch similarity", "LoRA r64 on the last 6 blocks",
                "Factors: multi-slice · resolution · model swap", "Combined run with every factor"])
            + arrow
            + node("picking", "Picking", "Which model to transfer?", [
                "Retrieval R@1 vs frozen heads vs fine-tuning", "Contrastive start vs pretrained start",
                "OmniRad frozen probe"])
            + arrow
            + node("phase2", "Phase 2", "Label fine-tuning", [
                "① multi-slice (3 planes × 8 × 3)", "② resolution 224 → 336", "③ soft LM labels",
                "OmniRad-base backbone → run 5"])
            + arrow
            + node("submission", "Ship", "Kaggle submission", ["Offline bundle", "Heads and protocols"])
            + "</div></section>")


def _next_card() -> str:
    text = data.read_text("next-model-candidates.md")
    if not text:
        return ""
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].isdigit():
            pair = markdown.markdown(cells[1])[3:-4]  # strip <p></p>
            rows.append(f"<li><strong>{pair}</strong>: {esc(cells[2])}</li>")
    ref = re.search(r"The current reference is \*\*(.+?)\*\*", text)
    ref_txt = f"Reference pair: <strong>{esc(ref.group(1))}</strong>. " if ref else ""
    return (f'<section class="card next-card" id="whats-next"><h2>What\'s next: the tower candidate queue</h2>'
            f'<p class="caption">{ref_txt}One tower changes at a time; gold differences are exploratory. '
            f'Full protocol in <a href="{L.doc("next-model-candidates.md")}">next-model-candidates</a>; '
            f'first-round results in <a href="{L.doc("next-model-pairs-v1.md")}">next-model-pairs-v1</a>.</p>'
            f'<ol>{"".join(rows[:5])}</ol></section>')


def render(summaries: dict) -> str:
    n = headline_numbers()
    f4 = lambda v: num(v, ".4f")  # noqa: E731
    lb = data.MD_ONLY["public_lb"]
    history = data.read_json("leaderboard-history-v1.json")
    best = next((r for r in history["submissions"] if r["bundle"] == history["best"]), None) if history else None
    kpis = kpi_row([
        {"label": "Gold probe, development", "value": f4(n["dev"]),
         "sub": f"4-block image tower, {n['seeds']} seeds, exposed gold. Quote with {f4(n['gen'])}."},
        {"label": "Gold probe, generalisation", "value": f4(n["gen"]),
         "sub": f"Same arm retrained without gold studies. Quote with {f4(n['dev'])}."},
        {"label": "Unadapted DINOv2 floor", "value": f4(n["floor"]),
         "sub": f"Frozen backbone, same probe and {n['pop']} studies."},
        {"label": "Submitted model, OOF", "value": f4(n["oof"]),
         "sub": f"LoRA holdout encoder, {n['pop']}-study {n['folds']}-fold OOF (single seed)."},
        {"label": "Kaggle public LB, best", "value": num(best["public"] if best else lb["value"], ".3f"),
         "sub": (f"{best['title']} (submission {best['submission']}, {best['recorded']}). "
                 if best else "") + f"The submitted OOF model scored {num(lb['value'], '.3f')}; different cohort, not paired."},
    ])
    caveats = (
        '<div class="caveat-strip">'
        f'<div><strong>{f4(n["dev"])} always travels with {f4(n["gen"])}.</strong> The first is read on gold studies the '
        f'encoder trained on; the second on studies it never saw. Exposure gap {num(data.POOLED["gold_holdout_exposure_gap"]["value"], ".4f")} '
        f'(pooled, <a href="{L.doc("gold-holdout-v1.md")}">gold-holdout-v1</a>).</div>'
        f'<div><strong>{n["pop"]}- and {n["gold"]}-case protocols differ.</strong> The probe and OOF numbers use {n["pop"]} gold studies; the LM-head '
        f'comparison ({f4(n["lm58"])} best) uses all {n["gold"]}. They are not paired.</div>'
        f'<div><strong>Public LB {num(lb["value"], ".3f")} vs OOF {num(n["oof"], ".3f")} is not a generalisation gap.</strong> '
        f'Different cohorts and scoring; see <a href="{L.doc("kaggle-submission-pipeline.md")}">the submission pipeline</a>.</div>'
        "</div>"
    )
    body = f"""<main>
<section class="hero" id="overview"><div class="site-kicker">Knee MRI · image–report contrastive learning → pathology labels</div>
<h1>Which model to transfer?</h1>
<p>Phase 1 trains contrastive image–report encoders; Phase 2 fine-tunes an image encoder on {n['labels']} pathology labels.
The open question is the picking step between them: does a better Phase-1 model make a better Phase-2 model?
Follow the main track left to right; the background pages hold the earlier work that fixed the recipe.</p></section>
{_phase_map(summaries)}
<section class="card" id="process"><h2>Every page, with its headline number</h2>
<p class="caption">Main track first, then background. Click a card to open the page.</p>
{_flow(summaries)}</section>
<section class="card" id="earlier"><h2>Earlier headline numbers (frozen-probe era)</h2>
<p class="caption">The first submitted model: DINOv2-small + E5-base LoRA encoder with gold-only logistic heads. Kept for the record; the current models are on the Phase 2 page.</p>
{kpis}{caveats}</section>
{_next_card()}
</main>"""
    return document(title="Which model to transfer?", current="index", links=L, body=body,
                    description="Knee MRI: Phase-1 contrastive encoders, the picking step, and Phase-2 label fine-tuning.")
