"""Heads & Kaggle submission: the submitted model, the offline bundle, and the LM-supervised head comparisons.

Two protocols live on this page and never share an axis:
- 56-case gold-only OOF (logistic heads fitted on gold, 8-fold out-of-fold): the submitted model, lm-mlp-v1.
- 58-case LM-head (heads trained on LM pseudo-labels, scored on all 58 gold studies): lm-head-tower-size-58-v1.
Absorbs the legacy lm-head-tower-size-58-v1.html (linked once as "original report").
"""

from __future__ import annotations

from .. import data
from ..shell import Page, caveat_badge, chart, esc, kpi_row, notice, num, section, table
from ..sitemap import BY_SLUG
from ._encoders_util import md_interval, md_number, md_table, section_script

SLUG = "submission"
SUBTITLE = "The submitted encoder and gold-only heads, the offline Kaggle bundle, and what LM-supervised heads change."

FACTORIAL = "tower-size-factorial-v1.json"
LM58 = "lm-head-tower-size-58-v1.json"
LM58_MD = "lm-head-tower-size-58-v1.md"
LMMLP = "lm-mlp-v1-results.json"
LORA_PROBE = "downstream-probe-lora-holdout-v1.json"
PIPELINE_MD = "kaggle-submission-pipeline.md"

# md-only leaderboard facts (no JSON carries them); the public LB value itself is data.MD_ONLY["public_lb"].
SUBMISSION_ID = "56479080"  # source: digest/kaggle-submission-pipeline.md:99
LB_RANK, LB_TEAMS = 3427, 4202  # source: digest/kaggle-submission-pipeline.md:100
LB_MEDIAN = 0.917  # source: digest/kaggle-submission-pipeline.md:102

IMG_GROUPS = {
    "small": {"label": "DINOv2-small image tower", "color": "var(--g-teal)"},
    "base": {"label": "DINOv2-base image tower", "color": "var(--g-violet)"},
}


def _ci(lo, hi, spec=".4f") -> str:
    return f"[{num(lo, spec, True)}, {num(hi, spec, True)}]"


def _excl(lo, hi) -> bool:
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


def _size(name: str) -> str:
    return "base" if "base" in name else "small"


def _arm_label(a: dict) -> str:
    return f"{a['image']} + {a['text']}"


def _compact(name: str) -> str:
    return name.replace("-small", "-S").replace("-base", "-B")


def _arm_short(a: dict) -> str:
    return f"{_compact(a['image'])} + {_compact(a['text'])}"


def _hf(model: str) -> str:
    m = model.split("/")[-1].replace("multilingual-", "")
    return m.replace("dinov2", "DINOv2").replace("e5", "E5")


def _arm_key(arms: dict, image: str, text: str) -> str | None:
    for k, a in arms.items():
        if a["image"].lower() == image.strip().lower() and a["text"].lower() == text.strip().lower():
            return k
    return None


SUBMITTED_RUN = "holdout-both-lora-depth6-r64-seed2026"  # source: digest/selected-kaggle-model.md:4


def _submitted_arm(j: dict | None) -> dict | None:
    if not j:
        return None
    return next((a for a in j["arms"].values() if SUBMITTED_RUN in (a.get("checkpoint") or "")), None)


def _oof() -> float | None:
    a = _submitted_arm(data.read_json(FACTORIAL))
    return a["gold_macro_auroc"] if a else None


# --------------------------------------------------------------------------- 1. protocol banner


def protocol_section(ctx, kpis_html: str = "") -> str:
    sid = "protocols"
    lm = data.read_json(LM58) or {}
    fac = data.read_json(FACTORIAL) or {}
    oof = _oof()
    top58 = max((a["mlp_macro_auroc"] for a in (lm.get("arms") or {}).values()), default=None)
    n56 = (fac.get("population") or {}).get("population_size")
    n58 = lm.get("gold_studies")
    lb = data.MD_ONLY["public_lb"]["value"]
    claim = f"Two protocols, two populations: {n56}-case gold-only OOF and {n58}-case LM-head scores never share an axis"
    body = (
        '<div class="caveat-strip">'
        f"<div><strong>{n56}-case gold-only OOF.</strong> Logistic heads fitted on gold labels, scored out-of-fold on the "
        f"{n56} held-out gold studies. The submitted model: {num(oof)}.</div>"
        f"<div><strong>{n58}-case LM-head.</strong> Heads trained on LM pseudo-labels, scored on all {n58} gold studies "
        f"(incl. the 2 former locked-test cases, now consumed). Highest: {num(top58)}.</div>"
        f"<div><strong>{num(oof)} → {num(top58)} is not an improvement.</strong> Different supervision and different "
        "populations; there is no paired before/after. No LM-head bundle has been submitted.</div>"
        f"<div><strong>First submission's public LB {num(lb, '.3f')} vs OOF {num(oof, '.3f')} is not a generalisation gap.</strong> "
        "Hidden test cohort vs gold OOF; different cohorts, not a paired estimate.</div>"
        "</div>" + kpis_html)
    caption = (f"Sources: <a href=\"{ctx.doc(PIPELINE_MD)}\">kaggle-submission-pipeline.md</a>, "
               f"<a href=\"{ctx.doc(LM58_MD)}\">lm-head-tower-size-58-v1.md</a>.")
    return section(sid, claim, body, caption=caption)


# --------------------------------------------------------------------------- 2. model card


def model_card_section(ctx) -> tuple[str, dict]:
    sid = "model-card"
    fac = data.read_json(FACTORIAL)
    probe = data.read_json(LORA_PROBE)
    rows = md_table(data.read_text(PIPELINE_MD), "| Candidate")
    arm = _submitted_arm(fac)
    if not arm:
        return section(sid, "Submitted model", notice(f"The submitted arm is not in <code>{FACTORIAL}</code>.")), {}
    lb = data.MD_ONLY["public_lb"]["value"]
    pop = fac["population"]["population_size"]
    # 56-case OOF ladder from the pipeline's verified-results table (parsed, not retyped)
    groups = {"floor": {"label": "Unadapted floor", "color": "var(--g-anchor)"},
              "holdout": {"label": "Gold-holdout encoders", "color": "var(--g-lora)"}}
    ladder = []
    for r in rows:
        name = r.get("Candidate", "").replace("*", "").strip()
        v = md_number(r.get(f"{pop}-study OOF macro AUROC", "") or list(r.values())[1])
        if v is None:
            continue
        ladder.append({"id": name, "label": (name.replace("Six-block LoRA holdout, default", "6-block LoRA (submitted)")
                                    .replace("Four-block holdout, three-seed probability ensemble", "4-block holdout, 3 seeds")),
                       "group": "floor" if "Unadapted" in name else "holdout", "value": v})
    lora_v = None
    if probe:
        lora_v = next((a.macro_auroc for a in data.probe_runs(probe)
                       if "holdout-both-lora-depth6-r64-seed2026" in a.run and a.feature == "uniform_mean_768"), None)
    card = table(["Component", "Setting"], [
        ["Image encoder", f"{_hf(arm['image'])} with LoRA (6 blocks, rank 64)"],
        ["Report encoder (training only)", _hf(arm["text"])],
        ["Checkpoint", f"{arm['checkpoint'].split('checkpoints/')[-1]}"],
        ["Checkpoint sha256", arm["checkpoint_sha256"][:12] + "…"],
        ["Selected epoch (validation R@1)", str(arm["best_epoch"])],
        ["Heads", f"12 logistic regressions on {pop} gold studies (gold-only), refit on all {pop} after OOF"],
        [f"{pop}-case OOF macro AUROC", num(arm["gold_macro_auroc"])],
        ["Kaggle submission", f"{SUBMISSION_ID}: public LB {num(lb, '.3f')}, rank {LB_RANK:,} of {LB_TEAMS:,}"],
    ])
    claim = (f"The submitted model pairs the {_hf(arm['image'])} + {_hf(arm['text'])} LoRA encoder with gold-only logistic heads: "
             f"{num(arm['gold_macro_auroc'])} OOF on {pop} cases")
    blob = {"groups": groups, "ladder": ladder}
    body = (
        '<div class="grid-2">\n'
        f"<div>{card}</div>\n"
        f"<div>{chart('ladder', f'{pop}-case gold OOF macro AUROC, local benchmark (public LB not on this axis)') if ladder else notice('Pipeline results table not found.')}</div>\n"
        "</div>\n"
        + section_script(sid, """
  if (d.ladder.length) Charts.barChart(sec.querySelector('[data-chart=ladder]'), d.ladder,
    {groups: d.groups, fmt: '.4f', axisFmt: '.2f', xLabel: 'Gold macro AUROC (56-case OOF)', store: store, labelWidth: 230});"""))
    caption = (f"Public LB {num(lb, '.3f')} (snapshot median {num(LB_MEDIAN, '.3f')}) scores a hidden test cohort and is shown only as a tile, "
               f"never on the OOF axis. Ladder values from the verified-results table in <a href=\"{ctx.doc(PIPELINE_MD)}\">the pipeline doc</a>.")
    notes = ("<ul>"
             f"<li>The same encoder's probe JSON <a href=\"{ctx.raw(LORA_PROBE)}\">{LORA_PROBE}</a> gives {num(lora_v)} for its uniform-mean "
             f"feature; the factorial arm gives {num(arm['gold_macro_auroc'])}.</li>"
             "<li>This is not the 4-block gold-holdout encoder from the probes page. The middle bar is a probability ensemble of those three "
             "4-block encoders; the probes page's 3-seed generalisation estimate is a pooled figure from a different aggregation, so do not subtract across them.</li>"
             "<li>OOF scores describe fold-trained heads, not the all-gold final head; gold has been examined repeatedly, so OOF is a "
             "development benchmark, not an external test. Study-level, not patient-level.</li>"
             f"<li>Model card detail: <a href=\"{ctx.doc('selected-kaggle-model.md')}\">selected-kaggle-model.md</a>.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 3. pipeline steps


def pipeline_section(ctx) -> str:
    sid = "bundle"
    steps = [
        ("Build bundle", "<code>scripts/build_kaggle_bundle.py</code> fits the heads, runs the OOF benchmark and exports "
                         "backbone weights, LoRA parameters, heads, preprocessing and <code>predict_kaggle.py</code>."),
        ("Validate offline", "Run the bundle with Hugging Face offline mode; the CSV must match the sample's study order, "
                             "12 label columns, finite [0,1] probabilities. No reports or labels are read."),
        ("Upload bundle only", "Only <code>bundle/</code> goes up as a private Kaggle dataset; the parent directory holds gold targets."),
        ("Internet-off notebook", "The Kaggle notebook attaches the bundle and competition data, runs on GPU with Internet disabled, "
                                  "and writes <code>submission.csv</code>."),
        ("Code submission", f"Submission {SUBMISSION_ID} scored the hidden test set; the recorded first baseline."),
    ]
    items = "".join(f'<li><strong>{esc(t)}</strong><span>{d}</span></li>' for t, d in steps)
    body = (
        "<style>#bundle ol.steps{list-style:none;counter-reset:s;display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));"
        "gap:10px;padding:0;margin:4px 0}#bundle ol.steps li{counter-increment:s;border:1px solid var(--border);border-radius:8px;"
        "padding:10px 12px;background:var(--plot-bg);font-size:13px;display:flex;flex-direction:column;gap:4px;min-width:0}"
        "#bundle ol.steps li::before{content:counter(s);font:700 11px var(--mono);color:var(--accent)}"
        "#bundle ol.steps li span{color:var(--muted);overflow-wrap:anywhere}</style>"
        f'<ol class="steps">{items}</ol>')
    claim = "The bundle runs fully offline: build, validate locally, upload the bundle only, run with Internet off, submit"
    caption = (f"Commands, schema checks and hosted-run validation in <a href=\"{ctx.doc(PIPELINE_MD)}\">kaggle-submission-pipeline.md</a>. "
               "Inference uses image features and the heads only; the report encoder and projection are not shipped.")
    notes = ("<ul><li>A missing plane or failed DICOM decode raises an explicit error; there is no silent row drop or constant fallback.</li>"
             "<li>Local three-study timings are smoke benchmarks, not estimates of hidden-test runtime.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes)


# --------------------------------------------------------------------------- 4. LM head: MLP vs linear (58)


def lm_heads_section(ctx) -> tuple[str, dict, dict]:
    sid = "lm-heads"
    lm = data.read_json(LM58)
    md = data.read_text(LM58_MD)
    if not lm:
        return section(sid, "LM-supervised heads", notice(f"<code>{LM58}</code> is missing.")), {}, {}
    arms = lm["arms"]
    groups = {"mlp": {"label": "MLP head (3-seed ensemble)", "color": "var(--g-lora)"},
              "linear": {"label": "Linear head (3-seed ensemble)", "color": "var(--g-imgft)"}}
    bars = []
    for k, a in arms.items():
        bars.append({"id": k, "label": f"{_arm_short(a)} · MLP", "group": "mlp", "value": a["mlp_macro_auroc"]})
        bars.append({"id": k, "label": f"{_arm_short(a)} · linear", "group": "linear", "value": a["linear_macro_auroc"]})
    diffs = []
    for r in md_table(md, "| Image / text | Linear gold AUROC"):
        pair = r.get("Image / text", "")
        if "/" not in pair:
            continue
        img, txt = [x.strip() for x in pair.split("/", 1)]
        key = _arm_key(arms, img, txt)
        iv = md_interval(r.get("Paired 95% interval", ""))
        est = md_number(r.get("MLP minus linear AUROC", ""))
        if key and iv and est is not None:
            diffs.append({"id": key, "label": _arm_short(arms[key]), "group": "mlp", "est": est, "lo": iv[0], "hi": iv[1]})
    n = lm.get("gold_studies")
    wins = sum(1 for a in arms.values() if a["mlp_macro_auroc"] > a["linear_macro_auroc"])
    excl = sum(1 for d_ in diffs if _excl(d_["lo"], d_["hi"]))
    claim = (f"Under the {n}-case LM head, the MLP beats the linear head in {wins} of {len(arms)} pairings; "
             f"{excl} of {len(diffs)} MLP − linear intervals exclude zero " + caveat_badge(f"{n}-case protocol"))
    blob = {"groups": groups, "bars": bars, "diffs": diffs}
    body = (
        '<div data-role="filter"></div>\n<div class="grid-2">\n'
        f"<div>{chart('bars', f'LM-head gold macro AUROC ({n} cases)')}</div>\n"
        f"<div>{chart('diffs', 'MLP − linear, paired 95% interval') if diffs else notice('MLP − linear table not found in the md.')}</div>\n"
        "</div>\n"
        + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  Charts.barChart(sec.querySelector('[data-chart=bars]'), d.bars, {groups: d.groups, fmt: '.4f', axisFmt: '.2f', xLabel: 'Gold macro AUROC (58 cases)', store: store, labelWidth: 230});
  if (d.diffs.length) Charts.ciDotPlot(sec.querySelector('[data-chart=diffs]'), d.diffs,
    {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.2f', xLabel: '\\u0394 macro AUROC (MLP \\u2212 linear)', store: store});"""))
    caption = (f"Frozen encoders, heads trained on LM verdicts ({lm.get('head_train_studies'):,} train / {lm.get('head_validation_studies')} val studies), "
               f"epochs selected on LM-label validation BCE; scored on all {n} gold studies. S/B = small/base; bars start at 0, so read gaps on the paired chart.")
    notes = ("<ul>"
             "<li>MLP − linear intervals are parsed from the md table (the JSON does not carry them); 5,000 paired study resamples.</li>"
             "<li>MLPs select very early epochs and overfit after them; the linear heads rebound less but score lower on gold "
             "(interactive head-loss curves in the next section).</li>"
             "<li>Four selected contrastive checkpoints, one seed each: intervals cover study sampling, not tower-training variation.</li>"
             f"<li>Sources: <a href=\"{ctx.doc(LM58_MD)}\">lm-head-tower-size-58-v1.md</a>, <a href=\"{ctx.raw(LM58)}\">{LM58}</a>, "
             f"<a href=\"{ctx.raw('lm-head-tower-size-58-v1.html')}\">original report</a>.</li></ul>")
    tbl = table(["Pairing", "LM-linear AUROC", "LM-MLP AUROC", "MLP − linear", "95% CI", "MLP selected epochs"],
                [[_arm_label(a), num(a["linear_macro_auroc"]), num(a["mlp_macro_auroc"]),
                  next((num(d_["est"], ".4f", True) for d_ in diffs if d_["id"] == k), "–"),
                  next((_ci(d_["lo"], d_["hi"]) for d_ in diffs if d_["id"] == k), "–"),
                  ", ".join(str(e) for e in a.get("mlp_epochs") or [])] for k, a in arms.items()],
                num_cols=(1, 2, 3, 4), row_attrs=[{"data-series-id": k, "data-group": "mlp"} for k in arms])
    top_k, top = max(arms.items(), key=lambda kv: kv[1]["mlp_macro_auroc"])
    return (section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Per-pairing table"),
            {f"data-{sid}": blob}, {"top": top["mlp_macro_auroc"], "top_label": _arm_label(top), "n": n})


# --------------------------------------------------------------------------- 4b. LM-head loss curves (58)

LOSS_SNAPSHOT = "lm-head-histories-58.json"
PAIR_HUES = ["var(--g-teal)", "var(--g-sky)", "var(--g-violet)", "var(--g-amber)"]


def _pair_label(key: str) -> str:
    # "dinov2-small-e5-base" -> "DINOv2-S + E5-B"
    img, _, txt = key.partition("-e5-")
    return f"{_compact(img).replace('dinov2', 'DINOv2')} + {_compact('e5-' + txt).replace('e5', 'E5')}"


def lm_loss_section(ctx) -> tuple[str, dict]:
    sid = "lm-loss"
    snap = data.site_data(LOSS_SNAPSHOT)
    lm = data.read_json(LM58) or {}
    n = lm.get("gold_studies")
    svg_links = (f"Static figures: <a href=\"{ctx.raw('lm-head-tower-size-58-curves.svg')}\">MLP</a>, "
                 f"<a href=\"{ctx.raw('lm-head-tower-size-58-linear-curves.svg')}\">linear</a>")
    if not snap or not snap.get("pairings"):
        return section(sid, "LM-head train and validation loss", notice(
            f"The loss-history snapshot <code>digest/site/data/{LOSS_SNAPSHOT}</code> is missing "
            f"(run <code>build_digest_site.py export</code>). {svg_links}.")), {}
    pairs = sorted(snap["pairings"])
    groups = {p: {"label": _pair_label(p), "color": PAIR_HUES[i % len(PAIR_HUES)]} for i, p in enumerate(pairs)}
    heads, stats = {}, {}
    for head in ("mlp", "linear"):
        series, best, rise = [], [], []
        for p in pairs:
            seeds = sorted((snap["pairings"][p].get(head) or {}).items())
            for i, (seed, h) in enumerate(seeds):
                for kind, key, dash in (("val", "validation_masked_bce", None), ("train", "train_masked_bce", "5 3")):
                    # one id per run: validation and train lines of a run highlight together
                    s_ = {"id": f"{p}-{seed}", "label": f"{_pair_label(p)} · s{seed} · {kind}", "kind": kind,
                          "group": p, "shade": i, "shadeOf": max(3, len(seeds)),
                          "points": [[e, v] for e, v in zip(h["epoch"], h[key])]}
                    if dash:
                        s_["dash"] = dash
                    series.append(s_)
                be = h.get("best_epoch")
                if be is not None:
                    best.append(be)
                    vals = dict(zip(h["epoch"], h["validation_masked_bce"]))
                    if vals.get(be) is not None and h["validation_masked_bce"][-1] is not None:
                        rise.append(h["validation_masked_bce"][-1] - vals[be])
        best_sorted = sorted(best)
        med = best_sorted[len(best_sorted) // 2] if best_sorted else None
        heads[head] = {"series": series, "vlines": [{"x": med, "label": f"median selected epoch {med}"}] if med else []}
        stats[head] = {"lo": min(best) if best else None, "hi": max(best) if best else None,
                       "rise": sum(rise) / len(rise) if rise else None, "runs": len(best)}
    m, li = stats["mlp"], stats["linear"]
    claim = (f"MLP heads hit their LM-label validation minimum by epoch {m['lo']}–{m['hi']}, then validation BCE climbs "
             f"{num(m['rise'], '.3f', True)} while train loss keeps falling; linear heads select epoch {li['lo']}–{li['hi']} "
             f"and rise only {num(li['rise'], '.3f', True)} " + caveat_badge(f"{n}-case protocol"))
    blob = {"groups": groups, "heads": heads}
    body = (
        '<div data-role="head"></div><div data-role="filter"></div>\n'
        f"{chart('loss', 'Masked BCE on LM pseudo-labels: solid = validation, dashed = train')}\n"
        '<div data-role="legend"></div>\n'
        + section_script(sid, """
  var head = 'mlp';
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var seg = sec.querySelector('[data-role=head]');
  seg.className = 'filter-bar'; seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', 'Head type');
  seg.innerHTML = '<span class="muted" style="font-size:12px">Head:</span>' +
    '<button type="button" class="pill-btn active" data-head="mlp" aria-pressed="true">MLP (256 hidden)</button>' +
    '<button type="button" class="pill-btn" data-head="linear" aria-pressed="false">Linear</button>';
  function draw() {
    var old = sec.querySelector('[data-chart=loss]'), el = document.createElement('div');
    el.className = 'chart'; el.setAttribute('data-chart', 'loss'); el.setAttribute('aria-label', old.getAttribute('aria-label') || '');
    old.parentNode.replaceChild(el, old);
    var h = d.heads[head], series = JSON.parse(JSON.stringify(h.series));
    store.set({solo: null, hover: null});
    Charts.lineChart(el, series, {groups: d.groups, yLabel: 'Masked BCE', yFmt: '.3f', xLabel: 'Epoch', best: 'min',
      vlines: h.vlines, store: store, ariaLabel: 'LM-head train and validation loss by epoch'});
    var val = series.filter(function (s) { return s.kind === 'val'; }).map(function (s) {
      return {id: s.id, label: s.label.replace(' \u00b7 val', ''), group: s.group, shade: s.shade, shadeOf: s.shadeOf, points: s.points}; });
    Charts.legend(sec.querySelector('[data-role=legend]'), val, store, {groups: d.groups, best: 'min', fmt: '.3f', valuePrefix: 'min val '});
  }
  seg.addEventListener('click', function (e) {
    var b = e.target.closest('[data-head]'); if (!b) return;
    head = b.getAttribute('data-head');
    Array.prototype.forEach.call(seg.querySelectorAll('[data-head]'), function (x) {
      var on = x === b; x.classList.toggle('active', on); x.setAttribute('aria-pressed', on ? 'true' : 'false'); });
    draw();
  });
  draw();"""))
    caption = (f"{len(pairs)} frozen-encoder pairings × 3 head seeds; loss is on LM pseudo-labels "
               f"({lm.get('head_train_studies'):,} train / {lm.get('head_validation_studies')} val studies), not gold. "
               "Click lines or legend entries to highlight runs (each with its validation and train curves); click several "
               "to compare them, click empty plot area to clear.")
    notes = ("<ul>"
             "<li>Each head is saved at its minimum validation masked BCE (patience 12); the curves continue past that "
             "epoch only because early stopping waits. The gold AUROCs above score the saved, early-stopped heads.</li>"
             f"<li>All {n} gold studies are excluded from head training and validation; gold is never used for epoch selection.</li>"
             "<li>Validation-BCE rise = last logged epoch minus the selected epoch, averaged over the "
             f"{m['runs']} MLP / {li['runs']} linear runs. The four contrastive towers are one seed each.</li>"
             f"<li>{svg_links}. Snapshot: <code>digest/site/data/{LOSS_SNAPSHOT}</code> "
             "(from <code>checkpoints/kaggle/lm-mlp-tower-size-58-*/{mlp,linear}-seed*-history.json</code>).</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 5. tower-size contrasts (58)


def lm_size_section(ctx) -> tuple[str, dict]:
    sid = "lm-size"
    lm = data.read_json(LM58)
    if not lm:
        return section(sid, "Tower size under the LM head", notice(f"<code>{LM58}</code> is missing.")), {}
    arms = lm["arms"]
    series = []
    for size in ("small", "base"):
        pts = sorted([1 if _size(a["text"]) == "base" else 0, a["mlp_macro_auroc"]]
                     for a in arms.values() if _size(a["image"]) == size)
        img = next(a["image"] for a in arms.values() if _size(a["image"]) == size)
        series.append({"id": f"img-{size}", "label": img, "group": size, "points": pts})
    rows = []
    for name, p in (lm.get("paired_mlp_auroc_differences") or {}).items():
        contrast, _, fixed = name.partition(",")
        fixed = fixed.strip()
        group = _size(fixed) if fixed.lower().startswith("dinov2") else "cross"
        lo, hi = p["ci95"]
        rows.append({"id": name, "label": f"{_compact(contrast.strip())} @ {_compact(fixed)}", "group": group,
                     "est": p["difference"], "lo": lo, "hi": hi})
    spans = sum(1 for r in rows if not _excl(r["lo"], r["hi"]))
    n = lm.get("gold_studies")
    claim = (f"{spans} of {len(rows)} paired tower-size contrasts under the LM head include zero: "
             "no size pairing is established on the 58 cases " + caveat_badge(f"{n}-case protocol"))
    groups = dict(IMG_GROUPS)
    groups["cross"] = {"label": "Image-size contrasts", "color": "var(--g-anchor)"}
    blob = {"groups": groups, "series": series, "rows": rows}
    body = (
        '<div data-role="filter"></div>\n<div class="grid-2">\n'
        f"<div>{chart('inter', f'LM-MLP gold AUROC ({n} cases) by text tower')}</div>\n"
        f"<div>{chart('forest', 'Paired contrasts (first named − second), 95% CI')}</div>\n</div>\n"
        '<div data-role="legend"></div>\n'
        + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var xf = function (v) { return v === 0 ? 'E5-small' : (v === 1 ? 'E5-base' : ''); };
  Charts.lineChart(sec.querySelector('[data-chart=inter]'), d.series,
    {groups: d.groups, yLabel: 'LM-MLP gold macro AUROC', yFmt: '.3f', xFmt: xf, xStep: 1, xLabel: 'Text tower', best: 'max', store: store, height: 300});
  Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {groups: d.groups, best: 'max', fmt: '.4f', valuePrefix: 'best '});
  Charts.ciDotPlot(sec.querySelector('[data-chart=forest]'), d.rows,
    {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.2f', sort: false, labelWidth: 250,
     xLabel: '\\u0394 LM-MLP macro AUROC', store: store});"""))
    p0 = next(iter((lm.get("paired_mlp_auroc_differences") or {}).values()), {})
    caption = (f"Same four checkpoints as the encoder factorial, re-scored with a matched LM-supervised MLP head on {n} gold studies; "
               f"{p0.get('resamples', '?'):,} paired study resamples. Sign follows each contrast's name (first − second).")
    notes = ("<ul>"
             "<li>This is not the 56-case factorial on the encoders page: different heads, supervision and population. "
             "Rankings may agree or differ; the two are never plotted on one axis.</li>"
             "<li>Mismatched-over-matched pairs hold for each head seed, but the contrastive towers are still one seed each.</li>"
             f"<li>Source: <a href=\"{ctx.raw(LM58)}\">{LM58}</a> <code>paired_mlp_auroc_differences</code>.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 6. per-label heatmap (58)


def per_label_section(ctx) -> tuple[str, dict]:
    sid = "per-label"
    md = data.read_text(LM58_MD)
    fac = data.read_json(FACTORIAL) or {}
    rows = md_table(md, "### AUROC by gold label")
    if not rows:
        return section(sid, "Per-label AUROC", notice("The per-label table is missing from the LM-head md.")), {}
    keys = list(rows[0].keys())
    pos_k = next((k for k in keys if "positives" in k.lower()), None)
    arm_cols = [k for k in keys if k not in ("Label", pos_k)]
    # guard against the fixed label-order bug: 58-case positives must be the 56-case positives plus 0-2
    p56 = {lab: v["positives"] for lab, v in
           (next(iter(fac.get("arms", {}).values()), {}) or {}).get("gold_per_label", {}).items()}
    bad = []
    for r in rows:
        p = md_number(r.get(pos_k, "")) if pos_k else None
        base = p56.get(r["Label"])
        if p is None or base is None or not (0 <= p - base <= 2):
            bad.append(r["Label"])
    if bad:
        return section(sid, "Per-label AUROC", notice(
            "Gold positive counts in the LM-head md do not align with the 56-case counts "
            f"({esc(', '.join(bad))}); the heatmap is withheld until the builder output is checked.")), {}
    matrix = [[md_number(r[c]) for c in arm_cols] for r in rows]
    labels = [f"{r['Label'].replace('Meniscus', 'men.').replace('Lateral ', 'Lat. ').replace('Medial ', 'Med. ')} ({int(md_number(r[pos_k]))}+)" for r in rows]
    leaders, lowest = {}, {}
    for r, vals in zip(rows, matrix):
        j = max(range(len(vals)), key=lambda i: vals[i])
        leaders[arm_cols[j]] = leaders.get(arm_cols[j], 0) + 1
    for ci, c in enumerate(arm_cols):
        lab = min(zip(rows, matrix), key=lambda rv: rv[1][ci])[0]["Label"]
        lowest[lab] = lowest.get(lab, 0) + 1
    top_col = max(arm_cols, key=lambda c: leaders.get(c, 0))
    hard, hard_n = max(lowest.items(), key=lambda kv: kv[1])
    claim = (f"No pairing leads on every label: {esc(top_col.lower())} leads {leaders.get(top_col, 0)} of {len(rows)} labels, "
             f"the other three share the rest; {esc(hard)} is the lowest label in {hard_n} of {len(arm_cols)} pairings")
    short_cols = [c.replace("Small image", "S img").replace("Base image", "B img").replace("small text", "S txt")
                  .replace("base text", "B txt") for c in arm_cols]
    blob = {"matrix": matrix, "rows": labels, "cols": short_cols, "rowIds": [r["Label"] for r in rows]}
    body = (chart("heat", "LM-MLP AUROC by gold label and pairing (58 cases); outlined cell = row leader")
            + section_script(sid, """
  Charts.heatmap(sec.querySelector('[data-chart=heat]'), d.matrix,
    {rows: d.rows, cols: d.cols, rowIds: d.rowIds, fmt: '.3f', domain: [0.5, 1], store: store, ariaLabel: 'Per-label AUROC heatmap'});"""))
    caption = ("Row labels show gold positives out of 58 (corrected counts; checked against the 56-case counts at build time). "
               "S/B = DINOv2-small/base image, E5-small/base text. Per-label values are much noisier than the macro.")
    notes = ("<ul>"
             "<li>An earlier builder mislabelled the positives column (alphabetical names zipped with canonical-order targets); "
             "it was fixed, and this page refuses to draw the heatmap if the counts drift again.</li>"
             "<li>MCL has 9 positives in 58 (8 in the 56-case set); single-label AUROCs rest on very few cases.</li>"
             f"<li>Source: <a href=\"{ctx.doc(LM58_MD)}\">lm-head-tower-size-58-v1.md</a> (AUROC by gold label).</li></ul>")
    tbl = table(["Label", "Positives / 58"] + arm_cols,
                [[r["Label"], r.get(pos_k, "")] + [num(v, ".4f") for v in m] for r, m in zip(rows, matrix)],
                num_cols=range(1, 2 + len(arm_cols)), row_attrs=[{"data-series-id": r["Label"]} for r in rows])
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl, table_summary="Per-label table"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 7. lm-mlp-v1 (56)


def lm_mlp_v1_section(ctx) -> tuple[str, dict, dict]:
    sid = "lm-mlp-v1"
    j = data.read_json(LMMLP)
    if not j:
        return section(sid, "LM-MLP on 56 cases", notice(f"<code>{LMMLP}</code> is missing.")), {}, {}
    res, pc = j["results"], j["paired_comparisons"]
    n = j["population"]["coverage"]["gold"]["studies"]
    groups = {"lm": {"label": "LM-supervised heads", "color": "var(--g-lora)"},
              "gold": {"label": "Gold-only logistic OOF (submitted heads)", "color": "var(--g-bothft)"}}
    bars = [
        {"id": "mlp", "label": "LM-MLP, 3-seed ensemble", "group": "lm", "value": res["mlp"]["macro_auroc"]},
        {"id": "linear", "label": "LM-linear, 3-seed ensemble", "group": "lm", "value": res["linear"]["macro_auroc"]},
        {"id": "gold", "label": "Gold-only logistic OOF", "group": "gold", "value": res["gold_only_logistic_oof"]["macro_auroc"]},
    ]
    names = {"mlp_minus_gold_only_logistic_oof": ("gold", "MLP − gold-only OOF"),
             "mlp_minus_lm_linear": ("linear", "MLP − LM-linear")}
    diffs = [{"id": names.get(k, (k, k))[0], "label": names.get(k, (k, k))[1], "group": "lm",
              "est": v["difference"], "lo": v["ci95"][0], "hi": v["ci95"][1], "B": v.get("resamples")}
             for k, v in pc.items()]
    g = pc.get("mlp_minus_gold_only_logistic_oof", {})
    claim = (f"On the same {n} cases, the LM-supervised MLP ({num(res['mlp']['macro_auroc'])}) ties the submitted gold-only heads "
             f"({num(res['gold_only_logistic_oof']['macro_auroc'])}): Δ {num(g.get('difference'), '.4f', True)}, "
             f"CI {_ci(*(g.get('ci95') or [None, None]))} " + caveat_badge(f"{n}-case protocol"))
    blob = {"groups": groups, "bars": bars, "diffs": diffs}
    body = (
        '<div class="grid-2">\n'
        f"<div>{chart('bars', f'Gold macro AUROC ({n} cases, same encoder)')}</div>\n"
        f"<div>{chart('diffs', 'Paired differences, 95% CI')}</div>\n</div>\n"
        + section_script(sid, """
  Charts.barChart(sec.querySelector('[data-chart=bars]'), d.bars, {groups: d.groups, fmt: '.4f', axisFmt: '.2f', xLabel: 'Gold macro AUROC (56 cases)', store: store});
  Charts.ciDotPlot(sec.querySelector('[data-chart=diffs]'), d.diffs, {groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.2f', sort: false,
    xLabel: '\\u0394 macro AUROC', store: store, tooltipExtra: function (r) { return '<div>' + r.B + ' paired resamples</div>'; }});"""))
    caption = (f"The one legitimate same-population comparison of head strategies: {n} held-out gold studies, submitted encoder, "
               f"{diffs[0]['B'] if diffs else '?'} paired study resamples; bars start at 0. The 58-case numbers above are a different population.")
    notes = ("<ul>"
             "<li>Different supervision, not an architecture-only comparison: gold-only OOF trains on other gold cases, LM heads never see gold.</li>"
             "<li>“Does not justify calling it a leaderboard improvement” (lm-mlp-v1.md); no LM-head bundle has been submitted.</li>"
             f"<li><code>{LMMLP}</code> is orphaned (unreferenced, not byte-identical to the checkpoint benchmark); its values match "
             f"<a href=\"{ctx.doc('lm-mlp-v1.md')}\">lm-mlp-v1.md</a>. It uses 2,000 resamples vs 5,000 for the 58-case run.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}, {"diff": g.get("difference")}


# --------------------------------------------------------------------------- page


# --------------------------------------------------------------------------- 0. public leaderboard history

LB_HISTORY = "leaderboard-history-v1.json"
LB_GROUPS = {"frozen": {"label": "Frozen encoder + heads", "color": "var(--g-anchor)"},
             "finetune": {"label": "Label fine-tuned image encoder", "color": "var(--g-sky)"}}


def _best_public() -> dict | None:
    j = data.read_json(LB_HISTORY)
    return next((r for r in j["submissions"] if r["bundle"] == j["best"]), None) if j else None


def leaderboard_section(ctx) -> tuple[str, dict]:
    sid = "leaderboard"
    j = data.read_json(LB_HISTORY)
    if not j:
        return "", {}
    subs = j["submissions"]
    best = _best_public()
    blob = {"groups": LB_GROUPS,
            "rows": [{"id": r["bundle"], "label": f"{r['recorded'] or 'undated'} · {r['title']}", "group": r["family"],
                      "value": r["public"]} for r in subs]}
    body = (chart("public", "Kaggle public macro ROC-AUC by submission, oldest first")
            + section_script(sid, """
  Charts.barChart(sec.querySelector('[data-chart=public]'), d.rows,
    {groups: d.groups, fmt: '.3f', axisFmt: '.2f', sort: false, labelWidth: 330});"""))
    rows = [[r["recorded"] or "—", esc(r["title"]), str(r["submission"] or "—"), num(r["public"], ".3f"),
             num(r["local_lm_test"]) if r["local_lm_test"] is not None else "—",
             num(r["local_gold58"]) if r["local_gold58"] is not None else "—", esc(r["outcome"] or "")] for r in subs]
    tbl = table(["Recorded", "Submission", "Kaggle ref", "Public", "Local LM-test", "Local gold-58", "Outcome"],
                rows, raw_html=True, num_cols=(3, 4, 5))
    claim = (f"Public leaderboard: {num(subs[0]['public'], '.3f')} → {num(best['public'], '.3f')} over "
             f"{len(subs)} submissions; best single model is {esc(best['title'])}")
    caption = ("Every recorded public-leaderboard submission, read from each bundle's leaderboard-submission.json. "
               "Public scores rest on the hidden test's public split; one submission per bundle, single seeds.")
    notes = (f'<p>Source: <a href="{ctx.raw(LB_HISTORY)}">{LB_HISTORY}</a>, built by '
             '<code>scripts/build_leaderboard_digest.py</code>. The sections below describe the first (frozen-encoder) '
             'pipeline in detail; the fine-tuned bundles reuse its Kaggle packaging.</p>')
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary="All submissions"), {f"data-{sid}": blob}


def build(ctx) -> Page:
    sections, blobs = [""], {}
    s, b = leaderboard_section(ctx); sections.append(s); blobs.update(b)
    s, b = model_card_section(ctx); sections.append(s); blobs.update(b)
    sections.append(pipeline_section(ctx))
    s, b, k_lm = lm_heads_section(ctx); sections.append(s); blobs.update(b)
    s, b = lm_loss_section(ctx); sections.append(s); blobs.update(b)
    s, b = lm_size_section(ctx); sections.append(s); blobs.update(b)
    s, b = per_label_section(ctx); sections.append(s); blobs.update(b)
    s, b, k_v1 = lm_mlp_v1_section(ctx); sections.append(s); blobs.update(b)
    lb = data.MD_ONLY["public_lb"]["value"]
    oof = _oof()
    kpis = [
        {"label": "Submitted model, 56-case OOF", "value": num(oof), "sub": "gold-only logistic heads"},
        {"label": "First submission, public LB", "value": num(lb, ".3f"),
         "sub": f"submission {SUBMISSION_ID} · rank {LB_RANK:,}/{LB_TEAMS:,} · different cohort from OOF"},
    ]
    best = _best_public()
    if best:
        kpis.insert(0, {"label": "Best public LB", "value": num(best["public"], ".3f"),
                        "sub": f"{esc(best['title'])} · submission {best['submission']}", "tone": "good"})
    if k_lm:
        kpis.append({"label": f"Best LM-MLP, {k_lm['n']}-case", "value": num(k_lm["top"]),
                     "sub": f"{esc(k_lm['top_label'])} · not submitted · not comparable to OOF"})
    if k_v1:
        kpis.append({"label": "LM-MLP − gold-only (56)", "value": num(k_v1["diff"], ".4f", True), "sub": "paired; CI spans zero"})
    # KPI tiles sit under the protocol banner, so the 56- and 58-case numbers are never read before the warning
    sections[0] = protocol_section(ctx, '<div style="margin-top:14px">' + kpi_row(kpis) + "</div>")
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=sections, data=blobs, kpis=None)


def summary() -> dict:
    info = BY_SLUG[SLUG]
    lb = data.MD_ONLY["public_lb"]["value"]
    best = _best_public()
    if best:
        return {"stage": info.number, "headline": f"Best public submission: {best['title']}",
                "value": num(best["public"], ".3f"), "value_label": "public leaderboard macro ROC-AUC",
                "caveat": f"Single seed; first submission scored {num(lb, '.3f')}", "href": "#leaderboard"}
    return {"stage": info.number, "headline": f"Kaggle submission {SUBMISSION_ID}",
            "value": num(lb, ".3f"), "value_label": "public leaderboard macro ROC-AUC",
            "caveat": f"Different cohort from the {num(_oof(), '.4f')} 56-case OOF: not a paired generalisation gap",
            "href": "#model-card"}
