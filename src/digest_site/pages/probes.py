"""Downstream probes & generalisation (stage 6).

Sections:
1. gold-holdout  - "Does it generalise?": floor / unexposed / exposed levels per seed from
                   downstream-probe-unexposed-v2.json + the paired decomposition from
                   gold-holdout-comparisons-v2.json; pooled gap from data.POOLED.
2. explorer      - one probe JSON at a time: Δ vs that file's own floor (ciDotPlot), per-label Δ
                   heatmap (arms x 12 labels, positives in the column header), and the paired
                   contrasts from that file's paired JSONs. Arms tagged exposed / unexposed by
                   checkpoint root; frozen-tower arms on the uniform feature are identity checks.
3. gold-set      - which 56 of 58 gold studies the probe reads, and the series-count leak check.

Every difference is read inside one file (one extraction pass). Floors are labelled per file.
"""

from __future__ import annotations

import re

from .. import catalog, data
from ..shell import Page, chart, esc, notice, num, section, status_badge, table
from ..sitemap import BY_SLUG

SLUG = "probes"
SUBTITLE = ("Gold-label probes on frozen features: how much each change moves label readout, and how much of the "
            "gain survives holding the gold studies out of training.")

HOLDOUT_PROBE = "downstream-probe-unexposed-v2.json"
HOLDOUT_PAIRED = "gold-holdout-comparisons-v2.json"

# md-only numbers (no JSON carries them)
# source: digest/acquisition-metadata-trust.md:104
LEAK_DETECTABLE = (0.149, 0.197)  # |AUROC - 0.5| the 58-study permutation test can detect (best / worst label)

# Checkpoint roots trained WITHOUT the gold studies (cohort contrastive-cohort-gold-holdout-v1).
# Exposure is a property of each arm's checkpoint, never of the file or its cohort_manifest field.
UNEXPOSED_ROOTS = {
    "gold-holdout-v1", "both-lora-holdout-v1", "label-similarity-v1", "label-similarity-v2",
    "tower-size-factorial-v1", "size-matched-holdout-v1",
}

GROUPS = {
    "exposed": {"label": "Exposed arm (development statistic)", "color": "var(--g-bothft)"},
    "unexposed": {"label": "Unexposed arm (generalisation estimate)", "color": "var(--g-holdout)"},
    "frozen": {"label": "Frozen image tower, learned head only", "color": "var(--g-anchor)"},
}

LABEL_SHORT = {
    "ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "Med men", "Lateral Meniscus": "Lat men", "Medial OA": "Med OA",
    "Lateral OA": "Lat OA", "PF OA": "PF OA", "Effusion": "Effusion", "Synovitis": "Synovitis", "Baker's": "Baker's",
    "Contusion": "Contusion", "Fracture": "Fracture",
}

FAMILIES = {
    "uniform": "Uniform mean (primary)",
    "plane": "Plane attention (secondary)",
    "projection": "Projection 256 (confounded)",
}

# One entry per probe file. `paired` is in priority order (newest first) for de-duplication.
EXPERIMENTS = [
    dict(file="downstream-probe-v1.json", title="First probe: one image block vs unadapted DINOv2", topic="pipeline"),
    dict(file="downstream-probe-both-block-v1.json", title="Image + text block vs image block alone", topic="text",
         paired=[f"downstream-both-vs-image-block-seed{s}.json" for s in (2026, 2027, 2028)]
         + [f"downstream-both-vs-image-block-planes-seed{s}.json" for s in (2026, 2027, 2028)]),
    dict(file="downstream-probe-extended-v3.json",
         title="Image-block variants: language, depth, 40 epochs, shuffled pairs", topic="adaptation",
         paired=["downstream-extended-comparisons-v3.json", "downstream-extended-comparisons-v2.json",
                 "downstream-extended-comparisons-v1.json", "downstream-original-vs-translated-v1.json"]),
    dict(file="downstream-probe-plane-pooling-v1.json", title="Uniform plane pooling vs learned attention",
         topic="adaptation", paired=["plane-pooling-comparisons-v1.json"]),
    dict(file=HOLDOUT_PROBE, title="Gold holdout: exposed, unexposed and data-size control", topic="probes",
         paired=[HOLDOUT_PAIRED], pooled="gold_holdout_exposure_gap"),
    dict(file="downstream-probe-lora-holdout-v1.json", title="LoRA and full FT retrained without gold (unexposed)",
         topic="adaptation"),
    dict(file="downstream-probe-crop150-v1.json", title="150 mm vs 130 mm crop (not adopted)", topic="overfitting",
         paired=["crop150-vs-crop130-comparisons-v1.json"], pooled="crop150_vs_crop130"),
    dict(file="downstream-probe-both-block-depth-v1.json", title="Both-tower depth and LoRA depth / rank sweep",
         topic="adaptation"),
    dict(file="downstream-probe-declip-v1.json", title="DeCLIP vs the LoRA and full-FT bests", topic="encoders"),
    dict(file="downstream-probe-label-similarity-v2.json", title="Label-similarity target, four arms (null)",
         topic="encoders", paired=["label-similarity-paired-v2.json"]),
    dict(file="tower-size-text-small-probe.json", title="Tower size: DINOv2-small / E5-small", topic="encoders"),
    dict(file="tower-size-image-base-probe.json", title="Tower size: DINOv2-base / E5-base", topic="encoders"),
    dict(file="size-matched-holdout-probe.json", title="Size-matched: DINOv2-base / E5-small", topic="encoders"),
    # superseded supersets (hidden behind "show superseded")
    dict(file="downstream-probe-cohort-compare-v1.json", title="Original vs translated reports, seed 2026",
         topic="text", paired=["downstream-original-vs-translated-v1.json"],
         superseded_by="downstream-probe-extended-v3.json"),
    dict(file="downstream-probe-extended-v1.json", title="Image-block variants, first pass", topic="adaptation",
         paired=["downstream-extended-comparisons-v1.json"], superseded_by="downstream-probe-extended-v3.json"),
    dict(file="downstream-probe-extended-v2.json", title="Image-block variants, second pass", topic="adaptation",
         paired=["downstream-extended-comparisons-v2.json", "downstream-extended-comparisons-v1.json"],
         superseded_by="downstream-probe-extended-v3.json"),
    dict(file="downstream-probe-unexposed-v1.json", title="Gold holdout before the data-size control",
         topic="probes", paired=["gold-holdout-comparisons-v1.json"], superseded_by=HOLDOUT_PROBE),
    dict(file="downstream-probe-label-similarity-v1.json", title="Label similarity, lambda 0 vs 0.3",
         topic="encoders", paired=["label-similarity-paired-v1.json"],
         superseded_by="downstream-probe-label-similarity-v2.json"),
]


# --------------------------------------------------------------------------- labels & tags


def _family(feature: str) -> str | None:
    if feature.startswith("uniform_mean"):
        return "uniform"
    if feature.startswith("plane_attention"):
        return "plane"
    if feature.startswith("projection"):
        return "projection"
    return None


def _seed(run: str) -> str | None:
    m = re.search(r"-seed(\d{4})$", run)
    return m.group(1) if m else None


def _base_labels(slot: str, base: str) -> tuple[str, str]:
    """(long, short) label for a run name without its seed suffix."""
    m = re.match(r"^(holdout-)?both-lora-depth(\d+)-r(\d+)(-bs(\d+))?(-reg)?$", base)
    if m:
        extra = (f" BS{m.group(5)}" if m.group(5) else "") + (" reg" if m.group(6) else "")
        pre = ("Unexposed " if m.group(1) else "")
        short = ("unexp " if m.group(1) else "") + f"LoRA{m.group(2)} r{m.group(3)}" + (" reg" if m.group(6) else
                                                                                       (" bs" + m.group(5) if m.group(5) else ""))
        return f"{pre}LoRA {m.group(2)} blk r{m.group(3)}{extra}", short
    m = re.match(r"^both-unfreeze(\d+)(-bs(\d+))?$", base)
    if m:
        bs = f" BS{m.group(3)}" if m.group(3) else ""
        return f"Both towers FT {m.group(1)} blk{bs}", f"both FT{m.group(1)}{bs.lower()}"
    m = re.match(r"^original-unfreeze(\d+)$", base)
    if m:
        n = m.group(1)
        if slot == "crop130":
            return f"crop130 FT {n} blk", f"crop130 FT{n}"
        if slot == "exposed":
            return f"Exposed FT {n} blk", f"exposed FT{n}"
        return f"Image FT {n} blk", f"img FT{n}"
    table_ = {
        "image-block": ("1 image block", "1 img blk"),
        "both-block": ("Image + text block", "img+txt blk"),
        "original-multilingual": ("1 blk, original reports", "1 blk orig"),
        "translated-english": ("1 blk, translated reports", "1 blk transl"),
        "original-40ep": ("1 blk, 40 epochs", "1 blk 40ep"),
        "holdout-unfreeze4": ("Unexposed FT 4 blk", "unexp FT4"),
        "control-unfreeze4": ("Data-size control FT 4 blk", "control FT4"),
        "crop150-unfreeze4": ("crop150 FT 4 blk", "crop150 FT4"),
        "shuffled-pairing-unfreeze4": ("Shuffled pairs FT 4 blk (last.pt)", "shuffled FT4"),
        "uniform-unfreeze4": ("Uniform pooling FT 4 blk", "uniform FT4"),
        "declip-lora-depth6-r64": ("DeCLIP + LoRA 6 blk r64", "DeCLIP LoRA6"),
        "declip-frozen": ("DeCLIP, frozen image tower", "DeCLIP frozen"),
        "declip-mv-only-frozen": ("DeCLIP MV-only, frozen image tower", "DeCLIP MV frz"),
        "lambda00": ("λ = 0 (plain contrastive)", "λ 0"),
        "lambda01": ("λ = 0.1", "λ 0.1"),
        "lambda03": ("λ = 0.3", "λ 0.3"),
        "lambda03-lowunk": ("λ = 0.3, low-UNK findings", "λ 0.3 lowUNK"),
        "dinov2-base-e5-base-depth6-r64": ("DINOv2-base / E5-base, LoRA 6 r64", "base / base"),
        "dinov2-small-e5-small-depth6-r64": ("DINOv2-small / E5-small, LoRA 6 r64", "small / small"),
        "dinov2-base-e5-small-depth6-r64": ("DINOv2-base / E5-small, LoRA 6 r64", "base / small"),
    }
    if base in table_:
        return table_[base]
    if slot == "frozen_tower_null":
        return f"Frozen image tower ({base})", f"frozen {base[:8]}"
    return base, base[:14]


def arm_labels(slot: str, run: str, with_seed: bool = True) -> tuple[str, str]:
    seed = _seed(run)
    base = run[: -len(f"-seed{seed}")] if seed else run
    long, short = _base_labels(slot, base)
    if seed and with_seed:
        return f"{long} · s{seed}", f"{short} s{seed[-2:]}"
    return long, short


def _ckpt_root(prov: dict, slot: str, run: str) -> str | None:
    p = prov.get(f"{slot}/{run}")
    ck = p.get("checkpoint") if isinstance(p, dict) else None
    if not ck:
        return None
    parts = ck.split("/")
    return parts[2] if len(parts) > 2 and parts[0] == "checkpoints" else None


def _ckpt(prov: dict, slot: str, run: str) -> str:
    p = prov.get(f"{slot}/{run}")
    return (p.get("checkpoint") if isinstance(p, dict) else None) or f"{slot}/{run}"


def exposure_of(prov: dict, slot: str, run: str) -> str:
    return "unexposed" if _ckpt_root(prov, slot, run) in UNEXPOSED_ROOTS else "exposed"


def _is_frozen(slot: str, run: str) -> bool:
    return slot == "frozen_tower_null" or "frozen" in run


def _r(v, d=4):
    return round(v, d) if isinstance(v, (int, float)) else None


# --------------------------------------------------------------------------- section 1: gold holdout


def _holdout_levels(probe: dict) -> dict:
    """{arm_slot: {seed: level}} for uniform-mean arms, plus the floor."""
    out: dict[str, dict[str, float]] = {}
    for a in data.probe_runs(probe):
        if a.feature != "uniform_mean_768" or a.slot == "before_cl":
            continue
        out.setdefault(a.slot, {})[_seed(a.run)] = a.macro_auroc
    return out


def holdout_numbers() -> dict | None:
    probe = data.read_json(HOLDOUT_PROBE)
    if not probe:
        return None
    lv = _holdout_levels(probe)
    floor = data.unadapted_level(probe)
    if not (lv.get("exposed") and lv.get("unexposed") and floor is not None):
        return None
    return {
        "probe": probe, "levels": lv, "floor": floor,
        "exposed": data.seed_mean(list(lv["exposed"].values())),
        "unexposed": data.seed_mean(list(lv["unexposed"].values())),
        "control": data.seed_mean(list((lv.get("control_datasize") or {}).values())),
        "pop": (probe.get("population") or {}).get("population_size"),
        "gold": (probe.get("population") or {}).get("gold_studies"),
    }


ARM_ORDER = [("unexposed", "Unexposed", "unexposed"), ("control_datasize", "Control", "control"),
             ("exposed", "Exposed", "exposed")]


def holdout_section(ctx) -> tuple[str, dict]:
    sid = "gold-holdout"
    h = holdout_numbers()
    paired_j = data.read_json(HOLDOUT_PAIRED)
    if not h or not paired_j:
        return section(sid, "Does it generalise?", notice(
            f"<code>{HOLDOUT_PROBE}</code> or <code>{HOLDOUT_PAIRED}</code> is missing.")), {}
    probe, floor = h["probe"], h["floor"]
    vs = {v.key: v for v in data.probe_vs_unadapted(probe)}
    groups = {
        "unexposed": {"label": "Unexposed: gold held out of training", "color": "var(--g-holdout)"},
        "control": {"label": "Control: 56 fewer non-gold studies, gold kept", "color": "var(--g-control)"},
        "exposed": {"label": "Exposed: gold studies were training pairs", "color": "var(--g-bothft)"},
    }
    levels = []
    for slot, name, g in ARM_ORDER:
        seeds = h["levels"].get(slot) or {}
        for seed in sorted(seeds):
            run = next((a.run for a in data.probe_runs(probe) if a.slot == slot and _seed(a.run) == seed), None)
            v = vs.get(f"{slot}/{run}/uniform_mean_768")
            levels.append({"id": f"s{seed}", "label": f"{name} · s{seed}", "group": g, "est": _r(seeds[seed]),
                           "d": _r(v.diff) if v else None, "dlo": _r(v.ci[0]) if v else None,
                           "dhi": _r(v.ci[1]) if v else None})
        if seeds:
            levels.append({"id": "mean", "label": f"{name} mean", "group": g,
                           "est": _r(data.seed_mean(list(seeds.values())))})

    kinds = {
        ("unexposed", "control_datasize"): ("exp", "Exposure alone (unexposed − control)"),
        ("unexposed", "exposed"): ("expdata", "Exposure + data loss (unexposed − exposed)"),
        ("control_datasize", "exposed"): ("data", "Data size alone (control − exposed)"),
    }
    pgroups = {
        "exp": {"label": kinds[("unexposed", "control_datasize")][1], "color": "var(--g-holdout)"},
        "expdata": {"label": kinds[("unexposed", "exposed")][1], "color": "var(--g-rose)"},
        "data": {"label": kinds[("control_datasize", "exposed")][1], "color": "var(--g-control)"},
    }
    order = {"exp": 0, "expdata": 1, "data": 2}
    pairs = []
    for p in data.paired_comparisons(paired_j):
        cs, cr, _ = data.split_key(p.candidate)
        bs, _, _ = data.split_key(p.baseline)
        k = kinds.get((cs, bs))
        if not k:
            continue
        seed = _seed(cr)
        short = {"exp": "unexp − ctrl", "expdata": "unexp − exp", "data": "ctrl − exp"}[k[0]]
        pairs.append({"id": f"s{seed}", "label": f"{short} s{seed[-2:]}", "full": f"{k[1]}, seed {seed}", "group": k[0], "est": _r(p.diff),
                      "lo": _r(p.ci[0]), "hi": _r(p.ci[1]), "p": _r(p.p_candidate, 3)})
    pairs.sort(key=lambda r: (order[r["group"]], r["label"]))

    split = ((data.read_json("gold-case-inventory.json") or {}).get("cohort_membership") or {}).get("by_split") or {}
    pooled = data.POOLED["gold_holdout_exposure_gap"]
    gain_u, gain_e = h["unexposed"] - floor, h["exposed"] - floor
    share = gain_u / gain_e if gain_e else None
    claim = (f"About {share:.0%} of the probe gain survives on gold studies the encoder never saw: "
             f"{num(h['unexposed'])} unexposed vs {num(h['exposed'])} exposed, from a {num(floor)} floor")
    excl = {k: sum(1 for r in pairs if r["group"] == k and (r["hi"] < 0 or r["lo"] > 0)) for k in order}
    n_seeds = {k: sum(1 for r in pairs if r["group"] == k) for k in order}

    key = "".join(f'<span><i style="background:{g["color"]}"></i>{esc(g["label"])}</span>' for g in pgroups.values())
    blob = {"groups": groups, "pgroups": pgroups, "levels": levels, "pairs": pairs, "floor": floor,
            "pooled": pooled["value"]}
    body = f"""<style>
#{sid} .px-key{{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--muted);margin-top:4px}}
#{sid} .px-key i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}}
</style>
<div class="grid-2">
<div>{chart('levels', 'Gold macro AUROC per arm and seed (levels, no interval exists on a level)')}</div>
<div>{chart('pairs', 'Paired differences, same 56 studies, folds and resamples')}
<div class="px-key">{key}</div></div>
</div>
<p class="legend-hint">Hover a seed in either chart to highlight the same seed in both. Pooled unexposed − exposed over
three seeds: <strong>{num(pooled['value'], '+.4f')}</strong> (from the <a href="{ctx.doc(pooled['doc'])}">write-up</a>;
per-study scores are not stored, so no pooled interval can be drawn here).</p>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  var f4 = Charts.fmt('+.4f');
  Charts.ciDotPlot(sec.querySelector('[data-chart=levels]'), d.levels,
    {{groups: d.groups, sort: false, fmt: '.4f', axisFmt: '.2f', ref: d.floor,
      refLabel: 'unadapted DINOv2 ' + Charts.fmt('.4f')(d.floor), xLabel: 'Gold macro AUROC (56 studies, 8-fold OOF)',
      estLabel: 'Level', store: store,
      tooltipExtra: function (r) {{ return r.d === undefined || r.d === null ? '<div>Arithmetic mean of the seed levels</div>' :
        '<div>vs unadapted ' + f4(r.d) + ' [' + f4(r.dlo) + ', ' + f4(r.dhi) + '] (bootstrap mean, 95% CI)</div>'; }}}});
  Charts.ciDotPlot(sec.querySelector('[data-chart=pairs]'), d.pairs,
    {{groups: d.pgroups, sort: false, fmt: '+.4f', axisFmt: '+.2f', ref: 0, refLabel: 'no difference',
      xLabel: '\\u0394 gold macro AUROC (95% CI)', estLabel: 'Difference', store: store,
      tooltipExtra: function (r) {{ return '<div>' + Charts.esc(r.full) + '</div><div>Share of resamples favouring the left arm ' + Charts.fmt('.3f')(r.p) + '</div>'; }}}});
}})();
</script>"""
    caption = (f"{h['pop']} of {h['gold']} gold studies, pooled 8-fold out-of-fold logistic probe on the uniform-mean feature, "
               f"3 seeds; intervals are {((probe.get('config') or {}).get('bootstrap') or {}).get('resamples', '?')} "
               f"study resamples. Exposure alone: {excl['exp']} of {n_seeds['exp']} seed intervals exclude 0; "
               f"data size alone: {excl['data']} of {n_seeds['data']}.")
    rows = [[r["label"], num(r["est"]), num(r.get("d"), "+.4f") if r.get("d") is not None else "–",
             f"[{num(r['dlo'], '+.4f')}, {num(r['dhi'], '+.4f')}]" if r.get("dlo") is not None else "–"] for r in levels]
    tbl = table(["Arm", "Gold macro AUROC", "Δ vs unadapted (bootstrap mean)", "95% CI"], rows, num_cols=(1, 2, 3),
                row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in levels])
    notes = f"""<ul>
<li><strong>{num(h['exposed'])} is a development statistic</strong> (the encoder trained on {split.get('train', '?')} of these studies and selected
its checkpoint on {split.get('validation', '?')} more); <strong>{num(h['unexposed'])} is the generalisation estimate</strong>. Quote them together.</li>
<li>Levels subtract exactly ({num(h['unexposed'])} − {num(h['exposed'])} = {num(h['unexposed'] - h['exposed'], '+.4f')});
the per-seed intervals are bootstrap means and differ from level differences by ~0.0003 by construction. Do not mix the two
in one subtraction. The pooled gap is never subtracted from a paired difference elsewhere: arms in one contrast share exposure.</li>
<li>Study-level bootstrap: patient identity was re-randomised at de-identification, so patient-disjointness is unverifiable.</li>
<li>One residual exposure channel: one training study carries a report identical to a held-out gold report
(see the <a href="{ctx.doc('gold-holdout-v1.md')}">result write-up</a>).</li>
<li>Pre-registration: <a href="{ctx.doc('gold-holdout-probe-protocol.md')}">gold-holdout-probe-protocol.md</a>. Data:
<a href="{ctx.raw(HOLDOUT_PROBE)}">{HOLDOUT_PROBE}</a>, <a href="{ctx.raw(HOLDOUT_PAIRED)}">{HOLDOUT_PAIRED}</a>
(v1 of both is the pre-control record and is reproduced exactly).</li>
<li>The Kaggle / LM-head results score 58 cases with fitted heads: a different protocol, on the
<a href="{ctx.page('submission')}">submission page</a>, never on these axes.</li>
</ul>"""
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Level table ({len(levels)} rows)"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- section 2: explorer


def _paired_rows(names: list[str]) -> tuple[dict[str, list], list[str]]:
    by_fam: dict[str, list] = {}
    seen: set[str] = set()
    used: list[str] = []
    for fname in names:
        j = data.read_json(fname)
        if not j:
            continue
        prov = j.get("provenance") or {}
        hit = False
        for p in data.paired_comparisons(j):
            if p.name in seen:
                continue
            cs, cr, cf = data.split_key(p.candidate)
            bs, br, _ = data.split_key(p.baseline)
            fam = _family(cf)
            if not fam:
                continue
            seen.add(p.name)
            hit = True
            sc, sb = _seed(cr), _seed(br)
            same = sc == sb
            cl = arm_labels(cs, cr, with_seed=not same)[1]
            bl = arm_labels(bs, br, with_seed=not same)[1]
            lab = f"{cl} − {bl}" + (f" · s{sc}" if same and sc else "")
            by_fam.setdefault(fam, []).append({
                "id": f"pair:{p.name}", "label": lab, "group": exposure_of(prov, cs, cr),
                "est": _r(p.diff), "lo": _r(p.ci[0]), "hi": _r(p.ci[1]), "p": _r(p.p_candidate, 3),
                "a": _r(p.macro_candidate), "b": _r(p.macro_baseline), "src": fname,
                "full": f"{arm_labels(cs, cr)[0]} minus {arm_labels(bs, br)[0]}",
            })
        if hit:
            used.append(fname)
    return by_fam, used


def _floor_label(key: str) -> str:
    m = re.search(r"@crop(\d+)mm", key)
    if m:
        return f"unadapted @ {m.group(1)} mm crop"
    return "unadapted DINOv2-base (1536-d)" if key.endswith("_1536") else "unadapted DINOv2"


def _experiment(ctx, exp: dict, labels: list[str]) -> dict | None:
    j = data.read_json(exp["file"])
    if not j:
        return None
    prov = j.get("provenance") or {}
    identity_check = j.get("frozen_backbone_identity_check") or {}
    results = j.get("results") or {}
    fams: dict[str, dict] = {}
    floors: dict[str, float] = {}
    positives = None
    seeds: set[str] = set()
    exposures: set[str] = set()
    for v in data.probe_vs_unadapted(j):
        fam = _family(v.feature)
        if not fam:
            continue
        slot = data.split_key(v.key)[0]
        f = fams.setdefault(fam, {"arms": [], "identity": [], "paired": []})
        if fam == "uniform" and v.before is not None:
            floors[_floor_label(v.baseline_key)] = v.before
        if positives is None and v.baseline_key in results:
            pl = results[v.baseline_key].get("per_label") or {}
            positives = [(pl.get(lab) or {}).get("positives") for lab in labels]
        long, short = arm_labels(slot, v.run)
        if _is_frozen(slot, v.run) and fam == "uniform":
            ic = identity_check.get(f"{slot}/{v.run}") or {}
            dev = ic.get("max_absolute_deviation_from_unadapted")
            if dev is None and v.after is not None and v.before is not None:
                dev = abs(v.after - v.before)
            f["identity"].append({"label": long, "dev": dev, "level": _r(v.after), "ck": _ckpt(prov, slot, v.run)})
            continue
        group = "frozen" if _is_frozen(slot, v.run) else exposure_of(prov, slot, v.run)
        if group != "frozen":
            exposures.add(group)
            if _seed(v.run):
                seeds.add(_seed(v.run))
        f["arms"].append({
            "id": f"{slot}/{v.run}", "label": long, "short": short, "group": group,
            "est": _r(v.diff), "lo": _r(v.ci[0]), "hi": _r(v.ci[1]), "after": _r(v.after), "before": _r(v.before),
            "base": _floor_label(v.baseline_key), "p": _r(v.p_favour, 3),
            "pl": [_r((v.per_label.get(lab) or (None,))[0]) for lab in labels],
            "ck": _ckpt(prov, slot, v.run),
        })
    paired, used = _paired_rows(exp.get("paired") or [])
    for fam, rows in paired.items():
        fams.setdefault(fam, {"arms": [], "identity": [], "paired": []})["paired"] = rows
    entry = catalog.BY_PATH.get(exp["file"])
    status = entry.status if entry else "current"
    if exp.get("superseded_by"):
        status = "superseded"
    single = len(seeds) <= 1
    pooled = data.POOLED.get(exp.get("pooled") or "")

    badges = [status_badge(status)]
    if exposures == {"exposed"}:
        badges.append('<span class="badge badge-amended">exposed arms only: development statistic</span>')
    elif exposures == {"unexposed"}:
        badges.append('<span class="badge badge-current">unexposed arms: generalisation estimate</span>')
    elif exposures:
        badges.append('<span class="badge badge-caveat">mixes exposed and unexposed arms (tagged by colour)</span>')
    if single:
        badges.append('<span class="badge badge-caveat">single seed: neighbouring arms are not separable</span>')
    if entry and entry.caveat:
        badges.append(f'<span class="badge badge-caveat">{esc(entry.caveat)}</span>')
    floor_txt = "; ".join(f"{esc(k)} = {num(v)}" for k, v in floors.items())
    links = [f'<a href="{ctx.raw(exp["file"])}">{esc(exp["file"])}</a>']
    links += [f'<a href="{ctx.raw(n)}">{esc(n)}</a>' for n in used]
    topic = exp["topic"]
    story = ("this page, above" if topic == SLUG else
             f'<a href="{ctx.page(topic)}">{esc(BY_SLUG[topic].short)}</a>')
    meta = (f'<div class="px-badges">{" ".join(badges)}</div>'
            f'<div class="px-line"><strong>Floor in this file:</strong> {floor_txt or "–"}'
            + (' · <strong>Superseded by</strong> ' + esc(exp["superseded_by"]) + ' (identical values)'
               if exp.get("superseded_by") else "")
            + (f' · <a href="#gold-holdout">generalisation estimate for comparison</a>' if exposures == {"exposed"} else "")
            + ('</div><div class="px-line"><strong>Two floors:</strong> differences against different floors are not '
               'comparable; read the paired contrasts below instead.' if len(floors) > 1 else "")
            + f'</div><div class="px-line"><strong>Story:</strong> {story} · <strong>Data:</strong> {", ".join(links)}</div>')
    return {
        "id": exp["file"].removesuffix(".json"), "title": exp["title"], "superseded": bool(exp.get("superseded_by")),
        "resamples": ((j.get("config") or {}).get("bootstrap") or {}).get("resamples"),
        "population": (j.get("population") or {}).get("population_size"), "floors": floors,
        "meta": meta, "features": fams, "positives": positives, "single": single, "multiFloor": len(floors) > 1,
        "pooled": ({"value": pooled["value"], "label": pooled["label"], "href": ctx.doc(pooled["doc"])}
                   if pooled else None),
    }


def explorer_section(ctx) -> tuple[str, dict]:
    sid = "explorer"
    labels = None
    for exp in EXPERIMENTS:
        j = data.read_json(exp["file"])
        if j and data.probe_labels(j):
            labels = data.probe_labels(j)
            break
    if not labels:
        return section(sid, "Downstream probe explorer", notice("No probe JSON found in digest/.")), {}
    files = [f for f in (_experiment(ctx, e, labels) for e in EXPERIMENTS) if f]
    current = [f for f in files if not f["superseded"]]
    # claim over DISTINCT checkpoints (one checkpoint is often read in several files); a checkpoint
    # "clears" only if its interval excludes 0 in every current file that reads it
    by_ck: dict[str, dict] = {}
    for f in current:
        for a in f["features"].get("uniform", {}).get("arms", []):
            if a["group"] == "frozen":
                continue
            c = by_ck.setdefault(a["ck"], {"group": a["group"], "clear": True})
            c["clear"] = c["clear"] and a["lo"] is not None and a["lo"] > 0
    idents = {i["ck"] for f in current for i in f["features"].get("uniform", {}).get("identity", [])}
    cnt = {g: (sum(1 for c in by_ck.values() if c["group"] == g and c["clear"]),
               sum(1 for c in by_ck.values() if c["group"] == g)) for g in ("exposed", "unexposed")}
    claim = (f"{cnt['unexposed'][0]} of {cnt['unexposed'][1]} unexposed and {cnt['exposed'][0]} of {cnt['exposed'][1]} "
             f"exposed checkpoints clear their file's unadapted floor; {len(idents)} frozen-tower checkpoints "
             f"reproduce it exactly")
    # derived caveat numbers
    floor_vals = {}
    for f in current:
        for k, v in f["floors"].items():
            floor_vals.setdefault(num(v), set()).add(k)
    floor_list = ", ".join(f"{v} ({'; '.join(sorted(ks))})" for v, ks in sorted(floor_vals.items()))
    resamples = sorted({f["resamples"] for f in current if f["resamples"]})
    pops = sorted({f["population"] for f in current if f["population"]})
    pos = next((f["positives"] for f in current if f["positives"]), None)
    min_pos = min(zip(pos, labels)) if pos else None
    spreads = {}
    for f in current:
        runs: dict[str, list] = {}
        for a in f["features"].get("uniform", {}).get("arms", []):
            m = re.match(r"(.*)-seed\d{4}$", a["id"])
            if m and a["after"] is not None:
                runs.setdefault(m.group(1), []).append((a["after"], a["label"].split(" · ")[0]))
        for k, vals in runs.items():
            if len(vals) >= 3:
                spreads[k] = (max(v[0] for v in vals) - min(v[0] for v in vals), vals[0][1])
    widest = max(spreads.values()) if spreads else None
    narrowest = min(spreads.values()) if spreads else None
    for f in files:
        for fam in f["features"].values():
            for row in fam["arms"] + fam["identity"]:
                row.pop("ck", None)
    default = next((f["id"] for f in current if f["id"] == "downstream-probe-extended-v3"), current[0]["id"])
    blob = {"files": files, "labels": labels, "short": [LABEL_SHORT.get(lab, lab) for lab in labels],
            "groups": GROUPS, "families": FAMILIES, "default": default}
    fam_btns = "".join(f'<button type="button" class="pill-btn" data-feat="{k}">{esc(v)}</button>'
                       for k, v in FAMILIES.items())
    body = f"""<style>
#{sid} .px-controls{{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin:4px 0 10px}}
#{sid} .px-controls select{{border:1px solid var(--border);border-radius:8px;padding:6px 10px;background:var(--card);max-width:100%;min-width:0}}
#{sid} .px-controls label{{font-size:13px;color:var(--muted);display:inline-flex;gap:6px;align-items:center}}
#{sid} .px-pick{{flex:1 1 320px;min-width:0;display:flex;flex-direction:column;gap:4px}}
#{sid} .px-feat{{display:flex;flex-wrap:wrap;gap:6px}}
#{sid} .px-feat .pill-btn[disabled]{{opacity:.4;cursor:not-allowed}}
#{sid} .px-meta{{font-size:13px;margin:6px 0 12px;display:flex;flex-direction:column;gap:4px;overflow-wrap:anywhere}}
#{sid} .px-badges{{display:flex;flex-wrap:wrap;gap:6px}}
#{sid} .px-note{{font-size:12.5px;color:var(--muted);margin:6px 0 0}}
#{sid} .px-ident{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;font-size:12.5px}}
#{sid} .px-ident span.chip{{border:1px dashed var(--border);border-radius:999px;padding:2px 10px;color:var(--muted)}}
</style>
<div class="px-controls">
<div class="px-pick"><label for="px-file">Experiment (one probe file; every difference stays inside it)</label>
<select id="px-file" data-role="pick"></select></div>
<label><input type="checkbox" data-role="superseded"> show superseded files</label>
</div>
<div class="px-feat" data-role="feature" role="group" aria-label="Probe feature">{fam_btns}</div>
<div class="px-meta" data-role="meta" aria-live="polite"></div>
<div data-role="filter"></div>
{chart('dots', 'Each arm minus this file’s unadapted floor')}<div class="px-ident" data-role="ident"></div>
{chart('pairs', 'Paired contrasts from this experiment’s comparison files (left arm minus right arm, same seed)')}
<p class="px-note" data-role="pooled"></p>
<div class="chart-title">Per-label AUROC difference vs the floor (column: label, positives of 56)</div>
<div class="chart" data-chart="heat"></div>
<p class="legend-hint">Pick an experiment, then hover an arm in the dot plot or heatmap to highlight it in both; click to pin.
Pills filter by exposure. Plane-attention and projection features are secondary and confounded; uniform mean is primary.</p>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  var pick = sec.querySelector('[data-role=pick]'), sup = sec.querySelector('[data-role=superseded]');
  var featBox = sec.querySelector('[data-role=feature]');
  var state = {{file: d['default'], feat: 'uniform'}};
  var f4 = Charts.fmt('+.4f'), l4 = Charts.fmt('.4f'), p3 = Charts.fmt('.3f');
  function byId(id) {{ return d.files.filter(function (f) {{ return f.id === id; }})[0]; }}
  function fresh(sel) {{
    var old = sec.querySelector(sel), n = old.cloneNode(false);
    n.removeAttribute('style'); n.className = old.getAttribute('data-chart') ? 'chart' : '';
    old.parentNode.replaceChild(n, old); return n;
  }}
  function fillPicker() {{
    var show = sup.checked; pick.innerHTML = '';
    if (!show && byId(state.file).superseded) state.file = d['default'];
    d.files.forEach(function (f) {{
      if (f.superseded && !show) return;
      var o = document.createElement('option'); o.value = f.id;
      o.textContent = (f.superseded ? '[superseded] ' : '') + f.title; pick.appendChild(o);
    }});
    pick.value = state.file;
  }}
  function render() {{
    var f = byId(state.file);
    if (!f.features[state.feat]) state.feat = 'uniform';
    var fam = f.features[state.feat] || {{arms: [], paired: [], identity: []}};
    store.set({{group: 'all', solo: null, hover: null}});
    Array.prototype.forEach.call(featBox.querySelectorAll('[data-feat]'), function (b) {{
      var k = b.getAttribute('data-feat'), has = !!f.features[k];
      b.disabled = !has; b.classList.toggle('active', k === state.feat); b.setAttribute('aria-pressed', k === state.feat);
    }});
    sec.querySelector('[data-role=meta]').innerHTML = f.meta +
      (state.feat !== 'uniform' ? '<div class="px-line">Reference for this feature is still the uniform-mean unadapted level: no same-feature floor exists.</div>' : '');
    var present = {{}};
    fam.arms.concat(fam.paired).forEach(function (r) {{ present[r.group] = d.groups[r.group]; }});
    Charts.filterBar(fresh('[data-role=filter]'), present, store);
    var dots = fresh('[data-chart=dots]');
    if (fam.arms.length) {{
      var small = (sec.clientWidth || 800) < 620;
      var armRows = fam.arms.map(function (r) {{ var c = {{}}; for (var k in r) c[k] = r[k]; if (small) c.label = r.short; c.full = r.label; return c; }});
      Charts.ciDotPlot(dots, armRows, {{groups: d.groups, ref: 0, fmt: '+.4f',
        refLabel: f.multiFloor ? '0 = each arm\\u2019s own crop floor' : '0 = this file\\u2019s floor',
        axisFmt: '+.2f', xLabel: '\\u0394 gold macro AUROC vs unadapted (bootstrap mean, 95% CI)', store: store,
        estLabel: 'Difference', labelWidth: 250,
        tooltipExtra: function (r) {{ return '<div>' + Charts.esc(r.full) + '</div><div>Level ' + l4(r.after) + ' vs ' + Charts.esc(r.base) + ' ' + l4(r.before) +
          '</div><div>' + Charts.esc(d.groups[r.group].label) + '</div>'; }}}});
    }} else dots.innerHTML = '<div class="notice">No adapted arms on this feature.</div>';
    var ident = sec.querySelector('[data-role=ident]'); ident.innerHTML = '';
    if (fam.identity.length) {{
      ident.innerHTML = '<span><strong>Identity checks passed</strong> (frozen image tower, structural zero, not a result):</span>' +
        fam.identity.map(function (r) {{ return '<span class="chip" title="level ' + l4(r.level) + '">' + Charts.esc(r.label) +
          ' · max dev ' + (r.dev === null || r.dev === undefined ? '–' : r.dev) + '</span>'; }}).join('');
    }}
    var pairs = fresh('[data-chart=pairs]');
    if (fam.paired.length) {{
      Charts.ciDotPlot(pairs, fam.paired, {{groups: d.groups, ref: 0, refLabel: 'no difference', fmt: '+.4f', axisFmt: '+.2f',
        xLabel: '\\u0394 macro AUROC, left minus right arm', store: store, estLabel: 'Difference', labelWidth: 260,
        tooltipExtra: function (r) {{ return '<div>' + Charts.esc(r.full) + '</div><div>' + l4(r.a) + ' vs ' + l4(r.b) +
          ' · resamples favouring left ' + p3(r.p) + '</div><div class="muted">' + Charts.esc(r.src) + '</div>'; }}}});
    }} else pairs.innerHTML = '<div class="notice">No seed-matched paired file exists for this experiment on this feature; ' +
      'compare arms only through their intervals against the floor.</div>';
    sec.querySelector('[data-role=pooled]').innerHTML = f.pooled && state.feat === 'uniform' ?
      'Pooled over seeds (write-up only, no interval stored): <strong>' + f4(f.pooled.value) + '</strong> · ' +
      Charts.esc(f.pooled.label) + ' · <a href="' + f.pooled.href + '">source</a>' : '';
    var heat = fresh('[data-chart=heat]');
    if (fam.arms.length) {{
      var narrow = (sec.clientWidth || 800) < 620, m = 0;
      fam.arms.forEach(function (r) {{ r.pl.forEach(function (v) {{ if (typeof v === 'number') m = Math.max(m, Math.abs(v)); }}); }});
      Charts.heatmap(heat, fam.arms.map(function (r) {{ return r.pl; }}), {{
        rows: fam.arms.map(function (r) {{ return narrow ? r.short : r.label; }}),
        cols: d.short.map(function (c, i) {{ return c + ' ' + (f.positives ? f.positives[i] : ''); }}),
        rowIds: fam.arms.map(function (r) {{ return r.id; }}), rowGroups: fam.arms.map(function (r) {{ return r.group; }}),
        diverging: true, domain: [-m, m], fmt: '+.2f', peakPerRow: false, store: store,
        labelWidth: narrow ? 112 : 250, ariaLabel: 'Per-label AUROC difference vs unadapted'}});
    }}
  }}
  fillPicker(); render();
  pick.addEventListener('change', function () {{ state.file = pick.value; render(); }});
  sup.addEventListener('change', function () {{ fillPicker(); render(); }});
  featBox.addEventListener('click', function (e) {{
    var b = e.target.closest('[data-feat]'); if (!b || b.disabled) return; state.feat = b.getAttribute('data-feat'); render();
  }});
}})();
</script>"""
    caption = (f"{' / '.join(str(p) for p in pops)} gold studies, pooled 8-fold out-of-fold L2 logistic regression per label, "
               f"{' / '.join(f'{r:,}' for r in resamples)} study resamples. Floors differ between files "
               f"({', '.join(sorted(floor_vals))}): read each file on its own.")
    rows, attrs = [], []
    for f in current:
        u = f["features"].get("uniform", {})
        for a in u.get("arms", []):
            rows.append([f["id"], a["label"], GROUPS[a["group"]]["label"].split(" (")[0], num(a["after"]),
                         num(a["before"]), num(a["est"], "+.4f"), f"[{num(a['lo'], '+.4f')}, {num(a['hi'], '+.4f')}]"])
    tbl = table(["File", "Arm", "Exposure", "Level", "Floor", "Δ (bootstrap mean)", "95% CI"], rows,
                num_cols=(3, 4, 5, 6), search="Filter arms")
    notes = f"""<ul>
<li><strong>Differences only within one file.</strong> Each file is one extraction pass with shared folds and resamples;
levels from different files are unpaired, and the floors differ: {esc(floor_list)}. The 0.6438 / 0.6440 split is two
extraction passes of the same model; the DINOv2-base floor uses 1,536-dimensional features. The crop150 config forbids
differencing its two floors.</li>
<li><strong>Exposure is tagged per arm by checkpoint root.</strong> Unexposed: <code>{"</code>, <code>".join(sorted(UNEXPOSED_ROOTS))}</code>
(all trained on the gold-held-out cohort). Everything else, including the data-size control, trained on the gold studies and
is a development statistic; the generalisation estimate for the 4-block recipe is in the section above.</li>
<li><strong>Identity checks are not results.</strong> Frozen-image-tower arms equal the floor exactly on the uniform-mean
feature (structural zero). On plane-attention and projection features they differ from the floor only through the learned
pooling / projection head.</li>
<li><strong>difference_mean is the bootstrap mean</strong>, not the point difference of levels (they differ in the 4th decimal).
There is no interval on a level. <code>fraction_of_replicates_favouring_right</code> is the share favouring the
<em>candidate</em> (left arm), despite its name.</li>
<li><strong>Seed noise.</strong> Three-seed level spreads in these files run from {num(narrowest[0], '.4f') if narrowest else '–'}
({esc(narrowest[1]) if narrowest else ''}) to {num(widest[0], '.4f') if widest else '–'} ({esc(widest[1]) if widest else ''});
single-seed families (depth / LoRA sweep, DeCLIP, label similarity, LoRA holdout, tower size) cannot resolve differences of that size. The shuffled-pairing control
uses <code>last.pt</code>, not <code>best.pt</code>.</li>
<li><strong>Per-label cells are noisy</strong>: {esc(min_pos[1]) if min_pos else '–'} has {min_pos[0] if min_pos else '–'} positives of {pops[0] if pops else '–'}. Study ≠ patient: the bootstrap resamples studies.</li>
<li><strong>Superseded files</strong> (hidden by default) are strict subsets with identical values: cohort-compare ⊂ extended
v1 ⊂ v2 ⊂ v3, unexposed v1 ⊂ v2, label-similarity v1 ⊂ v2. The extended-comparisons v1 / v2 / v3 files hold
<em>different</em> contrasts and are all shown with extended-v3 (duplicate pairs kept once, newest file).</li>
<li>Narratives: language and text block on <a href="{ctx.page('text')}">text</a>; depth, LoRA and pooling on
<a href="{ctx.page('adaptation')}">adaptation</a>; crop on <a href="{ctx.page('overfitting')}">overfitting</a>; label similarity,
DeCLIP and tower size on <a href="{ctx.page('encoders')}">encoders</a>. Write-ups:
<a href="{ctx.doc('downstream-probe-v1.md')}">first probe</a>,
<a href="{ctx.doc('downstream-both-block-v1.md')}">both block</a>,
<a href="{ctx.doc('downstream-original-vs-translated-v1.md')}">original vs translated</a> (amended).</li>
<li>Retrieval R@1 selects every checkpoint; it never shares a table with these AUROC values. The 58-case LM-head results are
a different protocol, on the <a href="{ctx.page('submission')}">submission page</a>.</li>
</ul>"""
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"All uniform-mean arms in current files ({len(rows)} rows)"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- section 3: gold set & leak check


def gold_set_section(ctx) -> tuple[str, dict]:
    sid = "gold-set"
    inv = data.read_json("gold-case-inventory.json")
    leak = data.read_json("protocol-leak-check.json")
    if not inv or not leak:
        return section(sid, "Gold set and protocol-leak check", notice("Gold inventory or leak-check JSON is missing.")), {}
    cm = inv.get("cohort_membership") or {}
    split = cm.get("by_split") or {}
    gold = cm.get("gold_studies")
    used = (split.get("train") or 0) + (split.get("validation") or 0)
    funnel = [
        {"id": "gold", "label": "Gold studies", "group": "all", "value": gold},
        {"id": "train", "label": "Train (training pairs)", "group": "used", "value": split.get("train")},
        {"id": "val", "label": "Validation (selection)", "group": "used", "value": split.get("validation")},
        {"id": "test", "label": "Locked test (not loaded)", "group": "excluded", "value": split.get("test")},
        {"id": "probe", "label": "Probe population", "group": "used", "value": used},
    ]
    fgroups = {"all": {"label": "All gold", "color": "var(--g-anchor)"},
               "used": {"label": "Read by the probe", "color": "var(--g-teal)"},
               "excluded": {"label": "Excluded", "color": "var(--g-rose)"}}
    prim = leak.get("primary") or {}
    lrows = [{"id": f"leak:{k}", "label": f"{LABEL_SHORT.get(k, k)} ({v.get('n_pos')}+)", "group": "leak",
              "est": _r(v.get("auroc"), 3), "p": _r(v.get("permutation_p"), 3)} for k, v in prim.items()]
    flagged = sum(1 for r in lrows if r["p"] is not None and r["p"] < 0.05)
    lo, hi = LEAK_DETECTABLE
    claim = (f"The probe reads {used} of {gold} gold studies; series count separates {flagged} of {len(lrows)} labels, "
             f"which rules out only large protocol leakage")
    blob = {"funnel": funnel, "fgroups": fgroups, "leak": lrows,
            "lgroups": {"leak": {"label": "Series count as a label score", "color": "var(--g-violet)"}}}
    body = f"""<div class="grid-2">
<div>{chart('funnel', 'Which gold studies the 56-case probe reads')}</div>
<div>{chart('leak', f"Series count → gold label AUROC (n = {leak.get('n_gold')}, not a model probe)")}</div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.barChart(sec.querySelector('[data-chart=funnel]'), d.funnel,
    {{groups: d.fgroups, fmt: '.0f', axisFmt: '.0f', xLabel: 'Studies', store: store, labelWidth: 240}});
  Charts.ciDotPlot(sec.querySelector('[data-chart=leak]'), d.leak,
    {{groups: d.lgroups, sort: 'est', fmt: '.3f', axisFmt: '.2f', ref: 0.5, refLabel: 'chance', domain: [0.3, 0.7],
      xLabel: 'AUROC of series count for the label', estLabel: 'AUROC', store: store,
      tooltipExtra: function (r) {{ return '<div>Permutation p = ' + Charts.fmt('.3f')(r.p) + '</div>'; }}}});
}})();
</script>"""
    caption = (f"Left: <a href=\"{ctx.raw('gold-case-inventory.json')}\">gold-case-inventory.json</a>. Right: "
               f"<a href=\"{ctx.raw('protocol-leak-check.json')}\">protocol-leak-check.json</a>, "
               f"{leak.get('resamples')} permutations; at this n the test detects only |AUROC − 0.5| above "
               f"{lo}–{hi}.")
    notes = f"""<ul>
<li>The 2 locked-test gold studies are counted, never loaded, in every probe on this page. Labels in the 58-case column
(leak check) therefore differ slightly from the 56-case positives in the explorer heatmap.</li>
<li>In every exposed run the 48 train-split gold studies were training pairs and the 8 validation studies selected the
checkpoint; the gold-holdout cohort removed all 56 from both (see <a href="{ctx.doc('gold-case-inventory.md')}">gold-case
inventory</a>, whose probe contract was later superseded in practice by the gold holdout).</li>
<li>The leak check can only detect a large effect ({lo}–{hi} from chance, best- to worst-powered label); a modest leak near
AUROC 0.60 is not ruled out. Recommended mitigation: carry series count as a control beside probe results
(<a href="{ctx.doc('acquisition-metadata-trust.md')}">acquisition-metadata-trust.md</a>, section 6).</li>
<li>Where the gold set, splits and integrity audits are explained: <a href="{ctx.page('data')}">data page</a>.</li>
</ul>"""
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- page


def build(ctx) -> Page:
    sections, blobs = [], {}
    for fn in (holdout_section, explorer_section, gold_set_section):
        s, b = fn(ctx)
        sections.append(s)
        blobs.update(b)
    kpis = None
    h = holdout_numbers()
    if h:
        pooled = data.POOLED["gold_holdout_exposure_gap"]
        kpis = [
            {"label": "Unadapted DINOv2 floor", "value": num(h["floor"]), "sub": f"{h['pop']} gold studies, 8-fold OOF"},
            {"label": "Unexposed: generalisation", "value": num(h["unexposed"]), "tone": "good",
             "sub": "gold held out of training, 3-seed mean"},
            {"label": "Exposed: development", "value": num(h["exposed"]),
             "sub": f"encoder trained on these studies; quote beside {num(h['unexposed'])}"},
            {"label": "Exposure gap, pooled", "value": num(pooled["value"], "+.4f"), "tone": "bad",
             "sub": "unexposed − exposed, 3 seeds (write-up)"},
        ]
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=sections, data=blobs, kpis=kpis)


def summary() -> dict:
    h = holdout_numbers()
    base = {"stage": BY_SLUG[SLUG].number, "href": "#gold-holdout"}
    if not h:
        return {**base, "headline": None, "value": None, "value_label": None, "caveat": None}
    return {**base,
            "headline": "Gold-probe AUROC on studies the encoder never saw",
            "value": num(h["unexposed"]),
            "value_label": f"unexposed, 3 seeds (floor {num(h['floor'])})",
            "caveat": f"Exposed (development) reads {num(h['exposed'])}; {h['pop']} studies, study-level bootstrap"}
