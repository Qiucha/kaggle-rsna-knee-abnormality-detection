"""Stage 1: how the pipeline works, and the first baselines.

Sections (each one claim + chart, numbers read from digest/* or digest/site/data/* at build time):
  architecture  interactive diagram of the submitted model (click a node for shapes / params; 3 modes)
  budget        trainable-parameter budget of the depth / rank options (batch-32 memory profile)
  timeline      pilot -> audit -> manifest -> frozen baseline -> reproduction (dated, linked)
  pilot         58-study CPU pilot checkpoints, each vs its own chance level; rescore 0/18 changed
  baseline      frozen-tower full-cohort baseline curves (full-multilingual-v1)
  reproduction  cleanup rerun of the translated-cohort baseline (bit-exact per the JSON booleans)
"""

from __future__ import annotations

import re

from .. import data
from ..shell import Page, chart, esc, kpi_row, notice, num, section, table
from ..sitemap import BY_SLUG

SLUG = "pipeline"
SUBTITLE = ("What the submitted model is, tensor by tensor, and the first baselines that showed the "
            "pipeline learns: the 58-study pilot, the frozen-tower full-cohort run and its exact rerun.")

# Architecture constants (model configs, not results). source: HF configs of the pinned
# facebook/dinov2-small (hidden 384, MLP 1536, 12 blocks, patch 14) and intfloat/multilingual-e5-base
# (hidden 768, intermediate 3072, 12 layers); LoRA wraps every nn.Linear of a block
# (src/contrastive_learning/lora.py apply_lora_to_linear_modules): q, k, v, attention out, MLP in, MLP out.
IMG_HIDDEN, IMG_MLP = 384, 1536
TXT_HIDDEN, TXT_MLP = 768, 3072
N_BLOCKS = 12

# Baseline bootstrap interval exists only in markdown. source: digest/full-projection-v1-results.md:19
BASELINE_CI = (0.0828, 0.1315)

BASELINE_ROOT, BASELINE_RUN = "full-multilingual-v1", "00-projection-full-mean-512"
ORIG_ROOT, ORIG_RUN = "claude-translated-dfd07db-q2-v1", "00-full-seed2026"
RERUN_ROOT, RERUN_RUN = "cleanup-reproduction-20260908", "full-seed2026"


def _pct(v, d=2):
    return num(v, f".{d}%")


def _m(v):
    return f"{v / 1e6:.2f}M"


def _lora_params(hidden: int, mlp: int, rank: int, blocks: int) -> int:
    """LoRA A+B parameters for q,k,v,out (hidden->hidden) and MLP in/out (hidden<->mlp)."""
    per_block = 4 * rank * (hidden + hidden) + 2 * rank * (hidden + mlp)
    return per_block * blocks


# --------------------------------------------------------------------------- shape table


def _shape_rows() -> dict[str, dict]:
    """Parse the operation table of image-shape-audit.md into {operation: {before, after, meaning}}."""
    text = data.read_text("image-shape-audit.md") or ""
    rows = {}
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip().replace("`", "") for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] == "Operation":
            continue
        rows[cells[0]] = {"before": cells[1], "after": cells[2], "meaning": cells[3]}
    return rows


# --------------------------------------------------------------------------- section 1: architecture


def architecture_section(ctx) -> tuple[str, dict, dict]:
    sid = "architecture"
    arch = data.read_json("architecture-parameter-audit.json")
    prof = data.read_json("lora-memory-profile.json")
    man = data.read_json("full-manifest-migration.json")
    shapes = _shape_rows()
    if not (arch and prof and man):
        return section(sid, "Architecture", notice("Parameter or manifest JSON missing.")), {}, {}

    r64 = next((r for r in prof if r.get("rank") == 64 and r.get("layer_count") == 6 and r.get("batch_size") == 32), None)
    base = arch["baseline"]
    heads = base["trainable_parameters"]
    img_bb, txt_bb, total = base["image_backbone_parameters"], base["text_backbone_parameters"], base["total_parameters"]
    split_ok = False
    img_lora = txt_lora = None
    if r64:
        img_lora = _lora_params(IMG_HIDDEN, IMG_MLP, 64, r64["layer_count"])
        txt_lora = _lora_params(TXT_HIDDEN, TXT_MLP, 64, r64["layer_count"])
        split_ok = (img_lora + txt_lora == r64["lora_parameters"]
                    and r64["trainable_parameters"] - r64["lora_parameters"] == heads
                    and r64["total_parameters"] == total + r64["lora_parameters"])
    cfg = man["metadata_before"]["config"]
    depth = r64["layer_count"] if r64 else 6
    first_lora = N_BLOCKS - depth + 1

    def sh(op):
        r = shapes.get(op)
        return r

    def shape_txt(*ops):
        out = []
        for op in ops:
            r = sh(op)
            if r:
                out.append({"op": op, "before": r["before"], "after": r["after"], "meaning": r["meaning"]})
        return out

    lora_img_txt = (f"{img_lora:,} LoRA parameters (rank 64, α 128) on q, k, v, attention-out and both MLP "
                    f"linears of {depth} blocks" if split_ok else "LoRA split unavailable")
    lora_txt_txt = (f"{txt_lora:,} LoRA parameters (rank 64, α 128) on the six linears of each of {depth} blocks"
                    if split_ok else "LoRA split unavailable")
    head_note = f"part of the {heads:,} non-backbone trainable parameters (plane attention, projections, temperature)"

    # state per mode: lora (adapter trained) / train (fully trainable) / frozen / off (not run) / data
    nodes = [
        {"id": "n-series", "lane": "image", "label": "Series & slice selection",
         "sub": f"3 planes × {cfg['slices_per_plane']} slices, {cfg['crop_mm']:.0f} mm crop, {cfg['image_size']} px",
         "detail": (f"Per study: the fluid-sensitive sagittal, coronal and axial series; {cfg['slices_per_plane']} slices "
                    f"from the central {cfg['slice_band_low']:.0%}–{cfg['slice_band_high']:.0%} band; centred "
                    f"{cfg['crop_mm']:.0f} mm crop; resized to {cfg['image_size']} px (manifest config)."),
         "shapes": shape_txt("Stack selected slices", "Resize (temporary batch axis)"),
         "state": {"frozen": "data", "lora": "data", "kaggle": "data"}},
        {"id": "n-stack", "lane": "image", "label": "Study tensor",
         "sub": "[B,3,3,224,224]",
         "detail": "Planes, then studies, are stacked; slices stay as the channel axis.",
         "shapes": shape_txt("Stack planes, then studies", "ImageTower reshape", "Channel normalization"),
         "state": {"frozen": "data", "lora": "data", "kaggle": "data"}},
        {"id": "n-img-low", "lane": "image", "label": f"DINOv2-small blocks 1–{first_lora - 1}",
         "sub": "patch embed + frozen blocks",
         "detail": f"Pretrained facebook/dinov2-small; image backbone {img_bb:,} parameters in total, all frozen here.",
         "shapes": shape_txt("DINOv2 patch convolution", "Flatten spatial grid, transpose", "Add CLS, transformer"),
         "params": img_bb, "state": {"frozen": "frozen", "lora": "frozen", "kaggle": "frozen"}},
        {"id": "n-img-lora", "lane": "image", "label": f"DINOv2-small blocks {first_lora}–{N_BLOCKS}",
         "sub": f"LoRA r64 · {_m(img_lora) if split_ok else '?'}",
         "detail": (f"Submitted model: {lora_img_txt}. Base weights stay frozen. In the 2026-09-05 baseline "
                    f"these blocks were frozen too. At Kaggle inference the trained adapters are part of the "
                    f"frozen feature extractor."),
         "params": img_lora, "state": {"frozen": "frozen", "lora": "lora", "kaggle": "frozen"}},
        {"id": "n-img-pool", "lane": "image", "label": "CLS + patch mean",
         "sub": "[3B,768] per plane",
         "detail": "CLS token and mean patch token concatenated for each plane.",
         "shapes": shape_txt("Mean patch tokens", "Concatenate CLS and patch mean", "Restore studies and planes"),
         "state": {"frozen": "data", "lora": "data", "kaggle": "data"}},
        {"id": "n-plane", "lane": "image", "label": "Plane attention",
         "sub": "[B,3,768] → [B,768]",
         "detail": (f"Learned attention over the three planes; {head_note}. The Kaggle bundle replaces it by a "
                    f"uniform mean over planes (src/contrastive_learning/submission.py image_features)."),
         "shapes": shape_txt("Attention scores / softmax", "Weighted plane aggregation"),
         "state": {"frozen": "train", "lora": "train", "kaggle": "off"}},
        {"id": "n-img-proj", "lane": "image", "label": "Image projection",
         "sub": "[B,768] → [B,256], L2",
         "detail": f"256-d contrastive projection; {head_note}. Not run at Kaggle inference.",
         "shapes": shape_txt("Projection, L2 normalization"),
         "state": {"frozen": "train", "lora": "train", "kaggle": "off"}},
        {"id": "n-tok", "lane": "text", "label": "Report tokens",
         "sub": "'query: ' prefix, ≤512 tokens",
         "detail": ("Raw multilingual report, E5 'query: ' prefix, truncated at 512 tokens "
                    "(truncation rates are on the data page)."),
         "state": {"frozen": "data", "lora": "data", "kaggle": "off"}},
        {"id": "n-txt-low", "lane": "text", "label": f"mE5-base layers 1–{first_lora - 1}",
         "sub": "frozen",
         "detail": f"Pretrained intfloat/multilingual-e5-base; text backbone {txt_bb:,} parameters in total.",
         "params": txt_bb, "state": {"frozen": "frozen", "lora": "frozen", "kaggle": "off"}},
        {"id": "n-txt-lora", "lane": "text", "label": f"mE5-base layers {first_lora}–{N_BLOCKS}",
         "sub": f"LoRA r64 · {_m(txt_lora) if split_ok else '?'}",
         "detail": f"Submitted model: {lora_txt_txt}. Frozen in the 2026-09-05 baseline. Not run at Kaggle inference.",
         "params": txt_lora, "state": {"frozen": "frozen", "lora": "lora", "kaggle": "off"}},
        {"id": "n-txt-pool", "lane": "text", "label": "Masked mean + projection",
         "sub": "[B,768] → [B,256], L2",
         "detail": f"Mean over non-padding tokens, then a 256-d projection; {head_note}.",
         "state": {"frozen": "train", "lora": "train", "kaggle": "off"}},
        {"id": "n-loss", "lane": "joint", "label": "Symmetric InfoNCE",
         "sub": "batch 32, learned temperature",
         "detail": ("Image→report and report→image cross-entropy over in-batch pairs, with a learned inverse "
                    "temperature. No pathology labels are used. Checkpoints are selected on validation R@1."),
         "state": {"frozen": "train", "lora": "train", "kaggle": "off"}},
        {"id": "n-heads", "lane": "joint", "label": "12 logistic heads",
         "sub": "fitted on 56 gold studies",
         "detail": ("Kaggle bundle only: image features (uniform plane mean) feed 12 standardized logistic "
                    "regressions (C = 1.0), one per finding, fitted after contrastive training on the 56 "
                    "held-out gold studies (selected-kaggle-model.md; feature pooling: src/contrastive_learning/submission.py)."),
         "state": {"frozen": "off", "lora": "off", "kaggle": "train"}},
    ]
    modes = {
        "lora": {"label": "Submitted model: contrastive training",
                 "trainable": r64["trainable_parameters"] if r64 else None,
                 "total": r64["total_parameters"] if r64 else None},
        "kaggle": {"label": "Kaggle inference", "trainable": None, "total": None},
        "frozen": {"label": "Frozen baseline (2026-09-05)", "trainable": heads, "total": total},
    }
    blob = {"nodes": nodes, "modes": modes, "total": total}

    body = """<div class="arch-modes filter-bar" role="group" aria-label="Diagram mode"></div>
<div class="chart arch" data-chart="arch">
  <div class="arch-lane" data-lane="image"><div class="arch-lane-title">Image tower (per study)</div><div class="arch-row"></div></div>
  <div class="arch-lane" data-lane="text"><div class="arch-lane-title">Report tower</div><div class="arch-row"></div></div>
  <div class="arch-lane" data-lane="joint"><div class="arch-lane-title">Objective / output</div><div class="arch-row"></div></div>
</div>
<div class="arch-key"><span class="k lora">LoRA adapter trained</span><span class="k train">trainable</span><span class="k frozen">frozen weights</span><span class="k data">data / reshape</span><span class="k off">not run</span></div>
<div class="arch-detail" aria-live="polite"></div>
<style>
#architecture .arch{display:flex;flex-direction:column;gap:12px}
#architecture .arch-lane-title{font-size:11.5px;font-weight:650;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);margin-bottom:6px}
#architecture .arch-row{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch}
#architecture .arch-node{position:relative;text-align:left;flex:1 1 150px;max-width:230px;min-width:130px;border:1.5px solid var(--border);border-radius:9px;background:var(--card);color:var(--text);padding:8px 10px;cursor:pointer;font:inherit}
#architecture .arch-node .t{display:block;font-weight:650;font-size:13px;color:var(--text-strong)}
#architecture .arch-node .s{display:block;font-size:11.5px;color:var(--muted);font-family:var(--mono);margin-top:2px}
#architecture .arch-node.st-lora{border-color:var(--g-lora);background:color-mix(in srgb,var(--g-lora) 14%,var(--card))}
#architecture .arch-node.st-train{border-color:var(--g-bothft);background:color-mix(in srgb,var(--g-bothft) 10%,var(--card))}
#architecture .arch-node.st-frozen{border-color:var(--muted)}
#architecture .arch-node.st-data{border-style:dotted}
#architecture .arch-node.st-off{border-style:dashed;opacity:.38}
#architecture .arch-node.highlighted{box-shadow:0 0 0 2px var(--text-strong)}
#architecture .arch-node.dimmed{opacity:.35}
#architecture .arch-key{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0 6px;font-size:12px;color:var(--muted)}
#architecture .arch-key .k:before{content:"";display:inline-block;width:12px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px;border:1.5px solid var(--muted)}
#architecture .arch-key .lora:before{border-color:var(--g-lora);background:color-mix(in srgb,var(--g-lora) 25%,var(--card))}
#architecture .arch-key .train:before{border-color:var(--g-bothft);background:color-mix(in srgb,var(--g-bothft) 18%,var(--card))}
#architecture .arch-key .data:before{border-style:dotted}
#architecture .arch-key .off:before{border-style:dashed;opacity:.5}
#architecture .arch-detail{border:1px solid var(--border);border-radius:9px;padding:10px 14px;background:var(--plot-bg);font-size:13px;min-height:60px}
#architecture .arch-detail table{font-size:12px;margin-top:6px}
#architecture .arch-detail code{font-family:var(--mono);font-size:11.5px}
</style>
<script>
(function () {
  var sec = document.getElementById('__SID__');
  var d = Charts.readJSON('data-__SID__');
  var store = Charts.selection(sec);
  var mode = 'lora';
  var byId = {};
  var bar = sec.querySelector('.arch-modes');
  Object.keys(d.modes).forEach(function (k) {
    var b = Charts.h('button', {type: 'button', class: 'pill-btn', 'data-mode': k, text: d.modes[k].label});
    bar.appendChild(b);
  });
  bar.addEventListener('click', function (e) {
    var b = e.target.closest('[data-mode]'); if (!b) return;
    mode = b.getAttribute('data-mode'); paint(); detail(store.get());
  });
  d.nodes.forEach(function (n) {
    byId[n.id] = n;
    var b = Charts.h('button', {type: 'button', class: 'arch-node', 'data-series-id': n.id, 'aria-label': n.label});
    b.appendChild(Charts.h('span', {class: 't', text: n.label}));
    b.appendChild(Charts.h('span', {class: 's', text: n.sub}));
    b.addEventListener('pointerenter', function () { if (!store.get().solo) store.set({hover: n.id}); });
    b.addEventListener('pointerleave', function () { store.set({hover: null}); });
    b.addEventListener('click', function () { store.toggleSolo(n.id); });
    sec.querySelector('[data-lane=' + n.lane + '] .arch-row').appendChild(b);
  });
  var NAMES = {lora: 'LoRA adapter trained', train: 'trainable', frozen: 'frozen weights', data: 'data / reshape', off: 'not run in this mode'};
  function paint() {
    sec.querySelectorAll('.arch-node').forEach(function (b) {
      var n = byId[b.getAttribute('data-series-id')];
      b.className = 'arch-node st-' + n.state[mode];
    });
    bar.querySelectorAll('[data-mode]').forEach(function (b) {
      var on = b.getAttribute('data-mode') === mode;
      b.classList.toggle('active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    Charts.applySelection(sec, store.get());
  }
  function detail(s) {
    var id = s.solo || s.hover, box = sec.querySelector('.arch-detail'), m = d.modes[mode];
    if (!id || !byId[id]) {
      var t = m.trainable ? ('<b>' + m.trainable.toLocaleString('en-US') + '</b> of ' + m.total.toLocaleString('en-US') +
        ' parameters trainable (' + Charts.fmt('.2%')(m.trainable / m.total) + ').') : 'No parameters are trained at inference.';
      box.innerHTML = '<b>' + Charts.esc(m.label) + '.</b> ' + t + ' Click a block for its tensor shapes and parameters.';
      return;
    }
    var n = byId[id];
    var h = '<div><b>' + Charts.esc(n.label) + '</b> <span class="muted">· ' + Charts.esc(NAMES[n.state[mode]]) + '</span></div>';
    h += '<div>' + Charts.esc(n.detail) + '</div>';
    if (n.params) h += '<div class="muted">Parameters: ' + n.params.toLocaleString('en-US') + '</div>';
    if (n.shapes && n.shapes.length) {
      h += '<div class="table-wrap"><table><thead><tr><th>Operation</th><th>Before</th><th>After</th><th>Meaning</th></tr></thead><tbody>';
      n.shapes.forEach(function (r) {
        h += '<tr><td>' + Charts.esc(r.op) + '</td><td><code>' + Charts.esc(r.before) + '</code></td><td><code>' +
          Charts.esc(r.after) + '</code></td><td>' + Charts.esc(r.meaning) + '</td></tr>';
      });
      h += '</tbody></table></div><div class="muted">Shapes: image-shape audit (synthetic inputs, random weights; amended 2026-09-10).</div>';
    }
    box.innerHTML = h;
  }
  store.on(detail);
  paint();
})();
</script>""".replace("__SID__", sid)

    tot_lora = r64["total_parameters"] if r64 else None
    frac = r64["trainable_parameters"] / tot_lora if r64 else None
    claim = (f"The submitted model trains {_m(r64['trainable_parameters'])} of {tot_lora / 1e6:.1f}M parameters "
             f"({_pct(frac)}): LoRA rank 64 in the last {depth} blocks of both towers") if r64 else "Architecture"
    caption = (f"DINOv2-small image tower + multilingual-E5-base report tower (selected-kaggle-model). Switch mode to "
               f"compare with the frozen-tower baseline ({heads:,} trainable) and the Kaggle inference path. "
               f"<a href=\"{ctx.doc('selected-kaggle-model.md')}\">Model card</a> · "
               f"<a href=\"{ctx.doc('image-shape-audit.md')}\">shape audit</a>.")
    notes = (
        "<ul>"
        f"<li>Parameter counts: backbone totals from <a href=\"{ctx.raw('architecture-parameter-audit.json')}\">architecture-parameter-audit.json</a> "
        f"(pre-LoRA arms); the {_m(r64['trainable_parameters']) if r64 else '?'} trainable total from the batch-32 row of "
        f"<a href=\"{ctx.raw('lora-memory-profile.json')}\">lora-memory-profile.json</a> for the same configuration, not read from the submitted checkpoint.</li>"
        + (f"<li>The per-tower LoRA split ({img_lora:,} image + {txt_lora:,} text) is computed from the hidden sizes "
           f"({IMG_HIDDEN}/{IMG_MLP} and {TXT_HIDDEN}/{TXT_MLP}) and checked to sum to the profiled "
           f"{r64['lora_parameters']:,}; trainable minus LoRA equals the baseline's {heads:,}. The LoRA model's total "
           f"({r64['total_parameters']:,}) is the {total:,} pretrained-plus-head parameters plus the adapters.</li>" if split_ok else
           "<li>The per-tower LoRA split did not reconcile with the profile and is not shown.</li>")
        + "<li>Tensor shapes are parsed from the image-shape audit table at build time. Its trace used synthetic inputs "
          "and random weights and cannot be regenerated (correction 2026-09-10); the runnable check is "
          "<code>tests/test_image_tower.py</code>. Text-tower shapes are not in that audit.</li>"
        + f"<li>The Kaggle path (12 logistic heads on image backbone features; no report encoder, no projections) comes from "
          f"<a href=\"{ctx.doc('selected-kaggle-model.md')}\">selected-kaggle-model.md</a>; the uniform mean of per-plane "
          f"CLS + patch-mean features is from <a href=\"{ctx.repo('src/contrastive_learning/submission.py')}\">submission.py</a> "
          "(<code>image_features</code>). Scores are on the "
          f"<a href=\"{ctx.page('submission')}\">submission page</a>.</li>"
        + f"<li>The architecture explorer in <a href=\"{ctx.raw('project-state.html')}\">project-state.html</a> shows the "
          "pre-LoRA model (frozen text tower, one image block); treat that page as a legacy run-status snapshot. The original "
          f"design reference <a href=\"{ctx.doc('contrastive-learning-pipeline.md')}\">contrastive-learning-pipeline.md</a> "
          "is superseded in its model choices (RadBERT, 256 tokens); its preprocessing sections still hold.</li>"
        "</ul>"
    )
    kp = {"total": tot_lora, "trainable": r64["trainable_parameters"] if r64 else None}
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}, kp


# --------------------------------------------------------------------------- section 2: budget


def budget_section(ctx) -> tuple[str, dict]:
    sid = "budget"
    arch = data.read_json("architecture-parameter-audit.json")
    prof = data.read_json("lora-memory-profile.json")
    if not (arch and prof):
        return section(sid, "Parameter budget", notice("Parameter JSON missing.")), {}
    groups = {
        "frozen": {"label": "Frozen backbones", "color": "var(--g-anchor)"},
        "imgft": {"label": "Image block only", "color": "var(--g-imgft)"},
        "lora": {"label": "Both towers, LoRA", "color": "var(--g-lora)"},
        "bothft": {"label": "Both towers, full FT", "color": "var(--g-bothft)"},
    }
    rows = [
        {"id": "b-frozen", "label": "Frozen towers (baseline)", "group": "frozen",
         "value": arch["baseline"]["trainable_parameters"] / 1e6, "mb": None, "fits": None},
        {"id": "b-imgblock", "label": "Image last block, text frozen", "group": "imgft",
         "value": arch["image_block"]["trainable_parameters"] / 1e6, "mb": None, "fits": None},
    ]
    for r in prof:
        if r.get("batch_size") != 32:
            continue
        lora = r.get("rank") is not None
        lab = (f"LoRA r{r['rank']}, {r['layer_count']} blocks" if lora else f"Full FT, {r['layer_count']} blocks")
        if not r.get("fits_in_8gb"):
            lab += " (OOM)"
        rows.append({"id": f"b-{'lora' if lora else 'ft'}-{r.get('rank')}-{r['layer_count']}", "label": lab,
                     "group": "lora" if lora else "bothft", "value": r["trainable_parameters"] / 1e6,
                     "mb": r.get("peak_allocated_mb"), "fits": r.get("fits_in_8gb")})
    lr64 = next((r for r in rows if r["id"] == "b-lora-64-6"), None)
    ft6 = next((r for r in rows if r["id"] == "b-ft-None-6"), None)
    claim = "Parameter budget"
    if lr64 and ft6:
        claim = (f"LoRA r64 in 6 blocks trains {lr64['value']:.2f}M parameters, {ft6['value'] / lr64['value']:.1f}× fewer "
                 f"than full fine-tuning of the same 6 blocks ({ft6['value']:.1f}M)")
    blob = {"groups": groups, "rows": rows}
    body = f"""<div data-role="filter"></div>
{chart('bars', 'Trainable parameters (millions), batch-32 profile')}
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var f = function (v) {{ return v === null || v === undefined ? '–' : v.toFixed(2) + 'M'; }};
  Charts.barChart(sec.querySelector('[data-chart=bars]'), d.rows,
    {{groups: d.groups, fmt: f, axisFmt: function (v) {{ return v.toFixed(0) + 'M'; }}, xLabel: 'Trainable parameters',
      store: store, estLabel: 'Trainable', labelWidth: 220,
      tooltipExtra: function (r) {{ return r.mb ? '<div>Peak allocated ' + r.mb.toFixed(0) + ' MB · ' +
        (r.fits ? 'fits' : 'does not fit') + ' 8 GB</div>' : '<div>Pre-LoRA audit arm</div>'; }}}});
}})();
</script>"""
    caption = ("Same DINOv2-small + mE5-base pair throughout. LoRA and full-FT rows come from the batch-32 memory profile "
               "(peak MB in the tooltip); the first two rows are the pre-LoRA parameter audit. Memory trade-offs are on the "
               f"<a href=\"{ctx.page('adaptation')}\">adaptation page</a>.")
    tbl = table(["Configuration", "Trainable parameters", "Peak allocated MB", "Fits 8 GB"],
                [[r["label"], f"{round(r['value'] * 1e6):,}", num(r["mb"], ".1f") if r["mb"] else "–",
                  "–" if r["fits"] is None else ("yes" if r["fits"] else "no")] for r in rows],
                num_cols=(1, 2), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = ("<ul><li>Sources: <a href=\"" + ctx.raw("architecture-parameter-audit.json") + "\">architecture-parameter-audit.json</a>, "
             "<a href=\"" + ctx.raw("lora-memory-profile.json") + "\">lora-memory-profile.json</a> (batch 32 only; the batch-16 "
             "sweep and OOM workarounds live on the adaptation page).</li>"
             "<li>Parameter count is not a result: which depth and rank generalise is decided on the adaptation and probes pages.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Parameter table ({len(rows)} configurations)"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- section 3: timeline


def timeline_section(ctx) -> str:
    sid = "timeline"
    man = data.read_json("full-manifest-migration.json") or {}
    migrated = (man.get("migrated_at") or "")[:10] or "2026-09-05"
    # Dates are from each document's own header/date line.
    steps = [
        ("2026-09-02", "58-study CPU pilot", "Five folds of 11–12 studies; text-encoder, unfreezing and token-length screens.",
         ctx.doc("experiment-session-log.md"), "session log"),
        ("2026-09-04", "Audit: the evaluator scored ties as hits",
         "Collapsed embeddings earned perfect Recall@1; fixed with expected credit for ties, then 18 checkpoints re-scored.",
         ctx.doc("contrastive-learning-current-state.md"), "current-state audit"),
        ("2026-09-04", "Hardening", "Multilingual E5 text contract, versioned manifest, resumable runs, 8-arm matrix.",
         ctx.doc("contrastive-learning-implementation-status.md"), "implementation status"),
        (migrated, "Full-cohort manifest", f"{man.get('rows', '?'):,} studies migrated with unchanged SHA-256." if man else "",
         ctx.raw("full-manifest-migration.json"), "migration record"),
        ("2026-09-05", "Frozen-tower baseline", "Only projections and temperature trained; 20 epochs on an RTX 3070.",
         ctx.doc("full-projection-v1-results.md"), "results"),
        ("2026-09-08", "Exact reproduction", "The cleaned code reran the translated-cohort baseline bit-for-bit.",
         ctx.doc("cleanup-reproduction-20260908.md"), "reproduction"),
    ]
    items = "".join(
        f'<li data-series-id="tl-{i}"><time>{esc(d)}</time><b>{esc(t)}</b><span>{esc(s)}</span>'
        f'<a href="{href}">{esc(lk)}</a></li>' for i, (d, t, s, href, lk) in enumerate(steps))
    body = f"""<ol class="tl">{items}</ol>
<style>
#timeline .tl{{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
#timeline .tl li{{border:1px solid var(--border);border-top:3px solid var(--g-anchor);border-radius:8px;padding:8px 10px;background:var(--card);display:flex;flex-direction:column;gap:3px;font-size:12.5px}}
#timeline .tl li:last-child{{border-top-color:var(--g-lora)}}
#timeline .tl time{{font-family:var(--mono);font-size:11.5px;color:var(--muted)}}
#timeline .tl b{{color:var(--text-strong);font-size:13px}}
#timeline .tl span{{color:var(--text)}}
#timeline .tl a{{font-size:12px;margin-top:auto}}
</style>
<script>Charts.selection(document.getElementById('{sid}'));</script>"""
    caption = (f"Everything after this (data audits, adaptation, probes, submission) is on the following stages; "
               f"start with <a href=\"{ctx.page('data')}\">data &amp; integrity</a>.")
    notes = ("<ul>"
             f"<li><a href=\"{ctx.doc('full-projection-v1-handoff.md')}\">full-projection-v1-handoff.md</a> records how the "
             "baseline was unblocked on CUDA (superseded ops note).</li>"
             f"<li><a href=\"{ctx.raw('contrastive-learning.html')}\">contrastive-learning.html</a> is the hand-maintained project "
             "narrative (original report); it predates LoRA and the Kaggle submission.</li>"
             "<li>The implementation-status note was amended in place on 2026-09-11: patient independence is unverifiable, "
             "not merely unproven.</li></ul>")
    from datetime import date
    span = (date.fromisoformat(steps[-1][0]) - date.fromisoformat(steps[0][0])).days
    return section(sid, f"{span} days took the project from a 58-study pilot to a reproducible full-cohort baseline",
                   body, caption=caption, notes=notes)


# --------------------------------------------------------------------------- section 4: pilot + rescore


def _pilot_label(path: str, text_model: dict) -> tuple[str, str] | None:
    m = re.search(r"cv/(biomedbert|radbert)-fold-(\d)", path)
    if m:
        return (("BiomedBERT" if m.group(1) == "biomedbert" else "Stanford RadBERT") + f" fold {m.group(2)}",
                "bio" if m.group(1) == "biomedbert" else "rad")
    m = re.search(r"unfreezing/biomedbert-blocks-(\d)", path)
    if m:
        return f"BiomedBERT, {m.group(1)} blocks unfrozen", "hold"
    if "token-length/biomedbert-512" in path:
        return "BiomedBERT, 512 tokens", "hold"
    m = re.search(r"gold-translated-([a-z-]+)/", path)
    if m:
        names = {"biomedbert": "BiomedBERT (first pilot)", "radbert-va": "UCSD RadBERT-RoBERTa (first pilot)",
                 "stanford-radbert": "Stanford RadBERT (first pilot)"}
        return names.get(m.group(1), m.group(1)), "hold"
    return None


def pilot_section(ctx) -> tuple[str, dict, dict]:
    sid = "pilot"
    j = data.read_json("contrastive-checkpoint-rescore.json")
    if not j:
        return section(sid, "Pilot", notice("contrastive-checkpoint-rescore.json missing.")), {}, {}
    groups = {
        "bio": {"label": "5-fold CV, BiomedBERT", "color": "var(--g-teal)"},
        "rad": {"label": "5-fold CV, Stanford RadBERT", "color": "var(--g-violet)"},
        "hold": {"label": "Single 12-study holdout", "color": "var(--g-amber)"},
    }
    rows, skipped = [], []
    for r in j["results"]:
        lab = _pilot_label(r["checkpoint"], {})
        if not lab:
            skipped.append(f"{r['checkpoint'].split('/')[-2]} ({r['studies']} studies, epoch {r['checkpoint_epoch']})")
            continue
        chance = r["random_ranking_baseline"]["r@1"]
        lo, hi = r["bootstrap_95_ci"]["mean_r@1"]
        rows.append({"id": "p-" + re.sub(r"[^a-z0-9]+", "-", lab[0].lower()), "label": lab[0], "group": lab[1],
                     "est": r["metrics"]["mean_r@1"] - chance, "lo": lo - chance, "hi": hi - chance,
                     "raw": r["metrics"]["mean_r@1"], "chance": chance, "studies": r["studies"],
                     "epoch": r["checkpoint_epoch"],
                     "max_delta": max(abs(v) for v in r["metric_deltas"].values())})
    fold = {g: [r["raw"] for r in rows if r["group"] == g] for g in ("bio", "rad")}
    mean = {g: sum(v) / len(v) for g, v in fold.items() if v}
    reach = sum(1 for r in rows if r["lo"] <= 0)
    changed = j.get("checkpoints_with_metric_changes")
    count = j.get("checkpoint_count")
    claim = (f"Pilot folds favoured BiomedBERT ({_pct(mean.get('bio'), 1)} vs {_pct(mean.get('rad'), 1)} fold-mean R@1), "
             f"but {reach} of {len(rows)} checkpoint intervals reach chance on 11–12-study folds")
    blob = {"groups": groups, "rows": rows}
    body = f"""<div data-role="filter"></div>
{chart('ci', 'Mean R@1 minus that fold’s chance level, bootstrap 95% CI')}
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  var pct = Charts.fmt('.1%');
  Charts.ciDotPlot(sec.querySelector('[data-chart=ci]'), d.rows,
    {{groups: d.groups, ref: 0, refLabel: 'chance (1 / fold size)', fmt: '+.1%', axisFmt: '+.0%', sort: false,
      xLabel: 'Mean R@1 above chance', store: store, estLabel: 'Above chance', labelWidth: 250,
      tooltipExtra: function (r) {{ return '<div>R@1 ' + pct(r.raw) + ' · chance ' + pct(r.chance) + ' · ' + r.studies +
        ' studies · epoch ' + r.epoch + '</div><div>Re-score change: ' + r.max_delta + '</div>'; }}}});
}})();
</script>"""
    sess = data.load_curves("session-20260902") or {}
    pc = next(iter((sess.get("runs") or {}).values()), {}).get("config", {})
    caption = (f"{len(rows)} pilot checkpoints (frozen DINOv2-small + medical text encoder, 58 translated gold studies, CPU, "
               f"{pc.get('epochs', '?')} epochs, batch {pc.get('batch_size', '?')}). Each is shifted by its own chance level (1/12 or 1/11). Re-scored with the "
               f"tie-aware evaluator: <b>{changed} of {count}</b> stored scores changed.")
    tbl = table(["Checkpoint", "Group", "Studies", "Epoch", "Mean R@1", "Chance", "95% CI", "Re-score Δ"],
                [[r["label"], groups[r["group"]]["label"], r["studies"], r["epoch"], _pct(r["raw"], 1), _pct(r["chance"], 1),
                  f"[{_pct(r['lo'] + r['chance'], 1)}, {_pct(r['hi'] + r['chance'], 1)}]", r["max_delta"]] for r in rows],
                num_cols=(2, 3, 4, 5, 7), row_attrs=[{"data-series-id": r["id"], "data-group": r["group"]} for r in rows])
    notes = (
        "<ul>"
        f"<li>Source: <a href=\"{ctx.raw('contrastive-checkpoint-rescore.json')}\">contrastive-checkpoint-rescore.json</a> "
        "(per-checkpoint metrics, bootstrap CI, random-ranking baseline, deltas). Fold means are computed here from the "
        f"five fold checkpoints; the <a href=\"{ctx.doc('experiment-session-log.md')}\">session log</a> gives the same means with SDs.</li>"
        f"<li>Not charted: {esc('; '.join(skipped)) or 'none'} (the smoke checkpoint of the rescore set).</li>"
        f"<li>Tie fix: ties now earn uniform expected credit; four collapsed embeddings score R@1 0.25, not 1.0. "
        f"Queries with tied candidates in this set: {j.get('queries_with_tied_candidates')}. See "
        f"<a href=\"{ctx.doc('contrastive-learning-implementation-status.md')}\">implementation status</a>.</li>"
        "<li>The same validation fold chose and reported each best epoch, and the pilot's 58 studies are the gold set, "
        "so these are feasibility numbers only. The pilot's encoders (RadBERT, BiomedBERT) and 256-token limit were later "
        "replaced by multilingual E5 at 512 tokens.</li>"
        f"<li>Original report: <a href=\"{ctx.raw('translated-report-training.html')}\">translated-report-training.html</a> "
        "(contains a real MRI preview; local only, do not host publicly).</li>"
        "</ul>"
    )
    kp = {"changed": changed, "count": count}
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"Checkpoint table ({len(rows)} pilot checkpoints)"), {f"data-{sid}": blob}, kp


# --------------------------------------------------------------------------- section 5: baseline


def baseline_section(ctx) -> tuple[str, dict, dict]:
    sid = "baseline"
    c = data.load_curves(BASELINE_ROOT)
    if not c or BASELINE_RUN not in c.get("runs", {}):
        return section(sid, "Frozen-tower baseline", notice(
            f"Curve snapshot <code>{BASELINE_ROOT}</code> missing; run <code>build_digest_site.py export</code>. "
            f"See <a href=\"{ctx.doc('full-projection-v1-results.md')}\">the results note</a>.")), {}, {}
    run = c["runs"][BASELINE_RUN]
    pool = run.get("val_candidates")
    chance = 1 / pool if pool else None
    groups = {"r1": {"label": "Recall@1", "color": "var(--g-anchor)"},
              "sim": {"label": "Similarity", "color": "var(--g-sky)"}}
    r1 = [
        {"id": "bl-mean", "label": "Mean R@1", "group": "r1", "color": "var(--text-strong)",
         "points": data.curve_series(c, BASELINE_RUN, "mean_r@1")},
        {"id": "bl-i2t", "label": "Image → report R@1", "group": "r1", "color": "var(--g-teal)", "dash": "5 3",
         "points": data.curve_series(c, BASELINE_RUN, "image_to_text_r@1")},
        {"id": "bl-t2i", "label": "Report → image R@1", "group": "r1", "color": "var(--g-violet)", "dash": "2 3",
         "points": data.curve_series(c, BASELINE_RUN, "text_to_image_r@1")},
    ]
    sim = [
        {"id": "bl-pos", "label": "Paired similarity", "group": "sim", "color": "var(--g-lora)",
         "points": data.curve_series(c, BASELINE_RUN, "positive_similarity_mean")},
        {"id": "bl-neg", "label": "Unpaired similarity", "group": "sim", "color": "var(--g-rose)",
         "points": data.curve_series(c, BASELINE_RUN, "negative_similarity_mean")},
    ]
    pts = [p for p in r1[0]["points"] if p[1] is not None]
    best = max(pts, key=lambda p: (p[1], p[0])) if pts else (None, None)
    cfg = run.get("config", {})
    claim = (f"With both towers frozen, trained projections already retrieve {_pct(best[1])} mean R@1 over {pool} "
             f"candidates, {best[1] / chance:.0f}× random ({_pct(chance)})") if chance and best[1] else "Frozen-tower baseline"
    blob = {"r1": r1, "sim": sim, "groups": groups, "refs": [{"y": chance, "label": f"random {_pct(chance)}"}] if chance else []}
    body = f"""<div class="grid-2">
<div>{chart('r1', f'Validation Recall@1 by epoch ({pool} candidates)')}<div data-role="legend-r1"></div></div>
<div>{chart('sim', 'Mean cosine similarity, validation pairs')}<div data-role="legend-sim"></div></div>
</div>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  Charts.lineChart(sec.querySelector('[data-chart=r1]'), d.r1, {{groups: d.groups, yLabel: 'Recall@1', yFmt: '.1%',
    best: 'max', refs: d.refs, store: store}});
  Charts.legend(sec.querySelector('[data-role=legend-r1]'), d.r1, store, {{groups: d.groups, best: 'max', fmt: '.2%'}});
  Charts.lineChart(sec.querySelector('[data-chart=sim]'), d.sim, {{groups: d.groups, yLabel: 'Cosine similarity', yFmt: '.2f',
    best: 'last', store: store}});
  Charts.legend(sec.querySelector('[data-role=legend-sim]'), d.sim, store, {{groups: d.groups, best: 'last', fmt: '.3f'}});
}})();
</script>"""
    caption = (f"full-projection-v1: {cfg.get('image_model', '?')} + {cfg.get('text_model', '?')}, both frozen, multilingual "
               f"raw reports, {cfg.get('epochs')} epochs, batch {cfg.get('batch_size')}, seed {cfg.get('seed')}; single run, "
               f"validation-selected (epoch {best[0]}). Descriptive 95% CI {_pct(BASELINE_CI[0])}–{_pct(BASELINE_CI[1])}.")
    notes = (
        "<ul>"
        f"<li>Curves: snapshot of <code>{BASELINE_ROOT}/{BASELINE_RUN}</code> (<code>metrics.jsonl</code>, 5 significant figures). "
        f"The CI is quoted from <a href=\"{ctx.doc('full-projection-v1-results.md')}\">full-projection-v1-results.md</a> "
        "(per-study bootstrap; no JSON carries it).</li>"
        "<li>This is validation data used to pick the epoch, not a locked-test estimate, and it cannot account for repeated "
        "patients (patient identity is unrecoverable; see the data page).</li>"
        "<li>Train and validation loss are not drawn: they use different candidate counts (32 in-batch vs "
        f"{pool} full), so their gap is not a generalisation gap. Overfitting evidence is on the "
        f"<a href=\"{ctx.page('overfitting')}\">overfitting page</a>.</li>"
        f"<li>Original report: <a href=\"{ctx.raw('full-projection-v1-curves.html')}\">full-projection-v1-curves.html</a>. "
        f"The accompanying <a href=\"{ctx.doc('full-projection-v1-data-review.md')}\">data review</a> is superseded by the "
        "integrity audit on the data page.</li>"
        "</ul>"
    )
    kp = {"best": best[1], "pool": pool, "chance": chance}
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}, kp


# --------------------------------------------------------------------------- section 6: reproduction


def reproduction_section(ctx) -> tuple[str, dict, dict]:
    sid = "reproduction"
    j = data.read_json("cleanup-reproduction-20260908.json")
    if not j:
        return section(sid, "Reproduction", notice("cleanup-reproduction-20260908.json missing.")), {}, {}
    deltas = j.get("metric_deltas") or {}
    n_metrics = len(deltas)
    n_zero = sum(1 for v in deltas.values() if v == 0)
    checks = [
        ("Epoch history (all epochs)", j.get("history_exact")),
        ("Selected checkpoint weights", j.get("selected_weights_exact")),
        (f"{j.get('validation_query_rows', '?')} query rankings", j.get("rerun_rankings_exact")),
        ("Bootstrap intervals", j.get("confidence_intervals_exact")),
        ("Original reloads under clean code", j.get("original_reload_rankings_exact")),
        ("Original artifacts unchanged", j.get("original_artifacts_unchanged")),
    ]
    c_o = data.load_curves(ORIG_ROOT)
    c_r = data.load_curves(RERUN_ROOT)
    series = []
    if c_o and ORIG_RUN in c_o.get("runs", {}):
        series.append({"id": "rp-orig", "label": "Original run", "group": "run", "color": "var(--g-anchor)",
                       "points": data.curve_series(c_o, ORIG_RUN, "mean_r@1")})
    if c_r and RERUN_RUN in c_r.get("runs", {}):
        series.append({"id": "rp-rerun", "label": "Clean-code rerun", "group": "run", "color": "var(--g-lora)",
                       "dash": "6 4", "points": data.curve_series(c_r, RERUN_RUN, "mean_r@1")})
    groups = {"run": {"label": "Runs", "color": "var(--g-anchor)"}}
    om = j.get("original_metrics", {})
    all_ok = all(v is True for _, v in checks) and n_zero == n_metrics
    claim = (f"The cleaned code reproduced the translated-cohort baseline exactly: {n_zero} of {n_metrics} metrics "
             f"unchanged, identical weights and rankings" if all_ok else
             f"Cleanup reproduction: {n_zero} of {n_metrics} metrics unchanged (see checks)")
    ticks = "".join(
        f'<li class="{"ok" if v else "bad"}"><span>{"✓" if v else "✗"}</span>{esc(t)}</li>' for t, v in checks)
    body = f"""<div class="grid-2">
<div>{chart('r1', 'Validation mean R@1 by epoch, original vs rerun')}<div data-role="legend"></div></div>
<div><div class="chart-title">Exact-match checks (from the reproduction JSON)</div><ul class="checks">{ticks}
<li class="{'ok' if n_zero == n_metrics else 'bad'}"><span>{'✓' if n_zero == n_metrics else '✗'}</span>{n_zero} / {n_metrics} evaluation metrics Δ = 0</li></ul>
<p class="caption">Selected epoch {j.get('original_selected_epoch')} → {j.get('rerun_selected_epoch')}; mean R@1 {_pct(om.get('mean_r@1'), 4)};
{j['training_summary'].get('train_studies'):,} train / {j['training_summary'].get('validation_studies')} validation studies;
{j.get('training_seconds', 0) / 60:.1f} min training, {j.get('peak_allocated_gib', 0):.2f} GiB peak.</p></div>
</div>
<style>
#reproduction .checks{{list-style:none;margin:6px 0 0;padding:0;display:grid;gap:6px}}
#reproduction .checks li{{display:flex;gap:8px;align-items:center;font-size:13px;border:1px solid var(--border);border-radius:8px;padding:6px 10px;background:var(--card)}}
#reproduction .checks li span{{font-weight:800;width:18px;text-align:center}}
#reproduction .checks li.ok span{{color:var(--good)}}
#reproduction .checks li.bad span{{color:var(--bad)}}
</style>
<script>
(function () {{
  var sec = document.getElementById('{sid}');
  var d = Charts.readJSON('data-{sid}');
  var store = Charts.selection(sec);
  if (d.series.length) {{
    Charts.lineChart(sec.querySelector('[data-chart=r1]'), d.series, {{groups: d.groups, yLabel: 'Mean R@1', yFmt: '.1%',
      best: 'max', store: store}});
    Charts.legend(sec.querySelector('[data-role=legend]'), d.series, store, {{groups: d.groups, best: 'max', fmt: '.2%'}});
  }}
}})();
</script>"""
    caption = ("Translated-report cohort (claude-dfd07db), frozen DINOv2-small + mE5-base, seed 2026. The dashed rerun line "
               "sits exactly on the original, so only one is visible. This is a different run from the multilingual baseline above; "
               "they share 10.66% only because both reach the same count of hits.")
    notes = (
        "<ul>"
        f"<li>The bit-exact claim rests on the booleans and zero deltas in <a href=\"{ctx.raw('cleanup-reproduction-20260908.json')}\">"
        "cleanup-reproduction-20260908.json</a> (<code>torch.equal</code> weights, all 882 query records). The curve snapshots are "
        "rounded to 5 significant figures, so the overlay alone would only show a match to that precision.</li>"
        "<li>The rerun reproduces the original run's existing behaviour too, including its epoch-2 infinite gradient-norm entry.</li>"
        f"<li>Original experiment: <a href=\"{ctx.doc('claude-translated-dfd07db-comparison-results.md')}\">translated-report controlled comparison</a>; "
        f"write-up: <a href=\"{ctx.doc('cleanup-reproduction-20260908.md')}\">cleanup-reproduction-20260908.md</a>.</li>"
        "<li>Bit-exact reproduction held for this configuration; it does not make later single-seed comparisons noise-free. "
        f"Seed spread is discussed on the <a href=\"{ctx.page('overfitting')}\">overfitting page</a>.</li>"
        "</ul>"
    )
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": {"series": series, "groups": groups}}, \
        {"n_zero": n_zero, "n_metrics": n_metrics, "all_ok": all_ok}


# --------------------------------------------------------------------------- page


def build(ctx) -> Page:
    a, a_blob, a_kp = architecture_section(ctx)
    b, b_blob = budget_section(ctx)
    t = timeline_section(ctx)
    p, p_blob, p_kp = pilot_section(ctx)
    bl, bl_blob, bl_kp = baseline_section(ctx)
    r, r_blob, r_kp = reproduction_section(ctx)
    blobs = {}
    for x in (a_blob, b_blob, p_blob, bl_blob, r_blob):
        blobs.update(x)
    kpis = []
    if a_kp.get("trainable"):
        kpis.append({"label": "Submitted model, trainable", "value": _m(a_kp["trainable"]),
                     "sub": f"of {a_kp['total'] / 1e6:.1f}M ({_pct(a_kp['trainable'] / a_kp['total'])}); LoRA r64, 6 blocks"})
    if bl_kp.get("best"):
        kpis.append({"label": "Frozen baseline mean R@1", "value": _pct(bl_kp["best"]),
                     "sub": f"{bl_kp['pool']} candidates; random {_pct(bl_kp['chance'])}; single run"})
    if r_kp:
        kpis.append({"label": "Clean-code rerun", "value": f"{r_kp['n_zero']}/{r_kp['n_metrics']}",
                     "sub": "metrics identical (translated cohort)", "tone": "good" if r_kp.get("all_ok") else "bad"})
    if p_kp.get("count") is not None:
        kpis.append({"label": "Tie-aware re-score", "value": f"{p_kp['changed']}/{p_kp['count']}",
                     "sub": "stored checkpoint scores changed"})
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE, sections=[a, b, t, p, bl, r], data=blobs,
                kpis=kpis)


def summary() -> dict:
    prof = data.read_json("lora-memory-profile.json") or []
    r64 = next((r for r in prof if r.get("rank") == 64 and r.get("layer_count") == 6 and r.get("batch_size") == 32), None)
    value = label = None
    if r64:
        value = _pct(r64["trainable_parameters"] / r64["total_parameters"])
        label = (f"of parameters trained: {_m(r64['trainable_parameters'])} of {r64['total_parameters'] / 1e6:.1f}M "
                 f"(LoRA r64, last {r64['layer_count']} blocks of both towers)")
    return {"stage": BY_SLUG[SLUG].number, "headline": "Two pretrained towers, adapted with small LoRA adapters",
            "value": value, "value_label": label,
            "caveat": "Parameter count of the submitted encoder; retrieval baselines are further down the page",
            "href": "#architecture"}
