"""Phase 1 · contrastive pre-training: where each mind-map box stands, then the evidence.

Lead card: the current recipe and one row per Phase-1 run the mind map tracks (factor-isolation arms,
plain-CLIP OmniRad, the combined runs), each with its retrieval selection metric and frozen-head scores.
Then the combined runs, factor isolation, the codec tower comparison and the candidate queue (all from encoders.py).
Sources: factor-isolation-v1.json, combined-lora-k8-336-v1.json, run5-followups-v1.json,
omnirad-clip-frozen-probe-v1.json.
"""

from __future__ import annotations

from .. import data
from ..shell import Page, esc, num, section, table
from ..sitemap import BY_SLUG
from . import encoders

SLUG = "phase1"
SUBTITLE = ("Train image–report contrastive encoders (CLIP loss over a 32 × 32 in-batch similarity matrix; LoRA on the "
            "last 6 blocks of both towers) and read each frozen image tower with LM-label MLP heads. Mean R@1 picks the "
            "checkpoint; the frozen heads judge it.")


def _row(box, run, change, r1, lm, gold, status, href=""):
    name = f'<a href="{href}">{esc(run)}</a>' if href else esc(run)
    return [esc(box), name, esc(change), num(r1), num(lm), num(gold), esc(status)]


def status_section(ctx) -> str:
    factor = data.read_json("factor-isolation-v1.json") or {}
    combined = data.read_json("combined-lora-k8-336-v1.json") or {}
    run5 = data.read_json("run5-followups-v1.json") or {}
    probe = data.read_json("omnirad-clip-frozen-probe-v1.json") or {}
    rows = []
    boxes = {"reference": "Reference", "dinov2-small-uniform-224": "Control",
             "dinov2-small-k8-224": "Multi-slice", "dinov2-small-336": "Resolution",
             "omnirad-base-lora-224": "Model swap"}
    for name, arm in factor.items():
        scored = arm["lm_test_mlp"] is not None
        rows.append(_row(boxes.get(name, ""), name, arm["purpose"], arm["contrastive"]["mean_r@1"], arm["lm_test_mlp"],
                         arm["gold_mlp"], "complete" if scored else "contrastive done · frozen head not scored",
                         "#factor-isolation-comparisons"))
    pre = run5.get("contrastive_pretraining") or {}
    mlp = ((probe.get("readout") or {}).get("mlp") or {})
    if pre:
        rows.append(_row("OmniRad + CLIP", "omnirad-clip-run5-v1", "OmniRad-base, plain CLIP, last 6 blocks unfrozen, K8 336",
                         pre.get("best_mean_recall_at_1"), (mlp.get("test") or {}).get("macro_auroc", {}).get("clip"),
                         (mlp.get("gold") or {}).get("macro_auroc", {}).get("clip"),
                         "complete" if pre.get("completed") else "running", ctx.page("picking") + "#frozen-probe"))
    for a in combined.get("arms", []):
        rows.append(_row("Combined", a["name"], f"{a['title']}, LoRA + K8 + 336", a["best_mean_recall_at_1"],
                         a["lm_test_mlp"], a["gold_mlp"],
                         "complete" if a["heads_status"] == "completed" else f"{a['status']} · heads {a['heads_status']}",
                         "#combined-lora-k8-336"))
    scored = [r for r in rows if r[4] != num(None)]
    body = ('<p>The mind map\'s Phase-1 boxes, one row each. Factor-isolation arms change one thing against the DINOv2-small '
            'codec reference (224 px, one stack per plane); the combined runs stack every factor. Frozen-head columns are '
            'MLP heads on the frozen image tower, three head seeds.</p>'
            + table(["Box", "Run", "Change", "Mean R@1", "LM-test MLP", "Gold-58 MLP", "Status"], rows,
                    raw_html=True, num_cols=(3, 4, 5)))
    notes = ("<p>Heads differ in protocol across rows: factor-isolation and combined arms use the lm-mlp head configs; the "
             "OmniRad plain-CLIP row is the frozen probe with the run-5 LM-head recipe. Compare within a block first. "
             "Mean R@1 is validation image–report retrieval and is only the checkpoint selector; whether it predicts "
             f'Phase-2 value is the question of the <a href="{ctx.page("picking")}">picking page</a>.</p>')
    return section("phase1-status", f"{len(scored)} of {len(rows)} Phase-1 runs have frozen-head scores", body, notes=notes)


def build(ctx) -> Page:
    factor_sections, blobs = encoders.factor_isolation_section(ctx)
    combined, combined_blob = encoders.combined_lora_section(ctx)
    blobs.update(combined_blob)
    codec, codec_blob, _ = encoders.codec_declip_section(ctx)
    nxt, next_blob = encoders.next_section(ctx)
    blobs.update(codec_blob)
    blobs.update(next_blob)
    page = encoders.build(ctx)  # for its KPI tiles only
    return Page(title=BY_SLUG[SLUG].title, subtitle=SUBTITLE,
                sections=[status_section(ctx), combined, factor_sections, codec, nxt], data=blobs, kpis=page.kpis)


def summary() -> dict:
    combined = data.read_json("combined-lora-k8-336-v1.json") or {}
    scored = [a for a in combined.get("arms", []) if a["lm_test_mlp"] is not None]
    if not scored:
        s = dict(encoders.summary())
        s["stage"] = BY_SLUG[SLUG].number
        return s
    best = max(scored, key=lambda a: a["lm_test_mlp"])
    return {"stage": BY_SLUG[SLUG].number, "headline": f"Combined {best['title']} leads the frozen heads",
            "value": num(best["lm_test_mlp"]), "value_label": "LM-test MLP AUROC, frozen tower",
            "caveat": f"Gold-58 {num(best['gold_mlp'])} (exploratory); one contrastive seed", "href": "#phase1-status"}
