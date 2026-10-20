"""Supervised fine-tuning ladder: R0 → R1–R3 → R1c → random slices → R4 multi-slice → K → backbones.

Every step is shown against the step it was built from (its "parent"), not the run before it, and
again against the fixed R0 baseline.
Sections: the step-by-step deltas, training curves, per-label heatmaps (gold-58 and LM-test,
Δ vs R0 or raw AUROC), and R4's window masking. Source: digest/finetune-ladder-v1.json, written by
scripts/digests/build_finetune_ladder_digest.py from checkpoints/finetune/.
"""

from __future__ import annotations

from .. import data
from ..shell import Page, caveat_badge, chart, esc, notice, num, section, table
from ..sitemap import BY_SLUG
from ._adaptation_sections import HELPERS
from ._encoders_util import section_script

SLUG = "finetune"
SRC = "finetune-ladder-v1.json"
DESIGN_MD = "finetune-r4-multislice-design.md"
BACKBONES_MD = "research/mri-pretrained-backbones.md"

GROUPS = {"baseline": {"label": "R0 baseline (frozen encoder)", "color": "var(--g-anchor)"},
          "ladder": {"label": "R1–R3, R1c: fine-tuning recipe", "color": "var(--g-violet)"},
          "slices": {"label": "Slice sampling and multi-slice (R4)", "color": "var(--g-teal)"},
          "seeds": {"label": "Seed replicates (noise floor)", "color": "var(--g-imgft)"},
          "labels": {"label": "Label targets: soft, then lateral-label fixes (ticket 18)", "color": "var(--g-rose)"},
          "resolution": {"label": "Resolution: 224 → 336 px", "color": "var(--g-bothft)"},
          "backbones": {"label": "Backbone swaps (R4a K8, pretrained start)", "color": "var(--g-amber)"},
          "combined": {"label": "Combined: OmniRad-base + soft + 336 px", "color": "var(--g-lime)"},
          "diagnostic": {"label": "Backbone diagnostic: CoAtNet-384 (ticket 16)", "color": "var(--g-sky)"},
          "followups": {"label": "Run-5 follow-ups", "color": "var(--g-sky)"}}

# The ladder JSON records steps in run order with coarse groups. Charts read them by concept instead: one group per
# question, groups in the order above, run order kept within a group. Unlisted steps keep their JSON group.
STEP_GROUP = {"R4a-K8-s27": "seeds", "OmniRad-B-s27": "seeds", "R4a-soft": "labels", "R4a-soft-unk": "labels",
              "R4a-soft-grouped": "labels", "DINOv2-S-336": "resolution"}

# Run-5 follow-ups grouped by the concept each one varies; members run from the smallest change to the largest,
# with run 5 itself as the implicit middle/zero degree of every concept. Unlisted runs fall into "followups".
RUN5_CONCEPTS = [
    ("r5-ref", "Run 5 and its seed replicate", "var(--g-lime)",
     ["r4a-k8-omnirad-base-soft-336", "r4a-k8-omnirad-base-soft-336-seed2027"]),
    ("r5-unfreeze", "Unfrozen image blocks: 1 → 3 (run 5) → 6 → 9 → 12", "var(--g-sky)",
     ["r4a-k8-omnirad-base-soft-336-u1", "r4a-k8-omnirad-base-soft-336-u6", "r4a-k8-omnirad-base-soft-336-u9",
      "r4a-k8-omnirad-base-soft-336-u12"]),
    ("r5-pretrain", "Contrastive pre-training first: none (run 5) → CLIP → CLIP + LoRA", "var(--g-teal)",
     ["r4a-k8-omnirad-base-soft-336-from-clip", "r4a-k8-omnirad-base-soft-336-from-clip-lora"]),
    ("r5-resolution", "Resolution: 336 px (run 5) → 384 px", "var(--g-amber)", ["r4a-k8-omnirad-base-soft-384"]),
    ("r5-labels", "Label targets: soft (run 5) → hard", "var(--g-rose)", ["r4a-k8-omnirad-base-hard-336"]),
    ("r5-epochs", "Epoch limit: run 5's → 30", "var(--g-violet)", ["r4a-k8-omnirad-base-soft-336-ep30"]),
    ("r5-backbone", "Backbone: OmniRad-base (run 5) → DINOv2-small → CoAtNet", "var(--g-holdout)",
     ["r4a-k8-dinov2-small-soft-336", "r4a-k8-coatnet-384-soft-s123"]),
]
RUN5_GROUPS = {key: {"label": label, "color": color} for key, label, color, _ in RUN5_CONCEPTS}
ALL_GROUPS = {**GROUPS, **RUN5_GROUPS}  # curve and heatmap sections serve both the ladder and the run-5 runs


def _ci(lo: float, hi: float) -> str:
    return f"[{num(lo, '.3f', True)}, {num(hi, '.3f', True)}]"


def _excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def _verdict(step: dict, bar_guard: float = -0.024) -> str:
    """The ticket-03 bar applied to this step against its parent."""
    v = step.get("vs_parent")
    if not v:
        return "baseline"
    lm, gold = v["lm_test"], v["gold"]
    guard = gold["difference"] >= bar_guard
    if _excludes_zero(*lm["ci95"]) and lm["difference"] > 0:
        return "passes" if guard else "fails gold guard"
    if _excludes_zero(*lm["ci95"]):
        return "worse"
    return "no clear change"


def _load():
    j = data.read_json(SRC)
    if not j:
        return j, {}
    order = list(GROUPS)
    steps = [{**s, "group": STEP_GROUP.get(s["id"], s["group"])} for s in j["steps"]]
    steps.sort(key=lambda s: order.index(s["group"]) if s["group"] in order else len(order))  # stable: run order kept
    j = {**j, "steps": steps}
    return j, {s["id"]: s for s in steps}


def _scored_runs(j):
    snapshot = data.read_json("run5-followups-v1.json") or {}
    return j["steps"] + [{"short": run["title"], "lm_test": {"macro": run["lm_test"]},
                          "gold": {"macro": run["gold"]}} for run in snapshot.get("completed", [])]


def _run5_snapshot() -> dict | None:
    """The run-5 follow-up snapshot with `completed` reordered by concept and each run tagged with its group."""
    snapshot = data.read_json("run5-followups-v1.json")
    if not snapshot:
        return None
    rank = {name: (gi, ni, key) for gi, (key, _, _, names) in enumerate(RUN5_CONCEPTS) for ni, name in enumerate(names)}
    runs = [{**run, "group": rank.get(run["name"], (0, 0, "followups"))[2]} for run in snapshot["completed"]]
    runs.sort(key=lambda run: rank.get(run["name"], (len(RUN5_CONCEPTS), 0))[:2])
    return {**snapshot, "completed": runs}


def _run5_groups(snapshot, extra=()) -> dict:
    used = {run["group"] for run in snapshot["completed"]} | set(extra)
    return {k: v for k, v in ALL_GROUPS.items() if k in used}


def _promising(snapshot) -> list[str]:
    """Run 5 plus, per concept group, the follow-up with the highest LM-test Δ vs run 5 when that Δ is positive
    (interval crossing zero or not). These start highlighted on the run-5 charts."""
    lm = snapshot.get("paired", {}).get("lm_test", {})
    best = {}
    for run in snapshot.get("completed", []):
        delta = (lm.get(run["name"]) or {}).get("difference")
        if run["name"] != snapshot.get("baseline") and delta is not None and delta > 0 \
                and delta > best.get(run["group"], (None, 0))[1]:
            best[run["group"]] = (run["name"], delta)
    keep = {snapshot.get("baseline")} | {name for name, _ in best.values()}
    return [run["name"] for run in snapshot.get("completed", []) if run["name"] in keep]


def run5_followups_section(ctx) -> str:
    snapshot = _run5_snapshot()
    if not snapshot:
        return ""
    completed = snapshot["completed"]
    rows = []
    for run in completed:
        deltas = []
        for split in ("lm_test", "gold"):
            delta = snapshot["paired"][split].get(run["name"])
            deltas.append(f"{num(delta['difference'], '.4f', True)} {_ci(*delta['ci95'])}" if delta else "baseline")
        rows.append([run["title"], num(run["lm_test"]), deltas[0], num(run["gold"]), deltas[1], run["best_epoch"]])
    score_table = table(["Completed run", "LM-test", "Δ vs run 5 [95% CI]", "Gold-58",
                         "Δ vs run 5 [95% CI]", "Best epoch"], rows, num_cols=(1, 3, 5))
    remaining = snapshot["unfinished"]
    queue_table = table(["Unfinished run", "Status", "Completed epochs"],
        [[run["title"], run["status"], f"{run['epochs_completed']}/{run['max_epochs']}"] for run in remaining])
    pre = snapshot["contrastive_pretraining"]
    retrieval = (f"K8 plain CLIP pre-training completed {pre['epochs_completed']}/{pre['max_epochs']} epochs. "
                 f"It selected epoch {pre['selected_epoch']} with mean retrieval Recall@1 "
                 f"{num(pre['best_mean_recall_at_1'])}. " if pre["completed"] else "K8 plain CLIP pre-training is unfinished. ")
    noise = snapshot["absolute_lm_test_seed_difference"]
    noise_text = (f"The measured absolute seed difference is <strong>{num(noise, '.6f')}</strong> LM-test AUROC. "
                  if noise is not None else "The seed-noise comparison is pending. ")
    confirmation = snapshot.get("contrastive_confirmation")
    if confirmation:
        verdict = "awaits the seed comparison" if confirmation["passed"] is None else "passes" if confirmation["passed"] else "does not pass"
        verdict_text = f"The completed contrastive confirmation <strong>{verdict}</strong> the predeclared pass rule."
    else:
        verdict_text = "The contrastive confirmation has no final pathology score until its supervised follow-up finishes."
    body = (f"<p>Updated {esc(snapshot['updated_at'])}. {len(completed) - 1} follow-up runs have final scores.</p>"
            + score_table + f"<p>{noise_text}{verdict_text}</p><p>{retrieval}"
            "Retrieval Recall@1 measures image–report matching; it is separate from pathology AUROC.</p>"
            + (queue_table if remaining else "<p>All queued runs are complete.</p>"))
    notes = (f"<p>{esc(snapshot['note'])} Paired intervals use 2,000 shared study bootstrap resamples, seed 2030. "
             "The contrastive pass rule requires a positive LM-test interval, a gold drop smaller than 0.024, "
             "and an LM-test gain larger than the seed difference. Interrupted supervised runs restart "
             "their original recipe because this trainer saves weights only at completion.</p>"
             f'<p>Sources: <a href="{ctx.raw("run5-followups-v1.json")}">result snapshot</a>, '
             f'<a href="{ctx.doc("run5-followups-v1.md")}">results and queue</a>, '
             f'<a href="{ctx.doc("omnirad-clip-run5-v1.md")}">CLIP protocol</a>.</p>')
    return section("run5-followups", "Run-5 follow-ups: completed scores and remaining work", body, notes=notes)


def run5_comparisons(ctx):
    snapshot = _run5_snapshot()
    if not snapshot:
        return "", {}
    sid = "run5-comparisons"
    blob = {"groups": _run5_groups(snapshot, extra=("baseline",)),
            "defaultPinned": _promising(snapshot)}
    for split in ("lm_test", "gold"):
        blob[split + "_level"] = [{"id": "R0", "label": "R0 frozen encoder", "group": "baseline",
            "value": snapshot["r0"][split]}] + [{"id": run["name"], "label": run["title"], "group": run["group"],
            "value": run[split]} for run in snapshot["completed"]]
        for comparator, values in (("run5", snapshot["paired"]), ("r0", snapshot["paired_vs_r0"])):
            blob[split + "_" + comparator] = [{"id": run["name"], "label": run["title"], "group": run["group"],
                "est": values[split][run["name"]]["difference"],
                "lo": values[split][run["name"]]["ci95"][0], "hi": values[split][run["name"]]["ci95"][1]}
                for run in snapshot["completed"] if run["name"] in values[split]]
    plots = '<div class="grid-2">' + ''.join('<div>' + chart(key, title) + '</div>' for key, title in (
        ("lm_test_level", "LM-test macro AUROC (432 studies)"), ("gold_level", "Gold-58 macro AUROC"),
        ("lm_test_run5", "Δ LM-test vs run 5, paired 95% CI"), ("gold_run5", "Δ gold-58 vs run 5, paired 95% CI"),
        ("lm_test_r0", "Δ LM-test vs R0 frozen baseline, paired 95% CI"),
        ("gold_r0", "Δ gold-58 vs R0 frozen baseline, paired 95% CI"))) + '</div>'
    body = plots + section_script(sid, """
    store.set({pinned: d.defaultPinned});
    ['lm_test_level', 'gold_level'].forEach(function (key) {
      Charts.barChart(sec.querySelector('[data-chart=' + key + ']'), d[key],
        {groups: d.groups, fmt: '.4f', axisFmt: '.2f', store: store, labelWidth: 245});
    });
    ['lm_test_run5', 'gold_run5', 'lm_test_r0', 'gold_r0'].forEach(function (key) {
      Charts.ciDotPlot(sec.querySelector('[data-chart=' + key + ']'), d[key],
        {groups: d.groups, ref: 0, refLabel: 'no difference', sort: false,
         fmt: '+.4f', axisFmt: '+.3f', xLabel: 'Candidate minus reference (95% CI)', store: store, labelWidth: 245});
    });""")
    rows = []
    for run in snapshot["completed"]:
        deltas = []
        for values in (snapshot["paired"], snapshot["paired_vs_r0"]):
            for split in ("lm_test", "gold"):
                d = values[split].get(run["name"])
                deltas.append(f"{num(d['difference'], '.4f', True)} {_ci(*d['ci95'])}" if d else "baseline")
        rows.append([run["title"], *deltas])
    tbl = table(["Run", "Δ LM-test vs run 5", "Δ gold vs run 5", "Δ LM-test vs R0", "Δ gold vs R0"], rows)
    return section(sid, "Run-5 follow-ups: absolute scores and paired reference deltas", body,
        caption="Run 5 is the matched soft-336 supervised baseline; R0 is the fixed frozen contrastive encoder with LM MLP heads. "
                "Highlighted on load: run 5 and, per concept, the follow-up with the highest LM-test Δ vs run 5 if it is positive "
                "(interval may cross zero). "
                "Paired study bootstrap: 2,000 resamples, seed 2030. Gold-58 is exploratory; LM-test scores LM-verdict agreement.",
        table_html=tbl, table_summary="All reference deltas and confidence intervals"), {"data-" + sid: blob}


def followup_graphs(ctx, j, by_id):
    snapshot = _run5_snapshot() or {}
    completed = snapshot.get("completed", [])
    steps = [by_id["R0"]]
    for run in completed:
        steps.append({"id": run["name"], "short": run["title"], "label": run["title"], "group": run["group"],
                      "history": run["history"],
                      **{split: {"macro": run[split], "per_label": run["per_label"][split]}
                         for split in ("gold", "lm_test")}})
    graph_data = {**j, "steps": steps}
    sections, blobs = [], {}
    root = "omnirad-clip-run5-v1"
    run = "omnirad-base-e5-base-clip-k8-336-u6-bs32-seed2026"
    curves = data.load_curves(root)
    metrics = {"train": "train_loss", "val": "validation_loss",
               "itot": "image_to_text_r@1", "ttoi": "text_to_image_r@1"}
    phase1 = {"groups": {"clip": {"label": "Plain CLIP · K8 · 336 px", "color": "var(--g-teal)"}}}
    combined = data.read_json("combined-lora-k8-336-v1.json") or {}
    lora = next((arm for arm in combined.get("arms", []) if arm["name"] == "omnirad-base-lora-k8-336"), None)
    if lora:
        phase1["groups"]["lora"] = {"label": "CLIP + LoRA · K8 · 336 px", "color": "var(--g-violet)"}
    for key, metric in metrics.items():
        phase1[key] = [{"id": "clip", "label": "Plain CLIP · K8 · 336 px", "group": "clip",
                        "points": data.curve_series(curves, run, metric)}]
        if lora:
            phase1[key].append({"id": "lora", "label": "CLIP + LoRA · K8 · 336 px", "group": "lora",
                                "points": [[row["epoch"], row[metric]] for row in lora["history"] if metric in row]})
    plots = "".join('<div>' + chart(key, title) + '</div>' for key, title in (
        ("train", "Phase 1 training in-batch loss"), ("val", "Phase 1 validation full-pool loss"),
        ("itot", "Phase 1 image → text Recall@1"), ("ttoi", "Phase 1 text → image Recall@1")))
    body = '<div class="grid-2">' + plots + '</div>' + section_script("run5-phase1", """
    [['train', 'Training InfoNCE', '.3f', 'min'], ['val', 'Validation InfoNCE', '.3f', 'min'],
     ['itot', 'Image to text R@1', '.1%', 'max'], ['ttoi', 'Text to image R@1', '.1%', 'max']].forEach(function (spec) {
      Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
        {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
    });""")
    pre = snapshot.get("contrastive_pretraining", {})
    caption = (f"Plain CLIP: {pre.get('epochs_completed', '?')}/{pre.get('max_epochs', '?')} epochs; "
               f"selected epoch {pre.get('selected_epoch', '?')} by mean retrieval R@1. ")
    if lora:
        caption += (f"CLIP + LoRA r64/a128, K8, 336 px: {lora['epochs_completed']}/{lora['max_epochs']} epochs; "
                    f"selected epoch {lora['selected_epoch']} with mean retrieval R@1 "
                    f"{num(lora['best_mean_recall_at_1'])}. ")
        followup = next((r for r in completed if r["name"] == "r4a-k8-omnirad-base-soft-336-from-clip-lora"), None)
        if followup:
            caption += (f"Its supervised fine-tune scores LM-test AUROC {num(followup['lm_test'])} "
                        f"and gold-58 AUROC {num(followup['gold'])}; selected fine-tuning epoch {followup['best_epoch']}. ")
    caption += ("Training is augmented in-batch loss; validation uses the full pool. "
                "Retrieval is separate from pathology AUROC.")
    notes = (f'<p>LoRA run: <a href="{ctx.doc("combined-lora-k8-336-v1.md")}">pre-training and frozen-head results</a>. '
             '<a href="#run5-followups">Supervised follow-up scores</a> and '
             '<a href="#run5-curves">fine-tuning curves</a> are shown separately below.</p>') if lora else ""
    sections.append(section("run5-phase1", "Phase 1: OmniRad K8 CLIP pre-training — plain and LoRA + 336", body,
                            caption=caption, notes=notes))
    blobs["data-run5-phase1"] = phase1
    for html, payload in (curves_section(ctx, graph_data, sid="run5-curves", default_pinned=_promising(snapshot)),
                          per_label_section(ctx, graph_data, {s["id"]: s for s in steps}, sid="run5-per-label",
                                            extra_reference=snapshot["baseline"])):
        sections.append(html)
        blobs.update(payload)
    return sections, blobs


# --------------------------------------------------------------------------- 1. steps


def _combined_note(by_id) -> str:
    """The stacked OmniRad-base + soft + 336 px run, which changes two things at once."""
    s = by_id.get("OmniRad-B-soft-336")
    if not s:
        return ""
    lm, gold = s["vs_parent"]["lm_test"], s["vs_parent"]["gold"]
    return ("<li><strong>OmniRad-B soft 336 (2026-10-02)</strong> stacks soft labels and 336 px on OmniRad-base. "
            f"It scores LM-test {num(s['lm_test']['macro'])} and gold-58 {num(s['gold']['macro'])}; "
            f"against OmniRad-base the LM-test gain is {num(lm['difference'], '.4f', True)} "
            f"{_ci(*lm['ci95'])}, which does not pass the bar on its own; gold {num(gold['difference'], '.3f', True)} "
            f"{_ci(*gold['ci95'])}. Submitted alone (2026-10-03) it scored "
            + (f"<strong>{num(s['public']['score'])} public</strong>, the best at the time and +0.040 over OmniRad-base's 0.839: "
               "far more than either local measure predicted (+0.007 LM-test, +0.013 gold). " if s.get("public") else "")
            + "It is also a member of the proposed 3-member ensemble (ticket 14).</li>")


def _coatnet_note(by_id) -> str:
    """Ticket 16: does swapping in CoAtNet-384 close the gap to the 0.924 public single model?"""
    s = by_id.get("CoAtNet-384")
    if not s:
        return ""
    lm, gold = s["vs_parent"]["lm_test"], s["vs_parent"]["gold"]
    r0 = s["vs_r0"]["lm_test"]
    epoch = s["best_epoch"]
    train = {k: next(h["train_masked_bce"] for h in by_id[k]["history"] if h["epoch"] == epoch)
             for k in ("CoAtNet-384", "OmniRad-B-soft-336")}
    return ("<li><strong>CoAtNet 384 (2026-10-02, ticket 16)</strong> asked whether the backbone explains the gap to "
            "a reported 0.924 public CoAtNet single model; a gain of about +0.05 LM-test would have said yes. It lands "
            f"the other way: {num(lm['difference'], '.3f', True)} {_ci(*lm['ci95'])} LM-test and "
            f"{num(gold['difference'], '.3f', True)} {_ci(*gold['ci95'])} gold against run 5 (OmniRad-B soft 336), "
            f"and {num(r0['difference'], '.3f', True)} {_ci(*r0['ci95'])} LM-test against the frozen R0. "
            "That rules out the plain swap, not the backbone. On the same soft targets its training BCE at epoch "
            f"{epoch} is {num(train['CoAtNet-384'], '.3f')} against run 5's {num(train['OmniRad-B-soft-336'], '.3f')}, "
            "so it underfits: only 3 of 24 blocks train, and stages 0–2 keep ImageNet features. The 0.924 reference "
            "fine-tuned every block for 16 epochs with top-3 weight averaging, on private labels. A deeper-unfreeze "
            "arm is queued.</li>")


def _lateral_note(by_id) -> str:
    """Ticket 18: Lateral Meniscus and Lateral OA are the two weakest gold-58 labels in every run."""
    arms = [by_id[k] for k in ("R4a-soft-unk", "R4a-soft-grouped") if k in by_id]
    if not arms:
        return ""
    parent = by_id["R4a-soft"]

    def lat(s):
        g = s["gold"]["per_label"]
        return f"Lateral Meniscus {num(g['Lateral Meniscus'])}, Lateral OA {num(g['Lateral OA'])}"
    rows = "; ".join(f"{esc(s['short'])}: {lat(s)} (macro LM-test {num(s['vs_parent']['lm_test']['difference'], '.4f', True)} "
                     f"{_ci(*s['vs_parent']['lm_test']['ci95'])})" for s in arms)
    return ("<li><strong>Lateral-label fixes (2026-10-03, ticket 18).</strong> Lateral Meniscus and Lateral OA are the two "
            "lowest gold-58 labels averaged over every run, while LM-test rates them as average. The LM labels disagree with "
            "gold on these two labels more than on others, and they tie Lateral OA to Medial OA (correlation 0.60 against 0.32 "
            "in gold), which the heads copy. "
            f"Parent R4a soft: {lat(parent)}. {rows}. Neither arm moves either lateral label beyond noise, and the "
            "Lateral OA head still ranks gold Medial OA and PF OA above its own label, so the lever left is the label "
            "source (ticket 18, option 2). Per-label paired intervals: "
            "<code>checkpoints/finetune/lateral-fixes-table.json</code>.</li>")


def _label_group_note(j, by_id) -> str:
    """One bullet that reads the soft-label, seed and 336 px steps against their parents."""
    def d(sid, split, fmt=".3f"):
        v = by_id[sid]["vs_parent"][split]
        lo, hi = v["ci95"]
        return f"{num(v['difference'], fmt, True)} [{num(lo, fmt, True)}, {num(hi, fmt, True)}]"
    needed = ("R4a-soft", "R4a-K8-s27", "OmniRad-B-s27", "DINOv2-S-336")
    if not all(k in by_id for k in needed):
        return ""
    return ("<li><strong>Soft labels, seeds and 336 px (2026-10-01).</strong> "
            f"Soft labels leave LM-test flat ({d('R4a-soft', 'lm_test')}) but lift gold-58 {d('R4a-soft', 'gold')}, "
            f"and move the best epoch from {by_id['R4a-K8']['best_epoch']} to {by_id['R4a-soft']['best_epoch']}. "
            f"The two seed-2027 replicates differ from seed 2026 by {d('R4a-K8-s27', 'lm_test')} (R4a K8) and "
            f"{d('OmniRad-B-s27', 'lm_test')} (OmniRad-B) on LM-test, so seed spread on this recipe is small. "
            f"336 px passes the bar by a hair: LM-test {d('DINOv2-S-336', 'lm_test', '.4f')}, gold {d('DINOv2-S-336', 'gold')}, "
            f"at {by_id['DINOv2-S-336']['minutes']:.0f} training minutes against {by_id['DINOv2-S']['minutes']:.0f}.</li>")


def steps_section(ctx, j, by_id) -> tuple[str, dict]:
    sid = "steps"
    rows = [s for s in j["steps"] if s.get("vs_parent")]
    blob = {"groups": {k: v for k, v in GROUPS.items() if k in {s["group"] for s in rows}},
            "lm": [{"id": s["id"], "label": f"{s['short']} vs {by_id[s['parent']]['short']}", "group": s["group"],
                    "est": s["vs_parent"]["lm_test"]["difference"], "lo": s["vs_parent"]["lm_test"]["ci95"][0],
                    "hi": s["vs_parent"]["lm_test"]["ci95"][1], "level": s["lm_test"]["macro"]} for s in rows],
            "gold": [{"id": s["id"], "label": f"{s['short']} vs {by_id[s['parent']]['short']}", "group": s["group"],
                      "est": s["vs_parent"]["gold"]["difference"], "lo": s["vs_parent"]["gold"]["ci95"][0],
                      "hi": s["vs_parent"]["gold"]["ci95"][1], "level": s["gold"]["macro"]} for s in rows]}
    # The same steps against one fixed baseline (R0), so their gains stack up on a common zero.
    for split, key in (("lm_test", "lm_r0"), ("gold", "gold_r0")):
        blob[key] = [{"id": s["id"], "label": s["short"], "group": s["group"],
                      "est": s["vs_r0"][split]["difference"], "lo": s["vs_r0"][split]["ci95"][0],
                      "hi": s["vs_r0"][split]["ci95"][1], "level": s[split]["macro"]} for s in rows]
    body = (HELPERS + '<div data-role="filter"></div><div class="grid-2">'
            f'<div>{chart("lm", "Δ LM-test macro AUROC vs the step it changes")}</div>'
            f'<div>{chart("gold", "Δ gold-58 macro AUROC vs the step it changes")}</div></div>'
            '<div class="grid-2">'
            f'<div>{chart("lm_r0", "Δ LM-test macro AUROC vs R0 (fixed frozen baseline)")}</div>'
            f'<div>{chart("gold_r0", "Δ gold-58 macro AUROC vs R0 (fixed frozen baseline)")}</div></div>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  [['lm', 'LM-test', 'the step it changes'], ['gold', 'gold-58', 'the step it changes'],
   ['lm_r0', 'LM-test', 'R0'], ['gold_r0', 'gold-58', 'R0']].forEach(function (spec) {
    Charts.ciDotPlot(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, ref: 0, sort: false, fmt: '+.3f', axisFmt: '+.2f',
       xLabel: '\\u0394 ' + spec[1] + ' macro AUROC vs ' + spec[2] + ' (95% CI)', store: store,
       tooltipExtra: function (r) { return '<div>Level ' + Charts.fmt('.4f')(r.level) + '</div>'; }});
  });"""))

    trs, attrs = [], []
    for s in j["steps"]:
        v = s.get("vs_parent") or {}
        lm, gold = v.get("lm_test"), v.get("gold")
        trs.append([f"<strong>{esc(s['short'])}</strong>", esc(s["change"]),
                    esc(by_id[s["parent"]]["short"]) if s.get("parent") else "—",
                    num(s["lm_test"]["macro"]),
                    f"{num(lm['difference'], '.4f', True)} {_ci(*lm['ci95'])}" if lm else "—",
                    num(s["gold"]["macro"]),
                    f"{num(gold['difference'], '.4f', True)} {_ci(*gold['ci95'])}" if gold else "—",
                    esc(_verdict(s)), str(s.get("best_epoch", "—")),
                    num(s["public"]["score"], ".3f") if s.get("public") else "—"])
        attrs.append({"data-series-id": s["id"], "data-group": s["group"]})
    tbl = table(["Step", "What it changes from its parent", "Parent", "LM-test", "Δ LM-test [95% CI]",
                 "Gold-58", "Δ gold [95% CI]", "Ticket-03 bar", "Best epoch",
                 "Public LB"],
                trs, raw_html=True, num_cols=(3, 5, 8, 9), row_attrs=attrs)

    best = max(j["steps"], key=lambda s: s["lm_test"]["macro"])
    r4 = by_id.get("R4a-K8")
    claim = (f"Multi-slice input (R4a) is the largest single gain, "
             f"{num(r4['vs_parent']['lm_test']['difference'], '.3f', True)} LM-test over random-slice R1c; "
             f"fine-tuning alone (R1–R3) lost to the frozen baseline"
             + " " + caveat_badge("single seed per step"))
    caption = (f"Top row: each dot is one step minus its parent (the run it was built from), so each row has its own zero. "
               f"Bottom row: the same steps minus R0, the frozen-encoder baseline, so every dot shares one zero and "
               f"the gains read cumulatively. Both on {j['lm_test_studies']} LM-test "
               f"and {j['gold_studies']} gold studies. {esc(j['bootstrap'])} The full table says what each step changes. "
               f"Best LM-test within this original ladder: {esc(best['short'])} ({num(best['lm_test']['macro'])}).")
    also = [s for s in j["steps"] if s.get("vs_also")]
    notes = ("<ul>"
             f"<li><strong>Bar.</strong> {esc(j['bar'])} A step only passes against its own parent, so a backbone swap "
             "is judged against the DINOv2-small arm with the same pretrained start, not against R4a.</li>"
             "<li><strong>Parents are not run order.</strong> R1c changes R1 (start from R0's encoder), not R3; "
             "the three backbone swaps and the 336 px run each change the pretrained-DINOv2 arm.</li>"
             + "".join(f"<li><strong>{esc(s['short'])} vs {esc(by_id[s['also']]['short'])}:</strong> LM-test "
                       f"{num(s['vs_also']['lm_test']['difference'], '.4f', True)} {_ci(*s['vs_also']['lm_test']['ci95'])}, "
                       f"gold {num(s['vs_also']['gold']['difference'], '.4f', True)} {_ci(*s['vs_also']['gold']['ci95'])}.</li>"
                       for s in also)
             + _label_group_note(j, by_id)
             + _combined_note(by_id)
             + _coatnet_note(by_id)
             + _lateral_note(by_id)
             + "<li>Gold-58 has been examined many times and rests on few positives per label: treat it as exploratory. "
               "LM-test scores agreement with LM-generated labels, not radiologists, and the two disagree on OmniRad-small.</li>"
             "<li>Every step is one training seed (2026), except the seed-2027 replicates, which repeat their parent "
             "exactly. A gap under ~0.05 needs a 3-seed rerun before it is trusted.</li>"
             "<li><strong>Public LB</strong> is the Kaggle public macro ROC-AUC for the steps that were submitted. "
             "OmniRad-base tied R4a on LM-test but led on gold-58, and it also led on the public board.</li>"
             f"<li>Sources: <a href=\"{ctx.raw(SRC)}\">{SRC}</a>, "
             f"<a href=\"{ctx.doc(DESIGN_MD)}\">R4 design</a>, <a href=\"{ctx.doc(BACKBONES_MD)}\">backbone survey</a>.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes, table_html=tbl,
                   table_summary=f"All {len(j['steps'])} steps: what changes, scores and deltas"), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 2. curves


# Line styles for the 1st, 2nd, ... run of a group: shades of one hue are hard to tell apart in dark mode.
DASHES = [None, "7 4", "2 3", "10 3 2 3", "4 4"]


def curves_section(ctx, j, sid="curves", default_pinned=None) -> tuple[str, dict]:
    series = {"train": [], "val": [], "auc": []}
    in_group = {}
    for s in j["steps"]:
        h = s.get("history")
        if not h:
            continue
        dash = DASHES[in_group.setdefault(s["group"], 0) % len(DASHES)]
        in_group[s["group"]] += 1
        for key, metric in (("train", "train_masked_bce"), ("val", "validation_masked_bce"),
                            ("auc", "validation_macro_auroc")):
            series[key].append({"id": s["id"], "label": s["label"], "group": s["group"], "dash": dash,
                                "points": [[r["epoch"], r[metric]] for r in h]})
    used_groups = {s["group"] for s in j["steps"] if s.get("history")}
    blob = {"groups": {k: v for k, v in ALL_GROUPS.items() if k in used_groups}, **series,
            "defaultPinned": default_pinned or []}
    plots = "".join(f"<div>{chart(k, t)}</div>" for k, t in (
        ("train", "Training masked BCE by epoch"), ("val", "Validation masked BCE by epoch (selects the epoch)"),
        ("auc", "Validation macro AUROC by epoch")))
    body = (HELPERS + '<div data-role="filter"></div>'
            f'<div class="grid-2">{plots}</div><div data-role="legend"></div>'
            '<p class="legend-hint">Click lines or legend entries to highlight runs in all three charts; click several '
            'to compare them, click a highlighted run again to drop it, or click empty plot area to clear.'
            + (' Highlighted on load: run 5 and, per concept, the follow-up with the highest LM-test Δ vs run 5 if it is '
               'positive (interval may cross zero).'
               if default_pinned else '') + '</p>'
            + section_script(sid, """
  Charts.filterBar(sec.querySelector('[data-role=filter]'), d.groups, store);
  store.set({pinned: d.defaultPinned});
  [['train', 'Training BCE', '.3f', 'min'], ['val', 'Validation BCE', '.3f', 'min'],
   ['auc', 'Validation AUROC', '.3f', 'max']].forEach(function (spec) {
    Charts.lineChart(sec.querySelector('[data-chart=' + spec[0] + ']'), d[spec[0]],
      {groups: d.groups, yLabel: spec[1], yFmt: spec[2], best: spec[3], store: store});
  });
  Charts.legend(sec.querySelector('[data-role=legend]'), d.val, store, {groups: d.groups, best: 'min', fmt: '.3f'});"""))
    mins = {}
    for s in j["steps"]:
        if s.get("history"):
            b = min(s["history"], key=lambda r: r["validation_masked_bce"])
            mins[s["id"]] = (b["epoch"], b["validation_masked_bce"])
    chain = [k for k in ("R1c", "R1c-rand", "R4a-K8") if k in mins]
    claim = ("Every rung overfits after its validation minimum; along R1c → random slices → R4a the minimum moves later "
             "and lower (" + ", ".join(f"epoch {mins[k][0]} at {num(mins[k][1], '.3f')}" for k in chain)
             + "), but it does not go away")
    if sid != "curves":
        claim = "Run-5 label fine-tuning: training loss, validation loss and validation AUROC"
    caption = ("Validation BCE is against hard LM verdicts for every rung (soft-label rungs included), and its minimum "
               "picks the epoch; training stops 5 epochs after it. R0 has no fine-tuning curve. "
               "These curves are on the 426-study selection split, so they show training behaviour, not a ranking: "
               "rank runs by LM-test (432 held-out studies) in the paired comparisons. Example: u6 has the lowest "
               "validation BCE of the unfreeze sweep (0.425 vs u9's 0.431), yet u9 leads on LM-test (0.883 vs 0.878), "
               "gold-58 (0.877 vs 0.862) and the public board (0.888).")
    notes = ("<ul><li>Training BCE is the masked macro BCE on each rung's own target (soft values for R2–R3), so compare "
             "it within a group; validation BCE is on one yardstick for all.</li>"
             "<li>R4 epochs read 8–16× more slices per study; the x axis is epochs, not time.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 3. per-label heatmaps


def per_label_section(ctx, j, by_id, sid="per-label", extra_reference=None) -> tuple[str, dict]:
    labels = j["labels"]
    r0 = by_id["R0"]
    steps = [s for s in j["steps"] if s["id"] != "R0"]
    blob = {"rows": labels, "cols": [s["short"] for s in steps], "colIds": [s["id"] for s in steps],
            "colGroups": [s["group"] for s in steps], "baseCol": r0["short"], "groups": ALL_GROUPS}
    for split in ("gold", "lm_test"):
        blob[split] = {
            "level": [[r0[split]["per_label"].get(l)] + [s[split]["per_label"].get(l) for s in steps] for l in labels],
            "delta": [[None] + [(s[split]["per_label"][l] - r0[split]["per_label"][l])
                                if l in s[split]["per_label"] and l in r0[split]["per_label"] else None
                                for s in steps] for l in labels]}
    blob["defaultMode"] = "reference" if extra_reference else "delta"
    blob["modeChoices"] = [{"key": "delta", "label": "Δ vs R0"}, {"key": "level", "label": "AUROC"}]
    if extra_reference:
        reference = by_id[extra_reference]
        blob["referenceCol"] = 1 + next(i for i, step in enumerate(steps) if step["id"] == extra_reference)
        blob["modeChoices"].insert(0, {"key": "reference", "label": "Δ vs run 5"})
        for split in ("gold", "lm_test"):
            blob[split]["reference"] = [[None] + [step[split]["per_label"].get(label) - reference[split]["per_label"][label]
                if step[split]["per_label"].get(label) is not None and reference[split]["per_label"].get(label) is not None else None
                for step in steps] for label in labels]
    body = (HELPERS + '<div data-role="split"></div><div data-role="mode"></div>'
            + '<div data-chart-scope>' + chart("heat", "Per-label AUROC by step") + '</div>'
            + section_script(sid, """
  var split = 'gold', mode = d.defaultMode;
  function draw() {
    var cols = [d.baseCol].concat(d.cols), m = d[split][mode];
    var keep = cols.map(function (c, j) { return j; }).filter(function (j) { return (mode === 'level' || j > 0) && !(mode === 'reference' && j === d.referenceCol); });
    var el = AdaptUI.remount(sec.querySelector('[data-chart=heat]'));
    var o = {rows: d.rows, cols: keep.map(function (j) { return cols[j]; }), rowIds: d.rows,
             fmt: mode === 'level' ? '.3f' : '+.3f',
             ariaLabel: 'per-label ' + (split === 'gold' ? 'gold-58' : 'LM-test') + ' AUROC heatmap'};
    if (mode !== 'level') { o.diverging = true; o.peakPerRow = false; }
    Charts.heatmap(el, m.map(function (row) { return keep.map(function (j) { return row[j]; }); }), o);
  }
  AdaptUI.seg(sec.querySelector('[data-role=split]'), 'Scored on',
    [{key: 'gold', label: 'Gold-58 (human)'}, {key: 'lm_test', label: 'LM-test (432)'}], 'gold',
    function (k) { split = k; draw(); });
  AdaptUI.seg(sec.querySelector('[data-role=mode]'), 'Cells',
    d.modeChoices, d.defaultMode, function (k) { mode = k; draw(); });
  draw();"""))

    def gain(split, step_id):
        s = by_id.get(step_id)
        if not s:
            return None
        ds = [s[split]["per_label"][l] - r0[split]["per_label"][l] for l in labels
              if l in s[split]["per_label"] and l in r0[split]["per_label"]]
        return sum(d > 0 for d in ds), len(ds)

    up = gain("lm_test", "R4a-K8")
    claim = (f"R4a lifts LM-test AUROC over R0 on {up[0]} of {up[1]} labels; per-label gold-58 moves are large and noisy"
             if up else "Per-label AUROC by step") + " " + caveat_badge("few gold positives per label")
    caption = ("Rows are labels; columns are steps in ladder order. Cells default to the step's AUROC minus R0's on the same "
               "label (the frozen-encoder baseline); switch to raw AUROC to see R0's column. Toggle between gold-58 and LM-test.")
    if extra_reference:
        claim = "Run-5 fine-tuning: per-label differences vs run 5 or R0, and absolute AUROC"
        caption = "Reference = run 5, soft targets at 336 px, seed 2026. Cells default to candidate minus run 5. " \
                  "Switch to Δ vs R0 for the fixed frozen baseline, or AUROC for raw scores. " \
                  "Both gold-58 and LM-test are available; per-label differences are point estimates, not intervals."
    notes = ("<ul><li>R0's per-label scores come from its 3-seed MLP heads: gold from <code>benchmark.json</code>, LM-test "
             "from <code>lm-proxy.json</code>, the same predictions the paired comparisons use.</li>"
             "<li>Per-label LM-test AUROC rests on each label's observed (non-UNK) verdicts, which vary widely by label.</li></ul>")
    return section(sid, claim, body, caption=caption, notes=notes), {f"data-{sid}": blob}


# --------------------------------------------------------------------------- 4. windows


def windows_section(ctx, j) -> tuple[str, dict]:
    sid = "windows"
    trs = []
    for s in j["steps"]:
        wc = s.get("window_counts")
        if not wc:
            continue
        for split in ("train", "validation", "lm_test", "gold"):
            c = wc.get(split)
            if c:
                planes = c["studies"] * 3
                trs.append([s["short"], split, str(c["studies"]), f"{c['masked_windows']:,} / {c['windows']:,}",
                            num(c["masked_windows"] / c["windows"], ".1%"),
                            f"{c['short_planes']} / {planes}", str(c["masked_planes"])])
    tbl = table(["Step", "Split", "Studies", "Masked windows", "Share", "Short planes", "Missing planes"],
                trs, num_cols=(2, 4, 6))
    k16 = next((s for s in j["steps"] if s["id"] == "R4a-K16"), None)
    share = (k16["window_counts"]["lm_test"]["masked_windows"] / k16["window_counts"]["lm_test"]["windows"]
             if k16 and k16.get("window_counts") else None)
    claim = ("At 8 windows no plane needs masking; at 16 about a fifth of planes are short stacks"
             + (f" and {num(share, '.1%')} of LM-test windows are masked" if share is not None else "")
             + "; no plane is ever fully missing")
    caption = ("A window is masked when its depth stratum holds no slice. A masked window gets zero weight in pooling "
               "and attention, instead of repeating a neighbour.")
    return section(sid, claim, tbl, caption=caption), {}


# --------------------------------------------------------------------------- page


def build(ctx) -> Page:
    j, by_id = _load()
    info = BY_SLUG[SLUG]
    if not j:
        return Page(title=info.title, subtitle="",
                    sections=[section("missing", "Fine-tuning ladder", notice(f"<code>{SRC}</code> is missing."))])
    followups = run5_followups_section(ctx)
    graphs, blobs = followup_graphs(ctx, j, by_id)
    comparisons, comparison_blob = run5_comparisons(ctx)
    blobs.update(comparison_blob)
    sections = ([followups] if followups else []) + ([comparisons] if comparisons else []) + graphs
    for s, b in (steps_section(ctx, j, by_id), curves_section(ctx, j), per_label_section(ctx, j, by_id),
                 windows_section(ctx, j)):
        sections.append(s)
        blobs.update(b)
    scored = _scored_runs(j)
    r0, best = by_id["R0"], max(scored, key=lambda s: s["lm_test"]["macro"])
    best_gold = max(scored, key=lambda s: s["gold"]["macro"])
    r4 = by_id.get("R4a-K8")
    kpis = [{"label": "Best LM-test", "value": num(best["lm_test"]["macro"]),
             "sub": f"{esc(best['short'])} · R0 {num(r0['lm_test']['macro'])}"},
            {"label": "Best gold-58", "value": num(best_gold["gold"]["macro"]),
             "sub": f"{esc(best_gold['short'])} · exploratory"}]
    if r4:
        d = r4["vs_parent"]["lm_test"]
        kpis.append({"label": "Multi-slice step (R4a)", "value": num(d["difference"], ".3f", True),
                     "sub": f"LM-test vs random-slice R1c {_ci(*d['ci95'])}", "tone": "good"})
    return Page(title=info.title,
                subtitle="Fine-tuning the image encoder on LM labels, one change per step, each against the step it builds on.",
                sections=sections, data=blobs, kpis=kpis)


def summary() -> dict:
    j, by_id = _load()
    info = BY_SLUG[SLUG]
    if not j:
        return {"stage": info.number, "headline": None, "value": None, "value_label": None, "caveat": None, "href": None}
    best = max(_scored_runs(j), key=lambda s: s["lm_test"]["macro"])
    return {"stage": info.number, "headline": f"Multi-slice fine-tuning ({best['short']}) leads",
            "value": num(best["lm_test"]["macro"]), "value_label": "LM-test macro AUROC",
            "caveat": "Gold-58 exploratory; run-5 follow-ups include its seed replicate", "href": "#run5-followups"}
