"""Recent completed runs and submissions, read from the same snapshots as their detailed cards."""

from .. import data
from ..shell import esc, num, section, table


def best_single():
    j = data.read_json("finetune-clip-init-u9-v1.json") or {}
    points = j.get("point", {}).get("lm_test", {})
    runs = j.get("runs", [])
    if not runs:
        return None
    run = max(runs, key=lambda r: points[r["key"]])
    return {"name": run["key"], "lm": points[run["key"]],
            "gold": j["point"]["gold"][run["key"]]}


def build(ctx):
    rows = []
    specs = [
        ("finetune-clip-init-u9-v1.json", "A30", "A30 reference", "lateral-rebalance", "Best single LM-test; longer-training controls remain paused"),
        ("finetune-image-augmentation-v1.json", "oversample+noise ep30 aug", "A30 + image augmentation", "image-augmentation", "No clear gain over A30"),
        ("finetune-slice-band-v1.json", "oversample+noise ep30 band05", "A30 + wide slice band", "slice-band", "No clear gain over A30"),
        ("finetune-clip-init-u9-v1.json", "CLIP-init u9", "A30 recipe + CLIP init", "clip-init-u9", "No clear solo gain; contributes to the best public ensemble"),
    ]
    for source, key, title, anchor, outcome in specs:
        j = data.read_json(source) or {}
        run = next((r for r in j.get("runs", []) if r["key"] == key), None)
        if run:
            rows.append([f'<a href="#{anchor}">{esc(title)}</a>', num(j["point"]["lm_test"][key]),
                         num(j["point"]["gold"][key]), f'{run["epochs_completed"]} / {run["best_epoch"]}',
                         "—", esc(outcome)])
    history = data.read_json("leaderboard-history-v1.json") or {}
    recent = {"r4a-k8-omb-soft-336-u9-a30": "A30 reference", "a30-ens4-rank": "First A30 ensemble",
              "a30-ens4-swap-rank": "Swap: CLIP-init u9 replaces u3", "a30-deep3-rank": "Deep three: remove both shallow members"}
    for r in history.get("submissions", []):
        if r["bundle"] not in recent:
            continue
        if r["bundle"] == "r4a-k8-omb-soft-336-u9-a30":
            if rows:
                rows[0][4] = num(r["public"], ".3f")
            continue
        href = ctx.page("submission") + "#leaderboard"
        rows.append([f'<a href="{href}">{esc(recent[r["bundle"]])}</a>', num(r["local_lm_test"]),
                     num(r["local_gold58"]), "Member checkpoints",
                     num(r["public"], ".3f"), f'Complete · {esc(r["recorded"])} · submission {r["submission"]}'])
    if not rows:
        return ""
    probes = data.read_json("a30-inference-probes-v1.json") or {}
    body = ('<style>#latest-results table{min-width:1000px}</style>'
            '<p>The completed training runs below each have training BCE, validation BCE, validation AUROC and '
            'per-label results in their linked sections. Ensembles reuse the selected member checkpoints.</p>'
            + table(["Completed run / submission", "LM-test", "Gold-58", "Epochs run / selected", "Public", "Result"],
                    rows, raw_html=True, num_cols=(1, 2, 4))
            + f'<p>Also complete: <a href="#a30-ensemble-sweep">17 A30 ensemble combinations</a> and '
            '<a href="#clip-init-ensemble-sweep">12 CLIP-init sweep combinations</a>, plus '
            f'<a href="#a30-inference-probes">{len(probes.get("variants", []))} inference-only variants</a> '
            '(checkpoint averaging, weight soups and window-jitter TTA; no clear gain). '
            'The 30- and 43-epoch controls and A50 remain paused; they have no completed result.</p>')
    return section("latest-results", "Latest completed results · 2026-10-17", body,
                   notes='<p>LM-test: 432 studies with LM labels. Gold-58: exploratory human labels. Public: '
                         'Kaggle hidden-test public split. These populations are reported separately; '
                         'public-split uncertainty is not estimated. An unsubmitted run has no public score.</p>')
