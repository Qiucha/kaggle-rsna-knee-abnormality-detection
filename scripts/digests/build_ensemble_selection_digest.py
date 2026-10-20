"""Export the ensemble-selection study (2026-10-09) to digest/ensemble-selection-v1.{json,md}.

Sources: .scratch/leaderboard-next/scratch/per-label-ensemble/ (per_label_ensemble_strategy.py, k5_coverage_vs_greedy.py,
ensemble_combo_sweep.py). This script recomputes per-label scores for each combination and the direct paired
comparisons; it re-reads the split-half and bootstrap results rather than re-running them.

This is a snapshot exporter: it needs the 2026-10-09 sweep outputs under .scratch/leaderboard-next (gitignored) and
the fine-tuned runs' predictions.npz under checkpoints/finetune; the committed JSON is the record. With only the 32
runs that existed on 2026-10-09 and the `generated` stamp pinned, a rerun reproduces it byte for byte; later runs
under checkpoints/finetune join the `singles` list.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.submission import LABELS  # predictions.npz column order

SRC = ROOT / ".scratch/leaderboard-next/scratch/per-label-ensemble"
FT = ROOT / "checkpoints/finetune"
DIGEST = ROOT / "digest"
REFERENCE = "u9"
PAIRS = [  # (A, B, question)
    (["u9", "clip", "384", "coat", "dS336"], ["u9", "clip", "384", "u6", "run5"], "5 members: diverse vs siblings"),
    (["u9", "clip", "coat", "dS336", "softR4a"], ["u9", "clip", "384", "u6", "run5"], "5: diverse (no 384) vs siblings"),
    (["u9", "clip", "384", "coat"], ["u9", "clip", "384", "u6"], "4th member: CoAtNet vs sibling u6"),
    (["u9", "clip", "coat", "dS336", "softR4a"], ["u9", "clip", "384", "coat"], "5 diverse vs 4 diverse"),
    (["u9", "clip", "coat", "dS336", "softR4a"], ["u9", "clip", "384", "coat", "dS336"], "Two diverse 5-member sets"),
]
RUNS = {  # short name -> run dir, as swept by ensemble_combo_sweep.py on 2026-10-09
    "u9": "r4a-k8-omnirad-base-soft-336-u9-b1", "clip": "r4a-k8-omnirad-base-soft-336-from-clip",
    "384": "r4a-k8-omnirad-base-soft-384", "u6": "r4a-k8-omnirad-base-soft-336-u6",
    "run5": "r4a-k8-omnirad-base-soft-336", "run5'": "r4a-k8-omnirad-base-soft-336-seed2027",
    "coat": "r4a-k8-coatnet-384-soft-s123", "dS336": "r4a-k8-dinov2-small-soft-336",
    "softR4a": "r4a-soft-224-k8-mean", "omb224": "r4a-k8-omnirad-base-pretrained",
    "omb224'": "r4a-k8-omnirad-base-pretrained-seed2027", "hard336": "r4a-k8-omnirad-base-hard-336",
}


def rk(a):
    return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])


def auc(p, t):
    n1 = t.sum(); n0 = len(t) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(p)[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ci(d):
    return {"difference": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_positive": float((d > 0).mean())}


def per_label(a):
    return {l: (None if np.isnan(v) else float(v)) for l, v in zip(LABELS, a)}


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ref = np.load(SRC / "lm-test-targets.npz")
    UIDS, Y, M = list(ref["uids"]), ref["y"] >= 0.5, ref["m"] > 0
    sweep = json.loads((SRC / "combo-sweep.json").read_text())

    all_runs, LMP, GP, GT = {}, {}, {}, None
    for d in sorted(p.parent for p in FT.glob("*/predictions.npz")):
        z = np.load(d / "predictions.npz")
        if list(z["lm_test_uids"]) != UIDS:
            continue
        LMP[d.name], GP[d.name] = rk(z["lm_test"]), rk(z["gold"]); GT = z["gold_targets"] >= 0.5
    L, n, ng = len(LABELS), len(UIDS), len(GT)

    def lm(pred, ix=None):
        ix = np.arange(n) if ix is None else ix
        return np.array([auc(pred[ix, j][M[ix, j]], Y[ix, j][M[ix, j]]) for j in range(L)])

    def gold(pred, ix=None):
        ix = np.arange(ng) if ix is None else ix
        return np.array([auc(pred[ix, j], GT[ix, j]) for j in range(L)])

    def ens(members, src):
        return np.mean([src[RUNS[m]] for m in members], axis=0)

    # 1. Single runs: the per-label landscape that "coverage" selection reads
    singles = []
    for name in LMP:
        a, g = lm(LMP[name]), gold(GP[name])
        singles.append({"run": name, "lm_test": {"macro": float(np.nanmean(a)), "per_label": per_label(a)},
                        "gold": {"macro": float(np.nanmean(g)), "per_label": per_label(g)}})
    singles.sort(key=lambda s: -s["lm_test"]["macro"])

    # 2. Combination sweep, with per-label scores recomputed
    ref_lm, ref_gold = lm(ens([REFERENCE], LMP)), gold(ens([REFERENCE], GP))
    combos = []
    for r in sweep:
        a, g = lm(ens(r["members"], LMP)), gold(ens(r["members"], GP))
        combos.append({
            "name": r["name"], "group": r["group"], "members": r["members"],
            "member_runs": [RUNS[m] for m in r["members"]],
            "lm_test": {"macro": r["lm"], "worst_label": r["worst"], "per_label": per_label(a),
                        "delta_per_label": per_label(a - ref_lm)},
            "gold": {"macro": r["gold"], "per_label": per_label(g), "delta_per_label": per_label(g - ref_gold)},
            "vs_reference": {"lm_test": {f"bootstrap_seed_{s}": {"difference": r[f"d{s}"][0], "ci95": r[f"d{s}"][1:]}
                                         for s in (101, 202, 303)},
                             "gold": {"difference": r["d_gold"][0], "ci95": r["d_gold"][1:]}},
            "vs_built_trio_lm_test": {"difference": r["d_prev"][0], "ci95": r["d_prev"][1:]},
            "member_seed_range": r["seed_variants"] or None,
        })

    # 3. Direct paired comparisons between designs
    pairs = []
    bs = np.random.default_rng(101).integers(0, n, (2000, n))
    gb = np.random.default_rng(101).integers(0, ng, (2000, ng))
    for a_m, b_m, q in PAIRS:
        pa, pb, ga, gbb = ens(a_m, LMP), ens(b_m, LMP), ens(a_m, GP), ens(b_m, GP)
        d = np.array([np.nanmean(lm(pa, ix)) - np.nanmean(lm(pb, ix)) for ix in bs])
        dg = np.array([np.nanmean(gold(ga, ix)) - np.nanmean(gold(gbb, ix)) for ix in gb])
        pairs.append({"question": q, "a": a_m, "b": b_m, "lm_test": ci(d), "gold": ci(dg)})

    # 4. Selection strategies judged out of sample (LM-test half-splits)
    summary = json.loads((SRC / "summary.json").read_text())
    strategies = [{"strategy": k.split("|")[0], "k": int(k.split("|")[1]), "held_out_macro": v["macro"],
                   "held_out_worst_label": v["worst"], "delta_vs_top_k": {"difference": v["delta"], "ci95": [v["lo"], v["hi"]]},
                   "p_beats_top_k": v["win"], "distinct_runs": v["runs"]} for k, v in summary.items()]
    k5_rows = {int(p.stem[7:]): json.loads(p.read_text()) for p in sorted(SRC.glob("k5-seed*.json"))}
    k5 = []
    for seed, rows in sorted(k5_rows.items()) + [("pooled", [r for rows in k5_rows.values() for r in rows])]:
        t, c, g = (np.array([r[k] for r in rows]) for k in ("top5", "coverage", "greedy"))
        k5.append({"split_seed": seed, "splits": len(rows), "top5": float(t.mean()), "coverage": float(c.mean()),
                   "greedy": float(g.mean()), "coverage_minus_greedy": ci(c - g), "coverage_minus_top5": ci(c - t),
                   "greedy_minus_top5": ci(g - t)})
    pooled = [r for rows in k5_rows.values() for r in rows]
    pick_rate = {k: {} for k in ("top5", "coverage", "greedy")}
    for k in pick_rate:
        for r in pooled:
            for m in r[f"{k}_members"]:
                pick_rate[k][m] = pick_rate[k].get(m, 0) + 1 / len(pooled)
    report = (SRC / "report.md").read_text()
    m = re.search(r"Per-label winners that are not the best-macro run: (\d+) cases over (\d+) splits\. Edge on the selection "
                  r"half ([+-][\d.]+); same runs on the held-out half ([+-][\d.]+) \(held-out edge > 0 in ([\d.]+); "
                  r"correlation ([+-][\d.]+)\)", report)
    assert m, "persistence line missing from report.md"
    persistence = {"cases": int(m.group(1)), "splits": int(m.group(2)), "selection_edge": float(m.group(3)),
                   "held_out_edge": float(m.group(4)), "held_out_positive": float(m.group(5)), "correlation": float(m.group(6))}

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"), "labels": LABELS, "reference": REFERENCE,
        "reference_run": RUNS[REFERENCE], "short_names": RUNS, "lm_test_studies": n, "gold_studies": ng,
        "ensemble": "per-label rank mean over members",
        "singles": singles, "combos": combos, "pairs": pairs,
        "selection": {"half_splits": 200, "strategies": strategies, "k5_seeds": k5, "k5_pick_rate": pick_rate,
                      "per_label_winner_persistence": persistence},
        "caveats": ["Combinations were hand-picked after the single-run LM-test scores were known; the split-half study "
                    "suggests about 0.004 shrinkage for in-sample picks.",
                    "Gold-58 is a guard (veto at -0.024) and tiebreaker only.",
                    "Member-seed range covers only members with a seed-2027 retrain (run 5, OmniRad-base 224)."],
    }
    (DIGEST / "ensemble-selection-v1.json").write_text(json.dumps(out, indent=1))

    f = lambda d: f"{d['difference']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
    md = [f"# Ensemble selection: how to choose members\n\nUpdated {out['generated']}\n",
          "All ensembles are per-label rank means of fine-tuned runs. LM-test is the criterion; gold-58 is the guard.\n",
          "## Hand-picked combinations vs u9 alone\n",
          "| Design | Combination | LM-test | Δ vs u9 [95% CI] | Gold-58 | Gold Δ vs u9 |", "|---|---|---:|---|---:|---|"]
    for c in combos:
        md.append(f"| {c['group']} | {c['name']} | {c['lm_test']['macro']:.4f} | "
                  f"{f(c['vs_reference']['lm_test']['bootstrap_seed_101'])} | {c['gold']['macro']:.4f} | {f(c['vs_reference']['gold'])} |")
    md += ["", "## Direct comparisons", "", "| Question | LM-test Δ (A − B) | P(A > B) | Gold-58 Δ |", "|---|---|---:|---|"]
    md += [f"| {p['question']} | {f(p['lm_test'])} | {p['lm_test']['p_positive']:.2f} | {f(p['gold'])} |" for p in pairs]
    md += ["", "## Selection rules, out of sample (k = 5, 5 split seeds x 200 half-splits)", "",
           "| Split seed | Top-5 | Coverage | Greedy | Coverage − greedy |", "|---|---:|---:|---:|---|"]
    md += [f"| {r['split_seed']} | {r['top5']:.4f} | {r['coverage']:.4f} | {r['greedy']:.4f} | {f(r['coverage_minus_greedy'])} |" for r in k5]
    md += ["", f"Per-label winners other than the best-macro run: edge {persistence['selection_edge']:+.4f} on the selection "
           f"half, {persistence['held_out_edge']:+.4f} held out (positive in {persistence['held_out_positive']:.0%}).",
           "", "[Submission page](site/pages/submission.html)", "[Machine-readable snapshot](ensemble-selection-v1.json)"]
    (DIGEST / "ensemble-selection-v1.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"combos": len(combos), "pairs": len(pairs), "singles": len(singles)}))


if __name__ == "__main__":
    main()
