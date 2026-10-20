"""Ensemble sweep with the CLIP-init u9 run (clipU9) swapped/added (2026-10-16).

Copy of ensemble_combo_sweep_a30.py; d_prev is now vs the submitted ens4 (A30+u9+clip+dS336).

Same method as ensemble_combo_sweep.py (rank-mean, paired LM-test bootstrap, gold-58 guard at -0.024),
but the baseline is A30 (best public single, 0.890) instead of u9, and the gold guard is judged against A30.
Adds a gold Lateral Meniscus column, since the shallow members dilute it (deep-unfreeze finding).
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/q/Projects/novo-project")
OUT = ROOT / ".scratch/leaderboard-next/scratch/per-label-ensemble"
FT = ROOT / "checkpoints/finetune"
ref = np.load(OUT / "lm-test-targets.npz")
UIDS, Y, M = list(ref["uids"]), ref["y"] >= 0.5, ref["m"] > 0
LATMEN = 3  # submission.LABELS order: "Lateral Meniscus"

RUNS = {
    "A30": "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30",
    "A30aug": "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-aug-ep30",
    "u9": "r4a-k8-omnirad-base-soft-336-u9-b1", "u12": "r4a-k8-omnirad-base-soft-336-u12-b1",
    "u6": "r4a-k8-omnirad-base-soft-336-u6", "clip": "r4a-k8-omnirad-base-soft-336-from-clip",
    "384": "r4a-k8-omnirad-base-soft-384", "coat": "r4a-k8-coatnet-384-soft-s123",
    "dS336": "r4a-k8-dinov2-small-soft-336", "softR4a": "r4a-soft-224-k8-mean",
    "clipU9": "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30-from-clip",
}
COMBOS = {
    "A30 alone": ("single", ["A30"]),
    "A30 + u9 + clip + dS336 (submitted ens4, 0.894)": ("submitted", ["A30", "u9", "clip", "dS336"]),
    "clip alone (shallow, u3)": ("single", ["clip"]),
    "clipU9 alone": ("single", ["clipU9"]),
    "A30 + clip": ("pair", ["A30", "clip"]),
    "A30 + clipU9": ("pair", ["A30", "clipU9"]),
    "A30 + u9 + clipU9": ("deep only", ["A30", "u9", "clipU9"]),
    "A30 + clipU9 + dS336": ("swap", ["A30", "clipU9", "dS336"]),
    "A30 + u9 + clipU9 + dS336 (ens4, clip -> clipU9)": ("swap", ["A30", "u9", "clipU9", "dS336"]),
    "A30 + u9 + clip + clipU9 + dS336": ("add", ["A30", "u9", "clip", "clipU9", "dS336"]),
    "A30 + u9 + clip + coat + dS336 (10-14 best 5)": ("5-member", ["A30", "u9", "clip", "coat", "dS336"]),
    "A30 + u9 + clipU9 + coat + dS336": ("5-member", ["A30", "u9", "clipU9", "coat", "dS336"]),
}
P, G, GT = {}, {}, None
def rk(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
for k, d in RUNS.items():
    z = np.load(FT / d / "predictions.npz"); assert list(z["lm_test_uids"]) == UIDS
    if GT is not None: assert (GT == (z["gold_targets"] >= 0.5)).all()
    P[k], G[k] = rk(z["lm_test"]), rk(z["gold"]); GT = z["gold_targets"] >= 0.5
L = Y.shape[1]

def auc(p, t):
    n1 = t.sum(); n0 = len(t) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(p)[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def lm(pred, ix): return np.array([auc(pred[ix, j][M[ix, j]], Y[ix, j][M[ix, j]]) for j in range(L)])
def gold(pred, ix): return np.array([auc(pred[ix, j], GT[ix, j]) for j in range(L)])
def ens(ms, src): return np.mean([src[m] for m in ms], axis=0)

n, ng = len(UIDS), len(GT)
SEEDS, B = (101, 202), 2000
boots = {s: np.random.default_rng(s).integers(0, n, (B, n)) for s in SEEDS}
gboot = np.random.default_rng(7).integers(0, ng, (B, ng))
base_lm, base_g = ens(["A30"], P), ens(["A30"], G)
prevbest = ens(COMBOS["A30 + u9 + clip + dS336 (submitted ens4, 0.894)"][1], P)
bl_base = {s: np.array([np.nanmean(lm(base_lm, ix)) for ix in boots[s]]) for s in SEEDS}
bl_prev = np.array([np.nanmean(lm(prevbest, ix)) for ix in boots[SEEDS[0]]])
gb_base = np.array([np.nanmean(gold(base_g, ix)) for ix in gboot])

rows = []
for name, (group, ms) in COMBOS.items():
    p, g = ens(ms, P), ens(ms, G); full = lm(p, np.arange(n)); gfull = gold(g, np.arange(ng))
    r = dict(name=name, group=group, members=ms, lm=float(np.nanmean(full)), worst=float(np.nanmin(full)),
             gold=float(np.nanmean(gfull)), gold_latmen=float(gfull[LATMEN]), lm_latmen=float(full[LATMEN]))
    for s in SEEDS:
        d = np.array([np.nanmean(lm(p, ix)) for ix in boots[s]]) - bl_base[s]
        r[f"d{s}"] = (float(d.mean()), *map(float, np.percentile(d, [2.5, 97.5])))
    dp = np.array([np.nanmean(lm(p, ix)) for ix in boots[SEEDS[0]]]) - bl_prev
    r["d_prev"] = (float(dp.mean()), *map(float, np.percentile(dp, [2.5, 97.5])))
    dg = np.array([np.nanmean(gold(g, ix)) for ix in gboot]) - gb_base
    r["d_gold"] = (float(dg.mean()), *map(float, np.percentile(dg, [2.5, 97.5])))
    rows.append(r); print(name, f"{r['lm']:.4f}", f"{r['gold']:.4f}", flush=True)

ci = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
lines = ["# Ensemble sweep with CLIP-init u9 (2026-10-16)", "",
         f"LM-test {n} studies; Δ vs A30 alone by paired bootstrap, {B} resamples per seed ({', '.join(map(str, SEEDS))}).",
         "Gold-58 guard: veto at Δ ≤ −0.024 vs A30. Rank-mean ensembles.", "",
         "| Group | Combination | n | LM-test | Δ vs A30 (seed 101) | Δ (seed 202) | Δ vs submitted ens4 | Gold-58 | Gold Δ vs A30 | Gold LatMen | LM LatMen | Worst label |",
         "|---|---|---:|---:|---|---|---|---:|---|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['group']} | {r['name']} | {len(r['members'])} | {r['lm']:.4f} | {ci(r['d101'])} | {ci(r['d202'])} | "
                 f"{ci(r['d_prev'])} | {r['gold']:.4f} | {ci(r['d_gold'])} | {r['gold_latmen']:.3f} | {r['lm_latmen']:.3f} | {r['worst']:.4f} |")
(OUT / "combo-sweep-clipu9.md").write_text("\n".join(lines) + "\n")
(OUT / "combo-sweep-clipu9.json").write_text(json.dumps(rows, indent=1))
print("\n".join(lines))
