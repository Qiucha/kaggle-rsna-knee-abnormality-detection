"""Hand-picked ensemble combinations, one design principle each, vs u9 alone (2026-10-09).

Paired bootstrap over LM-test studies (3 bootstrap seeds x 2000), gold-58 as guard, and member-seed swaps:
every member with a seed-2027 retrain is swapped for it to measure how much a combo depends on member seeds.
Rank-mean ensembles, same AUROC as per_label_ensemble_strategy.py.
"""
import itertools, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/q/Projects/novo-project")
OUT = ROOT / ".scratch/leaderboard-next/scratch/per-label-ensemble"
FT = ROOT / "checkpoints/finetune"
ref = np.load(OUT / "lm-test-targets.npz")
UIDS, Y, M = list(ref["uids"]), ref["y"] >= 0.5, ref["m"] > 0

RUNS = {  # short name -> run dir
    "u9": "r4a-k8-omnirad-base-soft-336-u9-b1", "clip": "r4a-k8-omnirad-base-soft-336-from-clip",
    "384": "r4a-k8-omnirad-base-soft-384", "u6": "r4a-k8-omnirad-base-soft-336-u6",
    "run5": "r4a-k8-omnirad-base-soft-336", "run5'": "r4a-k8-omnirad-base-soft-336-seed2027",
    "coat": "r4a-k8-coatnet-384-soft-s123", "dS336": "r4a-k8-dinov2-small-soft-336",
    "softR4a": "r4a-soft-224-k8-mean", "omb224": "r4a-k8-omnirad-base-pretrained",
    "omb224'": "r4a-k8-omnirad-base-pretrained-seed2027", "hard336": "r4a-k8-omnirad-base-hard-336",
}
SEED_TWIN = {"run5": "run5'", "omb224": "omb224'"}
COMBOS = {  # group, members
    "u9 alone": ("single", ["u9"]),
    "u9 + clip": ("sibling stack", ["u9", "clip"]),
    "u9 + clip + 384": ("sibling stack", ["u9", "clip", "384"]),
    "u9 + clip + 384 + u6": ("sibling stack", ["u9", "clip", "384", "u6"]),
    "u9 + clip + 384 + u6 + run5": ("sibling stack", ["u9", "clip", "384", "u6", "run5"]),
    "u9 + coat": ("backbone mix", ["u9", "coat"]),
    "u9 + dS336": ("backbone mix", ["u9", "dS336"]),
    "u9 + coat + dS336": ("backbone mix", ["u9", "coat", "dS336"]),
    "u9 + clip + 384 + coat": ("strong core + other backbone", ["u9", "clip", "384", "coat"]),
    "u9 + clip + 384 + dS336": ("strong core + other backbone", ["u9", "clip", "384", "dS336"]),
    "u9 + clip + 384 + softR4a": ("strong core + other backbone", ["u9", "clip", "384", "softR4a"]),
    "u9 + clip + 384 + coat + dS336": ("strong core + other backbone", ["u9", "clip", "384", "coat", "dS336"]),
    "u9 + clip + coat + dS336 + softR4a": ("strong core + other backbone", ["u9", "clip", "coat", "dS336", "softR4a"]),
    "u9 + 384 + omb224": ("input-size mix", ["u9", "384", "omb224"]),
    "u9 + hard336": ("label-type mix", ["u9", "hard336"]),
    "u9 + clip + hard336": ("label-type mix", ["u9", "clip", "hard336"]),
    "run5 + run5'": ("seed average", ["run5", "run5'"]),
    "u9 + run5 + run5'": ("seed average", ["u9", "run5", "run5'"]),
    "softR4a + omb224 + run5 (built, unsubmitted)": ("previous candidate", ["softR4a", "omb224", "run5"]),
    "softR4a + omb224 + u9": ("previous candidate", ["softR4a", "omb224", "u9"]),
}

P, G, GT = {}, {}, None
def rk(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
for k, d in RUNS.items():
    z = np.load(FT / d / "predictions.npz"); assert list(z["lm_test_uids"]) == UIDS
    P[k], G[k] = rk(z["lm_test"]), rk(z["gold"]); GT = z["gold_targets"] >= 0.5
L = Y.shape[1]

def auc(p, t):
    n1 = t.sum(); n0 = len(t) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(p)[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def lm(pred, ix): return np.array([auc(pred[ix, j][M[ix, j]], Y[ix, j][M[ix, j]]) for j in range(L)])
def gold(pred, ix): return np.array([auc(pred[ix, j], GT[ix, j]) for j in range(L)])
def ens(ms, src): return np.mean([src[m] for m in ms], axis=0)

n, ng = len(UIDS), len(GT)
SEEDS, B = (101, 202, 303), 2000
boots = {s: np.random.default_rng(s).integers(0, n, (B, n)) for s in SEEDS}
gboot = np.random.default_rng(7).integers(0, ng, (B, ng))
base_lm, base_g = ens(["u9"], P), ens(["u9"], G)
prev = ens(COMBOS["softR4a + omb224 + run5 (built, unsubmitted)"][1], P)
bl_base = {s: np.array([np.nanmean(lm(base_lm, ix)) for ix in boots[s]]) for s in SEEDS}
bl_prev = np.array([np.nanmean(lm(prev, ix)) for ix in boots[SEEDS[0]]])
gb_base = np.array([np.nanmean(gold(base_g, ix)) for ix in gboot])

def variants(ms):  # every combination of seed-twin swaps
    opts = [[m, SEED_TWIN[m]] if m in SEED_TWIN and SEED_TWIN[m] not in ms else [m] for m in ms]
    return [list(v) for v in itertools.product(*opts)]

rows = []
for name, (group, ms) in COMBOS.items():
    p, g = ens(ms, P), ens(ms, G); full = lm(p, np.arange(n))
    r = dict(name=name, group=group, members=ms, lm=float(np.nanmean(full)), worst=float(np.nanmin(full)),
             gold=float(np.nanmean(gold(g, np.arange(ng)))))
    for s in SEEDS:
        d = np.array([np.nanmean(lm(p, ix)) for ix in boots[s]]) - bl_base[s]
        r[f"d{s}"] = (float(d.mean()), *map(float, np.percentile(d, [2.5, 97.5])))
    dp = np.array([np.nanmean(lm(p, ix)) for ix in boots[SEEDS[0]]]) - bl_prev
    r["d_prev"] = (float(dp.mean()), *map(float, np.percentile(dp, [2.5, 97.5])))
    dg = np.array([np.nanmean(gold(g, ix)) for ix in gboot]) - gb_base
    r["d_gold"] = (float(dg.mean()), *map(float, np.percentile(dg, [2.5, 97.5])))
    vs = variants(ms)
    r["seed_variants"] = [float(np.nanmean(lm(ens(v, P), np.arange(n)))) for v in vs] if len(vs) > 1 else []
    rows.append(r); print(name, f"{r['lm']:.4f}", flush=True)

ci = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
lines = ["# Ensemble combination sweep (hand-picked, one principle each)", "",
         f"LM-test {n} studies; Δ vs u9 alone by paired bootstrap, {B} resamples per bootstrap seed ({', '.join(map(str, SEEDS))}).",
         "Gold-58 is the guard (veto at −0.024). Member-seed swing: LM-test range when run 5 / OmniRad-224 are swapped for their seed-2027 retrains.", "",
         "| Group | Combination | n | LM-test | Δ vs u9, seed 101 | Δ seed 202 | Δ seed 303 | Δ vs built trio | Gold-58 | Gold Δ vs u9 | Worst label | Member-seed swing |",
         "|---|---|---:|---:|---|---|---|---|---:|---|---:|---|"]
for r in rows:
    sw = f"{min(r['seed_variants']):.4f}–{max(r['seed_variants']):.4f}" if r["seed_variants"] else "—"
    lines.append(f"| {r['group']} | {r['name']} | {len(r['members'])} | {r['lm']:.4f} | {ci(r['d101'])} | {ci(r['d202'])} | "
                 f"{ci(r['d303'])} | {ci(r['d_prev'])} | {r['gold']:.4f} | {ci(r['d_gold'])} | {r['worst']:.4f} | {sw} |")
(OUT / "combo-sweep.md").write_text("\n".join(lines) + "\n")
(OUT / "combo-sweep.json").write_text(json.dumps(rows, indent=1))
print("\n".join(lines))
