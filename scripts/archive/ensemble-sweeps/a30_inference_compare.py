"""Paired comparison of the A30 inference probes against A30 (2026-10-14).

TTA variants average probabilities over the centre pass and the first k jitter passes.
Same AUROC and bootstrap conventions as ensemble_combo_sweep_a30.py.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/q/Projects/novo-project")
OUT = ROOT / "checkpoints/finetune/a30-inference-probes"
ref = np.load(ROOT / ".scratch/leaderboard-next/scratch/per-label-ensemble/lm-test-targets.npz")
UIDS, Y, M = list(ref["uids"]), ref["y"] >= 0.5, ref["m"] > 0
LATMEN = 3

def load(name):
    z = np.load(OUT / f"{name}.npz"); assert list(z["lm_test_uids"]) == UIDS
    return z["lm_test"], z["gold"], z["gold_targets"] >= 0.5

base_lm, base_g, GT = load("a30-best")
V = {k: load(k)[:2] for k in ("a30-best", "a30-last", "a30-avg-best-last", "soup-a30-a30aug", "soup-a30-u9")}
jit = sorted(p.stem for p in OUT.glob("a30-jitter-*.npz"))
J = [load(k)[:2] for k in jit]
for k in range(1, len(J) + 1):
    V[f"tta centre + {k} jitter"] = (np.mean([base_lm] + [j[0] for j in J[:k]], 0), np.mean([base_g] + [j[1] for j in J[:k]], 0))
if J:
    V[f"jitter only ({len(J)} passes)"] = (np.mean([j[0] for j in J], 0), np.mean([j[1] for j in J], 0))
    V["single jitter pass (mean of passes' AUROC shown via pass 0)"] = J[0]

def auc(p, t):
    n1 = t.sum(); n0 = len(t) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(p)[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
lm = lambda p, ix: np.array([auc(p[ix, j][M[ix, j]], Y[ix, j][M[ix, j]]) for j in range(Y.shape[1])])
gold = lambda p, ix: np.array([auc(p[ix, j], GT[ix, j]) for j in range(GT.shape[1])])

n, ng, B = len(UIDS), len(GT), 2000
boot = np.random.default_rng(101).integers(0, n, (B, n)); gboot = np.random.default_rng(7).integers(0, ng, (B, ng))
bl = np.array([np.nanmean(lm(base_lm, ix)) for ix in boot]); bg = np.array([np.nanmean(gold(base_g, ix)) for ix in gboot])
ci = lambda d: f"{d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
rows, lines = [], ["# A30 inference-only probes (2026-10-14)", "",
    f"Paired bootstrap vs A30 best (centre windows), LM-test {n} / gold-58, {B} resamples. Gold veto at −0.024.", "",
    "| Variant | LM-test | Δ vs A30 [95% CI] | Gold-58 | Gold Δ vs A30 [95% CI] | Gold LatMen |", "|---|---:|---|---:|---|---:|"]
for name, (p, g) in V.items():
    dl = np.array([np.nanmean(lm(p, ix)) for ix in boot]) - bl
    dg = np.array([np.nanmean(gold(g, ix)) for ix in gboot]) - bg
    full, gfull = lm(p, np.arange(n)), gold(g, np.arange(ng))
    rows.append(dict(variant=name, lm=float(np.nanmean(full)), d_lm=[float(dl.mean()), *np.percentile(dl, [2.5, 97.5]).tolist()],
                     gold=float(np.nanmean(gfull)), d_gold=[float(dg.mean()), *np.percentile(dg, [2.5, 97.5]).tolist()],
                     gold_latmen=float(gfull[LATMEN])))
    lines.append(f"| {name} | {np.nanmean(full):.4f} | {ci(dl)} | {np.nanmean(gfull):.4f} | {ci(dg)} | {gfull[LATMEN]:.3f} |")
(OUT / "compare.md").write_text("\n".join(lines) + "\n"); (OUT / "compare.json").write_text(json.dumps(rows, indent=1))
print("\n".join(lines))
