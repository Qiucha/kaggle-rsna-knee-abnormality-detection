"""k=5: coverage (mean of per-label best) vs greedy ensemble vs top-5, repeated over split seeds (2026-10-09).

Usage: k5_coverage_vs_greedy.py SEED [SPLITS]. Same candidates, ranks and AUROC as per_label_ensemble_strategy.py.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/q/Projects/novo-project")
OUT = ROOT / ".scratch/leaderboard-next/scratch/per-label-ensemble"
FT = ROOT / "checkpoints/finetune"
SEED, B, K = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200, 5

ref = np.load(OUT / "lm-test-targets.npz")
UIDS, Y, M = list(ref["uids"]), ref["y"] >= 0.5, ref["m"] > 0
runs, LM = [], []
for d in sorted(p.parent for p in FT.glob("*/predictions.npz")):
    z = np.load(d / "predictions.npz")
    if list(z["lm_test_uids"]) == UIDS: runs.append(d.name); LM.append(z["lm_test"])
LM = np.array(LM); R, L = LM.shape[0], LM.shape[2]
LMR = np.stack([np.column_stack([rankdata(x[:, j]) / len(x) for j in range(L)]) for x in LM])


def auc(p, t):
    n1 = t.sum(); n0 = len(t) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(p)[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_label(P, ix):
    return np.array([auc(P[ix, j][M[ix, j]], Y[ix, j][M[ix, j]]) for j in range(L)])


def macro(S, ix): return np.nanmean(per_label(LMR[list(S)].mean(0), ix))


def pick(A, ix):
    m = A.mean(1); top = list(np.argsort(-m)[:K]); cov = [int(np.argmax(m))]; gr = []
    while len(cov) < K: cov.append(max((r for r in range(R) if r not in cov), key=lambda r: A[cov + [r]].max(0).mean()))
    while len(gr) < K: gr.append(max((r for r in range(R) if r not in gr), key=lambda r: macro(gr + [r], ix)))
    return {"top5": top, "coverage": cov, "greedy": gr}


rng = np.random.default_rng(SEED); n = len(UIDS); rows = []
for b in range(B):
    perm = rng.permutation(n); a, h = perm[: n // 2], perm[n // 2:]
    A = np.array([per_label(LMR[r], a) for r in range(R)])
    sel = pick(A, a)
    rows.append({k: macro(S, h) for k, S in sel.items()} | {f"{k}_members": [runs[r] for r in S] for k, S in sel.items()})
(OUT / f"k5-seed{SEED}.json").write_text(json.dumps(rows))
print("done", SEED, flush=True)
