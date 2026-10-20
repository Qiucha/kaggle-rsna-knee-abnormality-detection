"""2026-10-15: A30 ensembles without CoAtNet (the exporter cannot package timm members yet), vs A30 and vs the
CoAtNet 5-member pick. Same rank-mean / paired-bootstrap conventions as ensemble_combo_sweep_a30.py."""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path("/home/q/Projects/novo-project"); FT = ROOT / "checkpoints/finetune"
ref = np.load(ROOT / ".scratch/leaderboard-next/scratch/per-label-ensemble/lm-test-targets.npz")
UIDS, Y, M = list(ref["uids"]), ref["y"] >= 0.5, ref["m"] > 0
RUNS = {"A30": "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30", "u9": "r4a-k8-omnirad-base-soft-336-u9-b1",
        "u12": "r4a-k8-omnirad-base-soft-336-u12-b1", "clip": "r4a-k8-omnirad-base-soft-336-from-clip",
        "384": "r4a-k8-omnirad-base-soft-384", "coat": "r4a-k8-coatnet-384-soft-s123",
        "dS336": "r4a-k8-dinov2-small-soft-336", "softR4a": "r4a-soft-224-k8-mean"}
P, G = {}, {}
rk = lambda a: np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
for k, d in RUNS.items():
    z = np.load(FT / d / "predictions.npz"); assert list(z["lm_test_uids"]) == UIDS
    P[k], G[k], GT = rk(z["lm_test"]), rk(z["gold"]), z["gold_targets"] >= 0.5
def auc(p, t):
    n1 = t.sum(); n0 = len(t) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(p)[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
lm = lambda p, ix: np.array([auc(p[ix, j][M[ix, j]], Y[ix, j][M[ix, j]]) for j in range(12)])
gd = lambda p, ix: np.array([auc(p[ix, j], GT[ix, j]) for j in range(12)])
ens = lambda ms, s: np.mean([s[m] for m in ms], 0)
n, ng = len(UIDS), len(GT)
boot = np.random.default_rng(101).integers(0, n, (2000, n)); gboot = np.random.default_rng(7).integers(0, ng, (2000, ng))
def bl(ms): return np.array([np.nanmean(lm(ens(ms, P), ix)) for ix in boot])
def bg(ms): return np.array([np.nanmean(gd(ens(ms, G), ix)) for ix in gboot])
A, C = ["A30"], ["A30", "u9", "clip", "coat", "dS336"]
bA, bC, gA = bl(A), bl(C), bg(A)
ci = lambda d: f"{d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
print("| Ensemble | n | LM-test | Δ vs A30 | Δ vs CoAtNet pick | Gold-58 | Gold Δ vs A30 | Gold LatMen |\n|---|---:|---:|---|---|---:|---|---:|")
for ms in (C, ["A30", "u9", "clip", "dS336"], ["A30", "u9", "clip", "384", "dS336"], ["A30", "clip", "384", "dS336"],
           ["A30", "u9", "clip", "dS336", "softR4a"], ["A30", "u9", "u12", "clip", "dS336"], ["A30", "u9", "dS336"],
           ["A30", "u9", "clip"]):
    b = bl(ms); g = bg(ms); full = gd(ens(ms, G), np.arange(ng))
    print(f"| {' + '.join(ms)} | {len(ms)} | {np.nanmean(lm(ens(ms, P), np.arange(n))):.4f} | {ci(b - bA)} | {ci(b - bC)} | "
          f"{np.nanmean(full):.4f} | {ci(g - gA)} | {full[3]:.3f} |", flush=True)
