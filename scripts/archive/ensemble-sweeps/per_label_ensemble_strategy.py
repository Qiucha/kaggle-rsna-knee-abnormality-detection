"""Per-label coverage vs best-macro ensemble selection, judged out of sample (2026-10-09).

Every fine-tuned run with predictions.npz is a candidate. LM-test (432 studies) is split in half many times:
strategies pick members on one half and are scored on the other, so per-label "winners" must persist to count.
Gold-58 is an external check: pick on all of LM-test, score on gold.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/q/Projects/novo-project")
OUT = ROOT / ".scratch/leaderboard-next/scratch/per-label-ensemble"
OUT.mkdir(exist_ok=True)
FT = ROOT / "checkpoints/finetune"
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.submission import LABELS  # predictions.npz column order (report.json keys are alphabetical)

ref_cache = OUT / "lm-test-targets.npz"
if not ref_cache.exists():
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
    from compare_finetune_rungs import r0_predictions
    from build_finetune_ladder_digest import R0_DIR
    ref = r0_predictions(R0_DIR, "cpu", 8)
    np.savez(ref_cache, uids=np.array(ref["lm_test_uids"]), y=ref["lm_test_targets"], m=ref["lm_test_masks"])
ref = np.load(ref_cache)
UIDS, Y, M = list(ref["uids"]), ref["y"] >= 0.5, ref["m"] > 0

runs, LM, GOLD, GT = [], [], [], None
for d in sorted(p.parent for p in FT.glob("*/predictions.npz")):
    z = np.load(d / "predictions.npz")
    if list(z["lm_test_uids"]) != UIDS:
        print("skip (uid order)", d.name); continue
    runs.append(d.name); LM.append(z["lm_test"]); GOLD.append(z["gold"])
    GT = z["gold_targets"] >= 0.5 if GT is None else GT
LM, GOLD = np.array(LM), np.array(GOLD)
R, L = len(runs), len(LABELS)


def ranks(a):  # per-label rank transform over all studies (labels are not used)
    return np.stack([np.column_stack([rankdata(x[:, j]) / len(x) for j in range(x.shape[1])]) for x in a])
LMR, GOLDR = ranks(LM), ranks(GOLD)


def auc(p, t):
    n1 = t.sum(); n0 = len(t) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (rankdata(p)[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_label(P, ix, gold=False):  # P: (studies, labels) -> (labels,)
    if gold: return np.array([auc(P[:, j], GT[:, j]) for j in range(L)])
    return np.array([auc(P[ix, j][M[ix, j]], Y[ix, j][M[ix, j]]) for j in range(L)])


def score(pred, ix, gold=False):
    a = per_label(pred, ix, gold); return np.nanmean(a), np.nanmin(a), a


def ens(members, R_):  # rank mean, all labels
    return R_[list(members)].mean(0)


def routed(choice, R_):  # choice: label -> list of members
    return np.column_stack([R_[choice[j]][:, :, j].mean(0) for j in range(L)])


def select(A, ix, k):
    """A: (runs, labels) per-label AUROC on the selection half. Returns {strategy: prediction-builder, members}."""
    macro = A.mean(1); out = {}
    out["top-k (best macro)"] = list(np.argsort(-macro)[:k])
    for name, obj in (("cover: mean of per-label best", lambda S: A[S].max(0).mean()),
                      ("cover: worst-label best", lambda S: A[S].max(0).min() + 1e-6 * A[S].max(0).mean())):
        S = [int(np.argmax(macro))]
        while len(S) < k:
            S.append(max((r for r in range(R) if r not in S), key=lambda r: obj(S + [r])))
        out[name] = S
    S = []  # Caruana forward selection on the ensemble itself
    while len(S) < k:
        S.append(max((r for r in range(R) if r not in S), key=lambda r: score(ens(S + [r], LMR), ix)[0]))
    out["greedy ensemble"] = S
    return out


def route(A, k):  # per label: rank-average that label's own top-k runs
    return {j: list(np.argsort(-A[:, j])[:k]) for j in range(L)}


rng = np.random.default_rng(0)
n, B, KS = len(UIDS), 200, (2, 3, 4)
res = {}
persist = []
for b in range(B):
    perm = rng.permutation(n); a_ix, b_ix = perm[: n // 2], perm[n // 2:]
    A = np.array([per_label(LMR[r], a_ix) for r in range(R)])
    Bv = np.array([per_label(LMR[r], b_ix) for r in range(R)])
    best = int(np.argmax(A.mean(1)))  # do per-label edges over the best-macro model persist?
    for j in range(L):
        w = int(np.argmax(A[:, j]))
        if w != best: persist.append((A[w, j] - A[best, j], Bv[w, j] - Bv[best, j]))
    for k in KS:
        for name, S in select(A, a_ix, k).items():
            res.setdefault((name, k), []).append((*score(ens(S, LMR), b_ix)[:2], len(set(S))))
        ch = route(A, k)
        res.setdefault((f"route: per-label top-{k}", k), []).append(
            (*score(routed(ch, LMR), b_ix)[:2], len({r for v in ch.values() for r in v})))
    ch = route(A, 1)
    res.setdefault(("route: per-label best single", 1), []).append(
        (*score(routed(ch, LMR), b_ix)[:2], len({r for v in ch.values() for r in v})))
    res.setdefault(("best single (macro)", 1), []).append((*score(LMR[best], b_ix)[:2], 1))
    if b % 50 == 0: print("split", b, flush=True)

lines = [f"# Per-label coverage vs best-macro selection\n\n{R} candidate runs, {B} random half-splits of LM-test "
         f"({n} studies): pick on one half, score on the other. Rank-mean ensembles.\n",
         "| Strategy | k | Held-out macro AUROC | Δ vs top-k [2.5%, 97.5%] | P(beats top-k) | Held-out worst label | Distinct runs |",
         "|---|---:|---:|---|---:|---:|---:|"]
summary = {}
for (name, k), v in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    v = np.array(v); base = np.array(res.get(("top-k (best macro)", k), res[("best single (macro)", 1)]))
    d = v[:, 0] - base[:, 0]
    summary[f"{name}|{k}"] = dict(macro=v[:, 0].mean(), worst=v[:, 1].mean(), delta=d.mean(),
                                  lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), win=(d > 0).mean(), runs=v[:, 2].mean())
    s = summary[f"{name}|{k}"]
    lines.append(f"| {name} | {k} | {s['macro']:.4f} | {s['delta']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] | "
                 f"{s['win']:.2f} | {s['worst']:.4f} | {s['runs']:.1f} |")

p = np.array(persist)
lines += ["", f"Per-label winners that are not the best-macro run: {len(p)} cases over {B} splits. "
          f"Edge on the selection half {p[:, 0].mean():+.4f}; same runs on the held-out half {p[:, 1].mean():+.4f} "
          f"(held-out edge > 0 in {(p[:, 1] > 0).mean():.2f}; correlation {np.corrcoef(p.T)[0, 1]:+.2f})."]

# Full LM-test selection, then gold-58 as the external check
full = np.arange(n)
A = np.array([per_label(LMR[r], full) for r in range(R)])
lines += ["", "## Selected on all of LM-test, scored on gold-58 (exploratory, guard only)", "",
          "| Strategy | k | Members | LM-test (in-sample) | Gold-58 macro | Gold-58 worst |", "|---|---:|---|---:|---:|---:|"]
short = lambda r: runs[r].replace("r4a-k8-", "").replace("omnirad-base", "omb").replace("-soft-336", "-s336")
for k in KS:
    for name, S in {**select(A, full, k), f"route: per-label top-{k}": None}.items():
        if S is None:
            ch = route(A, k); pl, pg = routed(ch, LMR), routed(ch, GOLDR); mem = sorted({r for v in ch.values() for r in v})
        else:
            pl, pg, mem = ens(S, LMR), ens(S, GOLDR), S
        g = score(pg, None, gold=True)
        lines.append(f"| {name} | {k} | {', '.join(short(r) for r in mem)} | {score(pl, full)[0]:.4f} | {g[0]:.4f} | {g[1]:.4f} |")

order = np.argsort(-A.mean(1))
lines += ["", "## Per-label LM-test AUROC, every run (sorted by macro; ★ = best on that label)", "",
          "| Run | macro | " + " | ".join(LABELS) + " |", "|---|---:|" + "---:|" * L]
for r in order:
    lines.append(f"| {runs[r]} | {A[r].mean():.4f} | " + " | ".join(
        f"{A[r, j]:.3f}{'★' if r == np.argmax(A[:, j]) else ''}" for j in range(L)) + " |")
(OUT / "report.md").write_text("\n".join(lines) + "\n")
np.savez(OUT / "per-label-lm-test.npz", runs=np.array(runs), labels=np.array(LABELS), auroc=A)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("\n".join(lines))
