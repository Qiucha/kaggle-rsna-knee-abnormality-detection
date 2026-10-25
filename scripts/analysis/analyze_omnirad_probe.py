"""Did plain CLIP help OmniRad-base? Readout and label grouping on frozen features.

Reads the output of ``extract_omnirad_probe_features.py`` (unadapted OmniRad vs its
CLIP-adapted tower, same windows) and reports, per arm:

* **Readout** -- masked-label heads on the frozen features, run 5's head recipe (linear
  control and one 256-unit MLP, seeds 2026/2027/2028, epoch chosen on validation masked
  BCE). Scored on the 58 gold studies and on the locked LM-test split, with paired
  study-bootstrap intervals on clip - base.
* **Grouping** -- per label, in the full standardized feature space: leave-one-out cosine
  kNN AUROC and cosine silhouette for pos vs neg. PCA and t-SNE figures are for
  inspection only (t-SNE fits differ per arm, so positions are not comparable).
* **Acquisition control** -- the same kNN statistic for scanner manufacturer and field
  strength. If one arm is organised by scanner and the other is not, label grouping
  differs for reasons unrelated to pathology.

The primary pool for grouping is the locked test split: CLIP never saw those images or
reports and no head was fitted on them. The train+validation pool is reported as
secondary because CLIP trained on those exact image/report pairs.

Usage: analyze_omnirad_probe.py --features DIR --output DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import rankdata  # noqa: E402

from contrastive_learning.lm_proxy import masked_macro_auroc  # noqa: E402
from contrastive_learning.submission import LABELS, predict_head, score_predictions  # noqa: E402
from contrastive_learning.supervised_head import fit_dense_head  # noqa: E402

ARMS = ("base", "clip")
SEEDS = (2026, 2027, 2028)
HEAD_ARMS = {"linear": [], "mlp": [256]}
TRAINING = {"dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 0.01, "batch_size": 128,
            "max_epochs": 100, "patience": 12, "min_delta": 1e-5}
KNN_K = 15
RESAMPLES = 2000
BOOTSTRAP_SEED = 2030
POS, NEG, UNK = "#D55E00", "#0072B2", "#C8C8C8"


def auroc(score, positive):
    n_pos = positive.sum()
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((rankdata(score)[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_label_auroc(pred, y, m):
    return [auroc(pred[m[:, j] > 0, j], y[m[:, j] > 0, j] > 0.5) for j in range(y.shape[1])]


def macro(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def paired_bootstrap(scorer, base_pred, clip_pred, y, m, seed=BOOTSTRAP_SEED):
    """clip - base deltas for macro and per label under shared study resamples."""
    rng = np.random.default_rng(seed)
    macro_d, label_d = [], []
    for _ in range(RESAMPLES):
        i = rng.integers(0, len(y), len(y))
        a, b = scorer(base_pred[i], y[i], m[i]), scorer(clip_pred[i], y[i], m[i])
        label_d.append([None if (p is None or q is None) else q - p for p, q in zip(a, b)])
        if macro(a) is not None and macro(b) is not None:
            macro_d.append(macro(b) - macro(a))
    label_ci = []
    for j in range(y.shape[1]):
        col = [row[j] for row in label_d if row[j] is not None]
        label_ci.append(np.quantile(col, [0.025, 0.975]).tolist() if len(col) > RESAMPLES // 2 else None)
    return np.quantile(macro_d, [0.025, 0.975]).tolist(), label_ci


def readout(features, population):
    """Fit heads on train, select on validation; predictions on test and gold."""
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for kind, hidden in HEAD_ARMS.items():
            heads, histories = [], []
            for seed in SEEDS:
                head, selection = fit_dense_head(
                    features[f"train_{arm}"], population["train_targets"], population["train_masks"],
                    features[f"validation_{arm}"], population["validation_targets"],
                    population["validation_masks"], hidden_dims=hidden, seed=seed, **TRAINING,
                )
                heads.append(head)
                histories.append(selection["best_epoch"])
            out[arm][kind] = {
                "heads": heads, "best_epochs": histories,
                "test_seeds": [predict_head(features[f"test_{arm}"], h) for h in heads],
                "gold_seeds": [predict_head(features[f"gold_{arm}"], h) for h in heads],
            }
            print(f"readout {arm}/{kind}: best epochs {histories}", flush=True)
    return out


def summarize_readout(heads, population):
    ty, tm = population["test_targets"], population["test_masks"]
    gy, gm = population["gold_targets"], np.ones_like(population["gold_targets"])
    result = {}
    for kind in HEAD_ARMS:
        preds = {arm: {"test": np.mean(heads[arm][kind]["test_seeds"], 0),
                       "gold": np.mean(heads[arm][kind]["gold_seeds"], 0)} for arm in ARMS}
        entry = {"best_epochs": {arm: heads[arm][kind]["best_epochs"] for arm in ARMS}}
        for split, y, m in (("test", ty, tm), ("gold", gy, gm)):
            scorer = lambda p, yy, mm: per_label_auroc(p, yy, mm)  # noqa: E731
            labels = {arm: per_label_auroc(preds[arm][split], y, m) for arm in ARMS}
            seed_macros = {arm: [macro(per_label_auroc(p, y, m)) for p in heads[arm][kind][f"{split}_seeds"]]
                           for arm in ARMS}
            macro_ci, label_ci = paired_bootstrap(scorer, preds["base"][split], preds["clip"][split], y, m)
            entry[split] = {
                "macro_auroc": {arm: macro(labels[arm]) for arm in ARMS},
                "delta_clip_minus_base": macro(labels["clip"]) - macro(labels["base"]),
                "delta_ci95": macro_ci,
                "head_seed_macro_auroc": seed_macros,
                "per_label": {
                    label: {"base": labels["base"][j], "clip": labels["clip"][j],
                            "delta": None if labels["base"][j] is None or labels["clip"][j] is None
                            else labels["clip"][j] - labels["base"][j],
                            "delta_ci95": label_ci[j]}
                    for j, label in enumerate(LABELS)
                },
            }
        result[kind] = entry
    return result


def standardize(train, pool):
    mean, std = train.mean(0), train.std(0)
    std[std < 1e-8] = 1.0
    z = (pool - mean) / std
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def knn_scores(sim, values, k=KNN_K):
    """Leave-one-out kNN: mean target among the k most similar other rows."""
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argpartition(-sim, k, axis=1)[:, :k]
    return values[nearest].mean(1)


def label_grouping(z, y, m):
    """Per-label kNN AUROC and silhouette over rows where the label is observed."""
    from sklearn.metrics import silhouette_score

    knn, sil = [], []
    for j in range(y.shape[1]):
        keep = m[:, j] > 0
        positive = y[keep, j] > 0.5
        if positive.sum() < 5 or (~positive).sum() < 5:
            knn.append(None)
            sil.append(None)
            continue
        zk = z[keep]
        knn.append(auroc(knn_scores(zk @ zk.T, positive.astype(float)), positive))
        sil.append(float(silhouette_score(zk, positive, metric="cosine")))
    return knn, sil


def knn_scores_by_label(z, y, m):
    """Per-label LOO kNN scores kept per study, for paired resampling."""
    scores = []
    for j in range(y.shape[1]):
        keep = m[:, j] > 0
        zk = z[keep]
        s = np.full(len(z), np.nan)
        s[keep] = knn_scores(zk @ zk.T, (y[keep, j] > 0.5).astype(float))
        scores.append(s)
    return np.stack(scores, 1)


def categorical_knn_accuracy(z, categories):
    """LOO kNN majority-vote accuracy for a categorical acquisition field."""
    labels = np.asarray(categories)
    known = labels != ""
    zk, lk = z[known], labels[known]
    classes, codes = np.unique(lk, return_inverse=True)
    if len(classes) < 2:
        return None
    sim = zk @ zk.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argpartition(-sim, KNN_K, axis=1)[:, :KNN_K]
    votes = np.apply_along_axis(lambda r: np.bincount(r, minlength=len(classes)).argmax(), 1, codes[nearest])
    return {"accuracy": float((votes == codes).mean()),
            "majority_baseline": float(np.bincount(codes).max() / len(codes)),
            "classes": {c: int((lk == c).sum()) for c in classes}, "studies": int(known.sum())}


def geometry(z):
    eig = np.clip(np.linalg.eigvalsh(np.cov(z.T))[::-1], 0, None)
    return {"effective_rank": float(eig.sum() ** 2 / (eig**2).sum()),
            "pc1_variance": float(eig[0] / eig.sum()), "top10_variance": float(eig[:10].sum() / eig.sum())}


def embed(z, method):
    from sklearn.decomposition import PCA

    if method == "pca":
        return PCA(2, random_state=0).fit_transform(z)
    from sklearn.manifold import TSNE

    reduced = PCA(min(50, len(z) - 1), random_state=0).fit_transform(z)
    return TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(reduced)


def draw_embedding(path, coords, y, m, color_by, method, pool):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 7, figsize=(21, 12.5))
    for row, arm in enumerate(ARMS):
        for col in range(13):
            ax = axes[row * 2 + col // 7, col % 7]
            xy = coords[arm]
            if col < 12:
                obs, pos = m[:, col] > 0, y[:, col] > 0.5
                ax.scatter(*xy[~obs].T, s=5, c=UNK, alpha=0.5, linewidths=0)
                ax.scatter(*xy[obs & ~pos].T, s=8, c=NEG, alpha=0.6, linewidths=0)
                ax.scatter(*xy[obs & pos].T, s=11, c=POS, alpha=0.85, linewidths=0)
                ax.set_title(f"{arm}: {LABELS[col]} (+{int((obs & pos).sum())}/-{int((obs & ~pos).sum())})", fontsize=9)
            else:
                cats = np.asarray(color_by)
                for k, c in enumerate(sorted(set(cats))):
                    sel = cats == c
                    ax.scatter(*xy[sel].T, s=8, alpha=0.7, linewidths=0, label=(c or "unknown")[:18],
                               c=None if c else UNK)
                ax.legend(fontsize=6, loc="best", markerscale=2)
                ax.set_title(f"{arm}: scanner manufacturer", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row * 2 + 1, 6].axis("off")
    fig.suptitle(f"{method.upper()} of frozen features, {pool} pool. orange = positive, blue = negative, "
                 "gray = unknown. Fit separately per arm: axes are not comparable.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_summary(path, grouping, readout_summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("kNN AUROC, held-out test pool", {a: grouping["test"]["knn_auroc"][a] for a in ARMS}),
              ("MLP readout AUROC, LM-test",
               {a: [readout_summary["mlp"]["test"]["per_label"][l][a] for l in LABELS] for a in ARMS}),
              ("MLP readout AUROC, gold-58",
               {a: [readout_summary["mlp"]["gold"]["per_label"][l][a] for l in LABELS] for a in ARMS})]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    y = np.arange(len(LABELS))
    for ax, (title, values) in zip(axes, panels):
        b = np.array([np.nan if v is None else v for v in values["base"]])
        c = np.array([np.nan if v is None else v for v in values["clip"]])
        ax.hlines(y, b, c, color="#888888", lw=1.5)
        ax.scatter(b, y, c=NEG, s=40, label="unadapted OmniRad", zorder=3)
        ax.scatter(c, y, c=POS, s=40, label="after CLIP", zorder=3)
        ax.axvline(0.5, color="#bbbbbb", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(y, LABELS)
        ax.invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--trainval-sample", type=int, default=1500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.features / "features.npz") as f:
        features = {k: f[k] for k in f.files}
    with np.load(args.features / "population.npz") as p:
        population = {k: p[k] for k in p.files}
    acquisition = json.loads((args.features / "acquisition.json").read_text()) \
        if (args.features / "acquisition.json").exists() else {}

    heads = readout(features, population)
    readout_summary = summarize_readout(heads, population)
    # Seed-ensembled head predictions, for paired comparison against fine-tuned runs.
    np.savez_compressed(
        args.output / "head-predictions.npz",
        test_uids=population["test_uids"], gold_uids=population["gold_uids"],
        **{f"{split}_{arm}_{kind}": np.mean(heads[arm][kind][f"{split}_seeds"], 0)
           for arm in ARMS for kind in HEAD_ARMS for split in ("test", "gold")},
    )

    rng = np.random.default_rng(0)
    pools = {}
    pools["test"] = {"feat": {a: features[f"test_{a}"] for a in ARMS}, "uids": population["test_uids"],
                     "y": population["test_targets"], "m": population["test_masks"]}
    tv_uids = np.concatenate([population["train_uids"], population["validation_uids"]])
    tv_pick = np.sort(rng.choice(len(tv_uids), min(args.trainval_sample, len(tv_uids)), replace=False))
    pools["trainval"] = {
        "feat": {a: np.concatenate([features[f"train_{a}"], features[f"validation_{a}"]])[tv_pick] for a in ARMS},
        "uids": tv_uids[tv_pick],
        "y": np.concatenate([population["train_targets"], population["validation_targets"]])[tv_pick],
        "m": np.concatenate([population["train_masks"], population["validation_masks"]])[tv_pick],
    }

    grouping = {}
    for name, pool in pools.items():
        z = {a: standardize(features[f"train_{a}"], pool["feat"][a]) for a in ARMS}
        y, m = pool["y"], pool["m"]
        entry = {"studies": len(y), "knn_k": KNN_K, "knn_auroc": {}, "silhouette": {}, "geometry": {},
                 "acquisition_knn": {}}
        per_study = {}
        for a in ARMS:
            entry["knn_auroc"][a], entry["silhouette"][a] = label_grouping(z[a], y, m)
            entry["geometry"][a] = geometry(z[a])
            per_study[a] = knn_scores_by_label(z[a], y, m)
        entry["knn_macro"] = {a: macro(entry["knn_auroc"][a]) for a in ARMS}
        entry["silhouette_macro"] = {a: macro(entry["silhouette"][a]) for a in ARMS}
        # Paired study bootstrap over the fixed LOO scores (neighbour sets held fixed).
        rng_b = np.random.default_rng(BOOTSTRAP_SEED)
        deltas = []
        for _ in range(RESAMPLES):
            i = rng_b.integers(0, len(y), len(y))
            vals = []
            for a in ARMS:
                vals.append(macro([auroc(per_study[a][i][m[i, j] > 0, j],
                                         y[i][m[i, j] > 0, j] > 0.5) for j in range(y.shape[1])]))
            if None not in vals:
                deltas.append(vals[1] - vals[0])
        entry["knn_macro_delta_clip_minus_base"] = entry["knn_macro"]["clip"] - entry["knn_macro"]["base"]
        entry["knn_macro_delta_ci95"] = np.quantile(deltas, [0.025, 0.975]).tolist()
        cats = {"manufacturer": [acquisition.get(u, {}).get("Manufacturer", "") for u in pool["uids"]],
                "field_strength": [acquisition.get(u, {}).get("MagneticFieldStrength", "") for u in pool["uids"]]}
        for field, values in cats.items():
            entry["acquisition_knn"][field] = {a: categorical_knn_accuracy(z[a], values) for a in ARMS}
        grouping[name] = entry
        for method in ("pca",) + (() if args.skip_tsne else ("tsne",)):
            coords = {a: embed(z[a], method) for a in ARMS}
            draw_embedding(args.output / f"{method}-{name}.png", coords, y, m, cats["manufacturer"], method, name)
            print(f"drew {method} {name}", flush=True)

    draw_summary(args.output / "per-label-summary.png", grouping, readout_summary)
    result = {"readout": readout_summary, "grouping": grouping,
              "notes": {"head_seeds": list(SEEDS), "heads_measure": "head-seed noise only; one CLIP training seed",
                        "knn_bootstrap": "neighbour sets held fixed, study resampling only",
                        "gold": "exploratory; 58 cases examined repeatedly"}}
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print("wrote", args.output / "results.json", flush=True)


if __name__ == "__main__":
    main()
