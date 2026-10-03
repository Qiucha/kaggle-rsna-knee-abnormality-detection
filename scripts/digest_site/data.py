"""Shared data loaders for page builders, plus the ``export`` snapshot step.

Page builders read ONLY ``digest/*`` and ``digest/site/data/*``:

    read_json("downstream-probe-v1.json")      -> dict | None   (digest/ root)
    site_data("text-block-v1.json")            -> dict | None   (digest/site/data/)
    load_curves("both-lora-v1")                -> dict | None   (digest/site/data/curves/<root>.json)
    curve_series(curves, run, metric)          -> [[epoch, value], ...]
    probe_runs(probe_json)                     -> list[ProbeArm]     (schema P and C "results")
    probe_vs_unadapted(probe_json)             -> list[VsFloor]      (comparisons_against_unadapted_dinov2)
    paired_comparisons(comparison_json)        -> list[Paired]       (paired_comparisons)
    POOLED                                     -> md-only pooled constants, with citations

``export()`` (``build_digest_site.py export``) is the only code that reads the
gitignored ``checkpoints/`` tree. It snapshots per-run metric histories and a
few results that exist nowhere in ``digest/`` into ``digest/site/data/``.
Missing inputs are skipped with a warning; a missing snapshot makes a chart
degrade to a notice, never fails the build.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DIGEST = ROOT / "digest"
SITE = DIGEST / "site"
DATA = SITE / "data"
CURVES = DATA / "curves"
CHECKPOINTS = ROOT / "checkpoints" / "contrastive"

UNADAPTED_KEY = "before_cl/dinov2-unadapted/uniform_mean_768"

# Pooled 3-seed values that exist only in markdown: per-study OOF scores are not stored,
# so they cannot be recomputed from any JSON. Quote them from here, never retype them.
POOLED = {
    "gold_holdout_exposure_gap": {
        "value": -0.0168,
        "label": "Unexposed minus exposed gold macro AUROC, pooled over 3 seeds",
        "source": "digest/gold-holdout-v1.md:49",
        "doc": "gold-holdout-v1.md",
    },
    "crop150_vs_crop130": {
        "value": -0.0201,
        "label": "crop150 minus crop130 gold macro AUROC, pooled over 3 seeds",
        "source": "digest/crop150-probe-v1.md:27",
        "doc": "crop150-probe-v1.md",
    },
}

# Other values that exist only in markdown (no JSON carries them).
MD_ONLY = {
    "public_lb": {
        "value": 0.717,
        "label": "Kaggle public leaderboard macro ROC-AUC, submission 56479080",
        "source": "digest/kaggle-submission-pipeline.md:100",
        "doc": "kaggle-submission-pipeline.md",
    },
}

_warned: set[str] = set()


def warn(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"warning: {message}", file=sys.stderr)


# --------------------------------------------------------------------------- readers


def read_json(name: str) -> Any:
    """A JSON file at the digest/ root, or None (with a warning) when missing."""
    path = DIGEST / name
    if not path.exists():
        warn(f"digest/{name} is missing")
        return None
    return json.loads(path.read_text())


def read_text(name: str) -> str | None:
    path = DIGEST / name
    if not path.exists():
        warn(f"digest/{name} is missing")
        return None
    return path.read_text()


def site_data(name: str) -> Any:
    """A snapshot under digest/site/data/, or None when `export` has not produced it."""
    path = DATA / name
    if not path.exists():
        warn(f"digest/site/data/{name} is missing; run `build_digest_site.py export`")
        return None
    return json.loads(path.read_text())


def load_curves(root: str) -> dict | None:
    """Per-run metric histories for one checkpoint root (e.g. 'both-lora-v1').

    Shape: {"source": "checkpoints/contrastive/<root>",
            "runs": {<run>: {"config": {...}, "val_candidates": int|None, "epoch": [...],
                             "metrics": {<metric>: [...]}, "rows": n, "duplicate_epochs": k}}}
    """
    return site_data(f"curves/{root}.json")


def curve_series(curves: dict | None, run: str, metric: str) -> list[list[float]]:
    """[[epoch, value], ...] for one run and metric ([] when absent)."""
    if not curves or run not in curves.get("runs", {}):
        return []
    r = curves["runs"][run]
    values = r["metrics"].get(metric)
    if values is None:
        return []
    return [[e, v] for e, v in zip(r["epoch"], values)]


# --------------------------------------------------------------------------- probe schemas


@dataclass
class ProbeArm:
    key: str  # full results key, e.g. after_cl/original-unfreeze4-seed2026/uniform_mean_768
    slot: str  # after_cl / before_cl / frozen_tower_null / ...
    run: str  # e.g. original-unfreeze4-seed2026
    feature: str  # uniform_mean_768 / plane_attention_768 / projection_256
    macro_auroc: float
    macro_auprc: float | None
    per_label: dict[str, float]
    positives: dict[str, int]
    checkpoint: str | None


@dataclass
class VsFloor:
    key: str
    run: str
    feature: str
    baseline_key: str
    before: float | None
    after: float | None
    diff: float  # bootstrap mean of (after - before)
    ci: tuple[float, float]
    p_favour: float | None  # share of bootstrap replicates favouring "after"
    per_label: dict[str, tuple[float, tuple[float, float]]]


@dataclass
class Paired:
    name: str  # "<candidate> vs <baseline>"
    candidate: str
    baseline: str
    macro_candidate: float | None
    macro_baseline: float | None
    diff: float  # bootstrap mean of (candidate - baseline)
    ci: tuple[float, float]
    p_candidate: float | None  # share of replicates favouring the candidate (field name says "right")
    per_label: dict[str, tuple[float, tuple[float, float]]]


def split_key(key: str) -> tuple[str, str, str]:
    """'slot/run/feature' -> (slot, run, feature). Two-part keys get an empty slot."""
    parts = key.split("/")
    if len(parts) >= 3:
        return parts[0], "/".join(parts[1:-1]), parts[-1]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    return "", key, ""


def probe_labels(j: dict) -> list[str]:
    return list((j.get("config") or {}).get("labels") or [])


def probe_runs(j: dict | None) -> list[ProbeArm]:
    """Every entry of results{} (schema P and schema C)."""
    if not j:
        return []
    prov = j.get("provenance") or {}
    out = []
    for key, r in (j.get("results") or {}).items():
        slot, run, feature = split_key(key)
        pl = r.get("per_label") or {}
        out.append(ProbeArm(
            key=key, slot=slot, run=run, feature=feature,
            macro_auroc=r.get("macro_auroc"), macro_auprc=r.get("macro_auprc"),
            per_label={k: v.get("auroc") for k, v in pl.items()},
            positives={k: v.get("positives") for k, v in pl.items()},
            checkpoint=(prov.get(f"{slot}/{run}") or {}).get("checkpoint"),
        ))
    return out


def unadapted_level(j: dict | None) -> float | None:
    """Macro AUROC of unadapted DINOv2 (the floor) from results{}; None if absent."""
    if not j:
        return None
    r = (j.get("results") or {}).get(UNADAPTED_KEY)
    return r.get("macro_auroc") if r else None


def _per_label_diff(d: dict) -> dict[str, tuple[float, tuple[float, float]]]:
    return {k: (v.get("difference"), tuple(v.get("ci95") or (None, None))) for k, v in (d or {}).items()}


def probe_vs_unadapted(j: dict | None) -> list[VsFloor]:
    """comparisons_against_unadapted_dinov2{}: difference vs the floor with 95% CI."""
    if not j:
        return []
    out = []
    for key, c in (j.get("comparisons_against_unadapted_dinov2") or {}).items():
        _, run, feature = split_key(key)
        out.append(VsFloor(
            key=key, run=run, feature=feature, baseline_key=c.get("baseline_key", UNADAPTED_KEY),
            before=c.get("macro_auroc_before"), after=c.get("macro_auroc_after"),
            diff=c["macro_auroc_difference_mean"], ci=tuple(c["macro_auroc_difference_ci95"]),
            p_favour=c.get("fraction_of_replicates_favouring_right"),
            per_label=_per_label_diff(c.get("per_label_auroc_difference")),
        ))
    return out


def paired_comparisons(j: dict | None) -> list[Paired]:
    """paired_comparisons{} (schema C). Sign: candidate (left of ' vs ') minus baseline."""
    if not j:
        return []
    out = []
    for name, p in (j.get("paired_comparisons") or {}).items():
        out.append(Paired(
            name=name, candidate=p.get("candidate_arm"), baseline=p.get("baseline_arm"),
            macro_candidate=p.get("macro_auroc_candidate"), macro_baseline=p.get("macro_auroc_baseline"),
            diff=p["macro_auroc_difference_mean"], ci=tuple(p["macro_auroc_difference_ci95"]),
            p_candidate=p.get("fraction_of_replicates_favouring_right"),
            per_label=_per_label_diff(p.get("per_label_auroc_difference")),
        ))
    return out


def seed_mean(values: list[float]) -> float | None:
    v = [x for x in values if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None


# --------------------------------------------------------------------------- export

METRICS = [
    "train_loss", "validation_loss", "mean_r@1",
    "image_to_text_r@1", "text_to_image_r@1", "image_to_text_r@5", "text_to_image_r@5",
    "image_to_text_r@10", "text_to_image_r@10", "image_to_text_mean_rank", "text_to_image_mean_rank",
    "image_to_text_median_rank", "text_to_image_median_rank", "inverse_temperature",
    "positive_similarity_mean", "negative_similarity_mean",
    "image_embedding_dimension_std_mean", "text_embedding_dimension_std_mean",
]
CONFIG_KEYS = [
    "seed", "epochs", "batch_size", "image_model", "text_model", "image_unfreeze_last", "text_unfreeze_last",
    "use_lora", "lora_rank", "lora_alpha", "image_drop_path_rate", "crop_mm", "image_plane_pooling",
    "label_similarity_weight", "report_view", "max_text_length", "lr_image", "lr_text", "cohort_manifest",
]


def _round(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return float(f"{v:.5g}")
    return v


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    path.write_text(text + "\n")


def _val_count(manifest: str | None, cache: dict) -> int | None:
    if not manifest:
        return None
    if manifest in cache:
        return cache[manifest]
    path = Path(manifest) if manifest.startswith("/") else ROOT / manifest
    count = None
    if path.exists():
        count = 0
        with path.open() as handle:
            for line in handle:
                if line.strip() and json.loads(line).get("split") == "validation":
                    count += 1
    cache[manifest] = count
    return count


def _history(path: Path) -> tuple[list[dict], int]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a writer may be mid-line
    by_epoch: dict[int, dict] = {}
    for r in rows:
        if "epoch" in r:
            by_epoch[int(r["epoch"])] = r  # keep the last row per epoch (resumed runs)
    dups = len(rows) - len(by_epoch)
    return [by_epoch[e] for e in sorted(by_epoch)], dups


def export_curves() -> dict:
    """Snapshot every checkpoints/contrastive/**/metrics.jsonl, one JSON per top-level root."""
    if not CHECKPOINTS.exists():
        warn("checkpoints/contrastive is missing; curve snapshots not refreshed")
        return {}
    cache: dict = {}
    roots: dict[str, dict] = {}
    for mfile in sorted(CHECKPOINTS.rglob("metrics.jsonl")):
        rel = mfile.parent.relative_to(CHECKPOINTS)
        parts = rel.parts
        root = parts[0]
        run = "/".join(parts[1:]) or root
        if "smoke" in str(rel):
            continue
        rows, dups = _history(mfile)
        if len(rows) < 3:
            continue
        if dups:
            warn(f"{rel}: {dups} duplicate epoch rows (kept the last row per epoch)")
        cfg = {}
        cpath = mfile.parent / "config.json"
        if cpath.exists():
            full = json.loads(cpath.read_text())
            cfg = {k: _round(full[k]) for k in CONFIG_KEYS if k in full}
        entry = {
            "config": cfg,
            "val_candidates": _val_count(cfg.get("cohort_manifest"), cache),
            "rows": len(rows),
            "duplicate_epochs": dups,
            "epoch": [int(r["epoch"]) for r in rows],
            "metrics": {m: [_round(r.get(m)) for r in rows] for m in METRICS if any(m in r for r in rows)},
        }
        roots.setdefault(root, {"source": f"checkpoints/contrastive/{root}", "runs": {}})["runs"][run] = entry
    for root, obj in roots.items():
        _write(CURVES / f"{root}.json", obj)
    index = {root: sorted(obj["runs"]) for root, obj in sorted(roots.items())}
    _write(CURVES / "index.json", index)
    return index


def export_text_block() -> None:
    """Text final-block adaptation results: best validation per seed vs matched frozen baselines."""
    sel = CHECKPOINTS / "claude-translated-dfd07db-text-block-v1" / "validation-selection.json"
    cfg = ROOT / "configs" / "contrastive" / "claude-translated-dfd07db-text-block-v1.json"
    if not (sel.exists() and cfg.exists()):
        warn("text-block validation-selection.json or its config is missing; skipped")
        return
    s = json.loads(sel.read_text())
    baselines = json.loads(cfg.read_text()).get("matched_baselines", {})
    seeds = []
    for exp in s.get("experiments", []):
        name = exp.get("name", "")
        seed = name.rsplit("seed", 1)[-1] if "seed" in name else None
        best = exp.get("best_validation") or {}
        cand = best.get("mean_r@1")
        base = (baselines.get(seed) or {}).get("mean_r@1_pp")
        seeds.append({
            "name": name, "seed": seed, "best_epoch": best.get("epoch"),
            "candidate_mean_r@1_pp": _round(cand * 100) if cand is not None else None,
            "baseline_mean_r@1_pp": _round(base),
            "difference_pp": _round(cand * 100 - base) if cand is not None and base is not None else None,
            "baseline_output": (baselines.get(seed) or {}).get("output"),
        })
    diffs = [x["difference_pp"] for x in seeds if x["difference_pp"] is not None]
    mean = sum(diffs) / len(diffs) if diffs else None
    sd = (math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else None)
    _write(DATA / "text-block-v1.json", {
        "source": ["checkpoints/contrastive/claude-translated-dfd07db-text-block-v1/validation-selection.json",
                   "configs/contrastive/claude-translated-dfd07db-text-block-v1.json#matched_baselines"],
        "note": "Validation-only readout (441 candidates); no confidence intervals exist for this arm.",
        "seeds": seeds, "mean_difference_pp": _round(mean), "sd_difference_pp": _round(sd),
    })


def export_text_hubs() -> None:
    """Retrieval-hub concentration per text-encoder screen (from the saved rank-1 rankings)."""
    script = ROOT / "scripts" / "analyze_text_screen_followups.py"
    if not script.exists():
        warn("scripts/analyze_text_screen_followups.py missing; hub table skipped")
        return
    try:
        spec = importlib.util.spec_from_file_location("_hubs", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        runs = []
        pool = None
        for label, path in mod.RUNS.items():
            rows = mod.load_rankings(path)
            for direction in ("text_to_image", "image_to_text"):
                subset = [r for r in rows if r["direction"] == direction]
                counts = Counter(r["top_uid"] for r in subset)
                common = counts.most_common()
                pool = len(subset)
                runs.append({
                    "run": label, "direction": direction, "queries": len(subset),
                    "r1_hits": sum(1 for r in subset if r["top_uid"] == r["positive_uid"]),
                    "distinct_rank1_targets": len(counts),
                    "biggest_hub": common[0][1], "top_two": common[0][1] + common[1][1],
                })
        null = mod.monte_carlo_null(pool, pool)
        _write(DATA / "text-hubs.json", {
            "source": "scripts/analyze_text_screen_followups.py hubs (saved rankings-validation-best.jsonl)",
            "runs": runs,
            "random_null": {k: list(v) for k, v in null.items()},
        })
    except Exception as exc:  # noqa: BLE001 - optional snapshot
        warn(f"hub table export failed ({exc.__class__.__name__}: {exc}); skipped")


LM_HEAD_ROOT = "kaggle"
LM_HEAD_PREFIX = "lm-mlp-tower-size-58-"


def export_lm_head_histories() -> None:
    """Per-epoch train/validation masked BCE of the 58-case LM-label heads (4 pairings x {mlp, linear} x 3 seeds)."""
    base = ROOT / "checkpoints" / LM_HEAD_ROOT
    dirs = sorted(p for p in base.glob(LM_HEAD_PREFIX + "*") if p.is_dir()) if base.exists() else []
    if not dirs:
        warn(f"checkpoints/{LM_HEAD_ROOT}/{LM_HEAD_PREFIX}* is missing; LM-head loss histories skipped")
        return
    pairings = {}
    for d in dirs:
        runs = {}
        for f in sorted(d.glob("*-seed*-history.json")):
            head, seed = f.name[: -len("-history.json")].split("-seed")
            j = json.loads(f.read_text())
            hist = sorted(j.get("history") or [], key=lambda r: r["epoch"])
            runs.setdefault(head, {})[seed] = {
                "best_epoch": j.get("best_epoch"),
                "parameter_count": j.get("parameter_count"),
                "selected_validation_masked_bce": _round(j.get("validation_masked_bce")),
                "epoch": [int(r["epoch"]) for r in hist],
                "train_masked_bce": [_round(r.get("train_masked_bce")) for r in hist],
                "validation_masked_bce": [_round(r.get("validation_masked_bce")) for r in hist],
            }
        if runs:
            pairings[d.name[len(LM_HEAD_PREFIX):]] = runs
    _write(DATA / "lm-head-histories-58.json", {
        "source": f"checkpoints/{LM_HEAD_ROOT}/{LM_HEAD_PREFIX}*/{{mlp,linear}}-seed*-history.json",
        "note": ("LM-label (pseudo-label) masked BCE on the head's own training and validation studies; "
                 "the 58 gold studies are excluded from both. Epoch selection = minimum validation BCE."),
        "pairings": pairings,
    })


def export() -> None:
    index = export_curves()
    print(f"curves: {sum(len(v) for v in index.values())} runs in {len(index)} roots -> {CURVES.relative_to(ROOT)}")
    export_text_block()
    export_text_hubs()
    export_lm_head_histories()
    print(f"snapshots written under {DATA.relative_to(ROOT)}")
