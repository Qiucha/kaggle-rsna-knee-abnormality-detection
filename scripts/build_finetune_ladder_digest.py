"""Build digest/finetune-ladder-v1.json: every fine-tuning rung, each against the step it was built from
and against the fixed R0 baseline.

Reads each run's report.json and history.jsonl under checkpoints/finetune/, R0's 58-case LM-head
scores, and the paired study-bootstrap intervals already saved by compare_finetune_rungs.py
(all with the same resamples, seed 2030). Nothing is retrained or rescored. Rerun after a new rung
or comparison lands, then rebuild the site:

    .venv/bin/python scripts/build_finetune_ladder_digest.py
    .venv/bin/python scripts/build_digest_site.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.artifacts import write_json  # noqa: E402

RUNS = ROOT / "checkpoints" / "finetune"
R0_DIR = ROOT / "checkpoints" / "kaggle" / "lm-mlp-tower-size-58-dinov2-small-e5-base"
OUT = ROOT / "digest" / "finetune-ladder-v1.json"

# Each step names the run it changes, and says exactly what it changes. The "parent" is the step
# the run was built from, which is not always the one run before it (R1c changes R1, not R3).
STEPS = [
    {"id": "R0", "run": None, "parent": None, "group": "baseline", "short": "R0",
     "kaggle": "lm-mlp-tower-size-58-dinov2-small-e5-base",
     "label": "R0 · frozen contrastive encoder + LM MLP head",
     "change": "Baseline. DINOv2-small adapted contrastively (LoRA, last 6 blocks) and then frozen; "
               "3-seed MLP head on LM verdicts. Not fine-tuned on labels."},
    {"id": "R1", "run": "r1-hard-224", "parent": "R0", "group": "ladder", "short": "R1",
     "label": "R1 · fine-tune on hard labels",
     "change": "Trains the encoder on the labels: the last 3 blocks of plain pretrained DINOv2-small and an MLP "
               "head, end to end, on hard LM verdicts. 3 slices per plane, 224 px."},
    {"id": "R2", "run": "r2-soft-224", "parent": "R1", "group": "ladder", "short": "R2",
     "label": "R2 · soft labels",
     "change": "Targets change from hard verdicts to the CSV's soft values. Selection still uses hard-label BCE."},
    {"id": "R3", "run": "r3-soft-336", "parent": "R2", "group": "ladder", "short": "R3",
     "label": "R3 · 336 px",
     "change": "Input resolution rises from 224 to 336 px."},
    {"id": "R1c", "run": "r1c-hard-224-from-r0", "parent": "R1", "group": "ladder", "short": "R1c",
     "label": "R1c · start from R0's encoder",
     "change": "R1, but initialised from R0's contrastive checkpoint (its LoRA over the last 6 blocks stays "
               "trainable) instead of plain DINOv2-small."},
    {"id": "R1c-rand", "run": "r1c-hard-224-from-r0-slice-random", "parent": "R1c", "group": "slices",
     "short": "R1c rand", "label": "R1c · random training slices",
     "change": "Each training read draws one random slice per third of the 0.2–0.8 band, per plane. "
               "Evaluation keeps the fixed slices."},
    {"id": "R4a-K8", "run": "r4a-hard-224-k8-mean", "parent": "R1c-rand", "group": "slices", "short": "R4a K8",
     "kaggle": "r4a-k8-mean",
     "label": "R4a · 8 windows per plane, masked mean",
     "change": "Each plane becomes 8 contiguous 3-slice windows, one per depth stratum (jittered in training, "
               "centred at evaluation), mean-pooled under a validity mask. Batch 4 × accumulation 4 keeps the "
               "effective batch at 16."},
    {"id": "R4b-K8", "run": "r4b-hard-224-k8-attention", "parent": "R4a-K8", "group": "slices", "short": "R4b K8",
     "label": "R4b · per-label attention",
     "change": "Mean pooling becomes single-head attention: 12 label queries over all 24 windows, with a plane "
               "embedding. The head is ~112k parameters against R4a's ~200k."},
    {"id": "R4a-K16", "run": "r4a-hard-224-k16-mean", "parent": "R4a-K8", "group": "slices", "short": "R4a K16",
     "label": "R4a · 16 windows per plane",
     "change": "16 windows per plane instead of 8. About 21% of planes are now short stacks with masked windows."},
    {"id": "DINOv2-S", "run": "r4a-k8-dinov2-small-pretrained", "parent": "R4a-K8", "group": "backbones",
     "short": "DINOv2-S pre", "label": "R4a K8 · DINOv2-small from pretrained",
     "change": "Starts from plain pretrained DINOv2-small (last 3 blocks trainable) instead of R0's contrastive "
               "checkpoint. The matched control for the backbone swaps below."},
    {"id": "OmniRad-S", "run": "r4a-k8-omnirad-small-pretrained", "parent": "DINOv2-S", "group": "backbones",
     "short": "OmniRad-S", "label": "R4a K8 · OmniRad-small",
     "change": "Swaps only the pretrained weights: OmniRad-small is the same ViT-S/14 trained with DINOv2 on "
               "RadImageNet (CT, MRI, ultrasound)."},
    {"id": "RadioDino", "run": "r4a-k8-radiodino-s16-pretrained", "parent": "DINOv2-S", "group": "backbones",
     "short": "RadioDino", "label": "R4a K8 · RadioDino-s16",
     "change": "RadImageNet weights trained with original DINO, at patch 16 (196 tokens per window instead of 256)."},
    {"id": "OmniRad-B", "run": "r4a-k8-omnirad-base-pretrained", "parent": "DINOv2-S", "group": "backbones",
     "kaggle": "r4a-k8-omnirad-base",
     "short": "OmniRad-B", "label": "R4a K8 · OmniRad-base",
     "change": "OmniRad at ViT-B/14 (86M parameters, 4× DINOv2-small). Changes both pretraining data and size; "
               "the DINOv2-base control is not run yet.",
     "also": "OmniRad-S"},
    {"id": "R4a-soft", "run": "r4a-soft-224-k8-mean", "parent": "R4a-K8", "group": "labels", "short": "R4a soft",
     "kaggle": "r4a-k8-soft",
     "label": "R4a K8 · soft labels",
     "change": "R2's change on R4a: targets become the CSV's soft LM values instead of hard verdicts."},
    {"id": "R4a-K8-s27", "run": "r4a-hard-224-k8-mean-seed2027", "parent": "R4a-K8", "group": "labels",
     "short": "R4a K8 s2027", "label": "R4a K8 · seed 2027",
     "change": "Replicate: identical to R4a K8 except the training seed (2027 instead of 2026). Measures seed noise."},
    {"id": "OmniRad-B-s27", "run": "r4a-k8-omnirad-base-pretrained-seed2027", "parent": "OmniRad-B", "group": "labels",
     "short": "OmniRad-B s2027", "label": "R4a K8 · OmniRad-base, seed 2027",
     "change": "Replicate: identical to OmniRad-base except the training seed (2027 instead of 2026).",
     "also": "DINOv2-S"},
    {"id": "DINOv2-S-336", "run": "r4a-k8-dinov2-small-pretrained-336", "parent": "DINOv2-S", "group": "labels",
     "short": "DINOv2-S 336", "label": "R4a K8 · DINOv2-small pretrained, 336 px",
     "change": "R3's change on R4a: input resolution rises from 224 to 336 px (576 tokens per window instead of 256), "
               "on a new K8 336 window cache. Starts from pretrained DINOv2-small because R0 is fixed at 224."},
    {"id": "OmniRad-B-soft-336", "run": "r4a-k8-omnirad-base-soft-336", "parent": "OmniRad-B", "group": "combined",
     "short": "OmniRad-B soft 336", "label": "R4a K8 · OmniRad-base, soft labels, 336 px",
     "change": "Stacks the two single-change wins on OmniRad-base: soft LM targets (from R4a soft) and 336 px input "
               "(from DINOv2-S 336), batch 2 × accumulation 8. Two changes at once, so the delta is not attributable.",
     "also": "R4a-soft"},
    {"id": "CoAtNet-384", "run": "r4a-k8-coatnet-384-soft", "parent": "OmniRad-B-soft-336", "group": "diagnostic",
     "short": "CoAtNet 384", "label": "R4a K8 · CoAtNet-384, soft labels (ticket 16)",
     "change": "Backbone diagnostic: timm coatnet_rmlp_2_rw_384 (ImageNet-12k → 1k, 73M parameters, the backbone of "
               "a reported 0.924 public single model) through the same K8 soft recipe at 384 px. Last 3 of 24 blocks "
               "trainable (26.6M parameters); stem and stages 0–2 stay at ImageNet weights with BatchNorm frozen. "
               "Pooled feature is [spatial max, mean] since CoAtNet has no CLS. Changes backbone and resolution "
               "against run 5.",
     "also": "OmniRad-B"},
    {"id": "R4a-soft-unk", "run": "r4a-soft-224-k8-mean-unk02", "parent": "R4a-soft", "group": "lateral",
     "short": "R4a soft UNK 0.2", "label": "R4a K8 soft · UNK cells trained at weight 0.2 (ticket 18)",
     "change": "UNK verdicts stop being masked: they train toward the CSV's UNK value (0.28) at loss weight 0.2, "
               "added on top of the observed cells (normalisation stays on the hard mask). Aimed at Lateral OA, "
               "33% UNK in training, whose gold-58 UNK cases are all negative."},
    {"id": "R4a-soft-grouped", "run": "r4a-soft-224-k8-grouped2", "parent": "R4a-soft", "group": "lateral",
     "short": "R4a soft grouped", "label": "R4a K8 soft · slice-position and plane-aware pooling (ticket 18)",
     "change": "Grouped pooling: each plane's 8 windows pool into 2 contiguous halves along the slice axis (on sagittal, "
               "medial vs lateral half; slice order is the same direction in every study and right knees are mirrored), and "
               "the 3 planes × 2 halves are concatenated instead of averaged, so the head (6× wider input) sees slice "
               "position and plane. Exports through the same dense_v1 head."},
]
KAGGLE = ROOT / "checkpoints" / "kaggle"


def _r0() -> dict:
    lm = json.loads((R0_DIR / "lm-proxy.json").read_text())
    gold = json.loads((R0_DIR / "benchmark.json").read_text())["results"]["mlp"]
    return {"lm_test": {"macro": lm["macro_auroc"], "per_label": lm["per_label"]},
            "gold": {"macro": gold["macro_auroc"],
                     "per_label": {k: v["auroc"] for k, v in gold["per_label"].items()}}}


def _run(name: str) -> dict:
    run = RUNS / name
    report = json.loads((run / "report.json").read_text())
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    gold = report["gold"]["per_label"]
    return {
        "lm_test": {"macro": report["lm_test"]["macro_auroc"], "per_label": report["lm_test"]["per_label"]},
        "gold": {"macro": report["gold"]["macro_auroc"],
                 "per_label": {k: (v["auroc"] if isinstance(v, dict) else v) for k, v in gold.items()}},
        "best_epoch": report["best_epoch"],
        "minutes": round(report["seconds"] / 60, 1),
        "window_counts": report.get("window_counts"),
        "history": [{k: h[k] for k in ("epoch", "train_masked_bce", "validation_masked_bce",
                                       "validation_macro_auroc")} for h in history],
    }


def _pairs() -> dict:
    """Every saved paired interval, keyed by (first, second) run name, as first − second."""
    pairs = {}
    for path in sorted(RUNS.glob("*compare*.json")):
        j = json.loads(path.read_text())
        for split in ("lm_test", "gold"):
            for key, v in j[split]["paired"].items():
                first, second = key.split(" - ")
                pairs.setdefault((split, first, second), {**v, "source": path.name})
    return pairs


def _delta(pairs: dict, split: str, child: str, parent: str) -> dict | None:
    """child − parent, from whichever direction the comparison saved."""
    if (split, child, parent) in pairs:
        v = pairs[(split, child, parent)]
        return {"difference": v["difference"], "ci95": list(v["ci95"]), "source": v["source"]}
    if (split, parent, child) in pairs:
        v = pairs[(split, parent, child)]
        return {"difference": -v["difference"], "ci95": [-v["ci95"][1], -v["ci95"][0]], "source": v["source"]}
    return None


def main() -> None:
    pairs = _pairs()
    run_of = {s["id"]: s["run"] or "R0" for s in STEPS}
    steps = []
    for s in STEPS:
        step = dict(s)
        step.update(_r0() if s["run"] is None else _run(s["run"]))
        if s.get("kaggle"):
            lb = json.loads((KAGGLE / s["kaggle"] / "leaderboard-submission.json").read_text())
            step["public"] = {"score": lb["public_score"], "submission": lb["submission_ref"]}
        for key in ("parent", "also"):
            if s.get(key):
                step[f"vs_{key}"] = {split: _delta(pairs, split, run_of[s["id"]], run_of[s[key]])
                                     for split in ("lm_test", "gold")}
        if s["run"] is not None:
            step["vs_r0"] = {split: _delta(pairs, split, s["run"], "R0") for split in ("lm_test", "gold")}
        steps.append(step)
    missing = [(s["id"], k) for s in steps for k in ("vs_parent", "vs_also", "vs_r0") if k in s
               for split, v in s[k].items() if v is None]
    if missing:
        raise SystemExit(f"No saved paired interval for: {missing}")
    labels = list(steps[0]["gold"]["per_label"])
    write_json(OUT, {
        "labels": labels,
        "steps": steps,
        "lm_test_studies": 432,
        "gold_studies": 58,
        "seed": 2026,
        "bar": "LM-test CI excludes zero; gold-58 difference >= -0.024 (ticket 03).",
        "bootstrap": "Paired study bootstrap, 2,000 resamples shared across arms (seed 2030).",
    })
    print(f"wrote {OUT.relative_to(ROOT)}: {len(steps)} steps")


if __name__ == "__main__":
    main()
