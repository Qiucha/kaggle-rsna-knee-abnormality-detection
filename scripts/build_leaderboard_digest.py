"""Collect every recorded Kaggle public-leaderboard submission into digest/leaderboard-history-v1.json.

Reads checkpoints/kaggle/*/leaderboard-submission.json. Submissions without a recorded date sort first.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Short titles and families for the bundles; anything unlisted keeps its directory name.
TITLES = {
    "lora-holdout-v1": ("First submission: six-block LoRA holdout contrastive + gold logistic heads", "frozen"),
    "lm-mlp-tower-size-58-dinov2-small-e5-small": ("R0-style frozen DINOv2-S + E5-small, LM MLP heads", "frozen"),
    "lm-mlp-tower-size-58-dinov2-small-e5-base": ("R0: frozen DINOv2-S + E5-base, LM MLP heads", "frozen"),
    "r4a-k8-mean": ("R4a K8 multi-slice fine-tune (DINOv2-S, hard)", "finetune"),
    "r4a-k8-omnirad-base": ("R4a K8 OmniRad-base (hard)", "finetune"),
    "r4a-k8-soft": ("R4a K8 soft labels (DINOv2-S)", "finetune"),
    "r4a-k8-omnirad-base-soft-336": ("Run 5: OmniRad-base, soft, 336 px", "finetune"),
    "r4a-k8-omb-soft-336-u9": ("u9-b1: run 5 with 9 image blocks unfrozen", "finetune"),
    "r4a-k8-omb-soft-336-u9-a30": ("A30: u9-b1 + lateral oversample + noise, 30 epochs", "finetune"),
}


def main():
    rows = []
    for path in sorted((ROOT / "checkpoints/kaggle").glob("*/leaderboard-submission.json")):
        j = json.loads(path.read_text())
        if j.get("public_score") is None:
            continue
        name = path.parent.name
        title, family = TITLES.get(name, (name, "finetune"))
        rows.append({"bundle": name, "title": title, "family": family, "run": j.get("run"),
                     "submission": j.get("submission_ref"), "recorded": j.get("recorded_at_utc"),
                     "public": j["public_score"], "private": j.get("private_score"),
                     "local_lm_test": j.get("local_lm_test_macro_auroc"),
                     "local_gold58": j.get("local_gold58_macro_auroc"),
                     "outcome": j.get("outcome"), "source": str(path.relative_to(ROOT))})
    rows.sort(key=lambda r: (r["recorded"] or "", r["submission"] or 0))
    out = ROOT / "digest/leaderboard-history-v1.json"
    best = max(rows, key=lambda r: r["public"], default=None)  # no recorded submissions -> no best
    out.write_text(json.dumps({"submissions": rows, "best": best and best["bundle"]}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
