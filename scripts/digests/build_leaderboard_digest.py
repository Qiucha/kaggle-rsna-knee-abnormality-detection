"""Collect every recorded Kaggle public-leaderboard submission into digest/leaderboard-history-v1.json.

Reads checkpoints/kaggle/*/leaderboard-submission.json. Submissions without a recorded date sort first.
Refuses to drop a bundle already recorded in the JSON (those records are gitignored) unless --allow-drop is given.

    .venv/bin/python scripts/digests/build_leaderboard_digest.py [--allow-drop]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
    "a30-ens4-rank": ("Rank ensemble: A30 + u9-b1 + CLIP-init OmniRad + DINOv2-S 336", "ensemble"),
    "a30-ens4-swap-rank": ("Rank ensemble swap: from-clip u3 replaced by CLIP-init u9 (A30 recipe)", "ensemble"),
    "a30-deep3-rank": ("Rank ensemble, deep three: A30 + u9-b1 + CLIP-init u9 (no shallow members)", "ensemble"),
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
    # The records live in git-ignored checkpoints/kaggle, so a checkout without them
    # would silently rewrite the history as empty. Refuse to drop a recorded bundle.
    if out.exists() and "--allow-drop" not in sys.argv[1:]:
        recorded = {r["bundle"] for r in json.loads(out.read_text())["submissions"]}
        if dropped := sorted(recorded - {r["bundle"] for r in rows}):
            raise SystemExit(f"Refusing to drop recorded submissions (no leaderboard-submission.json "
                             f"under checkpoints/kaggle): {', '.join(dropped)}. Pass --allow-drop to override.")
    best = max(rows, key=lambda r: r["public"], default=None)  # no recorded submissions -> no best
    out.write_text(json.dumps({"submissions": rows, "best": best and best["bundle"]}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
