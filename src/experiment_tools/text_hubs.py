"""Retrieval-hub concentration in the English/medical text screen's saved rank-1 rankings.

Shared by ``analyze_text_screen_followups.py hubs`` and the digest site's text-hub table.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from contrastive_learning.paths import repo_root

REPO = repo_root()

RUNS = {
    "baseline multilingual-e5": REPO / "checkpoints/contrastive/claude-translated-dfd07db-q2-v1/00-full-seed2026",
    "e5-base-v2": REPO / "checkpoints/contrastive/english-medical-text-v1/e5-base-v2-seed2026",
    "pubmedbert": REPO / "checkpoints/contrastive/english-medical-text-v1/pubmedbert-base-embeddings-seed2026",
}


def load_rankings(run_dir: Path) -> list[dict]:
    path = run_dir / "rankings-validation-best.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def monte_carlo_null(n_queries: int, n_candidates: int, draws: int = 20000, seed: int = 12345) -> dict:
    """Each query independently picks a top-1 uniformly at random."""
    rng = random.Random(seed)
    maxes, top_twos, distincts = [], [], []
    for _ in range(draws):
        counts = Counter(rng.randrange(n_candidates) for _ in range(n_queries))
        common = counts.most_common()
        maxes.append(common[0][1])
        top_twos.append(common[0][1] + common[1][1])
        distincts.append(len(counts))
    for series in (maxes, top_twos, distincts):
        series.sort()

    def pct(series, p):
        return series[min(len(series) - 1, int(p * len(series)))]

    return {
        "max p50/p95/p99": (pct(maxes, .50), pct(maxes, .95), pct(maxes, .99)),
        "top-two p50/p95/p99": (pct(top_twos, .50), pct(top_twos, .95), pct(top_twos, .99)),
        "distinct p05/p50": (pct(distincts, .05), pct(distincts, .50)),
    }
