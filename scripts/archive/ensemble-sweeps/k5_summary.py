"""Summarize k5-seed*.json: coverage vs greedy vs top-5 per split seed and pooled."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

OUT = Path("/home/q/Projects/novo-project/.scratch/leaderboard-next/scratch/per-label-ensemble")
files = sorted(OUT.glob("k5-seed*.json"), key=lambda p: int(p.stem[7:]))
lines = ["# k=5: coverage vs greedy (vs top-5), repeated over split seeds", "",
         "Each seed draws 200 random LM-test half-splits; members are picked on one half and scored on the other.", "",
         "| Split seed | Top-5 | Coverage | Greedy | Coverage − greedy [2.5%, 97.5%] | P(coverage > greedy) | Coverage − top-5 | Greedy − top-5 |",
         "|---|---:|---:|---:|---|---:|---:|---:|"]
allrows = []
def fmt(d): return f"{d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
for f in files + [None]:
    rows = allrows if f is None else json.loads(f.read_text())
    if f is not None: allrows += rows
    t, c, g = (np.array([r[k] for r in rows]) for k in ("top5", "coverage", "greedy"))
    lines.append(f"| {'pooled' if f is None else f.stem[7:]} | {t.mean():.4f} | {c.mean():.4f} | {g.mean():.4f} | {fmt(c - g)} | "
                 f"{(c > g).mean():.2f} | {fmt(c - t)} | {fmt(g - t)} |")
lines += ["", "Splits share the same 432 studies, so seeds are not independent samples; the per-seed spread shows the "
          "Monte-Carlo stability of the estimate, not new data.", "", "## How often each run is picked (pooled)", "",
          "| Run | Top-5 | Coverage | Greedy |", "|---|---:|---:|---:|"]
cnt = {k: Counter(m for r in allrows for m in r[f"{k}_members"]) for k in ("top5", "coverage", "greedy")}
n = len(allrows)
for run in sorted(set().union(*cnt.values()), key=lambda x: -(cnt["coverage"][x] + cnt["greedy"][x])):
    lines.append(f"| {run} | {cnt['top5'][run] / n:.2f} | {cnt['coverage'][run] / n:.2f} | {cnt['greedy'][run] / n:.2f} |")
(OUT / "k5-report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
