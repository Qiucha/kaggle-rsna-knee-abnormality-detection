"""Epoch curves for the static digest pages: a tolerant metrics.jsonl reader, a one-run summary, and an inline SVG line chart."""

import json
from html import escape

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 760, 300, 58, 18, 16, 40


def history(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A writer may be midway through the final line.
    return rows


def run_stat(path):
    """Lowest validation loss (with its epoch) and best mean R@1 of one contrastive run.

    ``path`` is the run's ``metrics.jsonl``. Returns two table cells, dashes until it has rows.
    """
    rows = history(path)
    if not rows:
        return ("—", "—")
    best_row = max(rows, key=lambda r: (r.get("mean_r@1", 0.0), -r.get("validation_loss", 999.0)))
    min_loss_row = min(rows, key=lambda r: r.get("validation_loss", 999.0))
    min_loss_str = f"{min_loss_row.get('validation_loss', 0):.3f} (ep {min_loss_row.get('epoch', 0)})"
    best_r1_str = f"{best_row.get('mean_r@1', 0)*100:.2f}%"
    return (min_loss_str, best_r1_str)


def line_chart(series, key, title, ylabel, yfmt='{:.2f}'):
    """One inline SVG; no external assets so the page works offline."""
    pts = [(lab, col, [(r['epoch'], r[key]) for r in rows if key in r]) for lab, col, rows in series if rows]
    if not pts:
        return f'<p class="muted">No data yet for {escape(title)}.</p>'
    xs = [x for _, _, p in pts for x, _ in p]
    ys = [y for _, _, p in pts for _, y in p]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = (y1 - y0) or 1.0
    y0, y1 = y0 - span * 0.08, y1 + span * 0.08
    def sx(x): return PAD_L + (x - x0) / max(x1 - x0, 1) * (W - PAD_L - PAD_R)
    def sy(y): return H - PAD_B - (y - y0) / (y1 - y0) * (H - PAD_T - PAD_B)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">']
    for i in range(5):  # gridlines and y axis labels
        gy = y0 + (y1 - y0) * i / 4
        parts.append(f'<line x1="{PAD_L}" y1="{sy(gy):.1f}" x2="{W-PAD_R}" y2="{sy(gy):.1f}" stroke="#dfe6ea"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{sy(gy)+4:.1f}" text-anchor="end" class="ax">{yfmt.format(gy)}</text>')
    step = 5 if x1 > 22 else 2
    for x in range(x0, x1 + 1):
        if x % step == 0 or x == x1:
            parts.append(f'<text x="{sx(x):.1f}" y="{H-PAD_B+18}" text-anchor="middle" class="ax">{x}</text>')
    parts.append(f'<text x="{(PAD_L+W-PAD_R)/2:.0f}" y="{H-4}" text-anchor="middle" class="ax">epoch</text>')
    parts.append(f'<text x="14" y="{H/2:.0f}" text-anchor="middle" class="ax" '
                 f'transform="rotate(-90 14 {H/2:.0f})">{escape(ylabel)}</text>')
    for lab, col, p in pts:
        d = ' '.join(('M' if i == 0 else 'L') + f'{sx(x):.1f},{sy(y):.1f}' for i, (x, y) in enumerate(p))
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
    parts.append('</svg>')
    legend = ' '.join(f'<span class="key"><i style="background:{c}"></i>{escape(l)}</span>' for l, c, _ in pts)
    return f'<figure><figcaption>{escape(title)}</figcaption>{"".join(parts)}<div class="legend">{legend}</div></figure>'
