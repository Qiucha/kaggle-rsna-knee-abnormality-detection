"""Interactive inline-SVG line chart for the HTML digest pages (hover tooltip, group filter pills, grouped legend).

The page needs its own CSS and JavaScript (classes ``chart-wrapper``, ``series-point``, ``legend-badge``, ...) to make the
chart interactive; the markup returned here is plain HTML and renders as a static chart without them.
"""

from html import escape

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 840, 340, 60, 24, 24, 46


def line_chart(
    series_list: list[dict],
    key: str,
    title: str,
    ylabel: str,
    yfmt: str = "{:.3f}",
    chart_idx: int = 1,
) -> str:
    """Generate an interactive SVG line chart with hover effects, tooltip, and grouped legend."""
    chart_id = f"chart-{key}-{chart_idx}"
    
    # Filter series that have data for this metric
    active_series = []
    for s in series_list:
        valid_pts = [(r["epoch"], r[key]) for r in s["rows"] if key in r and r[key] is not None]
        if valid_pts:
            active_series.append({**s, "pts": valid_pts})

    if not active_series:
        return f'<p class="muted">No data available yet for {escape(title)}.</p>'

    xs = [x for s in active_series for x, _ in s["pts"]]
    ys = [y for s in active_series for _, y in s["pts"]]
    if not xs or not ys:
        return f'<p class="muted">No points to plot for {escape(title)}.</p>'

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = (y1 - y0) or 1.0
    y0, y1 = y0 - span * 0.08, y1 + span * 0.08

    def sx(x: float) -> float:
        return PAD_L + (x - x0) / max(x1 - x0, 1) * (W - PAD_L - PAD_R)

    def sy(y: float) -> float:
        return H - PAD_B - (y - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

    # Build SVG content
    svg_parts = [
        f'<svg id="{chart_id}" class="interactive-chart" viewBox="0 0 {W} {H}" '
        f'role="img" aria-label="{escape(title)}" data-chart-id="{chart_id}">'
    ]
    
    # Background hover capture rect
    svg_parts.append(f'<rect x="{PAD_L}" y="{PAD_T}" width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}" fill="#fafbfc" rx="4"/>')

    # Horizontal grid lines and Y-axis labels
    for i in range(5):
        gy = y0 + (y1 - y0) * i / 4
        y_pos = sy(gy)
        svg_parts.append(f'<line x1="{PAD_L}" y1="{y_pos:.1f}" x2="{W-PAD_R}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-dasharray="2,2" stroke-width="1"/>')
        svg_parts.append(f'<text x="{PAD_L-8}" y="{y_pos+4:.1f}" text-anchor="end" class="chart-axis-text" font-size="11" fill="#64748b" font-family="system-ui,sans-serif">{yfmt.format(gy)}</text>')

    # X-axis ticks and labels
    step = 5 if x1 > 22 else (2 if x1 > 10 else 1)
    for x in range(int(x0), int(x1) + 1):
        if x % step == 0 or x == int(x1):
            x_pos = sx(x)
            svg_parts.append(f'<line x1="{x_pos:.1f}" y1="{H-PAD_B}" x2="{x_pos:.1f}" y2="{H-PAD_B+5}" stroke="#cbd5e1" stroke-width="1"/>')
            svg_parts.append(f'<text x="{x_pos:.1f}" y="{H-PAD_B+18}" text-anchor="middle" class="chart-axis-text" font-size="11" fill="#64748b" font-family="system-ui,sans-serif">{x}</text>')

    svg_parts.append(f'<text x="{(PAD_L+W-PAD_R)/2:.0f}" y="{H-10}" text-anchor="middle" font-size="12" font-weight="600" fill="#475569" font-family="system-ui,sans-serif">Epoch</text>')
    svg_parts.append(f'<text x="18" y="{H/2:.0f}" text-anchor="middle" font-size="12" font-weight="600" fill="#475569" font-family="system-ui,sans-serif" transform="rotate(-90 18 {H/2:.0f})">{escape(ylabel)}</text>')

    # Render series lines & points
    for s in active_series:
        sid = s["id"]
        group = s["group"]
        col = s["color"]
        sw = s.get("stroke_width", 2.4)
        dash_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        pts = s["pts"]

        svg_parts.append(f'<g class="chart-series group-{group}" data-series-id="{sid}" data-chart="{chart_id}">')
        if len(pts) == 1:
            cx, cy = sx(pts[0][0]), sy(pts[0][1])
            svg_parts.append(f'<circle class="series-point" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{col}" data-epoch="{pts[0][0]}" data-val="{pts[0][1]:.4f}" data-series="{escape(s["label"])}" data-group="{group}"/>')
        else:
            d = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
            svg_parts.append(f'<path class="series-path" d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>')
            for x, y in pts:
                cx, cy = sx(x), sy(y)
                svg_parts.append(f'<circle class="series-point" cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{col}" data-epoch="{x}" data-val="{y:.4f}" data-series="{escape(s["label"])}" data-group="{group}"/>')
        svg_parts.append("</g>")

    svg_parts.append("</svg>")

    # Render controls bar above the chart
    controls_html = f"""
    <div class="chart-controls-bar">
      <div class="filter-pills" data-chart="{chart_id}">
        <span class="filter-label">Filter:</span>
        <button class="pill-btn active" data-group="all" data-chart="{chart_id}">Show All</button>
        <button class="pill-btn" data-group="both_lora" data-chart="{chart_id}"><i class="dot" style="background:#047857;"></i> LoRA Series</button>
        <button class="pill-btn" data-group="both_ft" data-chart="{chart_id}"><i class="dot" style="background:#2563eb;"></i> Both Full FT</button>
        <button class="pill-btn" data-group="img_ft" data-chart="{chart_id}"><i class="dot" style="background:#64748b;"></i> Image-Only</button>
        <button class="pill-btn" data-group="holdout" data-chart="{chart_id}"><i class="dot" style="background:#ea580c;"></i> Holdout</button>
      </div>
      <div class="chart-hint">Hover line/legend to highlight &bull; Click legend to isolate</div>
    </div>
    """

    # Render grouped legend
    groups_dict: dict[str, list[dict]] = {
        "both_lora": [],
        "both_ft": [],
        "img_ft": [],
        "holdout": [],
    }
    for s in active_series:
        g = s["group"]
        if g in groups_dict:
            groups_dict[g].append(s)

    group_titles = {
        "both_lora": "⚡ Both-Tower LoRA (Emerald Hue)",
        "both_ft": "🔄 Both-Tower Full FT (Blue Hue)",
        "img_ft": "🖼️ Image-Only Full FT (Slate Hue)",
        "holdout": "🛡️ Generalizability Holdout (Amber/Coral Hue)",
    }

    legend_sections = []
    for g_key, g_items in groups_dict.items():
        if not g_items:
            continue
        badges = []
        for s in g_items:
            sid = s["id"]
            col = s["color"]
            lbl = s["label"]
            # get best or final stat for this key
            pts = s["pts"]
            if "loss" in key:
                best_val = min(y for _, y in pts)
                stat_str = f"min: {best_val:.3f}"
            elif "r@1" in key:
                best_val = max(y for _, y in pts)
                stat_str = f"peak: {best_val*100:.1f}%"
            else:
                best_val = pts[-1][1]
                stat_str = f"{best_val:.1f}"

            dash_style = 'border-style:dashed;' if s.get("dash") else ''
            badges.append(
                f'<button class="legend-badge" data-series-id="{sid}" data-chart="{chart_id}" data-group="{g_key}">'
                f'<span class="swatch" style="background:{col};{dash_style}"></span>'
                f'<span class="name">{escape(lbl)}</span>'
                f'<span class="val">{stat_str}</span>'
                f'</button>'
            )

        legend_sections.append(
            f'<div class="legend-group group-{g_key}">'
            f'<div class="legend-group-header">{group_titles.get(g_key, g_key)}</div>'
            f'<div class="legend-items">{"".join(badges)}</div>'
            f'</div>'
        )

    return f"""
    <div class="chart-wrapper" data-chart="{chart_id}">
      <div class="chart-header">
        <h3 class="chart-title">{escape(title)}</h3>
      </div>
      {controls_html}
      <div class="chart-svg-container" style="position:relative;">
        {"".join(svg_parts)}
        <div id="tooltip-{chart_id}" class="chart-tooltip" style="display:none;"></div>
      </div>
      <div class="grouped-legend">
        {"".join(legend_sections)}
      </div>
    </div>
    """
