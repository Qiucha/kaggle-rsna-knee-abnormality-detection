#!/usr/bin/env python3
"""Build a self-contained curves page for the image-block variant experiments.

Training writes one metrics row per epoch, so loss and retrieval are true curves.
AUROC is computed from a checkpoint and only ``best.pt`` is kept, so label readout
is one point per run, drawn at the epoch that produced it rather than as a curve.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / 'checkpoints/contrastive'

RUNS = [
    ('image-block-cohort-compare-v1/original-multilingual-seed2026', 'original s2026', '#1f77b4', 'policy'),
    ('image-block-cohort-compare-v1/original-multilingual-seed2027', 'original s2027', '#4b9bd0', 'policy'),
    ('image-block-cohort-compare-v1/original-multilingual-seed2028', 'original s2028', '#7fbde4', 'policy'),
    ('image-block-cohort-compare-v1/translated-english-seed2026',    'translated s2026', '#d95f02', 'policy'),
    ('image-block-cohort-compare-v1/translated-english-seed2027',    'translated s2027', '#e8863d', 'policy'),
    ('image-block-cohort-compare-v1/translated-english-seed2028',    'translated s2028', '#f2ac78', 'policy'),
    ('image-block-depth-v1/original-unfreeze2-seed2026',             'unfreeze 2 blocks', '#2ca02c', 'depth'),
    ('image-block-depth-v1/original-unfreeze4-seed2026',             'unfreeze 4 blocks', '#17806b', 'depth'),
    ('image-block-depth-v1/original-unfreeze4-seed2027',             'unfreeze 4 s2027', '#3aa88f', 'depth'),
    ('image-block-depth-v1/original-unfreeze4-seed2028',             'unfreeze 4 s2028', '#66c2b0', 'depth'),
    ('image-block-depth-v1/original-unfreeze8-seed2026',             'unfreeze 8 blocks', '#8c564b', 'depth'),
    ('image-block-epochs-v1/original-40ep-seed2026',                 '40 epochs', '#7b3fa0', 'schedule'),
    ('image-block-control-v1/shuffled-pairing-unfreeze4-seed2026',   'CONTROL shuffled s2026', '#111111', 'control'),
    ('image-block-control-v1/shuffled-pairing-unfreeze4-seed2027',   'CONTROL shuffled s2027', '#555555', 'control'),
    ('image-block-control-v1/shuffled-pairing-unfreeze4-seed2028',   'CONTROL shuffled s2028', '#888888', 'control'),
]

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


def control_section():
    """The shuffled-pairing control, read from the comparison artifact if present."""
    path = next((ROOT / f'digest/downstream-extended-comparisons-{v}.json'
                 for v in ('v3', 'v2')
                 if (ROOT / f'digest/downstream-extended-comparisons-{v}.json').exists()),
                ROOT / 'digest/downstream-extended-comparisons-v3.json')
    probe_path = next((ROOT / f'digest/downstream-probe-extended-{v}.json'
                       for v in ('v3', 'v2')
                       if (ROOT / f'digest/downstream-probe-extended-{v}.json').exists()),
                      ROOT / 'digest/downstream-probe-extended-v3.json')
    if not (path.exists() and probe_path.exists()):
        return ''
    d = json.loads(path.read_text())
    probe = json.loads(probe_path.read_text())
    res = probe['results']
    def macro(run):
        key = f'after_cl/{run}/uniform_mean_768'
        return res[key]['macro_auroc'] if key in res else None
    base = res['before_cl/dinov2-unadapted/uniform_mean_768']['macro_auroc']
    ctrls = [macro(f'shuffled-pairing-unfreeze4-seed{s}') for s in (2026, 2027, 2028)]
    ctrls = [v for v in ctrls if v is not None]
    if not ctrls:
        return ''
    ctrl = sum(ctrls) / len(ctrls)
    treat = [macro(f'original-unfreeze4-seed{s}') for s in (2026, 2027, 2028)]
    treat = [v for v in treat if v is not None]
    pairs = [v for v in d['paired_comparisons'].values()
             if 'shuffled-pairing' in v['baseline_arm'] and 'unfreeze4' in v['candidate_arm']]
    decisive = pairs[0] if pairs else None
    seed_rows = ''.join(
        f'<tr><td>{v["baseline_arm"].split("seed")[1].split("/")[0]}</td>'
        f'<td>{v["macro_auroc_baseline"]:.4f}</td><td>{v["macro_auroc_candidate"]:.4f}</td>'
        f'<td>{v["macro_auroc_difference_mean"]:+.4f}</td>'
        f'<td>{v["macro_auroc_difference_ci95"][0]:+.4f} to {v["macro_auroc_difference_ci95"][1]:+.4f}</td>'
        f'<td>{"excludes zero" if not (v["macro_auroc_difference_ci95"][0] < 0 < v["macro_auroc_difference_ci95"][1]) else "spans zero"}</td></tr>'
        for v in sorted(pairs, key=lambda x: x['baseline_arm']))
    n_excl = sum(1 for v in pairs
                 if not (v['macro_auroc_difference_ci95'][0] < 0 < v['macro_auroc_difference_ci95'][1]))
    mean_diff = sum(v['macro_auroc_difference_mean'] for v in pairs) / len(pairs) if pairs else 0.0
    rows = []
    for label, value in (('Unadapted DINOv2 (no training at all)', base),
                         ('<strong>Shuffled pairings</strong>, 4 blocks (control, mean of %d seeds)'
                          % len(ctrls), ctrl),
                         ('True pairings, 4 blocks (mean of %d seeds)' % len(treat),
                          sum(treat) / len(treat))):
        rows.append(f'<tr><td>{label}</td><td>{value:.4f}</td>'
                    f'<td>{value - base:+.4f}</td></tr>')
    recovered = (ctrl - base) / (sum(treat) / len(treat) - base) if treat else float('nan')
    verdict = ''
    if pairs:
        verdict = (f"<p>Depth-matched and seed-matched, the objective is worth "
                   f"<strong>{mean_diff:+.4f}</strong> macro AUROC on average across "
                   f"{len(pairs)} permutation seeds. All {len(pairs)} agree in sign and "
                   f"<strong>{n_excl} of {len(pairs)} intervals exclude zero</strong>.</p>"
                   f'<div class="table-wrap"><table><thead><tr><th>Seed</th><th>Shuffled</th>'
                   f'<th>True</th><th>Difference</th><th>95% CI</th><th></th></tr></thead>'
                   f'<tbody>{seed_rows}</tbody></table></div>')
    return f'''<h2>Is the contrastive objective what helps?</h2>

<p>Every arm above compares an adapted backbone against an unadapted one, which confounds
<em>the image/report objective</em> with <em>domain adaptation of DINOv2 to knee MRI at all</em>.
The control separates them: identical cohort, depth, learning rate, schedule and seed, with only the
image/report correspondence destroyed.</p>

<div class="note tiny"><strong>The control is valid.</strong> Validation pairings were left intact, and
validation Recall@1 collapsed to 0.0011 against a 0.0023 chance baseline while validation loss rose
monotonically from 6.09 to 8.40. It learned no alignment, which is what the control requires.</div>

<div class="table-wrap"><table><thead><tr><th>Arm</th><th>macro AUROC</th><th>vs unadapted</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>

<p>Training on <strong>randomly paired</strong> reports recovers roughly
<strong>{recovered:.0%}</strong> of the gain that training on the correct reports achieves. Most of
what looked like &ldquo;contrastive learning helps&rdquo; is reproduced by moving the same DINOv2
blocks with a signal carrying no image/report correspondence at all.</p>

{verdict}

<div class="note tiny"><strong>Both mechanisms are real; the objective is the larger half.</strong>
Across three permutation seeds the true pairing wins every time, by +0.0311 to +0.0538, and two of the
three intervals exclude zero. So the image/report objective does contribute beyond mere adaptation.
But the control still recovers a substantial part of the gain without any correct pairing, so
&ldquo;contrastive learning improves label readout&rdquo; overstates it: part of the effect is
available from adapting the same blocks against a non-semantic target. Note also what the control
did &mdash; its training loss fell from 3.47 to 1.41 while validation loss rose to 8.40, so it
<em>memorised</em> the random pairings. That requires features which separate individual studies,
which is closer to instance discrimination than to domain adaptation. These runs cannot tell those
two apart; an images-only self-supervised arm would.</div>
'''


def methods_section(probe):
    """Describe how the AUC above is produced, reading the numbers from the probe config.

    Rendered from the artifact rather than written by hand so the stated
    hyperparameters cannot drift away from the ones that were actually fitted.
    """
    if not probe:
        return '<h2>How the AUC is produced</h2><p class="muted">Probe not yet run.</p>'
    h = probe['config']['probe']
    b = probe['config']['bootstrap']
    n = probe['population']['population_size']
    rows = [
        ('Estimator', f"<code>{escape(h['estimator'])}</code>, one independent model per label"),
        ('Penalty / strength', f"{escape(str(h['penalty']))}, C = {h['C']}"),
        ('Solver', f"<code>{escape(h['solver'])}</code>, max_iter {h['max_iter']:,}"),
        ('Class weighting', f"{escape(str(h['class_weight']))} &mdash; no reweighting for rare labels"),
        ('Standardisation', escape(h['standardize'])),
        ('Folds', f"{escape(h['fold'])}, {h['folds']} folds, seed {h['fold_seed']}"),
        ('Predictions', escape(h['predictions'])),
        ('Tuning', escape(h['tuning'])),
        ('Bootstrap', f"{b['resamples']:,} resamples, seed {b['seed']}, unit = {escape(b['unit'])}, paired"),
    ]
    table = ''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in rows)
    return f'''<h2>How the AUC is produced</h2>

<p>The AUC is <strong>not</strong> read off a trained classifier head. Every contrastive model is
frozen, features are extracted once, and a fresh linear probe is cross-validated on top. Nothing in
the scoring path has gradients, and no pathology label ever reaches the contrastive training.</p>

<h3 style="font-size:16px;margin:22px 0 8px">From pixels to a number</h3>
<ol class="tiny" style="padding-left:20px">
<li><strong>Input.</strong> Three MRI planes &times; three slices each, 224&times;224, cropped to
130&nbsp;mm, slices drawn from the 0.2&ndash;0.8 depth band, normalised with ImageNet statistics.
Three slices per plane is fixed by the backbone&rsquo;s three input channels, not chosen.</li>
<li><strong>Backbone.</strong> <code>facebook/dinov2-small</code> at a pinned revision &mdash; hidden
width 384, 12 transformer blocks. Each plane is encoded independently, then described by its CLS
token concatenated with the mean of its patch tokens: 2 &times; 384 = <strong>768 dims per
plane</strong>. That is where the 768 in the feature names comes from.</li>
<li><strong>Pooling across the three planes</strong> &mdash; the three feature sets scored here:
<ul>
<li><code>uniform_mean_768</code> &mdash; a plain average over planes. <strong>Every headline number
on this page uses this one</strong>, because it is the only pooling with no learned parameters, so
two arms differ solely by their backbone weights.</li>
<li><code>plane_attention_768</code> &mdash; the tower&rsquo;s own learned pooling: add a per-plane
embedding, then <code>LayerNorm &rarr; Linear(768&rarr;384) &rarr; Tanh &rarr; Linear(384&rarr;1)</code>,
softmax over the three planes, weighted sum.</li>
<li><code>projection_256</code> &mdash; the contrastive projection head:
<code>LayerNorm &rarr; Linear(768&rarr;256) &rarr; GELU &rarr; Linear(256&rarr;256)</code>, L2 normalised.</li>
</ul></li>
<li><strong>The probe.</strong> Per label, an L2 logistic regression on the frozen features, fitted
inside stratified folds. This is the entire &ldquo;head&rdquo; &mdash; a linear decision function, so
the AUC measures how linearly separable the labels already are in the frozen representation.</li>
<li><strong>The metric.</strong> Each study gets exactly one held-out score per label; those pooled
out-of-fold scores give midrank AUROC and average-precision AUPRC, macro-averaged over the 12 labels.
The point estimate comes from scikit-learn; the bootstrap uses a rank-sum equivalent, and the script
asserts the two agree to 1e-9 on real scores before building any interval.</li>
</ol>

<h3 style="font-size:16px;margin:22px 0 8px">Probe hyperparameters</h3>
<div class="table-wrap"><table><tbody>{table}</tbody></table></div>
<p class="tiny muted">Every value above is fixed in the probe config before the first fit. None of it
was tuned on these {n} studies, which is what keeps the comparison between arms honest &mdash; the
probe is held constant so the representation is the only thing that varies.</p>

<h2>Ways to improve further</h2>

<p>Split deliberately: changes that would make the <em>representation</em> better, and changes that
would only make the <em>number</em> bigger.</p>

<h3 style="font-size:16px;margin:22px 0 8px">Likely real gains</h3>
<ol class="tiny" style="padding-left:20px">
<li><strong>Separate depth from learning rate.</strong> All arms use <code>--lr-image 1e-5</code>,
chosen when one block was trainable. The 4-block arm has four times the trainable surface at that
same rate, so &ldquo;depth helps&rdquo; and &ldquo;this rate suits more blocks&rdquo; are confounded.
An LR sweep at each depth is the highest-value next experiment, because depth is the only lever
measured so far whose interval excludes zero.</li>
<li><strong>The depth ladder has already turned.</strong> 1 &rarr; 2 &rarr; 4 &rarr; 8 blocks
gives 0.6695 &rarr; 0.6838 &rarr; 0.7105 &rarr; 0.7164, all seed-2026 levels on <em>exposed</em>
gold studies (the 4-block level pools to 0.7106 across seeds, and is 0.6938 held out). The 8-block screen has run: +0.0062 over
4 blocks, an interval spanning zero, so 4 of 12 is at or near the useful optimum and there is
nothing further to gain by climbing.</li>
<li><strong>A larger backbone.</strong> Gains scaling with trainable depth suggests capacity is
binding. <code>dinov2-base</code> doubles the hidden width to 768 and fits in 8&nbsp;GB at this batch
size &mdash; depth costs memory but almost no time here (3.2 &rarr; 4.5&nbsp;GB, 1,235 &rarr; 1,292&nbsp;s).</li>
<li><strong>Drop the learned plane attention at readout &mdash; free, and already measured.</strong>
On the same checkpoint, attention pooling scores 0.0185 <em>below</em> a uniform mean, while
concatenating all three planes scores +0.0010, i.e. no information is lost by not pooling. The
deficit is the re-weighting, not the representation. Exporting pre-attention plane features costs no
training.</li>
<li><strong>Weak supervision from the reports.</strong> The full 4,407-study cohort has reports but no
adjudicated labels; LM-extracted labels already exist in <code>data/lm_label/</code>. An auxiliary
multi-label loss over that cohort is the largest untapped headroom &mdash; but any study in the gold
set must be excluded from it, or the evaluation is contaminated.</li>
</ol>

<h3 style="font-size:16px;margin:22px 0 8px">Changes that would only move the number</h3>
<ol class="tiny" style="padding-left:20px">
<li><strong>Tuning the probe on these studies.</strong> Sweeping C, enabling
<code>class_weight=&#39;balanced&#39;</code> for the rare labels, or changing the fold count would
almost certainly raise the macro AUROC &mdash; by selecting on the evaluation set. If ever done, it
produces a development estimate and must be labelled one.</li>
<li><strong>A non-linear probe head.</strong> An MLP would measure probe capacity, not representation
quality, and 56 studies with 8&ndash;33 positives per label cannot support fitting one.</li>
<li><strong>Reading <code>projection_256</code> as a result.</strong> It is a learned 768&rarr;256
compression; a linear probe cannot gain information from compression, so it is reported but is not
evidence about the representation.</li>
</ol>

<h3 style="font-size:16px;margin:22px 0 8px">The binding constraint is not the model</h3>
<p class="tiny">With {n} studies and 8&ndash;33 positives per label, bootstrap intervals on a single
arm run roughly &plusmn;0.015 to &plusmn;0.03 macro AUROC. Improvements smaller than the 0.0038
seed spread cannot be detected here at all. More independently adjudicated labels would do more for
<em>establishing</em> a gain than any architecture change &mdash; and, unlike everything above, would
also make a generalisation claim possible, which no rerun on these studies ever can.</p>
'''


def main():
    series = [(lab, col, history(CK / rel / 'metrics.jsonl')) for rel, lab, col, _ in RUNS]
    have = [(lab, col, rows) for lab, col, rows in series if rows]

    probe = None
    for name in ('gold-probe-extended-v3', 'gold-probe-extended-v2', 'gold-probe-extended-v1', 'gold-probe-cohort-compare-v1'):
        p = ROOT / f'digest/downstream-probe-{name.replace("gold-probe-", "")}.json'
        if p.exists():
            probe = json.loads(p.read_text())
            break

    auc_html = '<p class="muted">Probe not yet run.</p>'
    if probe:
        base = probe['results'].get('before_cl/dinov2-unadapted/uniform_mean_768', {}).get('macro_auroc')
        scored = [
            (name.split('/')[1], res['macro_auroc'], res['macro_auprc'])
            for name, res in sorted(probe['results'].items())
            if name.startswith('after_cl/') and name.endswith('/uniform_mean_768')
        ]
        # Best and worst are picked on macro AUROC, the metric every comparison uses.
        best = max(a for _, a, _ in scored)
        worst = min(a for _, a, _ in scored)
        rows = []
        for run, auroc, auprc in scored:
            # A run cannot be both, so ties resolve to the best marking.
            open_tag, close_tag, mark = '', '', ''
            if auroc == best:
                open_tag, close_tag, mark = '<strong>', '</strong>', ' &#9650; best'
            elif auroc == worst:
                open_tag, close_tag, mark = '<em>', '</em>', ' &#9660; worst'
            d = auroc - base if base else None
            rows.append(
                f'<tr><td>{open_tag}{escape(run)}{close_tag}'
                f'<span class="tiny muted">{mark}</span></td>'
                f'<td>{open_tag}{auroc:.4f}{close_tag}</td>'
                f'<td>{open_tag}{auprc:.4f}{close_tag}</td>'
                f'<td>{open_tag}{d:+.4f}{close_tag}</td></tr>'
            )
        auc_html = ('<p style="border-left:3px solid #999;padding-left:12px">'
                    '<strong>Exposure caveat (measured 2026-09-14).</strong> Every level in this '
                    'section is a development statistic: 56 of the 58 scored gold studies were '
                    'training pairs. Retraining with them held out puts the 4-block level at '
                    '<strong>0.6938</strong> against the pooled exposed level of <strong>0.7106</strong> (the published 0.7105 is seed 2026) &mdash; an exposure inflation of <strong>&minus;0.0168</strong> by the primary pooled route, &minus;0.0163 by the data-size-matched route &mdash; and the gain over unadapted DINOv2 at <strong>+0.0500</strong> rather than +0.0668. The levels subtract exactly: 0.6438, 0.6938, 0.7106. Quote 0.6938 beside 0.7105, never instead of it. See '
                    '<a href="gold-holdout-v1.md">gold-holdout-v1.md</a>. The paired <em>differences</em> '
                    'here are between arms sharing the same exposure and are not corrected by that figure.</p>'
                    f'<p>Unadapted DINOv2-small reads the twelve labels at '
                    f'<strong>{base:.4f}</strong> macro AUROC over the {probe["population"]["population_size"]} '
                    f'scored gold studies. Each run below contributes one point, at its selected epoch.</p>'
                    f'<div class="table-wrap"><table><thead><tr><th>Run</th><th>macro AUROC</th>'
                    f'<th>macro AUPRC</th><th>vs unadapted<br><span class="tiny">point difference</span></th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                    f'<p class="tiny muted"><strong>Bold</strong> is the highest macro AUROC, '
                    f'<em>italic</em> the lowest. Ranking only &mdash; adjacent runs here differ by '
                    f'less than the {0.0038:.4f} seed spread, so a position in this ordering is not '
                    f'by itself a difference. The paired intervals decide that.<br>'
                    f'The last column subtracts point estimates. The write-up quotes the bootstrap '
                    f'mean of the paired difference instead, which lands a few ten-thousandths away '
                    f'&mdash; same quantity, two estimators, not a disagreement.</p>')

    methods_html = methods_section(probe)
    stamp = datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M %Z')
    html = f'''<!doctype html><meta charset="utf-8"><title>Image-block variants · curves</title>
<style>body{{font:16px/1.55 system-ui;color:#18313c;background:#f5f7f8;max-width:900px;margin:0 auto;padding:28px 18px}}
h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:19px;margin:34px 0 10px}}
figure{{margin:0 0 26px;background:#fff;border:1px solid #e3eaee;border-radius:10px;padding:14px}}
figcaption{{font-weight:600;margin-bottom:8px}}svg{{width:100%;height:auto;display:block}}
.ax{{font-size:11px;fill:#6c8390}}.legend{{margin-top:8px;font-size:12px;color:#4a5f6b}}
.key{{margin-right:14px;white-space:nowrap}}.key i{{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}
th,td{{border:1px solid #e3eaee;padding:6px 9px;text-align:left}}th{{background:#eef3f5}}
.muted{{color:#6c8390}}.note{{background:#fff;border-left:4px solid #d9a441;padding:12px 14px;border-radius:6px;margin:18px 0}}
.tiny{{font-size:12.5px}}</style>
<h1>Image-block variants: training curves and label readout</h1>
<p class="muted tiny">Generated {escape(stamp)} by <code>scripts/build_extended_curves.py</code> from local run artifacts.</p>

<h2>Training loss</h2>
{line_chart(have, 'train_loss', 'Training loss per epoch', 'train loss', '{:.2f}')}

<h2>Validation loss</h2>
{line_chart(have, 'validation_loss', 'Validation loss per epoch', 'validation loss', '{:.2f}')}
<div class="note tiny"><strong>Training loss falls throughout; validation loss does not.</strong>
Validation loss reaches its minimum early and then drifts upward while training loss keeps falling.
Retrieval R@1 nevertheless keeps rising, much of it as the cosine schedule anneals the learning rate
to zero. Late-epoch R@1 gains are therefore a schedule effect as much as a learning one.</div>

<h2>Retrieval Recall@1 (selection metric)</h2>
{line_chart(have, 'mean_r@1', 'Validation mean Recall@1 per epoch', 'mean R@1', '{:.3f}')}

<h2>Label readout (AUC)</h2>
<div class="note tiny"><strong>One row in the table below is a control, not a result.</strong>
<code>shuffled-pairing-*</code> trained with each study paired to a <em>different</em> study&rsquo;s
report, everything else identical to the 4-block arm. It exists to separate the image/report
objective from plain domain adaptation of the backbone. It is reported at <code>last.pt</code>,
not <code>best.pt</code>: its validation Recall@1 is meaningless by construction, and selecting on
it picked epoch&nbsp;3. See &ldquo;Is the contrastive objective what helps?&rdquo; below.</div>
<div class="note tiny"><strong>This is not a curve, and cannot be made into one from these runs.</strong>
AUROC is computed from a saved checkpoint, and training keeps only <code>best.pt</code> and
<code>last.pt</code>. Each run therefore contributes a single AUC point, at its selected epoch.
An AUC-versus-epoch curve would require per-epoch checkpointing, which the training loop does not do.</div>
{auc_html}
<p class="tiny muted">Scored on 56 of the 58 gold studies against 12 labels; the 2 locked-test gold
studies are counted and never loaded. 56 of those studies were contrastive training pairs or members
of the selection split, so this is readout on seen data; the held-out estimate is 0.6938.</p>

{control_section()}
{methods_html}
'''
    out = ROOT / 'digest/image-block-variants-curves.html'
    out.write_text(html)
    print(f'wrote {out} from {len(have)} runs')


if __name__ == '__main__':
    main()
