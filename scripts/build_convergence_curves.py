#!/usr/bin/env python3
"""Build the curves page that asks whether the regularised and crop150 arms converged.

The drop-path sweep and the crop150 re-baseline were each read as "this lever costs
retrieval".  Neither reading looked at the shape of the training curve, so neither
separated *the lever makes the representation worse* from *the lever slows optimisation
and a fixed 20-epoch budget stopped it early*.  This page draws the curves that
distinguish them.

Chart primitives are imported from ``build_extended_curves`` rather than re-implemented;
this script exists separately only because its output file and its run set are different,
and because mixing crop130 and crop150 arms into that page's shared panels would put
non-comparable pixels on one axis.

Runs are identified from each checkpoint's own ``config.json``.  Never parse a run
parameter out of a directory name -- that is what produced three silent failures in
``analyze_drop_path_sweep.py``.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from html import escape
from pathlib import Path

from build_extended_curves import line_chart, history

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / 'checkpoints/contrastive'
AUDIT = ROOT / 'digest/convergence-audit.json'
OUT = ROOT / 'digest/convergence-curves.html'

SEEDS = (2026, 2027, 2028)
# Arm label -> colour.  Graded by drop-path rate so the dose-response reads off the legend.
COLOURS = {
    'baseline (crop130, 4 blocks)': '#111111',
    'drop-path 0.005': '#cfe3f5',
    'drop-path 0.01': '#9ec8e8',
    'drop-path 0.02': '#5fa5d6',
    'drop-path 0.05': '#2b7cb8',
    'drop-path 0.1': '#0b4f7d',
    'crop150mm': '#d95f02',
}
ROOTS = ['image-block-depth-v1', 'drop-path-v1', 'drop-path-sweep-v1', 'crop150-v1']


def arm_of(cfg):
    """The arm's identity, from the fields the checkpoint itself records."""
    rate = float(cfg.get('image_drop_path_rate', 0.0) or 0.0)
    crop = float(cfg.get('crop_mm', 130.0))
    if crop != 130.0:
        return f'crop{crop:.0f}mm', 'crop', rate, crop
    if rate > 0:
        return f'drop-path {rate:g}', 'drop_path', rate, crop
    return 'baseline (crop130, 4 blocks)', 'baseline', rate, crop


def collect_arms():
    """Seed-averaged epoch history per arm; 21 individual lines would be unreadable."""
    buckets = {}
    for root in ROOTS:
        for d in sorted((CK / root).iterdir()):
            mpath, cpath = d / 'metrics.jsonl', d / 'config.json'
            if not (mpath.exists() and cpath.exists()):
                continue
            cfg = json.loads(cpath.read_text())
            if int(cfg.get('image_unfreeze_last', 0)) != 4 or int(cfg['seed']) not in SEEDS:
                continue
            rows = history(mpath)
            if len(rows) < 10:
                continue  # smoke runs
            arm, family, rate, crop = arm_of(cfg)
            buckets.setdefault(arm, {'family': family, 'rate': rate, 'crop': crop, 'runs': []})
            buckets[arm]['runs'].append(rows)

    arms = {}
    for arm, b in buckets.items():
        n_ep = min(len(r) for r in b['runs'])
        mean_rows = []
        for i in range(n_ep):
            # Index-wise averaging is only valid if the seeds are epoch-aligned. Assert rather
            # than trust it: a silent mis-plot is the failure mode this project keeps hitting.
            epochs = {r[i]['epoch'] for r in b['runs']}
            if len(epochs) != 1:
                raise ValueError(f'{arm}: seeds disagree on epoch at row {i}: {sorted(epochs)}')
            row = {'epoch': b['runs'][0][i]['epoch']}
            for key in ('train_loss', 'validation_loss', 'mean_r@1', 'inverse_temperature',
                        'image_to_text_r@5', 'image_to_text_r@10'):
                vals = [r[i][key] for r in b['runs'] if key in r[i]]
                if vals:
                    row[key] = sum(vals) / len(vals)
            mean_rows.append(row)
        arms[arm] = dict(b, rows=mean_rows, n_seeds=len(b['runs']))
    return arms


def series_for(arms, families):
    """(label, colour, rows) triples in dose order, for the requested arm families."""
    sel = [(a, v) for a, v in arms.items() if v['family'] in families]
    sel.sort(key=lambda kv: (kv[1]['crop'], kv[1]['rate']))
    return [(f"{a} (mean of {v['n_seeds']})", COLOURS.get(a, '#888'), v['rows']) for a, v in sel]


def diagnostics_table(audit):
    order = sorted(audit['per_arm'].items(), key=lambda kv: (kv[1]['crop_mm'], kv[1]['drop_path_rate']))
    rows = []
    for arm, s in order:
        n = s['n_runs']
        boundary = f"{s['n_best_at_boundary']}/{n}"
        falling = f"{s['n_val_still_falling']}/{n}"
        strong = s['drop_path_rate'] >= 0.05
        o, c = ('<strong>', '</strong>') if strong else ('', '')
        rows.append(
            f"<tr><td>{o}{escape(arm)}{c}</td><td>{s['mean_best_r@1']:.4f}</td>"
            f"<td>{s['mean_val_loss_min_epoch']:.1f}</td>"
            f"<td>{o}{s['mean_val_rise_from_min']:+.3f}{c}</td>"
            f"<td>{falling}</td><td>{boundary}</td>"
            f"<td>{s['mean_final_train_loss']:.3f}</td></tr>")
    return ('<div class="table-wrap"><table><thead><tr>'
            '<th>Arm</th><th>best R@1<br><span class="tiny">mean of 3 seeds</span></th>'
            '<th>val-loss<br>min epoch</th>'
            '<th>val-loss rise<br><span class="tiny">from its own min</span></th>'
            '<th>still falling<br><span class="tiny">at epoch 20</span><br>'
            '<span class="tiny muted">weaker &mdash; two-point sign test</span></th>'
            '<th>best epoch<br><span class="tiny">at boundary</span></th>'
            '<th>final<br>train loss</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def main():
    audit = json.loads(AUDIT.read_text())
    arms = collect_arms()
    dp = series_for(arms, {'baseline', 'drop_path'})
    crop = series_for(arms, {'baseline', 'crop'})
    stamp = datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M %Z')

    html = f'''<!doctype html><meta charset="utf-8"><title>Convergence audit · drop-path and crop150</title>
<style>body{{font:16px/1.55 system-ui;color:#18313c;background:#f5f7f8;max-width:900px;margin:0 auto;padding:28px 18px}}
h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:19px;margin:34px 0 10px}}h3{{font-size:16px;margin:22px 0 8px}}
figure{{margin:0 0 26px;background:#fff;border:1px solid #e3eaee;border-radius:10px;padding:14px}}
figcaption{{font-weight:600;margin-bottom:8px}}svg{{width:100%;height:auto;display:block}}
.ax{{font-size:11px;fill:#6c8390}}.legend{{margin-top:8px;font-size:12px;color:#4a5f6b}}
.key{{margin-right:14px;white-space:nowrap}}.key i{{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}
th,td{{border:1px solid #e3eaee;padding:6px 9px;text-align:left}}th{{background:#eef3f5}}
.muted{{color:#6c8390}}.note{{background:#fff;border-left:4px solid #d9a441;padding:12px 14px;border-radius:6px;margin:18px 0}}
.stop{{background:#fff;border-left:4px solid #c0392b;padding:12px 14px;border-radius:6px;margin:18px 0}}
.ok{{background:#fff;border-left:4px solid #2e8b57;padding:12px 14px;border-radius:6px;margin:18px 0}}
.tiny{{font-size:12.5px}}.table-wrap{{overflow-x:auto}}</style>

<h1>Did the regularised and crop150 arms converge?</h1>
<p class="muted tiny">Generated {escape(stamp)} by <code>scripts/build_convergence_curves.py</code>
from local run artifacts. Evidence: <code>digest/convergence-audit.json</code>.</p>

<p>Two results were recorded without looking at the shape of the training curve:
<a href="drop-path-sweep-v1.md">the stochastic-depth sweep</a> and
<a href="crop150-rebaseline-v1.md">the 150&nbsp;mm crop re-baseline</a>. Both were read as
&ldquo;this lever costs retrieval&rdquo;. Neither reading could separate two different
explanations:</p>
<ol class="tiny" style="padding-left:20px">
<li>the lever makes the representation <strong>worse</strong>; or</li>
<li>the lever <strong>slows optimisation</strong>, and a fixed 20-epoch budget stopped the run
before it finished improving.</li>
</ol>
<p>The epoch histories already on disk distinguish them, and <strong>they give different answers for
the two levers</strong>.</p>

<div class="ok"><strong>Answer, in one line each.</strong>
<strong>Drop-path: the hypothesis holds</strong> &mdash; at rate 0.1 the run is still sitting
<em>at</em> its own validation minimum when the budget ends (rise from minimum <strong>+0.010</strong>,
against the baseline&rsquo;s +0.259), with the tail slope still negative in 3 of 3 seeds. Those arms
had not finished improving, so the sweep compared rates at unequal degrees of completion.
<strong>crop150: the hypothesis fails</strong> &mdash; it reached its validation minimum
<em>earlier</em> than the baseline (epoch 9.0 vs 11.3) and then overfit at least as hard, so its
lower retrieval is not a truncation artifact.</div>

<h2>The diagnostic table</h2>
<p>Two diagnostics per run, both from <code>metrics.jsonl</code>: how far validation loss drifted
<em>up</em> from its own minimum by the last epoch (the overfitting signature), and whether it was
still falling at the end (the under-training signature).</p>

{diagnostics_table(audit)}
<p class="tiny muted"><strong>The fourth column is the load-bearing one, not the fifth.</strong>
&ldquo;Still falling at epoch 20&rdquo; is a two-point sign test on tail differences of roughly
0.001&ndash;0.015 &mdash; at the low rates that is noise around zero, so its 1/3, 0/3, 1/3, 1/3
counts should <em>not</em> be read as a dose-response. The <strong>rise from each arm&rsquo;s own
minimum</strong> is monotone across all six rates and is the column the finding rests on.</p>

<p><strong>Read the fourth column down the drop-path rows.</strong> The validation-loss rise collapses
monotonically &mdash; <strong>+0.259 &rarr; +0.184 &rarr; +0.125 &rarr; +0.102 &rarr; +0.057 &rarr;
+0.010</strong> &mdash; and from rate 0.01 upward the validation minimum moves steadily later,
10.3 &rarr; 14.0 &rarr; 14.0 &rarr; <strong>16.0</strong> at rate 0.1. (The two lowest rates sit at
10.3, marginally <em>earlier</em> than the baseline&rsquo;s 11.3, so the minimum-epoch column is
monotone only over the upper four rates; the rise column is monotone over all six.) The regulariser
is doing exactly what a regulariser
does: it delays the onset of overfitting and it slows optimisation. At rate 0.1 the run has barely
begun to overfit when the budget ends.</p>

<div class="stop"><strong>This is a confound in the sweep, not a correction to its arithmetic.</strong>
Every rate was trained for the same 20 epochs, but the higher rates need more epochs to reach their
own optimum. So the sweep did not compare six rates at convergence &mdash; it compared six rates at
<em>progressively earlier points along their own trajectories</em>. The measured retrieval deficit at
high rates is therefore a mixture of a genuine effect and a truncation artifact, and these runs
cannot say how much of it is which.</div>

<h2>Training loss &mdash; drop-path arms</h2>
{line_chart(dp, 'train_loss', 'Training loss per epoch, seed-averaged', 'train loss', '{:.2f}')}
<div class="stop"><strong>Do not compare these levels across arms, and do not read the distance to
the validation panel as a generalisation gap.</strong> Two separate reasons, both of which have
produced a wrong claim in this project before:
<ul class="tiny">
<li><code>train_loss</code> is computed in-batch over 32 with augmentation on, in <em>train mode</em>
(<code>training.py:158</code>). <code>validation_loss</code> is the full 441-candidate matrix,
unaugmented, in <em>eval mode</em> (<code>evaluation.py:200</code>). They are different quantities on
different denominators.</li>
<li>Drop-path is <strong>active</strong> in the arm&rsquo;s train-mode forward pass and
<strong>identity</strong> in the baseline&rsquo;s, so any forward-pass noise raises the arm&rsquo;s
train loss regardless of what it memorised. The higher final train loss at higher rates is
<em>partly</em> this artifact and partly real slower optimisation.</li>
</ul>
<strong>The curve&rsquo;s shape is readable; its level against another arm is not.</strong> The
admissible memorisation route is the matched-N eval-mode gap in
<a href="drop-path-sweep-v1.md">drop-path-sweep-v1.md</a>.</div>

<h2>Validation loss &mdash; drop-path arms</h2>
{line_chart(dp, 'validation_loss', 'Validation loss per epoch, seed-averaged', 'validation loss', '{:.2f}')}
<p>This is the panel that answers the question. The baseline turns upward around epoch 11 and climbs
for the rest of the run. Each successive rate turns later and more shallowly, and
<strong>rate 0.1 has essentially not turned at all</strong> by epoch 20.</p>

<div class="note"><strong>Part of the vertical separation here is temperature, not representation
quality.</strong> The learned &tau;&#8315;&sup1; drifts downward with rate, toward the
validation-optimal value of 11.50 &mdash; and rescaling &tau; at evaluation time changes retrieval by
<em>exactly zero</em>. So a lower validation-loss curve is not by itself a better representation. The
next panel shows the drift so it can be discounted by eye; the sweep&rsquo;s own analysis recomputes
the loss gap at matched &tau;.</div>

<h2>Learned inverse temperature</h2>
{line_chart(dp, 'inverse_temperature', 'Learned inverse temperature per epoch, seed-averaged', '1/tau', '{:.1f}')}

<h2>Retrieval Recall@1 &mdash; the selection metric</h2>
{line_chart(dp, 'mean_r@1', 'Validation mean Recall@1 per epoch, seed-averaged', 'mean R@1', '{:.3f}')}
<p class="tiny muted">All curves here are over <strong>441</strong> validation candidates &mdash;
<strong>verified, not assumed</strong>: all seven arms resolve the same cohort manifest
(<code>contrastive-cohort.jsonl</code>, sha <code>54bd803e&hellip;</code>) and every run logs
&ldquo;validation: 441 studies&rdquo; at startup, the crop150 arm included
(<code>queue10.log:227,233,239</code>). The 433-candidate gold-holdout and data-size-control figures
must never be placed on this axis.</p>

<h2>The crop150 re-baseline is a different story</h2>
<div class="stop"><strong>These curves share no axis with the drop-path panels above.</strong>
<code>crop_mm</code> 130 &rarr; 150 changes the pixels, so the crop150 runs are not comparable to any
crop130 run. They are drawn beside the baseline only to compare <em>curve shape</em>, which is a
statement about optimisation, not about score.</div>

{line_chart(crop, 'validation_loss', 'Validation loss per epoch, crop130 baseline vs crop150', 'validation loss', '{:.2f}')}
{line_chart(crop, 'mean_r@1', 'Validation mean Recall@1 per epoch, crop130 baseline vs crop150', 'mean R@1', '{:.3f}')}

<p><strong>crop150 converged and then overfit; it was not truncated.</strong> Its validation minimum
arrives at epoch <strong>9.0</strong> against the baseline&rsquo;s 11.3 &mdash; <em>earlier</em>, not
later &mdash; and it then drifts up <strong>+0.280</strong>, slightly more than the baseline&rsquo;s
+0.259. Its best epoch is at the boundary in 0 of 3 seeds. Every signature points the same way:
the 150&nbsp;mm crop is a harder optimisation problem that the model finishes early and then
memorises. <strong>The user&rsquo;s hypothesis does not rescue this result</strong>, and
<a href="crop150-rebaseline-v1.md">crop150-rebaseline-v1.md</a>&rsquo;s reading stands.</p>

<h2>Would more epochs actually help? The one run that bears on it</h2>
<p><code>image-block-epochs-v1/original-40ep-seed2026</code> is the project's only run at a budget
other than 20 epochs. Its config-exact twin at 20 epochs is
<code>image-block-cohort-compare-v1/original-multilingual-seed2026</code> &mdash; diffed field by
field, <strong>zero</strong> differences apart from <code>epochs</code>.</p>

<div class="table-wrap"><table><thead><tr><th>Run (seed 2026, 1 adapted block, unregularised)</th>
<th>epochs</th><th>best R@1</th><th>at epoch</th><th>val loss, min &rarr; final</th></tr></thead><tbody>
<tr><td><code>original-multilingual-seed2026</code></td><td>20</td><td>0.1111</td><td>19</td><td>3.666 (ep8) &rarr; 3.781</td></tr>
<tr><td><code>original-40ep-seed2026</code></td><td>40</td><td><strong>0.1190</strong></td><td>33</td><td>3.687 (ep13) &rarr; <strong>4.752</strong></td></tr>
</tbody></table></div>

<p><strong>Doubling the budget improved retrieval by +0.0079 &mdash; while validation loss ran away
from 3.687 to 4.752.</strong> That is a large divergence in the loss and a small gain in the
selection metric, in the same run.</p>

<div class="stop"><strong>This is the single most important caveat on this page, and it cuts against
the diagnostic the rest of the page is built on.</strong> A rising validation loss does
<em>not</em> imply that more epochs cannot help retrieval &mdash; here the loss nearly ran away and
R@1 still improved for another 14 epochs. It is another instance of the dissociation in
<a href="val-loss-vs-retrieval-audit.md">val-loss-vs-retrieval-audit.md</a>.
<strong>So the convergence diagnostics above establish that the high-rate drop-path arms were still
improving when the budget ended; they do not by themselves predict how much retrieval a longer budget
would buy.</strong> Only a budget arm measures that, and it must be decided on R@1, not on
validation loss.</div>

<div class="note"><strong>Three reasons this run cannot settle the question on its own.</strong>
It adapts <strong>1</strong> block, not 4, so it is not a comparator for any arm on this page &mdash;
it is excluded from every panel and from the diagnostic table for that reason. The cosine schedule is
stretched over the full 40 epochs, so this is a different learning-rate trajectory throughout rather
than the 20-epoch run continued. And it is <strong>one seed</strong>, against a seed spread of 0.023.</div>

<h2>What this page deliberately does not show</h2>
<div class="stop"><strong>Per-class accuracy and AUROC cannot be curves here, and for the drop-path
and crop150 arms they do not exist even as points.</strong>
<ul class="tiny">
<li>AUROC comes from <code>evaluate_downstream_probe.py</code> run against a saved checkpoint, and
training keeps only <code>best.pt</code> and <code>last.pt</code>. There are no per-epoch checkpoints
to probe, so an AUROC-versus-epoch curve would require <strong>retraining with per-epoch saves</strong>.
<code>build_extended_curves.py</code> records the same limitation for the arms it covers.</li>
<li><strong>No downstream probe has ever been run on the drop-path or crop150 checkpoints.</strong>
The sweep&rsquo;s probe box is unchecked by decision (selection is retrieval R@1, and the arm lost on
it), and the crop150 probe is flagged in the handoff as the first thing to run
<em>with a human present</em> because it needs a hand-built config. So there is no macro AUROC, no
per-class AUROC and no per-class accuracy for these arms at all &mdash; not a missing curve, a
missing measurement.</li>
<li>In place of &ldquo;accuracy&rdquo; this page reports <strong>Recall@1</strong>, which is the
project&rsquo;s actual selection metric and is logged every epoch.</li>
</ul>
<strong>A probe pass on the crop150 checkpoints would produce the macro and per-class AUROC, and is
the obvious next step &mdash; but it is a user decision, not something to launch unattended.</strong></div>

<h2>Follow-up: the budget arm ran, and it half-confirms this page</h2>
<div class="ok"><strong>Both arms retrained at 40 epochs, 3 seeds
(<a href="budget-40ep-v1.md">budget-40ep-v1.md</a>).</strong> Both converged inside the new budget
(best epochs 23&ndash;35 of 40), so the comparison is now at convergence rather than at a truncation.
<strong>Drop-path 0.1 gained +0.0150 from the longer budget against the baseline&rsquo;s +0.0062</strong>
&mdash; the asymmetry this page predicted &mdash; and the deficit fell from &minus;0.0255 to
<strong>&minus;0.0167</strong>, its interval moving from [&minus;0.0421, &minus;0.0094] to
<strong>[&minus;0.0337, +0.0005]</strong>, which now grazes zero instead of excluding it.
<strong>But the deficit did not vanish</strong>, and the pre-registered rule returns
<strong>ambiguous</strong>. Truncation was real and worth about a third of the measured harm; it was
not the whole of it.</div>

<div class="note"><strong>The caveat above was the right one.</strong> This page warned that its
diagnostics establish only that the high-rate arms were <em>still improving</em> &mdash; not how much
a longer budget would buy &mdash; because the 40-epoch precedent improved retrieval while validation
loss ran away. That held: the arms were genuinely truncated, and the recovery was real but partial.</div>

<h2>What follows from this</h2>
<ol class="tiny" style="padding-left:20px">
<li><strong>&ldquo;Stochastic depth is closed at 4 blocks&rdquo; is too strong &mdash; and so is
&ldquo;it was all truncation&rdquo;.</strong> The budget arm settled this in part: a third of the
measured harm was budget, and a directional deficit survives at convergence without excluding zero.
Both clean stories fail, which is the honest result.</li>
<li><strong>A fixed epoch budget is not a neutral choice when the lever is a regulariser.</strong>
Projection-head dropout and random erasing are both still untested; either must train to a
convergence criterion or report budget as a co-varying factor.</li>
<li><strong>The recorded 0.1340 baseline is a 20-epoch number, not a ceiling</strong> &mdash; it
gained +0.0062 at 40 epochs. Anything comparing against it should say which budget it means.</li>
<li><strong>The &ldquo;every lever that removes information fails the same way&rdquo; through-line
survives, but weakened on one leg.</strong> The crop leg is unaffected &mdash; crop150 converged. The
stochastic-depth leg is now partly confounded with budget.</li>
<li><strong>Selection stays retrieval R@1.</strong> Nothing here was selected on label AUROC, and
nothing on this page changes that.</li>
</ol>

<p class="tiny muted">Runs identified from each checkpoint&rsquo;s own <code>config.json</code>
(<code>seed</code>, <code>image_drop_path_rate</code>, <code>crop_mm</code>), never from a directory
name. Curves are seed means over 3 seeds per arm. Nothing was committed; the test split was never
loaded.</p>
'''
    OUT.write_text(html)
    print(f'wrote {OUT} from {len(arms)} arms')


if __name__ == '__main__':
    main()
