#!/usr/bin/env python3
"""Build digest/project-state.html, a self-contained snapshot of the claude-translated-dfd07db experiments.

Reads the image-block and text-block runs under checkpoints/contrastive/claude-translated-dfd07db-*, the matching configs, and
digest/*.json audits (architecture-parameter-audit, downstream-probe-v1, plane-pooling-analysis, tower-size-factorial-v1, ...).
The page shell is scripts/digests/templates/project-digest.html; __PROJECT_DATA__ and __DOWNSTREAM_PROBE__ are filled here.

    .venv/bin/python scripts/digests/build_project_digest.py
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / 'checkpoints/contrastive/claude-translated-dfd07db-image-block-execution'
OUT = ROOT / 'checkpoints/contrastive/claude-translated-dfd07db-image-block-v1'
OUT_TB = ROOT / 'checkpoints/contrastive/claude-translated-dfd07db-text-block-v1'

def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def history(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                # Only plot metrics known to be finite; gradient overflow is documented separately.
                rows.append({k: row[k] for k in ('epoch', 'mean_r@1', 'validation_loss', 'image_to_text_r@1', 'text_to_image_r@1')})
            except json.JSONDecodeError:
                continue  # A writer may be midway through the final line.
    return rows

def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    data = {
        'generated_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds'),
        'text_status': read(ROOT/'checkpoints/contrastive/english-medical-text-execution/status.json', {}),
        'status': read(EX/'status.json', {'status': 'not_started', 'stage': 'preflight', 'completed_runs': []}),
        'comparison': read(EX/'comparison.json', {}),
        'parameters': read(ROOT/'digest/architecture-parameter-audit.json', {}),
        'baseline': history(ROOT/'checkpoints/contrastive/claude-translated-dfd07db-q2-v1/00-full-seed2026/metrics.jsonl'),
        'ordering': history(ROOT/'checkpoints/contrastive/claude-translated-dfd07db-q2-v1/01-impression-first-seed2026/metrics.jsonl'),
        'candidate': history(OUT/'image-block-seed2026/metrics.jsonl'),
        'smoke_reload': read(OUT/'smoke-image-block/reload-validation.json'),
        'text_block': {
            seed: history(OUT_TB/f'text-block-seed{seed}/metrics.jsonl')
            for seed in ('2026', '2027', '2028')
        },
    }
    template = (ROOT/'scripts/digests/templates/project-digest.html').read_text()
    tower_head = read(ROOT/'digest/lm-head-tower-size-58-v1.json', {})
    tower_contrastive = read(ROOT/'digest/tower-size-factorial-v1.json', {})
    tower_html = ''
    if tower_head and tower_contrastive:
        order = ('small_image_base_text', 'small_image_small_text',
                 'base_image_base_text', 'base_image_small_text')
        gold_only = read(ROOT/'checkpoints/kaggle/lora-holdout-v1/benchmark.json', {})
        gold_only_auc = gold_only.get('results', {}).get('ensemble', {}).get('macro_auroc')
        rows = []
        for index, key in enumerate(('small_image_base_text', *order), 1):
            arm = tower_head['arms'][key]
            retrieval = tower_contrastive['arms'][key]
            auc = gold_only_auc if index == 1 else arm['mlp_macro_auroc']
            if auc is None:
                continue
            population = '56 gold, OOF' if index == 1 else '58 gold, held out'
            supervision = 'Gold-only logistic' if index == 1 else 'LM verdicts, 3-seed MLP'
            public = '0.717' if index == 1 else '—'
            linear = '—' if index == 1 else f'{arm["linear_macro_auroc"]:.4f}'
            rows.append(
                f'<tr><td>{index}</td><td>{auc:.4f}</td><td>{population}</td>'
                f'<td>{linear}</td>'
                f'<td>{retrieval["best_image_to_text_r1"]:.2%}</td>'
                f'<td>{retrieval["best_text_to_image_r1"]:.2%}</td>'
                f'<td>{public}</td><td>{escape(arm["image"])}</td>'
                f'<td>{escape(arm["text"])}</td><td>{supervision}</td></tr>'
            )
        matrix_source = (ROOT/'digest/lm-head-tower-size-58-v1.md').read_text()
        matrix_section = matrix_source.split('### Full settings and outcome matrix', 1)[1]
        matrix_lines = matrix_section.splitlines()
        first_line = next(i for i, line in enumerate(matrix_lines) if line.startswith('|'))
        matrix_lines = matrix_lines[first_line:]
        matrix_lines = matrix_lines[:next(i for i, line in enumerate(matrix_lines) if not line.startswith('|'))]
        matrix_cells = [[cell.strip() for cell in line.strip('|').split('|')] for line in matrix_lines]
        if len(matrix_cells) != 7 or any(len(row) != 22 for row in matrix_cells):
            raise ValueError('Expected a five-row, 22-column full settings matrix')
        matrix_header = ''.join(f'<th>{escape(value)}</th>' for value in matrix_cells[0])
        matrix_rows = ''.join(
            '<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in row) + '</tr>'
            for row in matrix_cells[2:]
        )
        full_matrix_html = (
            '<details style="margin-top:18px"><summary>Show the full 22-column settings '
            'and outcome matrix</summary><div class="table-wrap" style="margin-top:12px">'
            '<table><thead><tr>' + matrix_header + '</tr></thead><tbody>'
            + matrix_rows + '</tbody></table></div></details>'
        )
        tower_html = (
            '<section class="section" id="tower-comparison"><div class="section-head"><div>'
            '<div class="section-no">NEW / 58 GOLD CASES</div>'
            '<h2>Tower size and LM-head comparison.</h2>'
            '<p>Four frozen gold-holdout encoders, with the same LM-supervised pathology head settings. '
            'All 58 gold cases were excluded from head fitting and epoch selection.</p>'
            '</div></div><div class="card"><div class="table-wrap"><table><thead><tr>'
            '<th>Index</th><th>Local AUROC</th><th>Gold population</th><th>LM-linear AUROC</th>'
            '<th>Image→text R@1</th><th>Text→image R@1</th><th>Public</th>'
            '<th>Image</th><th>Text</th><th>Head supervision</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            + full_matrix_html +
            '<p class="tiny muted" style="margin-top:14px">Row 1 is the original 56-case '
            'gold-only out-of-fold benchmark and the only Kaggle public score. Rows 2–5 use '
            '58 held-out gold cases with heads trained on LM verdicts. The AUROCs across '
            'these two populations are not a paired improvement estimate. Two former gold '
            'test cases were scored, so they are no longer untouched.</p>'
            '<p style="margin-top:14px"><strong>Result:</strong> DINOv2-small + E5-base '
            'leads the 58-case LM-head scores at 0.7659. Both size-mismatched pairings '
            'lead the matched pairing with the same image size, but every paired 95% '
            'interval includes zero. This does not support a requirement to match tower '
            'sizes or a larger-text-specific overfit effect. Linear heads show much '
            'less late validation-loss rebound, yet score 0.7312–0.7362 on gold, '
            'below every early-stopped MLP.</p>'
            '<p style="margin-top:14px"><a href="lm-head-tower-size-58-v1.html">'
            'Full settings matrix, loss curves and per-label AUROC ↗</a> · '
            '<a href="lm-head-tower-size-58-v1.md">Markdown record ↗</a></p>'
            '</div></section>'
        )
    template = template.replace('__LM_TOWER_COMPARISON__', tower_html)
    audit = read(ROOT/'digest/claude-translated-dfd07db-image-block-audit.json')
    comparison_html = ''
    if audit:
        table_rows = []
        for pair in audit['paired_differences']:
            seed = pair['seed']
            base = next(r for r in audit['runs'] if r['seed'] == seed and r['arm'] == 'baseline')
            candidate = next(r for r in audit['runs'] if r['seed'] == seed and r['arm'] == 'image_block')
            low, high = pair['descriptive_paired_query_bootstrap_95_ci_pp']
            table_rows.append(f"<tr><td>{seed}</td><td>{base['mean_r@1']*100:.4f}%</td><td>{candidate['mean_r@1']*100:.4f}%</td><td>+{pair['mean_difference_pp']:.4f}</td><td>[{low:.4f}, {high:.4f}]</td></tr>")
        comparison_html = f'''<article class="card" style="margin-bottom:22px"><span class="pill">Completed · independently verified</span><h3 style="margin-top:14px">Image adaptation: every matched seed improved</h3><div class="table-wrap"><table><thead><tr><th>Seed</th><th>Frozen baseline</th><th>Image adaptation</th><th>Difference (pp)</th><th>Descriptive 95% interval (pp)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div><p style="margin-top:18px"><strong>Mean difference +{audit['mean_difference_pp']:.4f} pp</strong> · sample SD {audit['sample_sd_pp']:.4f} pp · range +{audit['range_pp'][0]:.4f} to +{audit['range_pp'][1]:.4f} pp.</p><p class="tiny muted" style="margin-top:10px">All six 20-epoch histories, selected checkpoint hashes, exact reloads, matched configs and telemetry passed verification. Positive direction is consistent across these three seeds; every paired-query interval still includes zero. These selected validation results do not establish test performance. Retain image adaptation as promising; finish its candidate error and subgroup review before the next experiment.</p><p class="tiny" style="margin-top:12px"><a href="claude-translated-dfd07db-image-block-audit.json">Verification artifact ↗</a> · <a href="claude-translated-dfd07db-image-block-results.md">All run results ↗</a></p></article>'''
    template = template.replace('__IMAGE_COMPARISON__', comparison_html)
    text_audit = read(ROOT/'digest/english-medical-text-results.json', {})
    text_rows = []
    for run in text_audit.get('results', []):
        assert run['independent_verification_passed']
        low, high = run['paired']['descriptive_paired_query_bootstrap_95_ci_pp']
        text_rows.append(f"<tr><td>{escape(run['candidate'])}</td><td>{run['seed']}</td><td>{run['best_epoch']}</td><td>{run['metrics']['mean_r@1']*100:.4f}%</td><td>{run['paired']['mean_difference_pp']:+.4f}</td><td>[{low:.4f}, {high:.4f}]</td></tr>")
    text_html = f'''<article class="card" style="margin-bottom:22px"><span class="pill">Text queue: {escape(text_audit.get('queue_status', 'unavailable'))}</span><h3 style="margin-top:14px">English / medical text replacement: neither screen beat the baseline</h3><p>Frozen DINOv2 and frozen text backbones; same quarantined full-report cohort, 512-token cap, batch 32 and 20 epochs. Matched multilingual E5 seed-2026 baseline: <strong>10.6576%</strong>.</p><div class="table-wrap"><table><thead><tr><th>Text encoder</th><th>Seed</th><th>Selected epoch</th><th>Mean R@1</th><th>Difference (pp)</th><th>Descriptive 95% interval (pp)</th></tr></thead><tbody>{''.join(text_rows)}</tbody></table></div><p style="margin-top:18px">Both smokes and both full runs completed; histories, configs, checkpoint hashes, exact reloads and paired rankings passed verification. Neither positive-gain repeat rule triggered. Retain multilingual E5 as the reference; no follow-up training is queued.</p><p class="tiny muted" style="margin-top:10px">One seed per candidate; both intervals include zero. This does not establish a general encoder disadvantage. Qualitative review of twenty worst pairs per candidate remains pending. AMP logged nonfinite gradient norms at English E5 epoch 19 and PubMedBERT epoch 7; skipped-step counts are unavailable. The full test cohort was not evaluated in this historical text screen; two gold cases from its split were scored later.</p><p class="tiny" style="margin-top:12px"><a href="english-medical-text-results.md">Verified results and subgroups ↗</a> · <a href="english-medical-text-results.json">Verification artifact ↗</a> · <a href="english-medical-text-error-review-packet.md">Pending error-review packet ↗</a></p></article>'''
    template = template.replace('__TEXT_COMPARISON__', text_html)

    # Text-only final-block adaptation. Unlike the arms above there is no separate
    # audit JSON to assert against -- this arm was finalized straight from the matrix,
    # so the card is built from validation-selection.json and the matrix's own frozen
    # matched baselines, and it claims no independent verification.
    tb_matrix = read(ROOT/'configs/contrastive/claude-translated-dfd07db-text-block-v1.json', {})
    tb_selection = read(OUT_TB/'validation-selection.json')
    text_block_html = ''
    if tb_selection:
        matched = tb_matrix.get('matched_baselines', {})
        rows, diffs = [], []
        for run in tb_selection['experiments']:
            seed = run['name'].rsplit('seed', 1)[-1]
            best = run['best_validation']
            value = best['mean_r@1'] * 100
            base = matched[seed]['mean_r@1_pp']
            difference = value - base
            diffs.append(difference)
            chosen = ' class="retained"' if run['name'] == tb_selection['selected_experiment'] else ''
            rows.append(
                f"<tr{chosen}><td>{seed}</td><td>{best['epoch']}</td><td>{value:.4f}%</td>"
                f"<td>{base:.4f}%</td><td>{difference:+.4f}</td></tr>"
            )
        mean_difference = sum(diffs) / len(diffs)
        spread = (sum((d - mean_difference) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
        text_block_html = f"""<article class="card" style="margin-bottom:22px"><span class="pill">Completed \u00b7 validation only</span><h3 style="margin-top:14px">Text adaptation: the mirror of the image arm, and it did not help</h3><p>Final text block unfrozen, DINOv2 frozen \u2014 every other setting matched to the image arm and to each seed's own frozen baseline. Three seeds, budgeted up front rather than triggered by a positive first seed.</p><div class="table-wrap"><table><thead><tr><th>Seed</th><th>Selected epoch</th><th>Text adaptation</th><th>Matched frozen baseline</th><th>Difference (pp)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p style="margin-top:18px"><strong>Mean difference {mean_difference:+.4f} pp</strong> \u00b7 sample SD {spread:.4f} pp \u00b7 three-seed interval spanning zero. The image arm on the same cohort and protocol gained +1.2094 pp, so the two halves of the final-block question answer with opposite signs \u2014 but neither interval excludes zero.</p><p class="tiny muted" style="margin-top:10px">Read this cautiously. Validation is 441 studies, so R@1 moves in steps of 1/441 = 0.2268 pp and repeated values across seeds are grid collisions, not agreement. Seed 2026 carries nearly the whole effect: its baseline is the highest of the three, and the other two seeds differ by under one retrieval hit. Selection used validation metrics alone (<code>locked_test_was_used_for_selection: false</code>); the 440 test studies were not evaluated or inspected. Unlike the arms above, no independent artifact verification was performed on this queue. AMP logged nonfinite gradient norms at 2\u20133 epochs per seed; selected metrics are finite and skipped-step counts were not recorded.</p><p class="tiny" style="margin-top:12px"><a href="claude-translated-dfd07db-text-block-protocol.md">Protocol \u2197</a></p></article>"""
    template = template.replace('__TEXT_BLOCK_COMPARISON__', text_block_html)

    # Downstream label readout. This is the first card on this page that is not a
    # retrieval number, so it states its own estimand rather than inheriting the
    # page's. Built from the probe artifacts; absent artifacts yield no card.
    probe = read(ROOT/'digest/downstream-probe-v1.json')
    planes = read(ROOT/'digest/plane-pooling-analysis.json')
    probe_html = ''
    if probe:
        results, labels = probe['results'], probe['config']['labels']
        identity = probe['frozen_backbone_identity_check']
        assert all(entry['identical'] for entry in identity.values()), 'frozen-tower identity check failed'
        base_key = 'before_cl/dinov2-unadapted/uniform_mean_768'
        before = results[base_key]['macro_auroc']
        seeds = ('2026', '2027', '2028')
        rows = []
        for seed in seeds:
            key = f'after_cl/image-block-seed{seed}/uniform_mean_768'
            comparison = probe['comparisons_against_unadapted_dinov2'][key]
            low, high = comparison['macro_auroc_difference_ci95']
            rows.append(
                f"<tr class=\"retained\"><td>{seed}</td><td>{before:.4f}</td>"
                f"<td>{results[key]['macro_auroc']:.4f}</td>"
                f"<td>+{comparison['macro_auroc_difference_mean']:.4f}</td>"
                f"<td>[{low:+.4f}, {high:+.4f}]</td></tr>"
            )
        # Labels ranked by how much the three-seed mean moved, to show the gain is
        # broad rather than carried by one column.
        moves = []
        for label in labels:
            after = sum(results[f'after_cl/image-block-seed{s}/uniform_mean_768']['per_label'][label]['auroc'] for s in seeds) / 3
            moves.append((after - results[base_key]['per_label'][label]['auroc'], label, results[base_key]['per_label'][label]['auroc'], after))
        moves.sort(reverse=True)
        improved = sum(1 for move, *_ in moves if move > 0)
        top = ''.join(
            f"<tr><td>{escape(label)}</td><td>{start:.3f}</td><td>{end:.3f}</td><td>{move:+.3f}</td></tr>"
            for move, label, start, end in moves[:3]
        )
        population = probe['population']['split_counts']
        probe_html = f'''<article class="card" style="margin-bottom:22px"><span class="pill">Downstream readout \u00b7 development statistic</span><h3 style="margin-top:14px">The first measurement against the actual goal: does this help predict pathology?</h3><p>Every card above reports retrieval. This one reports the twelve pathology labels, scored as pooled out-of-fold AUROC over the {probe['population']['population_size']} gold studies, under a probe contract fixed before the first fit. <strong>Unadapted DINOv2-small reads them at {before:.4f}.</strong></p><p style="border-left:3px solid #999;padding-left:12px"><strong>Exposure caveat, measured 2026-09-14.</strong> These are development statistics: 56 of the 58 gold studies were training pairs. Retraining with them held out puts the 4-block level at <strong>0.6938</strong> against the pooled exposed level of <strong>0.7106</strong> (the published 0.7105 is seed 2026) &mdash; an exposure inflation of <strong>&minus;0.0168</strong> by the primary pooled route, &minus;0.0163 by the data-size-matched route &mdash; and the gain over unadapted DINOv2 at <strong>+0.0500</strong> rather than +0.0668. The levels subtract exactly: 0.6438, 0.6938, 0.7106, with all three seed intervals excluding zero. Quote 0.6938 beside 0.7105, never instead of it. See <a href="gold-holdout-v1.md">gold-holdout-v1.md</a>.</p><div class="table-wrap"><table><thead><tr><th>Seed</th><th>Before contrastive learning</th><th>After image-block adaptation</th><th>Difference</th><th>Paired 95% interval</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p style="margin-top:18px"><strong>Every seed improves and every interval excludes zero</strong> \u2014 a more stable signal than the +1.2094 pp retrieval gain it came from, which had a sample SD of 0.6643. {improved} of 12 labels rise, so the gain is broad rather than one column:</p><div class="table-wrap"><table><thead><tr><th>Largest gains</th><th>Before</th><th>After</th><th>Difference</th></tr></thead><tbody>{top}</tbody></table></div><div class="result-note" style="margin-top:18px"><strong>Only adaptation that unfreezes an image block can do this.</strong><p class="muted">Every run with <code>--image-unfreeze-last 0</code> leaves DINOv2 byte-identical, verified here at 0.0 maximum deviation for all three frozen-tower checkpoints. Those runs trained the plane attention and the projection head and nothing else, so they could not have changed what the image tower sees \u2014 whatever they measured, it was not a change in the image representation.</p></div><p class="tiny muted" style="margin-top:14px"><strong>This is not generalisation evidence and cannot be made into any.</strong> {population['train']} of the {probe['population']['population_size']} scored studies were contrastive training pairs and {population['validation']} are in the split that selected every checkpoint. The only {population['test']} gold studies untouched by representation learning are in the locked test and were not loaded. The estimand is readout quality on a cohort the representation has already seen; cross-validation controls probe overfitting, not representation exposure. Studies are not patients: anonymisation is per study, so the bootstrap's independence assumption is unverifiable.</p><p class="tiny" style="margin-top:12px"><a href="downstream-probe-v1.md">Full probe report \u2197</a> \u00b7 <a href="downstream-probe-v1.json">Evidence artifact \u2197</a> \u00b7 <a href="gold-case-inventory.md">What 58 studies can support \u2197</a></p></article>'''
    if probe and planes:
        seeds = ('2026', '2027', '2028')
        weights = planes['plane_attention_weights']['image-block-seed2026']
        order = weights['plane_order']
        # Every seed learned this, so quote the range across seeds rather than one
        # seed's value beside a table that shows all three.
        axial_mass = [planes['plane_attention_weights'][f'image-block-seed{s}']['mean_weight'][2] for s in seeds]
        axial_argmax = [planes['plane_attention_weights'][f'image-block-seed{s}']['argmax_plane_counts'][2] for s in seeds]
        def mean_macro(key):
            return sum(planes['results'][f'image-block-seed{s}/{key}']['macro_auroc'] for s in seeds) / 3
        uniform, attention = mean_macro('uniform_mean_768'), mean_macro('attention_pooled_768')
        concat, no_embed = mean_macro('concat_planes_2304'), mean_macro('attention_no_embeddings_768')
        interval_rows = []
        for seed in seeds:
            comparison = planes['comparisons_against_uniform_mean_same_checkpoint'][f'image-block-seed{seed}/attention_pooled_768']
            low, high = comparison['macro_auroc_difference_ci95']
            interval_rows.append(
                f"<tr><td>{seed}</td><td>{comparison['macro_auroc_uniform_mean']:.4f}</td>"
                f"<td>{comparison['macro_auroc_this_arm']:.4f}</td>"
                f"<td>{comparison['macro_auroc_difference_mean']:+.4f}</td>"
                f"<td>[{low:+.4f}, {high:+.4f}]</td></tr>"
            )
        plane_rows = []
        for label in ('PF OA', 'MCL', 'Lateral Meniscus'):
            per_plane = [
                sum(planes['results'][f'image-block-seed{s}/plane_{name.lower()}_768']['per_label'][label]['auroc'] for s in seeds) / 3
                for name in order
            ]
            best = order[per_plane.index(max(per_plane))]
            delta = (sum(planes['results'][f'image-block-seed{s}/attention_pooled_768']['per_label'][label]['auroc'] for s in seeds)
                     - sum(planes['results'][f'image-block-seed{s}/uniform_mean_768']['per_label'][label]['auroc'] for s in seeds)) / 3
            cells = ''.join(f"<td>{value:.3f}</td>" for value in per_plane)
            plane_rows.append(f"<tr><td>{escape(label)}</td>{cells}<td>{escape(best)}</td><td>{delta:+.3f}</td></tr>")
        probe_html += f'''<article class="card" style="margin-bottom:22px"><span class="pill amber">Mechanism \u00b7 direction consistent, interval spans zero</span><h3 style="margin-top:14px">The pooling retrieval selected reads labels worse than a plain average</h3><p>The image tower pools its three planes with learned softmax attention rather than averaging them. Attention can represent the mean, so it should not lose to one \u2014 but on label readout it does, at all three seeds, paired against the uniform mean <em>of the same checkpoint</em>.</p><div class="table-wrap"><table><thead><tr><th>Seed</th><th>Uniform mean</th><th>Attention pooled</th><th>Difference</th><th>Paired 95% interval</th></tr></thead><tbody>{''.join(interval_rows)}</tbody></table></div><p style="margin-top:18px">Two candidate causes are eliminated outright. Removing the additive plane-embedding term changes the score by {abs(attention - no_embed):.4f} ({no_embed:.4f} against {attention:.4f}), so the nuisance vector is not it. Concatenating all three planes instead of pooling scores {concat:.4f} against the mean's {uniform:.4f}, so no recoverable information is being thrown away. What remains is the re-weighting.</p><p style="margin-top:14px">All three seeds independently learned to put about half the attention mass on <strong>{order[2]}</strong> ({min(axial_mass):.3f}\u2013{max(axial_mass):.3f} across seeds, against {weights['uniform_weight']:.3f} uniform; argmax is {order[2]} for {min(axial_argmax)}\u2013{max(axial_argmax)} of {probe['population']['population_size']} studies). Probing each plane alone shows why that is costly:</p><div class="table-wrap"><table><thead><tr><th>Label</th><th>{order[0]}</th><th>{order[1]}</th><th>{order[2]}</th><th>Best plane</th><th>Attention effect</th></tr></thead><tbody>{''.join(plane_rows)}</tbody></table></div><p style="margin-top:18px"><strong>One plane preference, twelve labels that disagree about planes.</strong> The labels the attention helps are the ones {order[2].lower()} reads best \u2014 patellofemoral OA is what that view is for. The ones it hurts most are read on the planes it de-weights: the medial collateral ligament is a coronal structure, the lateral meniscus a sagittal one. Averaging is not a weaker version of that choice; it is a refusal to make it, and refusing is better when twelve questions need twelve answers. Across the labels, how much {order[2].lower()} beats the other two planes tracks how much the attention helps at r = 0.62 \u2014 decoration rather than evidence, since both quantities come from the same per-plane scores and so share estimation noise. The anatomy in the table carries the argument on its own.</p><p class="tiny muted" style="margin-top:10px">Every interval above spans zero. Three seeds agree in sign and to within 0.003, and 90\u201395% of bootstrap replicates favour the mean, but at {probe['population']['population_size']} studies this is a direction, not a demonstrated deficit. The attention is not broken \u2014 it was trained to match one study to one report, which is a single question, and it fits that objective well. Exporting the pre-attention plane features costs no training and carries none of the risk.</p><p class="tiny" style="margin-top:12px"><a href="downstream-probe-v1.md#why-the-learned-plane-pooling-reads-labels-worse-than-averaging">Full decomposition \u2197</a> \u00b7 <a href="plane-pooling-analysis.json">Evidence artifact \u2197</a></p></article>'''
    template = template.replace('__DOWNSTREAM_PROBE__', probe_html)
    template = template.replace('Retain image adaptation as promising; finish its candidate error and subgroup review before the next experiment.', 'Image-adaptation candidate error and subgroup review is complete. Retain image adaptation as promising; text replacement results and their pending qualitative review guide the next decision.')
    payload = json.dumps(data, allow_nan=False).replace('<', '\\u003c')
    target = ROOT/'digest/project-state.html'
    target.write_text(template.replace('__PROJECT_DATA__', payload))
    print(target)

if __name__ == '__main__':
    main()
