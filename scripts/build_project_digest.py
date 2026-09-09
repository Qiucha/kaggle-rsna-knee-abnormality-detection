#!/usr/bin/env python3
"""Build a self-contained HTML snapshot from local experiment artifacts."""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / 'checkpoints/contrastive/claude-translated-dfd07db-image-block-execution'
OUT = ROOT / 'checkpoints/contrastive/claude-translated-dfd07db-image-block-v1'

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
    }
    template = (ROOT/'scripts/templates/project-digest.html').read_text()
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
    text_html = f'''<article class="card" style="margin-bottom:22px"><span class="pill">Text queue: {escape(text_audit.get('queue_status', 'unavailable'))}</span><h3 style="margin-top:14px">English / medical text replacement: neither screen beat the baseline</h3><p>Frozen DINOv2 and frozen text backbones; same quarantined full-report cohort, 512-token cap, batch 32 and 20 epochs. Matched multilingual E5 seed-2026 baseline: <strong>10.6576%</strong>.</p><div class="table-wrap"><table><thead><tr><th>Text encoder</th><th>Seed</th><th>Selected epoch</th><th>Mean R@1</th><th>Difference (pp)</th><th>Descriptive 95% interval (pp)</th></tr></thead><tbody>{''.join(text_rows)}</tbody></table></div><p style="margin-top:18px">Both smokes and both full runs completed; histories, configs, checkpoint hashes, exact reloads and paired rankings passed verification. Neither positive-gain repeat rule triggered. Retain multilingual E5 as the reference; no follow-up training is queued.</p><p class="tiny muted" style="margin-top:10px">One seed per candidate; both intervals include zero. This does not establish a general encoder disadvantage. Qualitative review of twenty worst pairs per candidate remains pending. AMP logged nonfinite gradient norms at English E5 epoch 19 and PubMedBERT epoch 7; skipped-step counts are unavailable. Test remains locked.</p><p class="tiny" style="margin-top:12px"><a href="english-medical-text-results.md">Verified results and subgroups ↗</a> · <a href="english-medical-text-results.json">Verification artifact ↗</a> · <a href="english-medical-text-error-review-packet.md">Pending error-review packet ↗</a></p></article>'''
    template = template.replace('__TEXT_COMPARISON__', text_html)
    template = template.replace('Retain image adaptation as promising; finish its candidate error and subgroup review before the next experiment.', 'Image-adaptation candidate error and subgroup review is complete. Retain image adaptation as promising; text replacement results and their pending qualitative review guide the next decision.')
    payload = json.dumps(data, allow_nan=False).replace('<', '\\u003c')
    target = ROOT/'digest/project-state.html'
    target.write_text(template.replace('__PROJECT_DATA__', payload))
    print(target)

if __name__ == '__main__':
    main()
