"""Export the 2026-10-05 OmniRad CLIP frozen-tower probe into digest/ (no GPU, no training).

Source: the probe session's analysis outputs (results.json, headonly-vs-finetune.json). The probe compares
the epoch-18 image tower of omnirad-clip-run5-v1 (plain CLIP, unfreeze 6, K8 336) with unadapted OmniRad-base on
the same K8 336 windows, each read by run-5-recipe LM heads (3 head seeds), and against the run-5 fine-tunes.
"""
import argparse
import json
import time
from pathlib import Path

DEFAULT_SOURCE = Path('.scratch/archive/omnirad-probe/full-analysis')

# Appended to the md, so a reader can rerun the probe from the committed scripts.
REPRODUCE = [
    '',
    '## Reproduce',
    '',
    'From the repository root on a CUDA host, with `paths.json` naming `data_root`:',
    '',
    '1. `python scripts/evaluation/extract_omnirad_probe_features.py --output <features>`: frozen K8 336 features from both towers, after identity checks.',
    '2. `python scripts/analysis/analyze_omnirad_probe.py --features <features> --output <analysis>`: heads, grouping and acquisition control (`results.json`, `head-predictions.npz`).',
    '3. `python scripts/analysis/compare_headonly_vs_finetune.py --heads <analysis>/head-predictions.npz --features <features> --output <analysis>/headonly-vs-finetune.json`.',
    '4. `python scripts/digests/build_omnirad_probe_digest.py --source <analysis>`. The default source is the archived `.scratch/archive/omnirad-probe/full-analysis`.',
    '',
    'To probe another checkpoint, change `CLIP_CHECKPOINT` and `CLIP_SHA256` together at the top of the extract script.',
    'Not in git: the CLIP checkpoint (`omnirad-clip-run5-v1`, epoch 18), the run-5 fine-tune predictions under `checkpoints/finetune/`, the cohort manifest, `data/lm_label/lm_labels_kaggle.csv` and the pinned Hugging Face snapshots.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--source', type=Path, default=None,
                        help='directory holding results.json and headonly-vs-finetune.json '
                             '(default: <project>/.scratch/archive/omnirad-probe/full-analysis)')
    args = parser.parse_args()
    ROOT = args.project
    SOURCE = args.source or ROOT / DEFAULT_SOURCE
    try:
        source_label = str(SOURCE.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        source_label = str(SOURCE)
    results = json.loads((SOURCE / 'results.json').read_text())
    versus = json.loads((SOURCE / 'headonly-vs-finetune.json').read_text())
    test, trainval = results['grouping']['test'], results['grouping']['trainval']
    snapshot = {
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'source': source_label,
        'checkpoint': 'checkpoints/contrastive/omnirad-clip-run5-v1/omnirad-base-e5-base-clip-k8-336-u6-bs32-seed2026/best.pt',
        'readout': results['readout'],
        'grouping': {split: {key: g[key] for key in ('studies', 'knn_k', 'knn_macro', 'knn_macro_delta_clip_minus_base',
                                                     'knn_macro_delta_ci95', 'silhouette_macro', 'geometry')}
                     | {'acquisition_knn': {attr: {arm: {k: v[arm][k] for k in ('accuracy', 'majority_baseline', 'studies')}
                                                   for arm in ('base', 'clip')}
                                            for attr, v in g['acquisition_knn'].items()}}
                     for split, g in (('test', test), ('trainval', trainval))},
        'versus_finetune': versus,
        'notes': results['notes'],
    }
    (ROOT / 'digest/omnirad-clip-frozen-probe-v1.json').write_text(json.dumps(snapshot, indent=2) + '\n')

    mlp, lin = results['readout']['mlp'], results['readout']['linear']
    f = lambda v: f'{v:.4f}'
    s = lambda v: f'{v:+.4f}'
    ci = lambda c: f'[{c[0]:+.4f}, {c[1]:+.4f}]'
    pair = lambda split, key: versus[split]['paired'][key]
    acq = test['acquisition_knn']
    lines = [
        '# OmniRad CLIP frozen-tower probe', '',
        'Updated ' + snapshot['updated_at'] + ' · analysis run 2026-10-05', '',
        'Question: does plain CLIP pre-training (Phase 1) make OmniRad-base a better *frozen* encoder, and does a head on '
        'that frozen tower reach a full label fine-tune (Phase 2)?', '',
        'Encoders: unadapted OmniRad-base vs the epoch-18 image tower of `omnirad-clip-run5-v1` (plain CLIP, last 6 blocks '
        'unfrozen, K8 windows at 336 px, batch 32, seed 2026). Same K8 336 inputs; heads use the run-5 LM-head recipe, '
        'three head seeds.', '',
        '| Readout | Split | Unadapted | CLIP | Δ CLIP − unadapted [95% CI] |', '|---|---|---:|---:|---|',
        f"| MLP head | LM-test (432) | {f(mlp['test']['macro_auroc']['base'])} | {f(mlp['test']['macro_auroc']['clip'])} | {s(mlp['test']['delta_clip_minus_base'])} {ci(mlp['test']['delta_ci95'])} |",
        f"| MLP head | Gold-58 | {f(mlp['gold']['macro_auroc']['base'])} | {f(mlp['gold']['macro_auroc']['clip'])} | {s(mlp['gold']['delta_clip_minus_base'])} {ci(mlp['gold']['delta_ci95'])} |",
        f"| Linear head | LM-test (432) | {f(lin['test']['macro_auroc']['base'])} | {f(lin['test']['macro_auroc']['clip'])} | {s(lin['test']['delta_clip_minus_base'])} {ci(lin['test']['delta_ci95'])} |",
        f"| Linear head | Gold-58 | {f(lin['gold']['macro_auroc']['base'])} | {f(lin['gold']['macro_auroc']['clip'])} | {s(lin['gold']['delta_clip_minus_base'])} {ci(lin['gold']['delta_ci95'])} |",
        f"| kNN (k={test['knn_k']}) | LM-test (432) | {f(test['knn_macro']['base'])} | {f(test['knn_macro']['clip'])} | {s(test['knn_macro_delta_clip_minus_base'])} {ci(test['knn_macro_delta_ci95'])} |",
        '', '## Frozen head vs full fine-tune', '',
        '| Comparison | Δ LM-test [95% CI] | Δ gold-58 [95% CI] |', '|---|---|---|']
    for key, label in (('clip-head - run5-ft', 'CLIP frozen head − run 5 fine-tune'),
                       ('clip-head - clip-then-ft', 'CLIP frozen head − CLIP then fine-tune'),
                       ('base-head - run5-ft', 'Unadapted frozen head − run 5 fine-tune'),
                       ('run5-ft - clip-then-ft', 'Run 5 fine-tune − CLIP then fine-tune'),
                       ('run5-ft - run5-ft-seed2027', 'Run 5 − its seed-2027 replicate (noise)')):
        a, b = pair('lm_test', key), pair('gold', key)
        lines.append(f"| {label} | {s(a['difference'])} {ci(a['ci95'])} | {s(b['difference'])} {ci(b['ci95'])} |")
    lines += [
        '', '## Acquisition control', '',
        f"CLIP also made scanner identity easier to read: manufacturer kNN accuracy {f(acq['manufacturer']['base']['accuracy'])} → "
        f"{f(acq['manufacturer']['clip']['accuracy'])} (majority {f(acq['manufacturer']['base']['majority_baseline'])}); field strength "
        f"{f(acq['field_strength']['base']['accuracy'])} → {f(acq['field_strength']['clip']['accuracy'])}. Effective rank "
        f"{test['geometry']['base']['effective_rank']:.1f} → {test['geometry']['clip']['effective_rank']:.1f}.", '',
        'Reading: LM labels come from the reports CLIP aligns to, so the LM-test gain may overstate what CLIP adds; gold-58 '
        'is the independent check and its interval includes zero. On gold, every frozen head trails every fine-tune. '
        'One CLIP training seed; head-seed noise only; gold-58 is exploratory.', '',
        '[Probes page](site/pages/probes.html) · [Machine-readable snapshot](omnirad-clip-frozen-probe-v1.json)']
    lines += REPRODUCE
    (ROOT / 'digest/omnirad-clip-frozen-probe-v1.md').write_text('\n'.join(lines) + '\n')
    print('Exported OmniRad frozen probe digest', flush=True)


if __name__ == '__main__':
    main()
