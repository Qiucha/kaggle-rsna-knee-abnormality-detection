"""Export saved combined CLIP + LoRA + K8 + 336 results without using the GPU.

Status comes from the run directories alone: an arm is completed once it has every
epoch of its recipe and a best.pt, incomplete if it has started, queued otherwise. The
recipe is configs/contrastive/combined-lora-k8-336-v1.json; the runs, LM heads and
feature caches are gitignored checkpoints (see experiment_tools.lm_heads.load_head).

Usage: build_combined_lora_digest.py [--project DIR] [--digest DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from contrastive_learning.checkpoints import selection_key  # noqa: E402
from contrastive_learning.config import _PATH_FIELDS  # noqa: E402
from experiment_tools.lm_heads import load_head  # noqa: E402

RUNS = 'checkpoints/contrastive/combined-lora-k8-336-v1'
MATRIX = ROOT / 'configs/contrastive/combined-lora-k8-336-v1.json'
# Top-level repository directories: a recorded path through one of them points into a checkout.
REPO_DIRS = ('checkpoints', 'configs', 'data')

# Appended to the md, so a reader can rerun an arm from the committed recipe.
REPRODUCE = [
    '',
    '## Reproduce',
    '',
    'From the repository root on a CUDA host, with `paths.json` naming `data_root` and, optionally, `train_series_mirror`:',
    '',
    '1. Train one arm: `python scripts/run_contrastive_matrix.py --matrix configs/contrastive/combined-lora-k8-336-v1.json --experiment <arm>`.',
    '2. Fit its heads: `python scripts/train_lm_head.py --config configs/kaggle/lm-mlp-combined-58-<arm>.json --features checkpoints/kaggle/lm-mlp-combined-58-<arm>-features.npz --output checkpoints/kaggle/lm-mlp-combined-58-<arm> --device cuda`. These head configs set `reload_tolerance` to 1e-3 for the fp16 K8 features.',
    '3. Optionally fine-tune from the OmniRad checkpoint: `python scripts/finetune_supervised.py --config configs/finetune/r4a-k8-omnirad-base-soft-336-from-clip-lora.json --output checkpoints/finetune/<name>`.',
    '4. Rebuild this digest: `python scripts/build_combined_lora_digest.py`.',
    '',
    "For a seed variant, add a matrix entry whose `args` set `--seed`. Copy the head config and point `checkpoint` at the new run's `best.pt`.",
    'Not in git: the cohort manifest (checked against `manifest_sha256`), `data/lm_label/lm_labels_kaggle.csv`, and the pinned Hugging Face snapshots.',
]


def history(path):
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
    return records


def repo_relative(value, root):
    """A recorded path into this repository, or into another checkout of it, made repo-relative.

    Runs record absolute paths resolved in whichever checkout trained them (a session
    worktree, for these arms). External mounts are left as recorded.
    """
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value
    path = Path(value)
    if path.is_relative_to(root):
        return str(path.relative_to(root))
    for index, part in enumerate(path.parts):
        if part in REPO_DIRS and (root / Path(*path.parts[index:])).exists():
            return str(Path(*path.parts[index:]))
    return value


def planned_epochs():
    base_args = json.loads(MATRIX.read_text())['base_args']
    return int(base_args[base_args.index('--epochs') + 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--project', type=Path, default=ROOT, help='Root holding the checkpoints (read only)')
    parser.add_argument('--digest', type=Path, default=ROOT / 'digest', help='Where the md and json are written')
    args = parser.parse_args()
    root = args.project
    arms = []
    for short, title in [('omnirad-base', 'OmniRad-base'), ('dinov2-small', 'DINOv2-small')]:
        name = short + '-lora-k8-336'
        run = root / RUNS / name
        metrics = history(run / 'metrics.jsonl')
        config = json.loads((run / 'config.json').read_text()) if (run / 'config.json').exists() else {}
        config = {key: repo_relative(value, root) if key in _PATH_FIELDS else value for key, value in config.items()}
        epochs = config.get('epochs', planned_epochs())
        completed = len(metrics) == epochs and (run / 'best.pt').exists()
        best = max(metrics, key=selection_key) if metrics else None
        head_path = root / 'checkpoints/kaggle' / ('lm-mlp-combined-58-' + name)
        head = load_head(head_path, run / 'best.pt', root) if completed else None
        arm = {'name': name, 'title': title,
               'status': 'completed' if completed else 'incomplete' if run.exists() else 'queued',
               'epochs_completed': len(metrics), 'max_epochs': epochs,
               'selected_epoch': best['epoch'] if completed and best else None,
               'best_mean_recall_at_1': best['mean_r@1'] if completed and best else None,
               'history': metrics, 'config': config,
               'heads_status': 'completed' if head else 'pending',
               'gold_mlp': head['benchmark']['results']['mlp']['macro_auroc'] if head else None,
               'lm_test_mlp': head['proxy']['macro_auroc'] if head else None,
               'head_provenance': head['signature'] if head else None,
               'gold_per_label': {label: value['auroc'] for label, value in
                                  head['benchmark']['results']['mlp']['per_label'].items()} if head else {},
               'lm_test_per_label': head['proxy']['per_label'] if head else {}}
        arms.append(arm)
    snapshot = {'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'arms': arms,
                'note': 'Retrieval uses validation image-report pairs. LM-test AUROC uses LM labels; gold-58 AUROC is exploratory. Pending arms have no final score. No Kaggle submission was made.'}
    args.digest.mkdir(parents=True, exist_ok=True)
    (args.digest / 'combined-lora-k8-336-v1.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    lines = ['# Combined CLIP + LoRA + K8 + 336', '', 'Updated ' + snapshot['updated_at'], '',
             '| Image tower | Contrastive status | Epochs | Selected epoch | Mean R@1 | LM-test MLP AUROC | Gold-58 MLP AUROC |',
             '|---|---|---:|---:|---:|---:|---:|']
    fmt = lambda v: f'{v:.4f}' if v is not None else 'pending'
    for a in arms:
        lines.append(f"| {a['title']} | {a['status']} | {a['epochs_completed']}/{a['max_epochs']} | {a['selected_epoch'] or 'pending'} | {fmt(a['best_mean_recall_at_1'])} | {fmt(a['lm_test_mlp'])} | {fmt(a['gold_mlp'])} |")
    lines += ['', snapshot['note'], '', 'Recipe: LoRA rank 64 / alpha 128 on the last six blocks of both towers; eight windows per plane at 336 px; batch size 32; 20 contrastive epochs; seed 2026; checkpoint selected by mean validation Recall@1.', '',
              'The OmniRad label fine-tune from this contrastive checkpoint is listed separately in [run-5 follow-ups](run5-followups-v1.md). Its AUROC is from an adapted image tower, whereas the MLP scores above use a frozen tower.', '',
              'The frozen-head exporter verifies checkpoint/bundle provenance and recomputes AUROC from existing feature and prediction caches. It does not extract images or train models.']
    lines += REPRODUCE
    (args.digest / 'combined-lora-k8-336-v1.md').write_text('\n'.join(lines) + '\n')
    print('Exported combined LoRA digest:', [(a['title'], a['status'], a['heads_status']) for a in arms], flush=True)


if __name__ == '__main__':
    main()
