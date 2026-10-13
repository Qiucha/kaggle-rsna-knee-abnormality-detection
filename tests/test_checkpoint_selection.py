from dataclasses import replace
import json

import pytest
import torch

from contrastive_learning.config import TrainConfig
from contrastive_learning.checkpoints import initialize_run


def test_resume_retains_loss_for_recall_ties(tmp_path):
    config = TrainConfig(output_dir=tmp_path / 'run', epochs=3)
    initialize_run(config)
    rows = [dict(epoch=1, **{'mean_r@1': 0.25, 'validation_loss': 2.0}),
            dict(epoch=2, **{'mean_r@1': 0.25, 'validation_loss': 1.0})]
    (config.output_dir / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    checkpoint = dict(epoch=2, metrics=rows[1])
    torch.save(checkpoint, config.output_dir / 'last.pt')
    torch.save(checkpoint, config.output_dir / 'best.pt')
    resumed = initialize_run(replace(config, resume=config.output_dir / 'last.pt'))
    assert resumed.best_metrics == rows[1]


def test_a_run_resumes_after_its_dicom_mount_moves(tmp_path):
    config = TrainConfig(output_dir=tmp_path / 'run', epochs=3, data_root=tmp_path / 'old-mount')
    initialize_run(config)
    row = dict(epoch=1, **{'mean_r@1': 0.25, 'validation_loss': 1.0})
    (config.output_dir / 'metrics.jsonl').write_text(json.dumps(row) + '\n')
    for name in ('last.pt', 'best.pt'):
        torch.save(dict(epoch=1, metrics=row), config.output_dir / name)
    moved = replace(config, data_root=tmp_path / 'new-mount', resume=config.output_dir / 'last.pt')
    assert initialize_run(moved).best_metrics == row
    with pytest.raises(ValueError, match='incompatible'):
        initialize_run(replace(moved, seed=config.seed + 1))


def test_training_resume_and_finalization_agree_on_ties(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from contrastive_learning import training
    from contrastive_learning.checkpoints import verified_best_epoch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *_args):
            return torch.eye(2) * self.logit_scale, None, None

    records = [SimpleNamespace(split=split) for split in ['train', 'train', 'validation']]
    monkeypatch.setattr(training, 'load_cohort', lambda *a, **kw: SimpleNamespace(
        records=records, summary={}, split=lambda name: [r for r in records if r.split == name]))
    monkeypatch.setattr(training, 'build_model', lambda *a, **kw: (TinyModel(), None, {}))
    monkeypatch.setattr(training, 'make_loader', lambda *a, **kw: [None])
    monkeypatch.setattr(training, 'move_batch', lambda *a: (None, None, None))
    monkeypatch.setattr(training, 'augment_study_images', lambda x: x)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    losses = iter([2.0, 1.0, 0.5, 0.5, 0.8])
    monkeypatch.setattr(training, 'evaluate', lambda *a: {'mean_r@1': 0.25, 'validation_loss': next(losses)})
    config = TrainConfig(output_dir=tmp_path / 'run', cohort_manifest=tmp_path / 'fake.jsonl',
                         epochs=2, batch_size=2, num_workers=0, trainable_checkpoints=True,
                         image_model_revision='test-image', text_model_revision='test-text')
    training.run_training(config)
    assert verified_best_epoch(config.output_dir / 'metrics.jsonl', config.output_dir / 'best.pt')['epoch'] == 2
    training.run_training(replace(config, epochs=5, resume=config.output_dir / 'last.pt'))
    selected = verified_best_epoch(config.output_dir / 'metrics.jsonl', config.output_dir / 'best.pt')
    assert selected['epoch'] == 3
    assert selected['validation_loss'] == 0.5
    checkpoint = torch.load(config.output_dir / 'best.pt', weights_only=True)
    assert checkpoint['state_dict_format'] == 'trainable_delta_v1'
    checkpoint['metrics']['validation_loss'] = 123
    torch.save(checkpoint, config.output_dir / 'best.pt')
    with pytest.raises(ValueError, match='does not match'):
        verified_best_epoch(config.output_dir / 'metrics.jsonl', config.output_dir / 'best.pt')
    with pytest.raises(ValueError, match='does not match'):
        initialize_run(replace(config, epochs=6, resume=config.output_dir / 'last.pt'))


@pytest.mark.parametrize('field', ['mean_r@1', 'validation_loss'])
def test_selection_rejects_nonfinite_metrics(field):
    from contrastive_learning.checkpoints import is_better
    metrics = {'epoch': 1, 'mean_r@1': 0.25, 'validation_loss': 1.0}
    metrics[field] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        is_better(metrics, None)


def test_frozen_baseline_matches_resolved_config(monkeypatch, tmp_path):
    # The frozen matrix names no DICOM mount; the paths configuration supplies it,
    # which is what lets those commands resolve to the same run on any machine.
    import sys
    from pathlib import Path
    from contrastive_learning import cli
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads((root / 'configs/contrastive/full-multilingual-matrix.json').read_text())
    expected = json.loads((root / 'configs/contrastive/full-projection-v1-resolved.json').read_text())
    paths_config = tmp_path / 'paths.json'
    paths_config.write_text(json.dumps({'data_root': expected['data_root']}))
    captured = []
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, 'argv', ['train', '--paths-config', str(paths_config),
                                      *matrix['base_args'], '--output-dir', expected['output_dir']])
    monkeypatch.setattr(Path, 'exists', lambda _: True)
    monkeypatch.setattr(cli, 'run_training', lambda config: captured.append(config.as_dict()))
    cli.main()
    assert captured == [expected]
