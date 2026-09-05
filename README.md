# Knee MRI contrastive experiments

The current experiment contract is [full-multilingual-v1](configs/contrastive/full-multilingual-matrix.json).
Use [ROADMAP.md](ROADMAP.md) for milestones and the
[frozen protocol](configs/contrastive/full-multilingual-v1-protocol.md) for selection and limitations.
Older translated-English pilot instructions in `digest/` describe historical experiments.

## Setup

Use Python 3.13.12 and install `requirements-lock.txt` in a virtual environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
.venv/bin/python -m pytest -q
```

The lock records the local CPU environment. A GPU host needs compatible CUDA builds
of the pinned PyTorch/torchvision versions and an NVIDIA driver. Record the actual
GPU environment with `pip freeze`, driver version, and GPU model before training;
the CPU lock alone does not establish a working CUDA environment.

Restore private/local artifacts listed with SHA-256 hashes in
[the artifact inventory](configs/contrastive/full-multilingual-v1-artifacts.json).
The full cohort and metadata belong in `data/processed/`; DICOM series are under
`/mnt/d/train_series`, and the original series CSV is `/mnt/d/train_series.csv`.
The manifest contains absolute series paths: preserve this mount layout.
Copy the two pinned Hugging Face snapshots (including referenced blob contents)
into the training host's Hugging Face cache. Model revisions are pinned in the matrix;
compact checkpoints require those exact frozen backbones at reload time.
`data/gold_translated.csv` is an external input recorded in the inventory.

The full manifest checksum was added by an audited migration; see
[digest/full-manifest-migration.json](digest/full-manifest-migration.json).
The reader now rejects missing or mismatched checksums. Do not recalculate checksums
silently to accept changed data; changes require a new experiment version.

## Run

```bash
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
.venv/bin/python scripts/run_contrastive_matrix.py --dry-run
```

Before the full run, perform the following short GPU smoke and resume check. It
uses the same pinned models and preprocessing, a separate output directory, two
training epochs in total, and at most 64 cohort rows. The full cohort's fixed splits
are retained. It never evaluates the test split.

```bash
.venv/bin/python - <<'PY'
import json, subprocess, sys
from pathlib import Path
import torch
assert torch.cuda.is_available()
m = json.loads(Path('configs/contrastive/full-multilingual-matrix.json').read_text())
out = 'checkpoints/contrastive/full-multilingual-v1-gpu-smoke'
command = [sys.executable, 'scripts/train_contrastive.py', *m['base_args'],
           '--output-dir', out, '--max-studies', '64', '--batch-size', '4',
           '--num-workers', '0', '--local-files-only']
subprocess.run([*command, '--epochs', '1'], check=True)
subprocess.run([*command, '--epochs', '2', '--resume', out + '/last.pt'], check=True)
subprocess.run([sys.executable, 'scripts/evaluate_contrastive.py', out + '/best.pt',
                '--split', 'validation', '--output', out + '/reload-validation.json'], check=True)
PY
.venv/bin/python scripts/run_contrastive_matrix.py --experiment 00-projection-full-mean-512
```

For an interrupted full run, use the exact command printed by `--dry-run` and append
`--resume checkpoints/contrastive/full-multilingual-v1/00-projection-full-mean-512/last.pt`.
Keep the original target of 20 epochs and all other settings. Resume rejects a
`best.pt` that disagrees with history; older runs with the former tie rule must be
audited separately, not silently finalized under the new policy.

Inspect `metrics.jsonl`, `run_manifest.json`, `best.pt`, and `last.pt` in the run
output directory. Record runtime, peak GPU memory, and the validation diagnostics
before deciding whether to run the remaining matrix. Batch size is 32 (31 in-batch
negatives per query); the final incomplete training batch is dropped.
Do not invoke `--evaluate-locked-test` during baseline development.
