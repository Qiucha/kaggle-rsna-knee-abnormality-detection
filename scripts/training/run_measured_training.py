#!/usr/bin/env python3
"""Run the training CLI unchanged and record CUDA memory and elapsed time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from contrastive_learning.artifacts import write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for measured training')
    if args.telemetry.exists():
        raise FileExistsError(args.telemetry)
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).resolve().with_name('train_contrastive.py')
    train_args = args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args
    record = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, str(train_script), *train_args],
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'device_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'status': 'running',
    }

    def publish() -> None:
        """Republish the telemetry record durably, so a killed run leaves a whole file."""
        write_text(args.telemetry, json.dumps(record, indent=2) + '\n')

    publish()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sys.argv = [str(train_script), *train_args]
    try:
        runpy.run_path(str(train_script), run_name='__main__')
        torch.cuda.synchronize()
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        publish()


if __name__ == '__main__':
    main()
