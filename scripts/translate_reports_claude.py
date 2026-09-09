#!/usr/bin/env python3
"""Prepare, run disjoint ranges, and merge origin/config's Claude CLI translations.

Preparation and dry runs are offline. Only `run --execute` invokes Claude.
Row numbers are zero-based in the pinned source CSV; stop is exclusive.
"""
from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import io
import json
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REVISION = '8e3bd6899d7a7c9e56a8d27110a57f0b711a0a59'
DEFAULT_BUNDLE = Path('checkpoints/translations/config-8e3bd68')


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(bundle):
    source = subprocess.check_output(['git', 'show', f'{REVISION}:data/train_translated_clean.csv'])
    notebook = json.loads(subprocess.check_output(['git', 'show', f'{REVISION}:notebooks/translation.ipynb']))
    constants = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'MODEL', 'TIMEOUT_S', 'MAX_ATTEMPTS', 'SYSTEM_PROMPT'}:
                        constants[target.id] = ast.literal_eval(node.value)
    if set(constants) != {'MODEL', 'TIMEOUT_S', 'MAX_ATTEMPTS', 'SYSTEM_PROMPT'}:
        raise ValueError('Notebook translation contract missing')
    contract = dict(constants, revision=REVISION, source_sha256=digest(source),
                    user_prefix='Translate this report into standard English:\n\n', source_column='Report')
    bundle.mkdir(parents=True, exist_ok=True)
    for name, data in [('source.csv', source), ('contract.json', (json.dumps(contract, ensure_ascii=False, indent=2)+'\n').encode())]:
        path = bundle / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different bundle input: {path}')
        path.write_bytes(data)
    print(json.dumps({'bundle': str(bundle), 'revision': REVISION, 'rows': len(read_csv(bundle/'source.csv')), 'model': contract['MODEL']}))


def load_bundle(bundle):
    contract_bytes = (bundle/'contract.json').read_bytes()
    contract = json.loads(contract_bytes)
    if digest((bundle/'source.csv').read_bytes()) != contract['source_sha256']:
        raise ValueError('Source checksum mismatch')
    rows = read_csv(bundle/'source.csv')
    ids = [r['StudyInstanceUID'] for r in rows]
    if len(ids) != len(set(ids)) or any(not uid for uid in ids):
        raise ValueError('Missing/duplicate study identity')
    return contract, digest(contract_bytes), rows


def identity(row, index, contract_hash):
    return {'row_index': index, 'study_uid': row['StudyInstanceUID'],
            'source_sha256': digest(row['Report'].encode()), 'contract_sha256': contract_hash}


def load_results(paths, rows, contract_hash):
    results = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            item = json.loads(line)
            index = item['row_index']
            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError('Invalid result row index')
            expected = identity(rows[index], index, contract_hash)
            if any(item.get(k) != v for k, v in expected.items()):
                raise ValueError('Result source/UID/contract mismatch')
            if item.get('status') != 'ok':
                continue
            if not isinstance(item.get('translation'), str) or not item['translation'].strip():
                raise ValueError('Empty successful translation')
            uid = item['study_uid']
            if uid in results and results[uid]['translation'] != item['translation']:
                raise ValueError(f'Conflicting translations for {uid}')
            results[uid] = item
    return results


def command(report, contract):
    # Same request flags and prompt as notebook cells 52–55; no API migration.
    cmd = ['claude', '-p', contract['user_prefix'] + report,
            '--model', contract['MODEL'], '--output-format', 'json', '--max-turns', '1',
            '--system-prompt', contract['SYSTEM_PROMPT'], '--tools', '',
            '--mcp-config', '{"mcpServers":{}}', '--strict-mcp-config']
    if contract.get('EFFORT'):
        cmd.extend(['--effort', contract['EFFORT']])
    return cmd


def translate(row, index, contract, contract_hash):
    record = identity(row, index, contract_hash)
    if not row['Report'].strip():
        return dict(record, status='failed', error='Empty source')
    error = 'No attempt'
    for attempt in range(contract['MAX_ATTEMPTS']):
        try:
            proc = subprocess.run(command(row['Report'], contract), capture_output=True,
                                  text=True, timeout=contract['TIMEOUT_S'], stdin=subprocess.DEVNULL)
            if proc.returncode:
                raise RuntimeError(f'Claude exit {proc.returncode}: {proc.stderr.strip()[:500]}')
            payload = json.loads(proc.stdout)
            if isinstance(payload, list):
                results = [event for event in payload if isinstance(event, dict) and event.get('type') == 'result']
                if len(results) != 1:
                    raise RuntimeError('Expected exactly one terminal CLI result')
                payload = results[0]
            if not isinstance(payload, dict):
                raise RuntimeError('Unexpected CLI JSON envelope')
            if payload.get('is_error') or not isinstance(payload.get('result'), str) or not payload['result'].strip():
                raise RuntimeError(f"Claude returned an error or empty result: {str(payload.get('result', ''))[:500]}")
            return dict(record, status='ok', translation=payload['result'],
                        model_usage=payload.get('modelUsage'), usage=payload.get('usage'),
                        cost_usd=payload.get('total_cost_usd'), requested_model=contract['MODEL'],
                        requested_effort=contract.get('EFFORT', 'CLI default'))
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt + 1 < contract['MAX_ATTEMPTS']:
                time.sleep(5 * 2**attempt)
    return dict(record, status='failed', error=error)


def run(args):
    contract, contract_hash, rows = load_bundle(args.bundle)
    if not 0 <= args.start < args.stop <= len(rows) or not 1 <= args.workers <= 8 or args.max_failed_rows < 1:
        raise ValueError('Invalid exclusive row range or worker count (1–8)')
    existing = list(args.completed)
    if args.output.exists():
        existing.append(args.output)
    done = load_results(existing, rows, contract_hash)
    todo = [(i, rows[i]) for i in range(args.start, args.stop) if rows[i]['StudyInstanceUID'] not in done]
    if args.reverse:
        todo.reverse()
    print(json.dumps({'pending': len(todo), 'start': args.start, 'stop_exclusive': args.stop,
                      'reverse': args.reverse, 'workers': args.workers, 'execute': args.execute, 'contract_sha256': contract_hash}), flush=True)
    if not args.execute:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # One writer per shard; independent shards must still have negotiated ranges.
    with args.output.with_suffix(args.output.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Refresh after taking the lock to prevent stale resume state.
        if args.output.exists():
            done.update(load_results([args.output], rows, contract_hash))
        todo = [(i, row) for i, row in todo if row['StudyInstanceUID'] not in done]
        version = subprocess.check_output(['claude', '--version'], text=True).strip()
        meta_path = args.output.with_suffix(args.output.suffix+'.run.json')
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if previous['claude_version'] != version or previous['contract_sha256'] != contract_hash:
                raise ValueError('CLI version or contract changed; use a separate result shard')
        write_json(meta_path, {'claude_version': version, 'contract_sha256': contract_hash,
                              'start': args.start, 'stop': args.stop, 'workers': args.workers,
                              'reverse': args.reverse,
                              'cwd': str(Path.cwd()), 'started_at_unix': time.time()})
        failures = 0
        with args.output.open('a') as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            remaining = iter(todo)
            pending = set()
            def submit_next():
                item = next(remaining, None)
                if item is not None:
                    i, row = item
                    pending.add(pool.submit(translate, row, i, contract, contract_hash))
            for _ in range(args.workers):
                submit_next()
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    failures += result['status'] != 'ok'
                    print(f"row {result['row_index']}: {result['status']}", flush=True)
                # Stop admitting work after repeated provider failures; drain only
                # requests already in flight rather than retrying a whole corpus.
                if failures < args.max_failed_rows:
                    for _ in completed:
                        submit_next()
        if failures:
            raise RuntimeError(f'{failures} rows failed; rerun the same shard to retry')


def merge(args):
    _, contract_hash, rows = load_bundle(args.bundle)
    results = load_results(args.results, rows, contract_hash)
    missing = [r for r in rows if r['Original Language'] != 'en' and r['StudyInstanceUID'] not in results]
    if missing:
        raise ValueError(f'{len(missing)} non-English rows missing; refusing partial training CSV')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['StudyInstanceUID', 'Report', 'Original Language', 'Translated Report'])
        writer.writeheader()
        for row in rows:
            translated = row['Report'] if row['Original Language'] == 'en' else results[row['StudyInstanceUID']]['translation']
            writer.writerow({k: translated if k == 'Translated Report' else row[k] for k in writer.fieldnames})
    write_json(args.output.with_suffix('.provenance.json'), {
        'contract_sha256': contract_hash, 'output_sha256': digest(args.output.read_bytes()),
        'result_files': {str(p): digest(p.read_bytes()) for p in args.results},
        'english_policy': 'retain original per training manifest contract',
        'language_source': 'Google detection from pinned origin/config; requires QC',
        'qc_status': 'unreviewed; not approved for training'})
    print(f'Exported {len(rows)} rows; semantic QC and a new cohort version still required.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    subs = parser.add_subparsers(dest='action', required=True)
    subs.add_parser('prepare')
    runner = subs.add_parser('run')
    runner.add_argument('--start', type=int, required=True)
    runner.add_argument('--stop', type=int, required=True)
    runner.add_argument('--workers', type=int, default=1)
    runner.add_argument('--reverse', action='store_true', help='Translate from stop - 1 down to start')
    runner.add_argument('--max-failed-rows', type=int, default=3, help='Stop scheduling after this many failed rows')
    runner.add_argument('--output', type=Path, required=True)
    runner.add_argument('--completed', type=Path, nargs='*', default=[])
    runner.add_argument('--execute', action='store_true')
    merger = subs.add_parser('merge')
    merger.add_argument('--results', type=Path, nargs='+', required=True)
    merger.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.bundle)
    elif args.action == 'run':
        run(args)
    else:
        merge(args)


if __name__ == '__main__':
    main()
