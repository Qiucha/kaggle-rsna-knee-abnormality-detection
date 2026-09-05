#!/usr/bin/env python3
"""Audit legacy cohort bytes and add a missing checksum without rebuilding splits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from contrastive_learning.manifest import _sha256_file, read_cohort_manifest


def migrate(path: Path, audit_path: Path) -> None:
    metadata_path = path.with_suffix('.meta.json')
    original = metadata_path.read_bytes()
    metadata = json.loads(original)
    if metadata.get('manifest_sha256'):
        raise ValueError('Manifest already has a checksum; refusing to replace it')
    if audit_path.exists():
        raise FileExistsError(audit_path)
    for name, expected in metadata['source_hashes'].items():
        if _sha256_file(Path(metadata['config'][name])) != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for row in rows:
        for field in ('report', 'source_report'):
            if hashlib.sha256(row[field].encode()).hexdigest() != row[field + '_sha256']:
                raise ValueError(f'{field} checksum mismatch')
        if row['training_eligible'] and row['preprocessing_status'] != 'passed':
            raise ValueError('Eligible row lacks saved pixel QC')
        if any(not Path(slot).is_dir() for slot in row['slots'].values()):
            raise ValueError('Manifest series path is unavailable')
    manifest_hash = _sha256_file(path)
    metadata['manifest_sha256'] = manifest_hash
    audit = {
        'schema_version': 1,
        'migrated_at': datetime.now(timezone.utc).isoformat(),
        'manifest_path': str(path),
        'manifest_sha256_before': manifest_hash,
        'manifest_sha256_after': manifest_hash,
        'metadata_before_sha256': hashlib.sha256(original).hexdigest(),
        'metadata_before': json.loads(original),
        'rows': len(rows),
        'split_counts': dict(Counter(row['split'] for row in rows)),
        'patient_groups': len({row['patient_group_id'] for row in rows}),
        'checks': ['source CSV hashes', 'report hashes', 'series directories',
                   'saved pixel QC', 'reader UID/group/split/integrity validation'],
        'limitations': 'Establishes integrity from migration onward; no historical checksum or fresh pixel decode. Patient independence remains unproven.',
    }
    temporary = metadata_path.with_suffix('.migration.tmp')
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    temporary.replace(metadata_path)
    try:
        read_cohort_manifest(path)
        if _sha256_file(path) != manifest_hash:
            raise ValueError('Manifest changed during migration')
        audit['metadata_after_sha256'] = _sha256_file(metadata_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    except Exception:
        metadata_path.write_bytes(original)
        raise
    print(json.dumps({key: audit[key] for key in ('rows', 'split_counts', 'manifest_sha256_after')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.manifest, args.audit)
