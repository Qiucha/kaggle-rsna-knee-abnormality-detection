#!/usr/bin/env bash
# Exit 1 unless the Windows C: drive has at least N GB free (default 25).
# The WSL disk file lives on C:, so a full C: crashes WSL even while `df /` inside Linux looks fine.
# Use in chain scripts before each heavy step:  scripts/require_free_space.sh 25 || exit 1
set -euo pipefail
need=${1:-25}
free=$(df --output=avail -BG /mnt/c | tail -1 | tr -dc '0-9')
if [ "$free" -lt "$need" ]; then
  echo "require_free_space: C: has ${free}G free, need ${need}G; archive cold runs with scripts/archive_checkpoints.sh" >&2
  exit 1
fi
