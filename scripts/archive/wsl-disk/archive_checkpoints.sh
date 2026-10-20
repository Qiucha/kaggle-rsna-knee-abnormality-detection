#!/usr/bin/env bash
# Move cold checkpoint data from C: (the WSL disk) to the D: archive disk, keeping small metadata in place.
#
#   scripts/archive_checkpoints.sh run   <path-under-checkpoints>...   # move files > 5 MB; keep json/jsonl/logs
#   scripts/archive_checkpoints.sh whole <path-under-checkpoints>...   # move the whole directory (caches)
#
# Each item is copied, verified with `rsync --checksum`, flushed, and only then deleted from C:.
# A note records where it went: <run>/MOVED_TO_D.txt, or <cache>.MOVED_TO_D.txt next to a moved directory.
set -euo pipefail

MODE=${1:?usage: archive_checkpoints.sh run|whole <path>...}; shift
SRC_ROOT=/home/q/Projects/novo-project/checkpoints
DST_ROOT=/mnt/wsl/novo-archive/novo-project/checkpoints
MIN_SIZE=+5M

"$HOME/.local/bin/ensure-novo-archive"

for rel in "$@"; do
  rel=${rel%/}
  src=$SRC_ROOT/$rel
  dst=$DST_ROOT/$rel
  [ -d "$src" ] || { echo "skip (missing): $rel"; continue; }
  echo "== $(date +%T) $rel"
  mkdir -p "$dst"
  if [ "$MODE" = whole ]; then
    rsync -a "$src/" "$dst/"
    diff=$(rsync -a --checksum --dry-run --itemize-changes "$src/" "$dst/")
    [ -z "$diff" ] || { echo "VERIFY FAILED: $rel"; echo "$diff" | head; exit 1; }
    sync -f "$dst"
    size=$(du -sh "$src" | cut -f1)
    rm -rf "$src"
    printf 'Archived %s (%s) on %s\nNow at: %s\n' "$rel" "$size" "$(date -I)" "$dst" > "$src.MOVED_TO_D.txt"
  else
    list=$(mktemp)
    (cd "$src" && find . -type f -size "$MIN_SIZE" -printf '%P\n' | sort) > "$list"
    [ -s "$list" ] || { echo "nothing > 5 MB"; rm -f "$list"; continue; }
    rsync -a --files-from="$list" "$src/" "$dst/"
    diff=$(rsync -a --checksum --dry-run --itemize-changes --files-from="$list" "$src/" "$dst/")
    [ -z "$diff" ] || { echo "VERIFY FAILED: $rel"; echo "$diff" | head; exit 1; }
    sync -f "$dst"
    (cd "$src" && xargs -d '\n' rm -f < "$list")
    { printf 'Files > 5 MB archived on %s\nNow at: %s\n\n' "$(date -I)" "$dst"; cat "$list"; } >> "$src/MOVED_TO_D.txt"
    rm -f "$list"
  fi
  echo "   done; C: free $(df -h --output=avail /mnt/c | tail -1)"
done
