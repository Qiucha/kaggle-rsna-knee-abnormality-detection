#!/usr/bin/env bash
# Kill process group PGID if the Windows C: drive drops below N GB free (default 15), checking every 60 s.
# A full C: crashes WSL, so stopping the run early is the lesser loss.
# Usage:  setsid nohup scripts/disk_watchdog.sh PGID [N] LOG < /dev/null &
set -uo pipefail
pgid=$1; need=${2:-15}; log=${3:-/dev/stderr}
while kill -0 -- "-$pgid" 2>/dev/null; do
  free=$(df --output=avail -BG /mnt/c | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "$need" ]; then
    echo "$(date '+%F %T') C: ${free}G free < ${need}G; killing process group $pgid" >> "$log"
    kill -TERM -- "-$pgid"; sleep 30; kill -KILL -- "-$pgid" 2>/dev/null
    exit 1
  fi
  sleep 60
done
echo "$(date '+%F %T') process group $pgid gone; watchdog exiting (C: ${free:-?}G free)" >> "$log"
