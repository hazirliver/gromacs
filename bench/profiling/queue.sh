#!/usr/bin/env bash
# Run a list of shell commands (one per line in FILE) sequentially, logging start/end times.
#   bench/profiling/queue.sh FILE LOG
set -u
file=${1:?}; log=${2:?}
while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    case "$cmd" in \#*) continue;; esac
    echo "[$(date -u +%H:%M:%S)] START $cmd" >> "$log"
    bash -c "$cmd" >> "$log" 2>&1
    echo "[$(date -u +%H:%M:%S)] END rc=$? $cmd" >> "$log"
done < "$file"
echo "[$(date -u +%H:%M:%S)] QUEUE DONE" >> "$log"
