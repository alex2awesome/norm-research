#!/usr/bin/env bash
set -euo pipefail

# Stop one owned GPU job if the server-wide occupied-GPU count exceeds a cap.
# A GPU counts as occupied above 100 MiB; idle devices on sk3 use only 4-6 MiB.

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 TARGET_PID [MAX_OCCUPIED_GPUS=4] [POLL_SECONDS=5]" >&2
  exit 2
fi

target_pid=$1
max_occupied=${2:-4}
poll_seconds=${3:-5}

if ! [[ $target_pid =~ ^[0-9]+$ && $max_occupied =~ ^[0-9]+$ ]]; then
  echo "PID and GPU cap must be nonnegative integers" >&2
  exit 2
fi

occupied_count() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 100 { count++ } END { print count + 0 }'
}

while kill -0 "$target_pid" 2>/dev/null; do
  occupied=$(occupied_count)
  if (( occupied > max_occupied )); then
    mapfile -t children < <(pgrep -P "$target_pid" || true)
    echo "$(date -Is) quota_exceeded=$occupied cap=$max_occupied stopping=$target_pid children=${children[*]:-none}"
    ((${#children[@]} == 0)) || kill -TERM "${children[@]}" 2>/dev/null || true
    kill -TERM "$target_pid" 2>/dev/null || true
    sleep 3
    ((${#children[@]} == 0)) || kill -KILL "${children[@]}" 2>/dev/null || true
    kill -KILL "$target_pid" 2>/dev/null || true
    exit 42
  fi
  sleep "$poll_seconds"
done

echo "$(date -Is) process_complete=$target_pid"
