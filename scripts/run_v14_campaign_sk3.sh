#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU_CSV <run_v14_value_campaign arguments...>" >&2
  exit 2
fi

physical_csv="$1"
shift
IFS=',' read -r -a physical_ids <<< "$physical_csv"
if [[ ${#physical_ids[@]} -eq 0 ]]; then
  echo "no physical GPUs declared" >&2
  exit 2
fi

for gpu in "${physical_ids[@]}"; do
  case "$gpu" in
    0|5|6|7) ;;
    1|2|3|4)
      echo "HARD STOP: sk3 physical GPU $gpu is permanently forbidden" >&2
      exit 64
      ;;
    *)
      echo "invalid sk3 physical GPU $gpu; v14 permits only 0,5,6,7" >&2
      exit 64
      ;;
  esac
done

# Hold one nonblocking process lock per physical device before CUDA is exposed.
lock_index=0
for gpu in "${physical_ids[@]}"; do
  lock_fd=$((200 + lock_index))
  eval "exec ${lock_fd}>/tmp/cr3-v14-sk3-gpu-${gpu}.lock"
  if ! flock -n "$lock_fd"; then
    echo "physical GPU $gpu already has a v14 lane lock" >&2
    exit 75
  fi
  lock_index=$((lock_index + 1))
done

export CUDA_VISIBLE_DEVICES="$physical_csv"
export V14_PHYSICAL_GPUS="$physical_csv"

exec python -m methods.metric_implementer.experiments.run_v14_value_campaign \
  --physical-gpus "$physical_csv" "$@"
