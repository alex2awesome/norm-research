#!/usr/bin/env bash
# launch_when_gpus_free.sh — Wait for N free GPUs, then run a command.
#
# Usage:
#   ./launch_when_gpus_free.sh <num_gpus> <command...>
#
# Example:
#   ./launch_when_gpus_free.sh 2 ./train_sweep.sh \
#     datasets/press-releases/press_release_modeling_dataset.csv.gz \
#     runs/press_release_sweep_llama-70b \
#     --use-optuna --optuna_trials 5 \
#     --model_name meta-llama/Llama-3.3-70B-Instruct \
#     --quantize --batch_size 1 --gradient_accumulation_steps 8
#
# A GPU is considered "free" if it has less than 100 MiB used.
# Once enough GPUs are free, CUDA_VISIBLE_DEVICES is set automatically
# and the command is executed.
set -euo pipefail

FREE_THRESHOLD_MIB=100
POLL_INTERVAL=60

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <num_gpus_needed> <command...>"
  exit 1
fi

NUM_NEEDED="$1"
shift

echo "Waiting for ${NUM_NEEDED} free GPUs (< ${FREE_THRESHOLD_MIB} MiB used)..."
echo "Polling every ${POLL_INTERVAL}s. Will run: $*"

while true; do
  # Query GPU memory usage; output: "index memory.used [MiB]" per line
  FREE_GPUS=()
  while IFS=', ' read -r idx used _unit; do
    used_int="${used%% *}"
    if [[ "${used_int}" -lt "${FREE_THRESHOLD_MIB}" ]]; then
      FREE_GPUS+=("${idx}")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  if [[ ${#FREE_GPUS[@]} -ge ${NUM_NEEDED} ]]; then
    # Take the first N free GPUs
    SELECTED=()
    for ((i = 0; i < NUM_NEEDED; i++)); do
      SELECTED+=("${FREE_GPUS[$i]}")
    done
    CUDA_STR=$(IFS=,; echo "${SELECTED[*]}")
    echo "$(date): Found ${#FREE_GPUS[@]} free GPUs. Using CUDA_VISIBLE_DEVICES=${CUDA_STR}"
    export CUDA_VISIBLE_DEVICES="${CUDA_STR}"
    exec "$@"
  fi

  echo "$(date): Only ${#FREE_GPUS[@]}/${NUM_NEEDED} GPUs free. Sleeping ${POLL_INTERVAL}s..."
  sleep "${POLL_INTERVAL}"
done
