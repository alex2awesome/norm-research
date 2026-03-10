#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <data_path> <output_root> [--use-optuna] [extra train_reward_model.py args...]"
  echo "Example: $0 datasets/creative-writing/LitBench-Train.csv.gz runs/sweep_01 --use-optuna --optuna_trials 20"
  exit 1
fi

DATA_PATH="$1"
OUTPUT_ROOT="$2"
shift 2
USE_OPTUNA=0
if [[ $# -gt 0 && "$1" == "--use-optuna" ]]; then
  USE_OPTUNA=1
  shift
fi

mkdir -p "${OUTPUT_ROOT}"
PERCENTAGES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for PCT in "${PERCENTAGES[@]}"; do
  PCT_LABEL="${PCT/./p}"
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/subset_${PCT_LABEL}"
  echo "Starting run for train_subset_percentage=${PCT} -> ${RUN_OUTPUT_DIR}"

  OPTUNA_ARGS=()
  if [[ "${USE_OPTUNA}" -eq 1 ]]; then
    OPTUNA_ARGS+=(--use_optuna)
  fi

  python methods/dense/train_reward_model.py \
    --data_path "${DATA_PATH}" \
    --train_subset_percentage "${PCT}" \
    --output_dir "${RUN_OUTPUT_DIR}" \
    "${OPTUNA_ARGS[@]}" \
    "$@"
done

echo "Sweep complete. Outputs in: ${OUTPUT_ROOT}"
