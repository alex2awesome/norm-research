#!/usr/bin/env bash
# train_sweep.sh — Run dense reward model training across data fractions (10%–100%).
#
# Usage:
#   ./train_sweep.sh <data_path> <output_root> [--use-optuna] [--parallel] [--gpu-mem-required MB] [extra train_reward_model.py args...]
#
# Examples:
#   # Sequential 8B sweep (one subset at a time)
#   ./train_sweep.sh datasets/press-releases/press_release_modeling_dataset.csv.gz \
#     runs/press_release_sweep_llama-8b --use-optuna --optuna_trials 5
#
#   # Parallel 8B sweep — auto-assigns subsets to GPUs with enough free memory
#   ./train_sweep.sh datasets/press-releases/press_release_modeling_dataset.csv.gz \
#     runs/press_release_sweep_llama-8b --use-optuna --parallel --optuna_trials 5
#
#   # Parallel with custom memory requirement (default: 25000 MB)
#   ./train_sweep.sh datasets/press-releases/press_release_modeling_dataset.csv.gz \
#     runs/press_release_sweep_llama-8b --use-optuna --parallel --gpu-mem-required 40000 --optuna_trials 5
#
#   # Llama-3.3-70B sweep with QLoRA on B200 GPUs
#   CUDA_VISIBLE_DEVICES=1,5 ./train_sweep.sh \
#     datasets/press-releases/press_release_modeling_dataset.csv.gz \
#     runs/press_release_sweep_llama-70b \
#     --use-optuna \
#     --optuna_trials 5 \
#     --model_name meta-llama/Llama-3.3-70B-Instruct \
#     --quantize \
#     --batch_size 1 \
#     --gradient_accumulation_steps 8
#
# Notes:
#   - For 70B models, --quantize enables 4-bit QLoRA (requires bitsandbytes).
#   - Set CUDA_VISIBLE_DEVICES to select GPUs. The 70B model in 4-bit fits on a
#     single B200 (~40 GB); use device_map="auto" (the default) to spread across
#     multiple GPUs if needed.
#   - --parallel mode checks free GPU memory and launches one subset per GPU.
#     It polls every 60s for newly freed GPUs until all subsets are done.
#   - Extra args are forwarded to train_reward_model.py.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <data_path> <output_root> [--use-optuna] [--parallel] [--gpu-mem-required MB] [extra args...]"
  exit 1
fi

DATA_PATH="$1"
OUTPUT_ROOT="$2"
shift 2

USE_OPTUNA=0
PARALLEL=0
GPU_MEM_REQUIRED=25000  # MB of free VRAM needed to launch a job

# Parse our flags (order-independent), pass the rest through
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-optuna)  USE_OPTUNA=1; shift ;;
    --parallel)    PARALLEL=1; shift ;;
    --gpu-mem-required) GPU_MEM_REQUIRED="$2"; shift 2 ;;
    *)             EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
PERCENTAGES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# ── Helper: get free memory (MB) for a given GPU index ──
gpu_free_mb() {
  local gpu_id="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | tr -d ' '
}

# ── Helper: find all GPUs with enough free memory ──
# Returns space-separated list of GPU indices
find_free_gpus() {
  local free_gpus=()
  local num_gpus
  num_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  for (( i=0; i<num_gpus; i++ )); do
    local free
    free=$(gpu_free_mb "$i")
    if [[ "${free}" -ge "${GPU_MEM_REQUIRED}" ]]; then
      free_gpus+=("$i")
    fi
  done
  echo "${free_gpus[*]:-}"
}

# ── Helper: launch a single subset training run ──
launch_subset() {
  local pct="$1"
  local gpu_id="$2"
  local pct_label="${pct/./p}"
  local run_output_dir="${OUTPUT_ROOT}/subset_${pct_label}"

  local optuna_args=()
  if [[ "${USE_OPTUNA}" -eq 1 ]]; then
    optuna_args+=(--use_optuna)
  fi

  echo "[$(date '+%H:%M:%S')] Launching subset ${pct} on GPU ${gpu_id} -> ${run_output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python methods/dense/train_reward_model.py \
    --data_path "${DATA_PATH}" \
    --train_subset_percentage "${pct}" \
    --output_dir "${run_output_dir}" \
    "${optuna_args[@]}" \
    "${EXTRA_ARGS[@]}" \
    > "${run_output_dir}/sweep_stdout.log" 2>&1 &

  local pid=$!
  echo "[$(date '+%H:%M:%S')] Started PID ${pid} for subset ${pct} on GPU ${gpu_id}"
  # Return pid:pct:gpu mapping
  echo "${pid}:${pct}:${gpu_id}"
}

# ── Build list of subsets that still need to run ──
PENDING=()
for PCT in "${PERCENTAGES[@]}"; do
  PCT_LABEL="${PCT/./p}"
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/subset_${PCT_LABEL}"

  if [[ -d "${RUN_OUTPUT_DIR}" ]]; then
    COMPLETED=$(find "${RUN_OUTPUT_DIR}" -maxdepth 2 -type d -name "best_model" 2>/dev/null | wc -l)
    if [[ "${COMPLETED}" -gt 0 ]]; then
      echo "Skipping subset ${PCT} (${COMPLETED} completed trials found in ${RUN_OUTPUT_DIR})"
      continue
    fi
  fi
  PENDING+=("${PCT}")
done

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "All subsets already completed!"
  exit 0
fi

echo "Subsets to run: ${PENDING[*]}"
echo "Mode: $(if [[ ${PARALLEL} -eq 1 ]]; then echo "parallel (need ${GPU_MEM_REQUIRED} MB free per GPU)"; else echo "sequential"; fi)"

# ── Sequential mode (original behavior) ──
if [[ "${PARALLEL}" -eq 0 ]]; then
  for PCT in "${PENDING[@]}"; do
    PCT_LABEL="${PCT/./p}"
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/subset_${PCT_LABEL}"
    mkdir -p "${RUN_OUTPUT_DIR}"
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
      "${EXTRA_ARGS[@]}"
  done
  echo "Sweep complete. Outputs in: ${OUTPUT_ROOT}"
  exit 0
fi

# ── Parallel mode ──
declare -A RUNNING=()   # pid -> "pct:gpu"
NEXT_IDX=0              # index into PENDING array

while true; do
  # Check for completed jobs
  for pid in "${!RUNNING[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      info="${RUNNING[${pid}]}"
      pct="${info%%:*}"
      gpu="${info##*:}"
      wait "${pid}" && status=0 || status=$?
      if [[ ${status} -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] ✓ Subset ${pct} completed successfully (GPU ${gpu})"
      else
        echo "[$(date '+%H:%M:%S')] ✗ Subset ${pct} FAILED with exit code ${status} (GPU ${gpu})"
        echo "  Check log: ${OUTPUT_ROOT}/subset_${pct/./p}/sweep_stdout.log"
      fi
      unset RUNNING["${pid}"]
    fi
  done

  # All done?
  if [[ ${NEXT_IDX} -ge ${#PENDING[@]} && ${#RUNNING[@]} -eq 0 ]]; then
    echo "All subsets finished!"
    break
  fi

  # Try to launch more jobs if subsets remain
  if [[ ${NEXT_IDX} -lt ${#PENDING[@]} ]]; then
    # Get GPUs currently used by our jobs
    USED_GPUS=()
    for pid in "${!RUNNING[@]}"; do
      info="${RUNNING[${pid}]}"
      USED_GPUS+=("${info##*:}")
    done

    # Find free GPUs not already running one of our jobs
    FREE_GPUS_STR=$(find_free_gpus)
    if [[ -n "${FREE_GPUS_STR}" ]]; then
      read -ra FREE_GPUS <<< "${FREE_GPUS_STR}"
      for gpu_id in "${FREE_GPUS[@]}"; do
        # Skip if we already have a job on this GPU
        skip=0
        for used in "${USED_GPUS[@]:-}"; do
          if [[ "${gpu_id}" == "${used}" ]]; then
            skip=1
            break
          fi
        done
        if [[ ${skip} -eq 1 ]]; then continue; fi

        # Launch next pending subset
        if [[ ${NEXT_IDX} -lt ${#PENDING[@]} ]]; then
          PCT="${PENDING[${NEXT_IDX}]}"
          mkdir -p "${OUTPUT_ROOT}/subset_${PCT/./p}"
          result=$(launch_subset "${PCT}" "${gpu_id}")
          pid=$(echo "${result}" | tail -1 | cut -d: -f1)
          RUNNING["${pid}"]="${PCT}:${gpu_id}"
          NEXT_IDX=$((NEXT_IDX + 1))
          USED_GPUS+=("${gpu_id}")
        fi
      done
    fi
  fi

  # Status update
  if [[ ${#RUNNING[@]} -gt 0 ]]; then
    running_info=""
    for pid in "${!RUNNING[@]}"; do
      info="${RUNNING[${pid}]}"
      running_info+=" subset_${info%%:*}(GPU${info##*:})"
    done
    remaining=$((${#PENDING[@]} - NEXT_IDX))
    echo "[$(date '+%H:%M:%S')] Running:${running_info} | Queued: ${remaining} | Polling in 60s..."
  fi

  sleep 60
done

echo "Sweep complete. Outputs in: ${OUTPUT_ROOT}"
