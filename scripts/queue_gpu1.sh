#!/bin/bash
# Queue worker for one "slice" of the dense sweep work list, pinned to one GPU.
# Usage: queue_gpu1.sh <math|humor|codereview>
# Sequential within a slice; run 3 instances concurrently to pack one GPU.
set -u
cd /lfs/skampere3/0/alexspan/norm-research || exit 1
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
export HOME=/lfs/skampere3/0/alexspan

SLICE="${1:?usage: queue_gpu1.sh <math|humor|codereview>}"
GPU_ID="${GPU_ID:-1}"
GPU_MEM_REQUIRED="${GPU_MEM_REQUIRED:-40000}"

DRIVERS_DIR=logs/sweep_drivers
mkdir -p "$DRIVERS_DIR"
PIDFILE="${DRIVERS_DIR}/queue_${SLICE}.pid"
DONEFILE="${DRIVERS_DIR}/queue_${SLICE}.done"
echo "$$" > "$PIDFILE"
cleanup() { rm -f "$PIDFILE"; }
trap cleanup EXIT

wait_for_gpu() {
  local req="$1"
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' ')
    if [[ -n "$free" && "$free" -ge "$req" ]]; then return 0; fi
    sleep 60
  done
}

run_one() {
  local data_path="$1" out_root="$2" frac="$3" max_len="$4" bs="$5" grad="$6" evalbs="$7"
  local label="${frac/./p}"
  local sub_dir="${out_root}/subset_${label}"

  if [[ -d "${sub_dir}/best_model" ]]; then
    echo "[$(date '+%F %T')] [skip] ${sub_dir} already done"; return 0
  fi

  mkdir -p "${sub_dir}"
  if ! ( set -C; echo "$$" > "${sub_dir}/.running" ) 2>/dev/null; then
    local owner
    owner=$(cat "${sub_dir}/.running" 2>/dev/null)
    if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
      echo "[$(date '+%F %T')] [skip] ${sub_dir} busy (PID $owner)"; return 0
    fi
    echo "$$" > "${sub_dir}/.running"   # stale lock -> reclaim
  fi

  echo "[$(date '+%F %T')] Waiting for GPU ${GPU_ID} free >= ${GPU_MEM_REQUIRED} MB ..."
  wait_for_gpu "${GPU_MEM_REQUIRED}"

  echo "[$(date '+%F %T')] Launching ${sub_dir} (frac=${frac} max_len=${max_len})"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u methods/dense/train_reward_model.py \
    --data_path "${data_path}" \
    --train_subset_percentage "${frac}" \
    --output_dir "${sub_dir}" \
    --model_name meta-llama/Llama-3.1-8B \
    --max_length "${max_len}" \
    --class_weight_auto \
    --epochs 3 \
    --batch_size "${bs}" \
    --gradient_accumulation_steps "${grad}" \
    --learning_rate 5e-5 \
    --lora_r 16 --lora_alpha 32 \
    --gradient-checkpointing \
    --eval_batch_size "${evalbs}" \
    --log_every 50 \
    > "${sub_dir}/sweep_stdout.log" 2>&1
  local status=$?
  rm -f "${sub_dir}/.running"
  echo "[$(date '+%F %T')] Done ${sub_dir} (exit=${status})"
}

MATH_DATA=datasets/math-stackexchange/math_se_modeling.csv.gz
HUMOR_DATA=datasets/humor/reddit_humor_modeling.csv.gz
CR_DATA=datasets/code-review/code_review_dense_4096tok.csv.gz

case "$SLICE" in
  math)
    for f in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
      run_one "$MATH_DATA" runs/math_se_sweep_llama8b "$f" 2048 2 8 16
    done ;;
  humor)
    for f in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
      run_one "$HUMOR_DATA" runs/reddit_humor_sweep_llama8b "$f" 512 4 4 32
    done ;;
  codereview)
    for f in 0.7 0.8 0.9; do
      run_one "$CR_DATA" runs/code_review_sweep_llama8b "$f" 2048 2 8 16
    done ;;
  *) echo "unknown slice: $SLICE" >&2; exit 1 ;;
esac

touch "$DONEFILE"
echo "[$(date '+%F %T')] Slice ${SLICE} complete."
