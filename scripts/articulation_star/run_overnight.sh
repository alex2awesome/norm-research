#!/bin/bash
# Overnight multi-iter STaR loop for articulation training.
#
# Per iter:
#   1. Generate K rationales per artifact using prev-iter LoRA (4-way DP
#      across GPU_A..GPU_D).
#   2. Score with weak judge (GPU_A) and strong judge (GPU_B) in parallel.
#   3. Combine: balanced top-K_PER_LABEL pos and neg by strong_margin,
#      with weak_margin < tau_weak constraint.
#   4. SFT 8B + LoRA on the balanced kept set (continued from prev LoRA).
#
# Env:
#   TASK             (default: creative_writing)
#   RUN_NAME         (default: v1_overnight_logprob)
#   N_ITERS          (default: 3)
#   N_TRAIN          (default: 10000) -- artifacts/iter
#   N_RATIONALES     (default: 4)     -- samples/artifact -> 40K rationales
#   K_PER_LABEL      (default: 1500)  -- balanced training-set size = 2 * K
#   GPU_A,B,C,D      (default: 1,5,6,7)
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_MOE_FP8=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

TASK="${TASK:-creative_writing}"
RUN_NAME="${RUN_NAME:-v1_overnight_logprob}"
N_ITERS="${N_ITERS:-3}"
N_TRAIN="${N_TRAIN:-10000}"
N_RATIONALES="${N_RATIONALES:-4}"
K_PER_LABEL="${K_PER_LABEL:-1500}"
TAU_STRONG="${TAU_STRONG:-0.0}"
TAU_WEAK="${TAU_WEAK:-0.0}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-5}"
GPU_C="${GPU_C:-6}"
GPU_D="${GPU_D:-7}"

LOG_DIR="logs/articulation_star/${RUN_NAME}"
mkdir -p "$LOG_DIR"

# Resume support: START_ITER (default 0) and PREV_LORA (init for that iter).
START_ITER="${START_ITER:-0}"
CUR_LORA="${PREV_LORA:-}"

for (( ITER=START_ITER; ITER<N_ITERS; ITER++ )); do
  ITER_TAG=$(printf "%02d" "$ITER")
  echo
  echo "=================================================================="
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : begin ==="
  echo "=================================================================="

  # ── 1. Generate (4-way DP) ──────────────────────────────────────
  GEN_EXTRA=""
  [[ -n "$CUR_LORA" ]] && GEN_EXTRA="--lora_path $CUR_LORA"

  for SHARD in 0 1 2 3; do
    case "$SHARD" in
      0) GPU=$GPU_A ;;
      1) GPU=$GPU_B ;;
      2) GPU=$GPU_C ;;
      3) GPU=$GPU_D ;;
    esac
    CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.generate_rationales \
      --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
      --shard_idx $SHARD --n_shards 4 \
      --n_train_subsample "$N_TRAIN" --n_rationales_per_input "$N_RATIONALES" \
      $GEN_EXTRA \
      > "$LOG_DIR/iter${ITER_TAG}_gen_shard${SHARD}.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : generation done ==="

  python -m methods.articulation_star.merge_shards \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    2>&1 | tee "$LOG_DIR/iter${ITER_TAG}_merge.log"

  # ── 2. Score weak (GPU_A) + strong (GPU_B) in parallel ───────────
  CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.judge_filter \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    --mode predict --judge weak \
    > "$LOG_DIR/iter${ITER_TAG}_weak.log" 2>&1 &
  PID_W=$!

  CUDA_VISIBLE_DEVICES=$GPU_B python -m methods.articulation_star.judge_filter \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    --mode predict --judge strong \
    > "$LOG_DIR/iter${ITER_TAG}_strong.log" 2>&1 &
  PID_S=$!

  if ! wait $PID_W; then echo "weak judge failed iter $ITER"; exit 1; fi
  if ! wait $PID_S; then echo "strong judge failed iter $ITER"; exit 1; fi
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : scoring done ==="

  # ── 3. Combine balanced ─────────────────────────────────────────
  python -m methods.articulation_star.judge_filter \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    --mode combine \
    --tau_strong "$TAU_STRONG" --tau_weak "$TAU_WEAK" \
    --balanced_k_per_label "$K_PER_LABEL" \
    2>&1 | tee "$LOG_DIR/iter${ITER_TAG}_combine.log"

  # ── 4. Train (GPU_A) ────────────────────────────────────────────
  TRAIN_EXTRA=""
  [[ -n "$CUR_LORA" ]] && TRAIN_EXTRA="--prev_lora $CUR_LORA"

  CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.train_sft \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    --n_train_subsample "$N_TRAIN" \
    $TRAIN_EXTRA \
    > "$LOG_DIR/iter${ITER_TAG}_train.log" 2>&1
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : training done ==="

  CUR_LORA="$(pwd)/outputs/articulation_star/$TASK/$RUN_NAME/iter_${ITER_TAG}/lora"
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : LoRA at $CUR_LORA ==="

  # quick audit summary
  python -m methods.articulation_star.audit_rationales \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" --n_samples 0 \
    > "$LOG_DIR/iter${ITER_TAG}_audit.log" 2>&1 || true
done

echo
echo "=================================================================="
echo "[$(date '+%F %T')] === ALL $N_ITERS ITERS COMPLETE ==="
echo "  final LoRA: $CUR_LORA"
echo "=================================================================="
