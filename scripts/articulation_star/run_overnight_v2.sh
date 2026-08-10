#!/bin/bash
# v2 overnight: 3-iter STaR loop on creative_writing with WEAK = Llama-3.2-1B
# (was 3B in v1). Plus held-out test eval + leakage detection at the end.
#
# Pipeline:
#   for iter in 0..N_ITERS-1:
#     generate (4-way DP) -> score weak + strong (parallel) -> combine balanced -> train
#   test_eval for each stage (base, iter00, iter01, iter02)
#   leakage_detect auto for each stage
#   leakage_detect llm for each stage (60-sample LLM-judged rubric)
#   final summary
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_MOE_FP8=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

TASK="${TASK:-creative_writing}"
RUN_NAME="${RUN_NAME:-v2_weak1b_logprob}"
N_ITERS="${N_ITERS:-3}"
N_TRAIN="${N_TRAIN:-10000}"
N_RATIONALES="${N_RATIONALES:-4}"
K_PER_LABEL="${K_PER_LABEL:-1500}"
TAU_STRONG="${TAU_STRONG:-0.0}"
TAU_WEAK="${TAU_WEAK:-0.0}"
N_TEST="${N_TEST:-500}"
N_LLM_SAMPLE="${N_LLM_SAMPLE:-60}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-5}"
GPU_C="${GPU_C:-6}"
GPU_D="${GPU_D:-7}"

LOG_DIR="logs/articulation_star/${RUN_NAME}"
mkdir -p "$LOG_DIR" "$LOG_DIR/test_eval"

CUR_LORA=""
for (( ITER=0; ITER<N_ITERS; ITER++ )); do
  ITER_TAG=$(printf "%02d" "$ITER")
  echo
  echo "=================================================================="
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : begin ==="
  echo "=================================================================="

  # 1. Generate (4-way DP)
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
  python -m methods.articulation_star.merge_shards \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    2>&1 | tee "$LOG_DIR/iter${ITER_TAG}_merge.log"
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : generation done ==="

  # 2. Score weak (1B, GPU_A) + strong (122B, GPU_B) parallel
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

  # 3. Combine balanced
  python -m methods.articulation_star.judge_filter \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    --mode combine --tau_strong "$TAU_STRONG" --tau_weak "$TAU_WEAK" \
    --balanced_k_per_label "$K_PER_LABEL" \
    2>&1 | tee "$LOG_DIR/iter${ITER_TAG}_combine.log"

  # 4. Train SFT
  TRAIN_EXTRA=""
  [[ -n "$CUR_LORA" ]] && TRAIN_EXTRA="--prev_lora $CUR_LORA"
  CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.train_sft \
    --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
    --n_train_subsample "$N_TRAIN" \
    $TRAIN_EXTRA \
    > "$LOG_DIR/iter${ITER_TAG}_train.log" 2>&1
  CUR_LORA="$(pwd)/outputs/articulation_star/$TASK/$RUN_NAME/iter_${ITER_TAG}/lora"
  echo "[$(date '+%F %T')] === iter ${ITER_TAG} : training done; LoRA at $CUR_LORA ==="
done

# ── HELD-OUT TEST EVAL (all stages) ─────────────────────────────
echo
echo "=================================================================="
echo "[$(date '+%F %T')] === TEST EVAL ==="
echo "=================================================================="
python -m methods.articulation_star.test_eval \
  --task "$TASK" --run_name "$RUN_NAME" --mode build_split --n_test "$N_TEST" \
  2>&1 | tee "$LOG_DIR/test_eval/build_split.log"

LORA_ROOT="$(pwd)/outputs/articulation_star/${TASK}/${RUN_NAME}"
for STAGE in base iter00 iter01 iter02; do
  if [[ "$STAGE" == "base" ]]; then
    LORA_FLAG=""
  else
    ITER_TAG="${STAGE#iter}"
    LORA_FLAG="--lora_path ${LORA_ROOT}/iter_${ITER_TAG}/lora"
  fi
  echo "[$(date '+%F %T')] === test eval: stage ${STAGE} generate ==="
  CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.test_eval \
    --task "$TASK" --run_name "$RUN_NAME" --mode generate \
    --stage "$STAGE" $LORA_FLAG \
    > "$LOG_DIR/test_eval/gen_${STAGE}.log" 2>&1
  echo "[$(date '+%F %T')] === test eval: stage ${STAGE} score ==="
  CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.test_eval \
    --task "$TASK" --run_name "$RUN_NAME" --mode score \
    --stage "$STAGE" \
    > "$LOG_DIR/test_eval/score_${STAGE}.log" 2>&1
done

python -m methods.articulation_star.test_eval \
  --task "$TASK" --run_name "$RUN_NAME" --mode summarize \
  2>&1 | tee "$LOG_DIR/test_eval/summary.log"

# ── LEAKAGE DETECTION (auto + LLM) ──────────────────────────────
echo
echo "=================================================================="
echo "[$(date '+%F %T')] === LEAKAGE DETECTION ==="
echo "=================================================================="
for STAGE in base iter00 iter01 iter02; do
  python -m methods.articulation_star.leakage_detect \
    --task "$TASK" --run_name "$RUN_NAME" --mode auto --stage "$STAGE" \
    2>&1 | tee "$LOG_DIR/test_eval/leakage_auto_${STAGE}.log"
done
python -m methods.articulation_star.leakage_detect \
  --task "$TASK" --run_name "$RUN_NAME" --mode aggregate \
  2>&1 | tee "$LOG_DIR/test_eval/leakage_auto_summary.log"

for STAGE in base iter00 iter01 iter02; do
  python -m methods.articulation_star.leakage_detect \
    --task "$TASK" --run_name "$RUN_NAME" --mode llm --stage "$STAGE" \
    --n_sample "$N_LLM_SAMPLE" \
    > "$LOG_DIR/test_eval/leakage_llm_${STAGE}.log" 2>&1
done
python -m methods.articulation_star.leakage_detect \
  --task "$TASK" --run_name "$RUN_NAME" --mode llm_aggregate \
  2>&1 | tee "$LOG_DIR/test_eval/leakage_llm_summary.log"

echo
echo "=================================================================="
echo "[$(date '+%F %T')] === v2 RUN COMPLETE ==="
echo "  final LoRA: $CUR_LORA"
echo "=================================================================="
