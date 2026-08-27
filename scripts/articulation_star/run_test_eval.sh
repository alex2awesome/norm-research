#!/bin/bash
# Held-out test eval for the articulation-STaR loop.
# For each LoRA stage (base, iter00, iter01, iter02): generate one rationale
# per held-out test artifact, score with strong judge, summarize.
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_MOE_FP8=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

TASK="${TASK:-creative_writing}"
RUN_NAME="${RUN_NAME:-v1_overnight_logprob}"
N_TEST="${N_TEST:-500}"
GPU="${GPU:-1}"

LOG_DIR="logs/articulation_star/${RUN_NAME}/test_eval"
mkdir -p "$LOG_DIR"

LORA_ROOT="$(pwd)/outputs/articulation_star/${TASK}/${RUN_NAME}"

# 1. Build the held-out split
python -m methods.articulation_star.test_eval \
  --task "$TASK" --run_name "$RUN_NAME" --mode build_split --n_test "$N_TEST" \
  2>&1 | tee "$LOG_DIR/build_split.log"

# 2. For each stage: generate then score
for STAGE in base iter00 iter01 iter02; do
  if [[ "$STAGE" == "base" ]]; then
    LORA_FLAG=""
  else
    ITER_TAG="${STAGE#iter}"
    LORA_FLAG="--lora_path ${LORA_ROOT}/iter_${ITER_TAG}/lora"
  fi
  echo "[$(date '+%F %T')] === stage ${STAGE}: generate (GPU $GPU) ==="
  CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.test_eval \
    --task "$TASK" --run_name "$RUN_NAME" --mode generate \
    --stage "$STAGE" $LORA_FLAG \
    > "$LOG_DIR/gen_${STAGE}.log" 2>&1

  echo "[$(date '+%F %T')] === stage ${STAGE}: score (GPU $GPU) ==="
  CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.test_eval \
    --task "$TASK" --run_name "$RUN_NAME" --mode score \
    --stage "$STAGE" \
    > "$LOG_DIR/score_${STAGE}.log" 2>&1
done

# 3. Summarize
python -m methods.articulation_star.test_eval \
  --task "$TASK" --run_name "$RUN_NAME" --mode summarize \
  2>&1 | tee "$LOG_DIR/summary.log"
