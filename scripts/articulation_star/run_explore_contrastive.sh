#!/bin/bash
# Exploratory contrastive (weak/strong judge) run for the articulation-STaR
# pipeline. NOT a training iter -- this writes rationales + per-rationale
# weak/strong predictions and a kept set, for manual audit BEFORE deciding
# whether to launch a real training iter.
#
# Stages (each is its own python subprocess so vLLM tears down between):
#   1. Generate    -- Llama-8B, data-parallel across GPU_A + GPU_B.
#   2. Predict     -- weak judge on GPU_A, strong judge on GPU_B, in parallel.
#   3. Combine     -- CPU-only merge; contrastive keep rule applied.
#
# Env:
#   TASK             (default: creative_writing)
#   RUN_NAME         (default: explore_contrastive_cw)
#   ITER             (default: 0)
#   GPU_A            (default: 1) -- generation shard A + weak judge
#   GPU_B            (default: 5) -- generation shard B + strong judge
#   N_TRAIN          (default: 500)
#   N_RATIONALES     (default: 4)
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_MOE_FP8=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

TASK="${TASK:-creative_writing}"
RUN_NAME="${RUN_NAME:-explore_contrastive_cw}"
ITER="${ITER:-0}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-5}"
N_TRAIN="${N_TRAIN:-500}"
N_RATIONALES="${N_RATIONALES:-4}"

LOG_DIR="logs/articulation_star/${RUN_NAME}"
mkdir -p "$LOG_DIR"

# ── 1. Generate (DP across 2 GPUs) ─────────────────────────────
echo "[$(date '+%F %T')] === iter ${ITER}: generate (DP across ${GPU_A},${GPU_B}) ==="
CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.generate_rationales \
  --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
  --shard_idx 0 --n_shards 2 \
  --n_train_subsample "$N_TRAIN" --n_rationales_per_input "$N_RATIONALES" \
  > "$LOG_DIR/iter${ITER}_gen_a.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=$GPU_B python -m methods.articulation_star.generate_rationales \
  --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
  --shard_idx 1 --n_shards 2 \
  --n_train_subsample "$N_TRAIN" --n_rationales_per_input "$N_RATIONALES" \
  > "$LOG_DIR/iter${ITER}_gen_b.log" 2>&1 &
PID_B=$!

if ! wait $PID_A; then echo "gen shard A failed"; exit 1; fi
if ! wait $PID_B; then echo "gen shard B failed"; exit 1; fi

python -m methods.articulation_star.merge_shards \
  --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
  2>&1 | tee "$LOG_DIR/iter${ITER}_merge.log"

# ── 2. Predict weak + strong in parallel on 2 GPUs ─────────────
echo "[$(date '+%F %T')] === iter ${ITER}: predict weak (GPU $GPU_A) + strong (GPU $GPU_B) ==="
CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.judge_filter \
  --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
  --mode predict --judge weak \
  > "$LOG_DIR/iter${ITER}_predict_weak.log" 2>&1 &
PID_W=$!

CUDA_VISIBLE_DEVICES=$GPU_B python -m methods.articulation_star.judge_filter \
  --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" \
  --mode predict --judge strong \
  > "$LOG_DIR/iter${ITER}_predict_strong.log" 2>&1 &
PID_S=$!

if ! wait $PID_W; then echo "weak judge failed"; exit 1; fi
if ! wait $PID_S; then echo "strong judge failed"; exit 1; fi

# ── 3. Combine (CPU-only) ───────────────────────────────────────
echo "[$(date '+%F %T')] === iter ${ITER}: combine ==="
python -m methods.articulation_star.judge_filter \
  --task "$TASK" --run_name "$RUN_NAME" --iter "$ITER" --mode combine \
  2>&1 | tee "$LOG_DIR/iter${ITER}_combine.log"

OUT_DIR="outputs/articulation_star/$TASK/$RUN_NAME/iter_$(printf %02d "$ITER")"
echo "[$(date '+%F %T')] === done; inspect: $OUT_DIR ==="
