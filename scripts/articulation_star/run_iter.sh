#!/bin/bash
# Run one full articulation-STaR iteration with per-step subprocesses.
#
# Each step gets its own python invocation so vLLM tears down before the
# next step starts (avoids GPU OOM from coexisting engines).
#
# Generation runs as 2-way data-parallel across GPU_A and GPU_B.
# Judge and train each use one GPU.
#
# Env:
#   TASK            (default: peer_review)
#   RUN_NAME        (default: v0)
#   ITER            (required)
#   GPU_A           (default: 1)
#   GPU_B           (default: 5)
#   PREV_LORA       (optional, abs path; passed to generate + train)
#   TEACHER_MODEL   (optional, e.g. meta-llama/Llama-3.3-70B-Instruct for iter 0)
#   N_TRAIN         (default: 4000)
#   N_RATIONALES    (default: 4)
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# Required for Qwen3.5-122B FP8 MoE on B200 — see [[reference_qwen35_vllm_sk3]].
export VLLM_USE_FLASHINFER_MOE_FP8=0
# Known sk3 env quirk: flashinfer-cubin and flashinfer pip versions are skewed.
export FLASHINFER_DISABLE_VERSION_CHECK=1

TASK="${TASK:-peer_review}"
RUN_NAME="${RUN_NAME:-v0}"
ITER="${ITER:?must set ITER=<integer>}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-5}"
PREV_LORA="${PREV_LORA:-}"
TEACHER_MODEL="${TEACHER_MODEL:-}"
N_TRAIN="${N_TRAIN:-4000}"
N_RATIONALES="${N_RATIONALES:-4}"

LOG_DIR="logs/articulation_star/${RUN_NAME}"
mkdir -p "$LOG_DIR"

GEN_EXTRA=""
[[ -n "$PREV_LORA" ]]     && GEN_EXTRA+=" --lora_path $PREV_LORA"
[[ -n "$TEACHER_MODEL" ]] && GEN_EXTRA+=" --teacher_model $TEACHER_MODEL"

# ── 1. Generate (data-parallel across 2 GPUs) ─────────────────
echo "[$(date '+%F %T')] === iter ${ITER}: generate (DP across GPUs ${GPU_A},${GPU_B}) ==="
CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.generate_rationales \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  --shard_idx 0 --n_shards 2 \
  --n_train_subsample $N_TRAIN --n_rationales_per_input $N_RATIONALES \
  $GEN_EXTRA \
  > "$LOG_DIR/iter${ITER}_gen_a.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=$GPU_B python -m methods.articulation_star.generate_rationales \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  --shard_idx 1 --n_shards 2 \
  --n_train_subsample $N_TRAIN --n_rationales_per_input $N_RATIONALES \
  $GEN_EXTRA \
  > "$LOG_DIR/iter${ITER}_gen_b.log" 2>&1 &
PID_B=$!

# Wait for both shards. Fail fast if either dies.
if ! wait $PID_A; then echo "shard A failed"; exit 1; fi
if ! wait $PID_B; then echo "shard B failed"; exit 1; fi

python -m methods.articulation_star.merge_shards \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  2>&1 | tee "$LOG_DIR/iter${ITER}_merge.log"

# ── 2. Judge filter (one GPU; Qwen-122B is too big for two) ───
echo "[$(date '+%F %T')] === iter ${ITER}: judge (Qwen-122B on GPU $GPU_A) ==="
CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.judge_filter \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  2>&1 | tee "$LOG_DIR/iter${ITER}_judge.log"

# ── 3. SFT (one GPU; 8B + LoRA fits cleanly) ─────────────────
echo "[$(date '+%F %T')] === iter ${ITER}: train (8B LoRA on GPU $GPU_A) ==="
TRAIN_EXTRA=""
[[ -n "$PREV_LORA" ]] && TRAIN_EXTRA+=" --prev_lora $PREV_LORA"

CUDA_VISIBLE_DEVICES=$GPU_A python -m methods.articulation_star.train_sft \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  $TRAIN_EXTRA \
  2>&1 | tee "$LOG_DIR/iter${ITER}_train.log"

NEW_LORA="outputs/articulation_star/$TASK/$RUN_NAME/iter_$(printf %02d $ITER)/lora"
echo "[$(date '+%F %T')] === iter ${ITER} done; LoRA at: $NEW_LORA ==="
