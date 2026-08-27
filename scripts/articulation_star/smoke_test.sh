#!/bin/bash
# Tiny end-to-end smoke test for the articulation-STaR loop.
#
# Runs 1 iter on 20 artifacts × 2 rationales each, on a single GPU.
# Goal: catch wiring bugs before scaling, not produce a useful artifact.
#
# Usage from repo root on sk3:
#   bash scripts/articulation_star/smoke_test.sh
set -euo pipefail

# Per [[feedback_sk3_afs_tokens]]: pin HOME to /lfs so nohup keeps tokens.
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# Required for Qwen3.5-122B FP8 MoE on B200 — see [[reference_qwen35_vllm_sk3]].
export VLLM_USE_FLASHINFER_MOE_FP8=0
# Known sk3 env quirk: flashinfer-cubin and flashinfer pip versions are skewed.
export FLASHINFER_DISABLE_VERSION_CHECK=1

TASK="${TASK:-peer_review}"
RUN_NAME="${RUN_NAME:-smoke}"
ITER="${ITER:-0}"
GPU="${GPU:-1}"

LOG_DIR="logs/articulation_star/${RUN_NAME}"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] === smoke: generate (8B, GPU $GPU) ==="
CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.generate_rationales \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  --n_train_subsample 20 --n_rationales_per_input 2 \
  2>&1 | tee "$LOG_DIR/iter${ITER}_generate.log"

echo "[$(date '+%F %T')] === smoke: judge (Qwen-122B, GPU $GPU) ==="
CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.judge_filter \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  2>&1 | tee "$LOG_DIR/iter${ITER}_judge.log"

echo "[$(date '+%F %T')] === smoke: train (8B LoRA, GPU $GPU) ==="
CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.train_sft \
  --task $TASK --run_name $RUN_NAME --iter $ITER \
  2>&1 | tee "$LOG_DIR/iter${ITER}_train.log"

echo "[$(date '+%F %T')] === smoke complete; inspect: ==="
echo "  outputs/articulation_star/$TASK/$RUN_NAME/iter_$(printf %02d $ITER)/"
