#!/usr/bin/env bash
# Chain dedup-first canonical selection (GPU 5) then Steps 4+5 (GPU 7, Llama-70B).
# Run from /lfs/skampere3/0/alexspan/norm-research

set -euo pipefail

OUT_DIR=outputs/local_explanations/peer_review_star_local_llama33_v2
DEDUP_LOG=$OUT_DIR/dedup_first.log
STEP45_LOG=$OUT_DIR/steps4_5.log

export HOME=/lfs/skampere3/0/alexspan
export OPENAI_API_KEY=$(cat /lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt)
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HF_MODULES_CACHE=/lfs/skampere3/0/alexspan/hf_modules
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "== Phase 1: dedup-first canonical selection on GPU 5 ==" > "$DEDUP_LOG"
date >> "$DEDUP_LOG"
CUDA_VISIBLE_DEVICES=5 /lfs/skampere3/0/alexspan/miniconda3/bin/python -u scripts/select_canonical_via_dedup.py \
  --dataset peer-review \
  --output-dir "$OUT_DIR" \
  --proposer-model openai/gpt-5-mini \
  --top-n-candidates 200 \
  --top-k-families 30 \
  --hdbscan-min-cluster-size 100 \
  >> "$DEDUP_LOG" 2>&1

echo "" >> "$DEDUP_LOG"
echo "== Phase 1 complete at $(date) ==" >> "$DEDUP_LOG"

echo "== Phase 2: Steps 4+5 with Llama-3.3-70B-FP8 on GPU 7 ==" > "$STEP45_LOG"
date >> "$STEP45_LOG"
CUDA_VISIBLE_DEVICES=7 /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python -u scripts/run_local_explanations.py \
  --dataset peer-review \
  --approach star_local \
  --model /lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362 \
  --proposer-model openai/gpt-5-mini \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --n-canonical-features 30 \
  --max-text-tokens 512 \
  --output-dir "$OUT_DIR" \
  >> "$STEP45_LOG" 2>&1

echo "== Phase 2 complete at $(date) ==" >> "$STEP45_LOG"
