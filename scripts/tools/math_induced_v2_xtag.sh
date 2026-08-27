#!/bin/bash
# Cross-tag replication scoring of the WAVE-2 induced candidates (induced-bank-v2, 12 rubrics,
# GLM-proposed practice metrics). Queues behind the CW/humor community-infill supervisor
# (waits for the SUPERVISOR to exit, not per-leg gaps, to avoid grabbing the GPU mid-batch).
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES=$GPU
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

while pgrep -f cw_humor_community_infill >/dev/null 2>&1; do sleep 300; done
while pgrep -f run_arm_comparison >/dev/null 2>&1; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 2000 ]; do
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] CW/humor batch done — v2 x-tag replication scoring START"
$PY scripts/tools/score_math_bank_by_tag.py \
  --bank-dir datasets/math/stackexchange/induced-bank-v2 \
  --judge-model "$M70" --executor-label llama-3.3-70b-fp8 \
  --per-tag 2400 \
  --out outputs/ctree/math_induced_v2_xtag
echo "[$(date '+%m-%d %H:%M')] INDUCED V2 XTAG COMPLETE rc=$?"
