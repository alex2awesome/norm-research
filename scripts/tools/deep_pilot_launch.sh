#!/bin/bash
# Deep-metric pilot legs (multi-step programs): general-topology + probability (hottest tails).
# Queues behind the v2 x-tag replication scoring (waits for its supervisor to exit + GPU free).
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES=$GPU
export ANTHROPIC_API_KEY=$(cat /lfs/skampere3/0/alexspan/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

while pgrep -f math_induced_v2_xtag >/dev/null 2>&1; do sleep 300; done
while pgrep -f cw_humor_community_infill >/dev/null 2>&1; do sleep 300; done
while pgrep -f "run_arm_comparison|score_math_bank_by_tag" >/dev/null 2>&1; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 2000 ]; do
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] queue clear — DEEP METRIC PILOT START"
for TAG in general-topology probability; do
  $PY scripts/tools/deep_metric_pilot.py --tag $TAG --judge-model "$M70" \
    >> outputs/ctree/deep_pilot_${TAG}.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] deep-pilot $TAG DONE rc=$?"
done
echo "[$(date '+%m-%d %H:%M')] DEEP PILOT COMPLETE"
