#!/bin/bash
# Cross-tag validity matrix of the INDUCED candidate metrics (1 keep + top confirm-tail):
# score induced-bank-v1 (10 rubrics) on all 12 tags x 2,400 items. Transfer reads (11
# non-source tags) are clean out-of-sample replications; home-tag reads are optimistic
# (sample overlaps the proposing leg). Queues behind the pooled-control arm run.
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES=$GPU
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

while pgrep -f run_arm_comparison >/dev/null 2>&1; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 2000 ]; do
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] pooled control done, GPU $GPU free — induced x-tag scoring START"
$PY scripts/tools/score_math_bank_by_tag.py \
  --bank-dir datasets/math/stackexchange/induced-bank-v1 \
  --judge-model "$M70" --executor-label llama-3.3-70b-fp8 \
  --per-tag 2400 \
  --out outputs/ctree/math_induced_xtag
echo "[$(date '+%m-%d %H:%M')] INDUCED XTAG COMPLETE rc=$?"
