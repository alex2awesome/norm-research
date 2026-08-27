#!/bin/bash
# 3a Qwen3 family staircase (+2b de-censoring rung at 32B): battery -> humor285 -> 9-task mbar2
# per rung, mirroring run_newmodels.sh exactly (no-think: backend pins enable_thinking=False).
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; OSL=$B/outputs/osl; ST=$O/QWEN3_STATUS
GPU=$1
[ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | awk '$1>2000')" ] && { echo "GPU$GPU BUSY - abort $(date)" >> $ST; exit 1; }
TASKS="creative_writing press_releases math news_homepages peer_review notice_and_comment patents humor_sup code_review"
for EX in qwen3-8b qwen3-1.7b qwen3-4b qwen3-14b qwen3-32b; do
  EXTRA_ENV="HF_HUB_OFFLINE=1"
  if [ ! -s $OSL/$EX.json ]; then
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --battery-only \
      --executor $EX --battery $OSL/battery_humor_v1.json --out $OSL/$EX.json >> $O/logs/bat_$EX.log 2>&1
    echo "[$EX] battery rc=$? $(date)" >> $ST
  fi
  if [ ! -s $OSL/mbar285_$EX.npz ]; then
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --executor $EX --freeze $OSL/freeze_humor285_v2.json --out $OSL/mbar285_$EX.npz >> $O/logs/h285_$EX.log 2>&1
    echo "[$EX] humor285 rc=$? $(date)" >> $ST
  fi
  for TD in $TASKS; do
    OUT=$O/mbar2_${TD}_${EX}.npz
    [ -s $OUT ] && continue
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --executor $EX --freeze $O/freeze_${TD}_v2.json --out $OUT >> $O/logs/${TD}_${EX}.log 2>&1
    echo "[$EX] $TD rc=$? $(date)" >> $ST
  done
  echo "[$EX] ALL DONE $(date)" >> $ST
done
echo "QWEN3-STAIR-DONE $(date)" >> $ST
