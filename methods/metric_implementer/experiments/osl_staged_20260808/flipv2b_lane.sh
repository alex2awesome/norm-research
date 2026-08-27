#!/bin/bash
# 1c-v3b NULL-ALL lane: $1=gpu $2=selection-executor
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1; EX=$2
for i in $(seq 1 240); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$U" -lt 2000 ] && break
  sleep 20
done
echo "FLIPV2B GPU$GPU $EX START $(date)" >> $O/FLEET_STATUS
CUDA_VISIBLE_DEVICES=$GPU $PY $O/flip_functional_v2b.py $EX $O/flip_functional_v2_nullall_$EX.json \
  >> $O/logs/flipv2b_$EX.log 2>&1
echo "FLIPV2B GPU$GPU $EX END rc=$? $(date)" >> $O/FLEET_STATUS
