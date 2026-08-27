#!/bin/bash
# 1c-v3 FULL-BANK lane: $1=gpu $2=selection-executor
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
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
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
[ "$U" -ge 2000 ] && { echo "FLIPV3 GPU$GPU $EX ABORT busy $(date)" >> $O/FLEET_STATUS; exit 1; }
echo "FLIPV3 GPU$GPU $EX START $(date)" >> $O/FLEET_STATUS
CUDA_VISIBLE_DEVICES=$GPU $PY $O/flip_functional_v3.py $EX $O/flip_functional_v3_$EX.json \
  >> $O/logs/flipv3_$EX.log 2>&1
echo "FLIPV3 GPU$GPU $EX END rc=$? $(date)" >> $O/FLEET_STATUS
