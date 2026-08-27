#!/bin/bash
# GPU3 queue: functional-rubric ladder (12 execs) then full-bank flip-v3
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=${1:-3}
while pgrep -f "executor magistral-24b|magfix_test" >/dev/null; do sleep 120; done
for i in $(seq 1 240); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$U" -lt 2000 ] && break
  sleep 30
done
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
[ "$U" -ge 2000 ] && { echo "FLIPQ GPU$GPU ABORT busy $(date)" >> $O/FLEET_STATUS; exit 1; }
for EX in llama1b llama3b qwen25-3b qwen25-7b mistral7b llama8b phi4 qwen25-14b gemma2-27b qwen25-32b llama70b qwen25-72b; do
  echo "FLIPQ GPU$GPU ladder-$EX START $(date)" >> $O/FLEET_STATUS
  timeout 7200 env CUDA_VISIBLE_DEVICES=$GPU $PY $O/flip_ladder.py $EX >> $O/logs/flipladder_$EX.log 2>&1
  echo "FLIPQ GPU$GPU ladder-$EX END rc=$? $(date)" >> $O/FLEET_STATUS
done
echo "FLIPQ GPU$GPU v3 START $(date)" >> $O/FLEET_STATUS
env CUDA_VISIBLE_DEVICES=$GPU $PY $O/flip_functional_v3.py llama70b $O/flip_functional_v3_llama70b.json >> $O/logs/flipv3_llama70b.log 2>&1
echo "FLIPQ GPU$GPU v3 END rc=$? $(date)" >> $O/FLEET_STATUS
