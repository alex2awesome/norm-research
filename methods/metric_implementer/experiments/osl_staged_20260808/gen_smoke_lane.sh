#!/bin/bash
# 3b smoke chain: $1=gpu; qwen3-8b (toggle) then r1-qwen-14b (native think)
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1
for i in $(seq 1 90); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$U" -lt 2000 ] && break
  sleep 20
done
for EX in qwen3-8b r1-qwen-14b; do
  echo "GENSMOKE GPU$GPU $EX START $(date)" >> $O/FLEET_STATUS
  timeout 3600 env CUDA_VISIBLE_DEVICES=$GPU $PY $O/gen_readout_smoke.py $EX $O/gen_smoke_$EX.json \
    >> $O/logs/gen_smoke_$EX.log 2>&1
  echo "GENSMOKE GPU$GPU $EX END rc=$? $(date)" >> $O/FLEET_STATUS
done
