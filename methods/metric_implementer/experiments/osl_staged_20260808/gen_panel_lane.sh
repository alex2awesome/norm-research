#!/bin/bash
# 3b think/no-think panel lane: $1=gpu, $2..=executor chain
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1; TASK=$2; shift 2
for i in $(seq 1 90); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$U" -lt 2000 ] && break
  sleep 20
done
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
[ "$U" -ge 2000 ] && { echo "GENPANEL GPU$GPU ABORT busy $(date)" >> $O/FLEET_STATUS; exit 1; }
for EX in "$@"; do
  echo "GENPANEL GPU$GPU $TASK-$EX START $(date)" >> $O/FLEET_STATUS
  timeout 14400 env CUDA_VISIBLE_DEVICES=$GPU $PY $O/gen_think_panel.py $EX $TASK \
    >> $O/logs/gen_panel_${TASK}_$EX.log 2>&1
  echo "GENPANEL GPU$GPU $TASK-$EX END rc=$? $(date)" >> $O/FLEET_STATUS
done
