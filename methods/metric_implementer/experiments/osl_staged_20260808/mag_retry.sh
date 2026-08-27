#!/bin/bash
# solo magistral retry: waits for triage chain to drain, then 90-min-timeout run on GPU7
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; OSL=$B/outputs/osl
while pgrep -f "battery_triage_lane.sh" >/dev/null; do sleep 60; done
for i in $(seq 1 90); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)
  [ "$U" -lt 2000 ] && break
  sleep 20
done
echo "MAGRETRY GPU7 START $(date)" >> $O/FLEET_STATUS
timeout 5400 env CUDA_VISIBLE_DEVICES=7 $PY -m methods.metric_implementer.experiments.osl_sweep --battery-only \
  --executor magistral-24b --battery $OSL/battery_humor_v1.json --out $OSL/magistral-24b.json \
  >> $O/logs/bat_triage_magistral-24b.log 2>&1
echo "MAGRETRY GPU7 END rc=$? $(date)" >> $O/FLEET_STATUS
