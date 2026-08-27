#!/bin/bash
# 1c exemplar-arm ladder lane: $1=gpu $2=space-separated executor list
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
export HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1; shift
mkdir -p $O/logs
[ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | awk '$1>2000')" ] && { echo "EX2ZXA GPU$GPU BUSY - abort $(date)" >> $O/FLEET_STATUS; exit 1; }
for EX in "$@"; do
  OUT=$O/mbar_zxaex2_humor_${EX}.npz
  [ -s $OUT ] && continue
  echo "EX2ZXA GPU$GPU START $EX $(date)" >> $O/FLEET_STATUS
  CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
    --n-forms 1 --executor $EX --freeze $O/freeze_zxa_ex2_humor_v1.json --out $OUT \
    >> $O/logs/zxaex2_humor_${EX}.log 2>&1
  echo "EX2ZXA GPU$GPU END $EX rc=$? $(date)" >> $O/FLEET_STATUS
done
echo "EX2ZXA GPU$GPU LANE-DONE $(date)" >> $O/FLEET_STATUS
