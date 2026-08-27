#!/bin/bash
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1
while read -r TD EX; do
  OUT=$O/mbar2_${TD}_${EX}.npz
  [ -s $OUT ] && continue
  echo "GPU$GPU START $TD $EX $(date)" >> $O/FLEET_STATUS
  CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
    --executor $EX --freeze $O/freeze_${TD}_v2.json --out $OUT >> $O/logs/${TD}_${EX}.log 2>&1
  echo "GPU$GPU END $TD $EX rc=$? $(date)" >> $O/FLEET_STATUS
done < $O/jobs2_cr
echo "GPU$GPU CR-RETRY-DONE $(date)" >> $O/FLEET_STATUS
