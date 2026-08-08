#!/bin/bash
# qwen25-32b priority z×a (2026-07-09): THE falsification rung for the first forecast row
# (Self-deprecation z*=2.22 ~ 34B qwen-equiv). GPU2 (idle, small residents stay). Battery
# first (x-axis), then 4-task z×a + mechbat. Resumable.
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
export HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; OSL=$B/outputs/osl; GPU=$1; EX=qwen25-32b
if [ ! -s $OSL/$EX.json ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --battery-only \
    --executor $EX --battery $OSL/battery_humor_v1.json --out $OSL/$EX.json >> $O/logs/bat_$EX.log 2>&1
  echo "GPU$GPU QWEN32 battery rc=$? $(date)" >> $O/FLEET_STATUS
fi
for TD in humor creative_writing peer_review math mechbat_humor mechbat_peer_review; do
  OUT=$O/mbar_zxa_${TD}_${EX}.npz
  [ -s $OUT ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
    --n-forms 1 --executor $EX --freeze $O/freeze_zxa_${TD}_v1.json --out $OUT >> $O/logs/zxa_${TD}_${EX}.log 2>&1
  echo "GPU$GPU QWEN32 zxa $TD rc=$? $(date)" >> $O/FLEET_STATUS
done
$PY $O/zxa_fit.py > $O/zxa_fit_after_qwen32.log 2>&1
echo "GPU$GPU QWEN32-ZXA-DONE (fit rc=$?) $(date)" >> $O/FLEET_STATUS
