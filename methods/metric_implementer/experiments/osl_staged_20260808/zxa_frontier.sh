#!/bin/bash
# z×a frontier-local arms (morning harvest 2026-07-09): llama70b + qwen25-72b on the 4-task
# z×a freezes + mechbat, then auto-refit, then resume the llama70b v2 panels whose driver
# died 23:11 Jul 8. Resumable ([ -s OUT ] skips). One GPU.
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1

for EX in llama70b qwen25-72b; do
  for TD in humor creative_writing peer_review math mechbat_humor mechbat_peer_review; do
    OUT=$O/mbar_zxa_${TD}_${EX}.npz
    [ -s $OUT ] && continue
    echo "GPU$GPU ZXA-FRONTIER START $TD $EX $(date)" >> $O/FLEET_STATUS
    CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --n-forms 1 --executor $EX --freeze $O/freeze_zxa_${TD}_v1.json --out $OUT \
      >> $O/logs/zxa_${TD}_${EX}.log 2>&1
    echo "GPU$GPU ZXA-FRONTIER END $TD $EX rc=$? $(date)" >> $O/FLEET_STATUS
  done
  # refit as soon as each frontier-local executor lands (metric-B cells + multi-frontier y_ref)
  $PY $O/zxa_fit.py > $O/zxa_fit_after_${EX}.log 2>&1
  echo "GPU$GPU ZXA-FIT after $EX rc=$? $(date)" >> $O/FLEET_STATUS
done

# resume llama70b 9-task v2 panels (remaining; driver resume_fleet_after_405.sh died Jul 8 23:11)
for TD in math news_homepages peer_review notice_and_comment patents humor_sup code_review; do
  OUT=$O/mbar2_${TD}_llama70b.npz
  [ -s $OUT ] && continue
  echo "GPU$GPU 70BPANEL START $TD $(date)" >> $O/FLEET_STATUS
  CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
    --executor llama70b --freeze $O/freeze_${TD}_v2.json --out $OUT \
    >> $O/logs/${TD}_llama70b.log 2>&1
  echo "GPU$GPU 70BPANEL END $TD rc=$? $(date)" >> $O/FLEET_STATUS
done
echo "GPU$GPU ZXA-FRONTIER-CHAIN-DONE $(date)" >> $O/FLEET_STATUS
