#!/bin/bash
# z×a articulation-capability exchange sweep, small ladder (spec:
# notes/2026-07-08__zxa-articulation-capability-exchange-spec.md). Arms are freeze rubric
# strings; --n-forms 1 so arm text is shown VERBATIM (orbit reformulation would rewrite it).
# Task-outer loop: full humor ladder lands first for the earliest beta fit. Resumable.
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1
export HF_HUB_OFFLINE=1  # NAT64 HEAD-revalidation flake 2026-07-09; cache complete
# wait up to 3h for the freeze files (uploaded from laptop after authoring validation)
for i in $(seq 1 36); do
  [ -s $O/freeze_zxa_humor_v1.json ] && break
  echo "GPU$GPU ZXA waiting for freezes ($i/36) $(date)" >> $O/FLEET_STATUS
  sleep 300
done
[ -s $O/freeze_zxa_humor_v1.json ] || { echo "GPU$GPU ZXA NO FREEZES - SKIPPED $(date)" >> $O/FLEET_STATUS; exit 0; }
for TD in humor creative_writing peer_review math; do
  for EX in llama1b llama3b llama8b qwen25-3b qwen25-7b qwen25-14b gemma2-27b gemma2-9b; do
    EXTRA_ENV=""; [ "$EX" = "gemma2-9b" ] && EXTRA_ENV="VLLM_BLOCK_SIZE=32"
    if [ "$EX" = "gemma2-9b" ] && [ ! -d /lfs/skampere3/0/shared_hf_cache/models--google--gemma-2-9b-it ]; then
      continue
    fi
    OUT=$O/mbar_zxa_${TD}_${EX}.npz
    [ -s $OUT ] && continue
    echo "GPU$GPU ZXA START $TD $EX $(date)" >> $O/FLEET_STATUS
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --n-forms 1 --executor $EX --freeze $O/freeze_zxa_${TD}_v1.json --out $OUT \
      >> $O/logs/zxa_${TD}_${EX}.log 2>&1
    echo "GPU$GPU ZXA END $TD $EX rc=$? $(date)" >> $O/FLEET_STATUS
  done
done
echo "GPU$GPU ZXA-SMALL-DONE $(date)" >> $O/FLEET_STATUS
