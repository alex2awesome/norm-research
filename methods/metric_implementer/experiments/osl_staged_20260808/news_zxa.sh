#!/bin/bash
# Journalism z×a lane (news_homepages, 2026-07-09). Waits for the freeze (built after
# Sonnet authoring lands + validates), then runs executors with the CURATED probe pool.
# Usage: news_zxa.sh <gpu> <mem_util> <ex...>
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
export HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi
GPU=$1; UTIL=$2; shift 2
mkdir -p $O/logs
export OSL_PROBES_FILE=$O/news_probes.jsonl

for i in $(seq 1 48); do
  [ -s $O/freeze_zxa_news_homepages_v1.json ] && break
  sleep 300
done
[ -s $O/freeze_zxa_news_homepages_v1.json ] || { echo "GPU$GPU NEWS no freeze - exit $(date)" >> $O/FLEET_STATUS; exit 0; }

for EX in "$@"; do
  OUT=$O/mbar_zxa_news_homepages_${EX}.npz
  [ -s $OUT ] && continue
  for TRY in 1 2 3; do
    echo "GPU$GPU NEWS START $EX try$TRY $(date)" >> $O/FLEET_STATUS
    VLLM_GPU_MEM_UTIL=$UTIL CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep \
      --mbar-only --n-forms 1 --executor $EX --freeze $O/freeze_zxa_news_homepages_v1.json --out $OUT \
      >> $O/logs/zxa_news_${EX}.log 2>&1
    RC=$?
    echo "GPU$GPU NEWS END $EX rc=$RC $(date)" >> $O/FLEET_STATUS
    [ $RC -eq 0 ] && break
    sleep 600
  done
done
echo "GPU$GPU NEWS-LANE-DONE ($*) $(date)" >> $O/FLEET_STATUS
