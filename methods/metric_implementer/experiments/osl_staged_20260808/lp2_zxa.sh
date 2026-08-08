#!/bin/bash
# LP lane, split after GPU6 was claimed by another workstream (engine-init 8.9GiB free).
# Usage: lp2_zxa.sh <gpu> <mem_util> <ex...>  — runs the given executors on the given GPU
# at the given VLLM_GPU_MEM_UTIL, retrying each executor up to 4 times with a 10-min wait
# (engine-init memory contention is transient state, not config — see
# reference_flashinfer_workspace_oom_is_contention).
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
export HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi
GPU=$1; UTIL=$2; shift 2
mkdir -p $O/logs
export OSL_PROBES_FILE=$O/humor_long_probes.jsonl

for EX in "$@"; do
  OUT=$O/mbar_zxaLP_humor_${EX}.npz
  [ -s $OUT ] && continue
  for TRY in 1 2 3 4; do
    echo "GPU$GPU LP2 START humor $EX try$TRY util$UTIL $(date)" >> $O/FLEET_STATUS
    VLLM_GPU_MEM_UTIL=$UTIL CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep \
      --mbar-only --n-forms 1 --executor $EX --freeze $O/freeze_zxa_humor_v1.json --out $OUT \
      >> $O/logs/zxaLP_humor_${EX}.log 2>&1
    RC=$?
    echo "GPU$GPU LP2 END humor $EX rc=$RC $(date)" >> $O/FLEET_STATUS
    [ $RC -eq 0 ] && break
    sleep 600
  done
done
echo "GPU$GPU LP2-DONE ($*) $(date)" >> $O/FLEET_STATUS
