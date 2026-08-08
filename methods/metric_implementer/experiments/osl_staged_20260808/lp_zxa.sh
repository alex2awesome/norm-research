#!/bin/bash
# Long-form humor probe lane (tacit-support fix, 2026-07-09). Same freeze_zxa_humor_v1
# arms, NEW probe support: humor_long_probes.jsonl (800-4000 chars, topic-stratified,
# med 1018 chars vs v1's 81). Tests whether cluster-A tacit candidates (host presence,
# voice, delivery: all-NO on one-liners) develop variance + frontier agreement on long
# support. Frontier execs first (kappa question), then mid rungs (curve question).
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
export HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1
mkdir -p $O/logs
export OSL_PROBES_FILE=$O/humor_long_probes.jsonl
[ -s $OSL_PROBES_FILE ] || { echo "GPU$GPU LP no probes file $(date)" >> $O/FLEET_STATUS; exit 1; }

for EX in llama70b qwen25-72b qwen25-32b llama8b qwen25-14b qwen25-7b llama3b qwen25-3b llama1b; do
  OUT=$O/mbar_zxaLP_humor_${EX}.npz
  [ -s $OUT ] && continue
  echo "GPU$GPU LP START humor $EX $(date)" >> $O/FLEET_STATUS
  CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
    --n-forms 1 --executor $EX --freeze $O/freeze_zxa_humor_v1.json --out $OUT \
    >> $O/logs/zxaLP_humor_${EX}.log 2>&1
  echo "GPU$GPU LP END humor $EX rc=$? $(date)" >> $O/FLEET_STATUS
done
echo "GPU$GPU LP-LANE-DONE $(date)" >> $O/FLEET_STATUS
