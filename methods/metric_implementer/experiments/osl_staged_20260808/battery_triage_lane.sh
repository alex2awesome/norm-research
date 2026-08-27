#!/bin/bash
# Reasoning-slate battery triage: z + instrument-compatibility per model (no-think logprob
# readout; degenerate/nan results ARE findings — they map where 3b's generative readout is
# mandatory). Resumable; skips existing FRESH files only (era guard: require per_family key).
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; OSL=$B/outputs/osl; ST=$O/TRIAGE_STATUS
GPU=$1
[ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | awk '$1>2000')" ] && { echo "GPU$GPU BUSY $(date)" >> $ST; exit 1; }
for EX in gpt-oss-120b r1-qwen-32b gpt-oss-20b r1-qwen-14b phi4-reasoning magistral-24b seed-oss-36b glm-z1-32b r1-llama-8b r1-qwen-7b r1-qwen-1.5b; do
  FRESH=$($PY -c "
import json,os
p='$OSL/$EX.json'
print(1 if os.path.exists(p) and json.load(open(p)).get('battery',{}).get('per_family') else 0)" 2>/dev/null)
  [ "$FRESH" = "1" ] && continue
  echo "[$EX] battery start $(date)" >> $ST
  timeout 2400 env CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --battery-only \
    --executor $EX --battery $OSL/battery_humor_v1.json --out $OSL/$EX.json >> $O/logs/bat_triage_$EX.log 2>&1
  echo "[$EX] battery rc=$? $(date)" >> $ST
done
echo "TRIAGE-DONE $(date)" >> $ST
