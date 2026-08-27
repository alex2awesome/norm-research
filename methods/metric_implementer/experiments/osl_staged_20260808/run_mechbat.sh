#!/bin/bash
# MECHBAT local ladder: ~19 code-verifiable rules x 300 probes per task — what can an LLM do
# that a regex can? Truth lives inside the freeze (mech.truth). Tiny: ~1-3 min per exec-task.
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1
export HF_HUB_OFFLINE=1  # NAT64 HEAD-revalidation flake 2026-07-09; cache complete
for TD in mechbat_humor mechbat_peer_review; do
  for EX in llama1b llama3b llama8b qwen25-3b qwen25-7b qwen25-14b gemma2-27b gemma2-9b mistral-24b qwen25-32b; do
    EXTRA_ENV=""; [ "$EX" = "gemma2-9b" ] && EXTRA_ENV="VLLM_BLOCK_SIZE=32"
    case $EX in gemma2-9b) D=models--google--gemma-2-9b-it;; mistral-24b) D=models--mistralai--Mistral-Small-24B-Instruct-2501;; qwen25-32b) D=models--Qwen--Qwen2.5-32B-Instruct;; *) D="";; esac
    if [ -n "$D" ] && [ ! -d /lfs/skampere3/0/shared_hf_cache/$D ]; then continue; fi
    OUT=$O/mbar_zxa_${TD}_${EX}.npz
    [ -s $OUT ] && continue
    echo "GPU$GPU MECHBAT START $TD $EX $(date)" >> $O/FLEET_STATUS
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --n-forms 1 --executor $EX --freeze $O/freeze_zxa_${TD}_v1.json --out $OUT \
      >> $O/logs/${TD}_${EX}.log 2>&1
    echo "GPU$GPU MECHBAT END $TD $EX rc=$? $(date)" >> $O/FLEET_STATUS
  done
done
echo "GPU$GPU MECHBAT-DONE $(date)" >> $O/FLEET_STATUS
