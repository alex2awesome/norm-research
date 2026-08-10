#!/bin/bash
# Pooled control for the within-tag infilling legs: same arms/recipe/item universe (12 tags
# pooled, unconditioned). Queues behind the running math_tag_infill supervisor, then takes
# the same GPU once it frees (teardown-lag tolerant: waits for <2 GB used).
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES=$GPU
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

while pgrep -f run_arm_comparison >/dev/null 2>&1; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 2000 ]; do
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] tag legs done, GPU $GPU free — pooled control START"
$PY scripts/tools/run_arm_comparison.py --task math-pooled-12tags \
  --rubrics-dir datasets/math/stackexchange/medoid-bank-clean --n 900 --max-rounds 15 \
  --min-auc-gain 0.005 --min-bits-gain 0.003 --acceptance-eval cv --confirm-repeats 5 \
  --content-only --measure-reliability --min-bank-auc-residual 0.55 \
  --judge-backend vllm_offline --judge-model "$M70" \
  --proposer-backend vllm_offline --proposer-model "$M70" \
  --executor-label llama-3.3-70b-fp8 \
  --out outputs/ctree/arm_comparison/math-pooled-12tags \
  >> outputs/ctree/arm_math-pooled-12tags.log 2>&1
echo "[$(date '+%m-%d %H:%M')] pooled control DONE rc=$? MATH POOLED COMPLETE"
