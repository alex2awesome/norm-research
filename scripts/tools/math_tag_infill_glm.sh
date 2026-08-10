#!/bin/bash
# WAVE-2 within-subtask infilling: GLM-5.2 PROPOSER (the walkthrough showed the 70B proposer
# mode-collapses onto tonal/coherence variants while GLM proposes tag-local practices on the
# same prompt). Judge/executor stays offline 70B (certificate E unchanged). Residual +
# metric_tree arms only (quota discipline: ~24 GLM calls/leg). De-steered proposer prompt
# (feature_gen._PROMPT now names domain practices alongside aesthetic/structural).
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-5}
export ANTHROPIC_API_KEY=$(cat /lfs/skampere3/0/alexspan/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU:-5}")
if [ "$USED" -gt 2000 ]; then echo "GPU ${GPU:-5} busy (${USED} MiB) — aborting"; exit 1; fi

for TASK in math-general-topology math-combinatorics math-geometry math-probability \
            math-linear-algebra math-real-analysis math-pooled-12tags; do
  OUT=outputs/ctree/arm_comparison/${TASK}-glmprop
  if [ -f $OUT/summary.json ]; then echo "[$(date '+%m-%d %H:%M')] $TASK-glmprop exists — skip"; continue; fi
  echo "[$(date '+%m-%d %H:%M')] $TASK (GLM proposer) START"
  $PY scripts/tools/run_arm_comparison.py --task $TASK \
    --rubrics-dir datasets/math/stackexchange/medoid-bank-clean --n 900 --max-rounds 12 \
    --arms residual,metric_tree \
    --min-auc-gain 0.005 --min-bits-gain 0.003 --acceptance-eval cv --confirm-repeats 5 \
    --content-only --measure-reliability --min-bank-auc-residual 0.55 \
    --judge-backend vllm_offline --judge-model "$M70" \
    --proposer-backend anthropic --proposer-model glm-5.2 \
    --executor-label llama-3.3-70b-fp8 \
    --out $OUT \
    >> outputs/ctree/arm_${TASK}_glmprop.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $TASK-glmprop DONE rc=$?"
done
echo "[$(date '+%m-%d %H:%M')] MATH GLM WAVE-2 COMPLETE"
