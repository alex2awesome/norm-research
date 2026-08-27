#!/bin/bash
# Within-SUBTASK metric infilling (task #60, decisive leg): all 5 arms + certificate gate
# WITHIN each of the 6 scored math tags. Same recipe as cw_cleanbank_expert.sh (clean bank,
# executor-closed 70B-FP8 offline, content-only guard, reliability, confirm stage).
# Hypothesis sharpened by the 2026-07-07 powered read: reweighting the general bank is
# provably insufficient within-tag, so any kept proposal must be tag-local articulable signal.
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-3}
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU:-3}")
if [ "$USED" -gt 2000 ]; then echo "GPU ${GPU:-3} busy (${USED} MiB) — aborting"; exit 1; fi

for TAG in general-topology combinatorics geometry probability linear-algebra real-analysis; do
  TASK=math-$TAG
  if [ -f outputs/ctree/arm_comparison/$TASK/summary.json ]; then
    echo "[$(date '+%m-%d %H:%M')] $TASK exists — skip"; continue
  fi
  echo "[$(date '+%m-%d %H:%M')] $TASK START"
  $PY scripts/tools/run_arm_comparison.py --task $TASK \
    --rubrics-dir datasets/math/stackexchange/medoid-bank-clean --n 900 --max-rounds 15 \
    --min-auc-gain 0.005 --min-bits-gain 0.003 --acceptance-eval cv --confirm-repeats 5 \
    --content-only --measure-reliability --min-bank-auc-residual 0.55 \
    --judge-backend vllm_offline --judge-model "$M70" \
    --proposer-backend vllm_offline --proposer-model "$M70" \
    --executor-label llama-3.3-70b-fp8 \
    --out outputs/ctree/arm_comparison/$TASK \
    >> outputs/ctree/arm_${TASK}.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $TASK DONE rc=$?"
done
echo "[$(date '+%m-%d %H:%M')] MATH TAG INFILL COMPLETE"
