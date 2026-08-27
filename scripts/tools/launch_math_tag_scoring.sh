#!/bin/bash
# Math tag-stratified bank scoring (task #60): clean math bank x top-12 primary_tag
# communities, offline vLLM Llama-3.3-70B-FP8, SINGLE GPU. Same engine recipe as
# cw_cleanbank_expert.sh (executor-closed, post-audit judge semantics).
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-3}"
export CUDA_VISIBLE_DEVICES=$GPU
cd /lfs/skampere3/0/alexspan/norm-research

# refuse to start on an occupied GPU (verify, never assume)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$USED" -gt 2000 ]; then
  echo "GPU $GPU busy (${USED} MiB) — aborting"; exit 1
fi

PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

echo "[$(date '+%m-%d %H:%M')] math tag scoring START gpu=$GPU per_tag=${PER_TAG:-550}"
$PY scripts/tools/score_math_bank_by_tag.py \
  --judge-model "$M70" --executor-label llama-3.3-70b-fp8 \
  --per-tag "${PER_TAG:-550}" \
  ${TAGS:+--tags "$TAGS"} \
  --out "${OUT:-outputs/ctree/math_tag_bank}"
echo "[$(date '+%m-%d %H:%M')] math tag scoring DONE rc=$?"
