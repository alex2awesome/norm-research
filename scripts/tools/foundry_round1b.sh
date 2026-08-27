#!/bin/bash
# Foundry round 1b: redo articulation (with retries/resume after the z.ai 529 crash) +
# the solo-dense stage-2 legs. solo-vs-panel test already completed in round 1.
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-5}
export ANTHROPIC_API_KEY=$(cat /lfs/skampere3/0/alexspan/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
while pgrep -f "foundry_round1.sh|replicate_candidates|articulate_candidates" >/dev/null 2>&1; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU:-5}")" -lt 2000 ]; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] FOUNDRY 1b START"
# (b) articulation ladder (GLM, ~30 calls total)
$PY scripts/tools/articulate_candidates.py --legs 'outputs/ctree/arm_comparison/cw-genre-*' \
  --skip-pooled --domain-hint 'short fiction on WritingPrompts' \
  --out outputs/ctree/foundry/dense_cw.json >> outputs/ctree/foundry_r1.log 2>&1
$PY scripts/tools/articulate_candidates.py --legs 'outputs/ctree/arm_comparison/humor-topic-*' \
  --skip-pooled --domain-hint 'jokes on r/Jokes' \
  --out outputs/ctree/foundry/dense_humor.json >> outputs/ctree/foundry_r1.log 2>&1
$PY scripts/tools/articulate_candidates.py --legs 'outputs/ctree/arm_comparison/math-*-glmprop' \
  --skip-pooled --domain-hint 'answers on Math StackExchange' \
  --out outputs/ctree/foundry/dense_math.json >> outputs/ctree/foundry_r1.log 2>&1
echo "[$(date '+%m-%d %H:%M')] articulation DONE rc=$?"

# (c) stage-2 SOLO x dense rubrics (the corrected protocol)
$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/cw-genre-*' --skip-pooled \
  --data-template 'datasets/creative-writing/by_genre/{community}.csv.gz' \
  --leg-prefix cw-genre- --id-col prompt --solo \
  --dense-rubrics outputs/ctree/foundry/dense_cw.json \
  --bank-dir datasets/creative-writing/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/cw-genres-solo-dense >> outputs/ctree/foundry_r1.log 2>&1
echo "[$(date '+%m-%d %H:%M')] stage2 cw solo-dense DONE rc=$?"
$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/humor-topic-*' --skip-pooled \
  --data-template 'datasets/humor/by_topic/{community}.csv.gz' \
  --leg-prefix humor-topic- --id-col text --solo \
  --dense-rubrics outputs/ctree/foundry/dense_humor.json \
  --bank-dir datasets/humor/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/humor-topics-solo-dense >> outputs/ctree/foundry_r1.log 2>&1
echo "[$(date '+%m-%d %H:%M')] stage2 humor solo-dense DONE rc=$?"
$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/math-*-glmprop' --skip-pooled \
  --data-template 'datasets/math/stackexchange/by_tag/{community}.csv.gz' \
  --leg-prefix math- --leg-suffix=-glmprop --id-col question_id --solo \
  --dense-rubrics outputs/ctree/foundry/dense_math.json \
  --bank-dir datasets/math/stackexchange/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/math-glmprop-solo-dense >> outputs/ctree/foundry_r1.log 2>&1
echo "[$(date '+%m-%d %H:%M')] stage2 math solo-dense DONE rc=$? FOUNDRY 1b COMPLETE"
