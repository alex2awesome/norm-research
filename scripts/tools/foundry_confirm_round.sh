#!/bin/bash
# Confirmation round (sample-3): the solo-dense near-keeps (rep_p<0.05) re-tested on a THIRD
# disjoint sample per community, solo x dense, Bonferroni over the confirmation set.
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-5}
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
while pgrep -f "foundry_round1b|replicate_candidates|articulate_candidates" >/dev/null 2>&1; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU:-5}")" -lt 2000 ]; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] CONFIRM ROUND START"
$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/cw-genre-*' --skip-pooled \
  --data-template 'datasets/creative-writing/by_genre/{community}.csv.gz' \
  --leg-prefix cw-genre- --id-col prompt --solo \
  --dense-rubrics outputs/ctree/foundry/dense_cw.json \
  --only-from outputs/ctree/stage2/cw-genres-solo-dense/stage2_ledger.json \
  --salt rep3 --exclude-salts rep2 \
  --bank-dir datasets/creative-writing/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/cw-confirm3 >> outputs/ctree/foundry_confirm.log 2>&1
echo "[$(date '+%m-%d %H:%M')] confirm cw DONE rc=$?"
$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/humor-topic-*' --skip-pooled \
  --data-template 'datasets/humor/by_topic/{community}.csv.gz' \
  --leg-prefix humor-topic- --id-col text --solo \
  --dense-rubrics outputs/ctree/foundry/dense_humor.json \
  --only-from outputs/ctree/stage2/humor-topics-solo-dense/stage2_ledger.json \
  --salt rep3 --exclude-salts rep2 \
  --bank-dir datasets/humor/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/humor-confirm3 >> outputs/ctree/foundry_confirm.log 2>&1
echo "[$(date '+%m-%d %H:%M')] confirm humor DONE rc=$?"
$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/math-*-glmprop' --skip-pooled \
  --data-template 'datasets/math/stackexchange/by_tag/{community}.csv.gz' \
  --leg-prefix math- --leg-suffix=-glmprop --id-col question_id --solo \
  --dense-rubrics outputs/ctree/foundry/dense_math.json \
  --only-from outputs/ctree/stage2/math-glmprop-solo-dense/stage2_ledger.json \
  --salt rep3 --exclude-salts rep2 \
  --bank-dir datasets/math/stackexchange/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/math-confirm3 >> outputs/ctree/foundry_confirm.log 2>&1
echo "[$(date '+%m-%d %H:%M')] confirm math DONE rc=$? CONFIRM ROUND COMPLETE"
