#!/bin/bash
# GEPA re-pass behind wave-3: optimize every stage-2 hot candidate's rubric against label-free
# diagnostics (retest + MI-recovery), then rep4 confirm with the optimized rubrics.
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-5}
export ANTHROPIC_API_KEY=$(cat /lfs/skampere3/0/alexspan/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
while pgrep -f "wave3_communities|run_arm_comparison|replicate_candidates|articulate_candidates" >/dev/null 2>&1; do sleep 600; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU:-5}")" -lt 2000 ]; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] GEPA REPASS START"

$PY scripts/tools/gepa_optimize_rubrics.py --stage2 outputs/ctree/stage2/w3-cw/stage2_ledger.json \
  --dense outputs/ctree/foundry/dense_cw.json \
  --data-template 'datasets/creative-writing/by_genre/{community}.csv.gz' \
  --leg-prefix cw-genre- --id-col prompt --judge-model "$M70" \
  --out outputs/ctree/foundry/dense_cw_gepa.json >> outputs/ctree/gepa_repass.log 2>&1
$PY scripts/tools/gepa_optimize_rubrics.py --stage2 outputs/ctree/stage2/w3-humor/stage2_ledger.json \
  --dense outputs/ctree/foundry/dense_humor.json \
  --data-template 'datasets/humor/by_topic/{community}.csv.gz' \
  --leg-prefix humor-topic- --id-col text --judge-model "$M70" \
  --out outputs/ctree/foundry/dense_humor_gepa.json >> outputs/ctree/gepa_repass.log 2>&1
echo "[$(date '+%m-%d %H:%M')] GEPA optimize DONE rc=$?"

$PY scripts/tools/replicate_candidates.py --legs 'outputs/ctree/arm_comparison/cw-genre-*' --skip-pooled \
  --data-template 'datasets/creative-writing/by_genre/{community}.csv.gz' \
  --leg-prefix cw-genre- --id-col prompt --solo \
  --dense-rubrics outputs/ctree/foundry/dense_cw_gepa.json \
  --only-from outputs/ctree/stage2/w3-cw/stage2_ledger.json \
  --salt rep4 --exclude-salts rep2,rep3 \
  --bank-dir datasets/creative-writing/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/w3-cw-gepa4 >> outputs/ctree/gepa_repass.log 2>&1
$PY scripts/tools/replicate_candidates.py --legs 'outputs/ctree/arm_comparison/humor-topic-*' --skip-pooled \
  --data-template 'datasets/humor/by_topic/{community}.csv.gz' \
  --leg-prefix humor-topic- --id-col text --solo \
  --dense-rubrics outputs/ctree/foundry/dense_humor_gepa.json \
  --only-from outputs/ctree/stage2/w3-humor/stage2_ledger.json \
  --salt rep4 --exclude-salts rep2,rep3 \
  --bank-dir datasets/humor/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/w3-humor-gepa4 >> outputs/ctree/gepa_repass.log 2>&1
echo "[$(date '+%m-%d %H:%M')] GEPA REPASS COMPLETE rc=$?"
