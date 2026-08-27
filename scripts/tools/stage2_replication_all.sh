#!/bin/bash
# Stage-2 replication across all community legs (math wave-2, CW genres, humor topics):
# strictly fresh samples, Bonferroni over the replicated set only — the formal keeper stage.
# Queues behind the deep-metric pilot (end of the current GPU chain).
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES=$GPU
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

while pgrep -f "deep_pilot_launch|deep_metric_pilot|cw_humor_community_infill|math_induced_v2_xtag|run_arm_comparison|score_math_bank_by_tag" >/dev/null 2>&1; do
  sleep 300
done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 2000 ]; do
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] queue clear — STAGE-2 REPLICATION START"

$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/math-*-glmprop' --skip-pooled \
  --data-template 'datasets/math/stackexchange/by_tag/{community}.csv.gz' \
  --leg-prefix math- --leg-suffix=-glmprop --id-col question_id \
  --bank-dir datasets/math/stackexchange/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/math-glmprop >> outputs/ctree/stage2_math.log 2>&1
echo "[$(date '+%m-%d %H:%M')] stage2 math DONE rc=$?"

$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/cw-genre-*' --skip-pooled \
  --data-template 'datasets/creative-writing/by_genre/{community}.csv.gz' \
  --leg-prefix cw-genre- --id-col prompt \
  --bank-dir datasets/creative-writing/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/cw-genres >> outputs/ctree/stage2_cw.log 2>&1
echo "[$(date '+%m-%d %H:%M')] stage2 cw DONE rc=$?"

$PY scripts/tools/replicate_candidates.py \
  --legs 'outputs/ctree/arm_comparison/humor-topic-*' --skip-pooled \
  --data-template 'datasets/humor/by_topic/{community}.csv.gz' \
  --leg-prefix humor-topic- --id-col text \
  --bank-dir datasets/humor/medoid-bank-clean --judge-model "$M70" \
  --out outputs/ctree/stage2/humor-topics >> outputs/ctree/stage2_humor.log 2>&1
echo "[$(date '+%m-%d %H:%M')] stage2 humor DONE rc=$?  STAGE-2 ALL COMPLETE"
