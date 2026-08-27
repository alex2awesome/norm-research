#!/bin/bash
# WAVE-3: the validated recipe (GLM proposer -> within-community -> dense articulation ->
# SOLO scoring -> disjoint replication) over 18 NEW subcommunities: 6 CW genres, 6 humor
# topics, 6 remaining math tags. Full chain: stage-1 legs -> articulation -> stage-2
# solo-dense (rep2) -> sample-3 confirm for p<.05 survivors.
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-5}
export ANTHROPIC_API_KEY=$(cat /lfs/skampere3/0/alexspan/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU:-5}")
if [ "$USED" -gt 2000 ]; then echo "GPU busy — abort"; exit 1; fi

leg () {  # $1 task $2 bank
  OUT=outputs/ctree/arm_comparison/$1
  [ -f $OUT/summary.json ] && { echo "[skip] $1"; return; }
  echo "[$(date '+%m-%d %H:%M')] $1 START"
  $PY scripts/tools/run_arm_comparison.py --task $1 --rubrics-dir $2 --n 900 --max-rounds 12 \
    --arms residual,metric_tree \
    --min-auc-gain 0.005 --min-bits-gain 0.003 --acceptance-eval cv --confirm-repeats 5 \
    --content-only --measure-reliability --min-bank-auc-residual 0.55 \
    --judge-backend vllm_offline --judge-model "$M70" \
    --proposer-backend anthropic --proposer-model glm-5.2 \
    --executor-label llama-3.3-70b-fp8 --out $OUT >> outputs/ctree/arm_$1.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $1 DONE rc=$?"
}

CWB=datasets/creative-writing/medoid-bank-clean
HUB=datasets/humor/medoid-bank-clean
MAB=datasets/math/stackexchange/medoid-bank-clean
for G in aliens villain soulmate ai time-travel meta-experimental; do leg cw-genre-$G $CWB; done
for T in political-classroom police chicken-crossing everyday-observational absurd-wordplay topical-corona; do leg humor-topic-$T $HUB; done
for M in calculus abstract-algebra algebra-precalculus sequences-and-series complex-analysis integration; do leg math-$M $MAB; done
echo "[$(date '+%m-%d %H:%M')] WAVE3 STAGE-1 ALL DONE"

$PY scripts/tools/articulate_candidates.py --legs 'outputs/ctree/arm_comparison/cw-genre-*' \
  --skip-pooled --domain-hint 'short fiction on WritingPrompts' \
  --out outputs/ctree/foundry/dense_cw.json >> outputs/ctree/wave3.log 2>&1
$PY scripts/tools/articulate_candidates.py --legs 'outputs/ctree/arm_comparison/humor-topic-*' \
  --skip-pooled --domain-hint 'jokes on r/Jokes' \
  --out outputs/ctree/foundry/dense_humor.json >> outputs/ctree/wave3.log 2>&1
$PY scripts/tools/articulate_candidates.py --legs 'outputs/ctree/arm_comparison/math-*' \
  --skip-pooled --domain-hint 'answers on Math StackExchange' \
  --out outputs/ctree/foundry/dense_math.json >> outputs/ctree/wave3.log 2>&1
echo "[$(date '+%m-%d %H:%M')] WAVE3 ARTICULATION DONE"

s2 () {  # $1 glob $2 template $3 prefix $4 idcol $5 bank $6 dense $7 out $8 extra
  $PY scripts/tools/replicate_candidates.py --legs "$1" --skip-pooled \
    --data-template "$2" --leg-prefix $3 --id-col $4 --solo --dense-rubrics $6 \
    --bank-dir $5 --judge-model "$M70" --out $7 $8 >> outputs/ctree/wave3.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] s2 $7 rc=$?"
}
W3CW='outputs/ctree/arm_comparison/cw-genre-{aliens,villain,soulmate,ai,time-travel,meta-experimental}'
s2 "outputs/ctree/arm_comparison/cw-genre-*" 'datasets/creative-writing/by_genre/{community}.csv.gz' cw-genre- prompt $CWB outputs/ctree/foundry/dense_cw.json outputs/ctree/stage2/w3-cw ""
s2 "outputs/ctree/arm_comparison/humor-topic-*" 'datasets/humor/by_topic/{community}.csv.gz' humor-topic- text $HUB outputs/ctree/foundry/dense_humor.json outputs/ctree/stage2/w3-humor ""
s2 "outputs/ctree/arm_comparison/math-*-glmprop" 'datasets/math/stackexchange/by_tag/{community}.csv.gz' math- question_id $MAB outputs/ctree/foundry/dense_math.json outputs/ctree/stage2/w3-math-old "--leg-suffix=-glmprop"
for M in calculus abstract-algebra algebra-precalculus sequences-and-series complex-analysis integration; do
  s2 "outputs/ctree/arm_comparison/math-$M" 'datasets/math/stackexchange/by_tag/{community}.csv.gz' math- question_id $MAB outputs/ctree/foundry/dense_math.json outputs/ctree/stage2/w3-math-$M ""
done
echo "[$(date '+%m-%d %H:%M')] WAVE3 STAGE-2 DONE"

s2 "outputs/ctree/arm_comparison/cw-genre-*" 'datasets/creative-writing/by_genre/{community}.csv.gz' cw-genre- prompt $CWB outputs/ctree/foundry/dense_cw.json outputs/ctree/stage2/w3-cw-confirm3 "--only-from outputs/ctree/stage2/w3-cw/stage2_ledger.json --salt rep3 --exclude-salts rep2"
s2 "outputs/ctree/arm_comparison/humor-topic-*" 'datasets/humor/by_topic/{community}.csv.gz' humor-topic- text $HUB outputs/ctree/foundry/dense_humor.json outputs/ctree/stage2/w3-humor-confirm3 "--only-from outputs/ctree/stage2/w3-humor/stage2_ledger.json --salt rep3 --exclude-salts rep2"
echo "[$(date '+%m-%d %H:%M')] WAVE3 COMPLETE"
