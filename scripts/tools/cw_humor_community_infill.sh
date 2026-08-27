#!/bin/bash
# Cross-domain replication of the within-subtask infilling result (#60): CW prompt-genres and
# humor topics, each with its own pooled control over the same item universe. GLM-5.2 proposer
# (walkthrough-validated) + offline 70B judge/executor. Queues behind the math GLM wave-2.
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES=$GPU
export ANTHROPIC_API_KEY=$(cat /lfs/skampere3/0/alexspan/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
M70=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362

while pgrep -f run_arm_comparison >/dev/null 2>&1; do sleep 180; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 2000 ]; do
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] wave-2 done, GPU $GPU free — CW/humor community infill START"

run_leg () {  # $1=task  $2=rubrics-dir
  OUT=outputs/ctree/arm_comparison/$1
  if [ -f $OUT/summary.json ]; then echo "[$(date '+%m-%d %H:%M')] $1 exists — skip"; return; fi
  echo "[$(date '+%m-%d %H:%M')] $1 START"
  $PY scripts/tools/run_arm_comparison.py --task $1 \
    --rubrics-dir $2 --n 900 --max-rounds 12 \
    --arms residual,metric_tree \
    --min-auc-gain 0.005 --min-bits-gain 0.003 --acceptance-eval cv --confirm-repeats 5 \
    --content-only --measure-reliability --min-bank-auc-residual 0.55 \
    --judge-backend vllm_offline --judge-model "$M70" \
    --proposer-backend anthropic --proposer-model glm-5.2 \
    --executor-label llama-3.3-70b-fp8 \
    --out $OUT >> outputs/ctree/arm_$1.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $1 DONE rc=$?"
}

CWB=datasets/creative-writing/medoid-bank-clean
HUB=datasets/humor/medoid-bank-clean
for G in abstract-premise immortality wakeup-mystery hell-deal pooled-4genres; do
  run_leg cw-genre-$G $CWB
done
for T in marriage bar-jokes family doctor pooled-4topics; do
  run_leg humor-topic-$T $HUB
done
echo "[$(date '+%m-%d %H:%M')] CW-HUMOR COMMUNITY INFILL COMPLETE"
