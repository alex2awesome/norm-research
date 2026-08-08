#!/bin/bash
# Step-4 surgical model additions (user 2026-07-08): qwen25-32b (fills qwen 14->72 gap),
# gemma2-9b + mistral-24b (second rungs for singleton families). Per executor:
# ensure weights -> battery (z for the x-axis) -> humor285 flagship mbar -> 9-task mbar2
# (incl. code_review curves-only freeze). Resumable; runs on one GPU.
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; OSL=$B/outputs/osl; ST=$O/NEWMODEL_STATUS
GPU=$1
bash $O/run_zxa_small.sh $GPU   # z-x-a main-objective insertion 2026-07-08 (runs before new-model panels)
declare -A REPO=( [gemma2-9b]="google/gemma-2-9b-it" [mistral-24b]="mistralai/Mistral-Small-24B-Instruct-2501" [qwen25-32b]="Qwen/Qwen2.5-32B-Instruct" )
TASKS="creative_writing press_releases math news_homepages peer_review notice_and_comment patents humor_sup code_review"

for EX in gemma2-9b mistral-24b qwen25-32b; do
  EXTRA_ENV="HF_HUB_OFFLINE=1"; [ "$EX" = "gemma2-9b" ] && EXTRA_ENV="VLLM_BLOCK_SIZE=32 HF_HUB_OFFLINE=1"   # FlashInfer head-256
  echo "[$EX] ensure weights $(date)" >> $ST
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('${REPO[$EX]}', max_workers=8)" >> $O/logs/dl_$EX.log 2>&1 || { echo "[$EX] DOWNLOAD FAILED" >> $ST; continue; }
  if [ ! -s $OSL/$EX.json ]; then
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --battery-only \
      --executor $EX --battery $OSL/battery_humor_v1.json --out $OSL/$EX.json >> $O/logs/bat_$EX.log 2>&1
    echo "[$EX] battery rc=$? $(date)" >> $ST
  fi
  if [ ! -s $OSL/mbar285_$EX.npz ]; then
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --executor $EX --freeze $OSL/freeze_humor285_v2.json --out $OSL/mbar285_$EX.npz >> $O/logs/h285_$EX.log 2>&1
    echo "[$EX] humor285 rc=$? $(date)" >> $ST
  fi
  # z-x-a + mechbat arms (morning 2026-07-09): run right after battery, weights complete.
  # qwen25-32b = the 14B->72B gap rung that tests the first zxa forecast (z*~2.2 Self-deprecation).
  for TD in humor creative_writing peer_review math mechbat_humor mechbat_peer_review; do
    OUT=$O/mbar_zxa_${TD}_${EX}.npz
    [ -s $OUT ] && continue
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --n-forms 1 --executor $EX --freeze $O/freeze_zxa_${TD}_v1.json --out $OUT >> $O/logs/zxa_${TD}_${EX}.log 2>&1
    echo "[$EX] zxa $TD rc=$? $(date)" >> $ST
  done
  for TD in $TASKS; do
    OUT=$O/mbar2_${TD}_${EX}.npz
    [ -s $OUT ] && continue
    env $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --executor $EX --freeze $O/freeze_${TD}_v2.json --out $OUT >> $O/logs/${TD}_${EX}.log 2>&1
    echo "[$EX] $TD rc=$? $(date)" >> $ST
  done
  echo "[$EX] ALL DONE $(date)" >> $ST
done
echo "NEWMODELS-DONE $(date)" >> $ST
