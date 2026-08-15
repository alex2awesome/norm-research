#!/bin/bash
# ladder stage A: score all pooled forms + definitions on probes with one executor
# args: $1=gpu $2=model-dir $3=tag
export HOME=/lfs/skampere1/0/alexspan
cd $HOME/ocl
PY=$HOME/norm-research/datasets/prompt-optimality-test/.venv/bin/python
GPU=$1; MODEL=$2; TAG=$3
for T in peer cw pr humor; do
  CUDA_VISIBLE_DEVICES=$GPU VLLM_GPU_MEM_UTIL=0.55 $PY score_within_metric.py \
    --texts ${T}_probe_texts.jsonl --manifest ocl_${T}_manifest.json --key probe_id \
    --model $MODEL --chunk 20 --out ocl_${TAG}_${T}_probes.json >> locl_${TAG}.log 2>&1
  echo "$(date -u +%FT%TZ) $TAG $T probes done" >> locl_${TAG}.log
done
echo "$(date -u +%FT%TZ) OCL ${TAG} STAGE-A DONE" >> locl_${TAG}.log
