#!/bin/bash
# GEPA round K: score dev set with bank_rK on sk2, fetch npz, diagnose+propose bank_r{K+1}.
# Usage: bash run_round.sh <K> [gpu]
set -e
K=$1; GPU=${2:-6}
HERE="$(cd "$(dirname "$0")" && pwd)"
V4="$(dirname "$HERE")"
REMOTE=/lfs/skampere2/0/alexspan/nc_vat/gepa
MODEL=/lfs/skampere2/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/518276fb130dc81caf9a4f772e65e63ef2526493
ENVPY=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python

# dev.jsonl rows lack y fields the scorer tolerates via .get defaults
ssh sk2 "mkdir -p $REMOTE"
scp -q "$HERE/bank_r$K.jsonl" "$HERE/dev.jsonl" "$V4/score_va_gemma_nc.py" sk2:$REMOTE/
ssh sk2 "cd $REMOTE && env HOME=/lfs/skampere2/0/alexspan VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPU VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_DEVICE_ORDER=PCI_BUS_ID $ENVPY score_va_gemma_nc.py --shard dev.jsonl --rubrics bank_r$K.jsonl --out dev_scores_r$K.npz --model $MODEL --util 0.90 > round$K.log 2>&1; grep -c SCORE_DONE round$K.log"
scp -q sk2:$REMOTE/dev_scores_r$K.npz "$HERE/"
python3 -W ignore "$HERE/diagnose_propose.py" --round $((K+1)) --dev-npz "$HERE/dev_scores_r$K.npz"
echo "ROUND_${K}_DONE"
