#!/usr/bin/env bash
# Chained debias-pilot trainings on ONE ledger-claimed sk3 GPU.
# usage:  bash run_chain.sh <GPU> <stageN_order.txt>
set -u
GPU="$1"; ORDER="$2"
export HOME=/lfs/skampere3/0/alexspan
ROOT=/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/debias_pilot
PY=$HOME/envs/ai_usage/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || exit 1
mkdir -p runs logs

while read -r TAG; do
  [ -z "$TAG" ] && continue
  if [ -f "runs/$TAG/result.json" ]; then
    echo "=== SKIP $TAG (already done) ==="; continue
  fi
  echo "=== START $TAG $(date -u +%FT%TZ) ==="
  $PY train_grl.py --config "configs/$TAG.json" > "logs/$TAG.log" 2>&1
  rc=$?
  echo "=== TRAIN $TAG rc=$rc $(date -u +%FT%TZ) ==="
  if [ $rc -ne 0 ]; then tail -30 "logs/$TAG.log"; continue; fi
  $PY probe_reps.py --run_dir "runs/$TAG" --nuisance build/nuisance.npz \
      --targets plant,realtok,char_len,docket_year > "logs/${TAG}_probe.log" 2>&1
  echo "=== PROBE $TAG rc=$? $(date -u +%FT%TZ) ==="
  tail -3 "logs/${TAG}_probe.log"
done < "$ORDER"
echo "=== CHAIN DONE $(date -u +%FT%TZ) ==="
