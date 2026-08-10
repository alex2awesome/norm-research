#!/usr/bin/env bash
# DIAGNOSTIC chain after V2' FAIL (coordinator-ordered full table 2026-08-08).
# NOT gate-certified: V2' failed on the utility legs (a,c); reliance leg (b)
# passed.  Runs the remaining arms as diagnostics + the D03 attribution arm.
# usage: bash run_diagnostics_decor.sh <GPU>
set -u
GPU="$1"
export HOME=/lfs/skampere3/0/alexspan
ROOT=/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/debias_pilot
PY=$HOME/envs/ai_usage/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || exit 1
mkdir -p runs logs results

run() {
  TAG="$1"
  if [ -f "runs/$TAG/result.json" ]; then echo "=== SKIP $TAG (already done)"; return 0; fi
  echo "=== START $TAG $(date -u +%FT%TZ)"
  $PY train_decor.py --config "configs/$TAG.json" > "logs/$TAG.log" 2>&1
  rc=$?
  echo "=== TRAIN $TAG rc=$rc $(date -u +%FT%TZ)"
  if [ $rc -ne 0 ]; then tail -30 "logs/$TAG.log"; fi
  return $rc
}
probe() {
  TAG="$1"
  $PY probe_reps.py --run_dir "runs/$TAG" --nuisance build/nuisance.npz \
      --targets plant,realtok,char_len,docket_year > "logs/${TAG}_probe.log" 2>&1
  echo "=== PROBE $TAG rc=$?"
}
release() {
  echo "$(date -u +%FT%TZ) | GPU=$GPU | agent=claude-decor-battery-fable | job=decor_diagnostics | RELEASE $1" >> "$HOME/norm-research/gpu_ledger.txt"
}

run D03_decor_unplanted_plantweights && probe D03_decor_unplanted_plantweights
run D07_decor_realtok_standard && probe D07_decor_realtok_standard
run D09_decor_v3b_date && probe D09_decor_v3b_date
run D10_decor_length_real && probe D10_decor_length_real
run D20_cap_vanilla
run D21_cap_decor_jointB
$PY analyze_decor.py --gate full > logs/analyze_decor_full.log 2>&1
echo "=== ANALYZE FULL rc=$?"
release "rc=0 diagnostic chain complete (NOT gate-certified; V2p failed a,c / passed b)"
echo "=== DECOR DIAGNOSTICS DONE $(date -u +%FT%TZ)"
