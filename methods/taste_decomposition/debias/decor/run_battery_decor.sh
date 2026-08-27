#!/usr/bin/env bash
# DECORRELATED-TRAINING planted battery (debias instrument #3), auto-gated:
#   gradcheck -> stage 1 (V2') -> gate -> stage 2 (V3'a+b) -> gate ->
#   stage 3 (V4') -> gate -> stage 4 (cap_crowd bonus arm) -> full report.
# ONE ledger-claimed sk3 GPU.  usage: bash run_battery_decor.sh <GPU>
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
  tail -2 "logs/${TAG}_probe.log"
}
release() {
  echo "$(date -u +%FT%TZ) | GPU=$GPU | agent=claude-decor-battery-fable | job=decor_battery | RELEASE $1" >> "$HOME/norm-research/gpu_ledger.txt"
}

# 0. gradient check on the real model + real batches (fatal on failure)
if [ ! -f runs/D02_decor_planted_plant/gradcheck_report.json ]; then
  $PY train_decor.py --config configs/D02_decor_planted_plant.json --gradcheck > logs/decor_gradcheck.log 2>&1
  if [ $? -ne 0 ]; then
    echo "=== GRADCHECK FAIL"; tail -20 logs/decor_gradcheck.log
    release "rc=2 gradcheck FAIL"; exit 2
  fi
fi
echo "=== GRADCHECK PASS"

# ---- stage 1: V2' REMOVAL-OF-RELIANCE -------------------------------------
run D02_decor_planted_plant && probe D02_decor_planted_plant
run D00_vanilla_real_s1
run D00_vanilla_real_s2
$PY analyze_decor.py --gate v2p > logs/gate_v2p.log 2>&1; rc2=$?
echo "=== GATE V2' rc=$rc2"; tail -25 logs/gate_v2p.log
if [ $rc2 -ne 0 ]; then release "gate=v2p rc=$rc2 STOP"; exit $rc2; fi

# ---- stage 2: V3' SPECIFICITY ---------------------------------------------
run D07_decor_realtok_standard && probe D07_decor_realtok_standard
run D09_decor_v3b_date && probe D09_decor_v3b_date
$PY analyze_decor.py --gate v3p > logs/gate_v3p.log 2>&1; rc3=$?
echo "=== GATE V3' rc=$rc3"; tail -25 logs/gate_v3p.log
if [ $rc3 -ne 0 ]; then release "gate=v3p rc=$rc3 STOP"; exit $rc3; fi

# ---- stage 3: V4' CONSISTENCY ----------------------------------------------
run D10_decor_length_real && probe D10_decor_length_real
$PY analyze_decor.py --gate v4p > logs/gate_v4p.log 2>&1; rc4=$?
echo "=== GATE V4' rc=$rc4"; tail -25 logs/gate_v4p.log
if [ $rc4 -ne 0 ]; then release "gate=v4p rc=$rc4 STOP"; exit $rc4; fi

# ---- stage 4: cap_crowd bonus arm (scientific payoff; no gate) --------------
run D20_cap_vanilla
run D21_cap_decor_jointB
$PY analyze_decor.py --gate cap > logs/gate_cap.log 2>&1
echo "=== CAP readout rc=$?"; tail -15 logs/gate_cap.log

$PY analyze_decor.py --gate full > logs/analyze_decor_full.log 2>&1
release "rc=0 all gates passed + cap bonus complete"
echo "=== DECOR BATTERY DONE $(date -u +%FT%TZ)"
