#!/usr/bin/env bash
# Bottleneck-architecture battery, auto-gated: stage B1 (V1/V2) -> gate -> stage B2
# (V3/V4/final) on ONE ledger-claimed sk3 GPU.  usage: bash run_battery_bn.sh <GPU>
set -u
GPU="$1"
export HOME=/lfs/skampere3/0/alexspan
ROOT=/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/debias_pilot
PY=$HOME/envs/ai_usage/bin/python
cd "$ROOT" || exit 1
mkdir -p runs logs results

bash run_chain.sh "$GPU" configs/stageB1_order.txt >> logs/chain_B1.log 2>&1

# scope probe on pooled h for the GRL arms (bottleneck runs only)
for T in B02_grl_planted_full_l0.1 B03_grl_planted_full_l0.5 B04_grl_planted_full_l1.0; do
  [ -f "runs/$T/reps.npz" ] && CUDA_VISIBLE_DEVICES="$GPU" $PY probe_reps.py \
      --run_dir "runs/$T" --nuisance build/nuisance.npz --targets plant \
      --rep_key rep_h > "logs/${T}_probe_h.log" 2>&1
done

$PY gate_bn.py > logs/gate_bn.log 2>&1
rc=$?
echo "=== GATE rc=$rc $(date -u +%FT%TZ) ===" >> logs/gate_bn.log
if [ $rc -eq 0 ]; then
  LAM=$($PY -c "import json;print(json.load(open('results/v2_gate_bn.json'))['lam_star'])")
  $PY make_configs.py --stage 2 --arch bn --lam "$LAM" >> logs/gate_bn.log 2>&1
  bash run_chain.sh "$GPU" configs/stageB2_order.txt >> logs/chain_B2.log 2>&1
fi
$PY analyze_battery.py --arch bn > logs/analyze_bn.log 2>&1
echo "$(date -u +%FT%TZ) | GPU=$GPU | agent=claude-debias-audit-fable | job=bn_battery RELEASE gate_rc=$rc" >> $HOME/norm-research/gpu_ledger.txt
echo "=== BN BATTERY DONE gate_rc=$rc $(date -u +%FT%TZ) ==="
