#!/bin/bash
# Phase-2 pipeline on sk3: pool build -> propensity balance -> pairwise companion.
# Each stage logs to logs/, manifests/ live next to outputs.
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
ROOT=/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python3

cd $ROOT
mkdir -p pool balanced pairwise logs

echo "===== [stage 1] pool build ====="
$PY scripts/build_so_python_pool_v2.py 2>&1 | tee logs/build_pool.log

echo "===== [stage 2] propensity balance ====="
$PY scripts/propensity_balance_v2.py \
    --pool pool/so_python_v2_pool.csv.gz \
    --out  balanced/so_python_v2_propensity_balanced.csv.gz \
    2>&1 | tee logs/balance.log

echo "===== [stage 3] pairwise later-wins ====="
$PY scripts/build_pairwise_laterwins_v2.py 2>&1 | tee logs/pairwise.log

echo "===== ALL DONE ====="
