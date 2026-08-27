#!/usr/bin/env bash
# Run after >=80% of the 2,495 AC pair_ids have L1 relabel results in
# outputs/v2_analysis/ac_l1_relabel/results/*.jsonl.
#
# This will:
#   1. Re-run build_cells.py locally — produces cell_ac_l1.parquet automatically
#      once coverage passes the 80% threshold.
#   2. Re-ship cell_ac_l1.parquet to sk3.
#   3. Re-run dense ceiling on the ac_l1 cell only.
#   4. Merge ac_l1 numbers into report.md / report.json.
#
# Usage: bash scripts/dense_ceiling/ac_l1_rerun.sh

set -euo pipefail

REPO_LAPTOP="/Users/spangher/Projects/stanford-research/norm-research"
REPO_SK3="/lfs/skampere3/0/alexspan/norm-research"

# 1. rebuild
python3 "$REPO_LAPTOP/scripts/dense_ceiling/build_cells.py"

if [[ ! -f "$REPO_LAPTOP/outputs/v2_analysis/dense_ceiling/cell_ac_l1.parquet" ]]; then
  echo "cell_ac_l1.parquet not produced — coverage still <80%. Aborting."
  exit 1
fi

# 2. ship
scp "$REPO_LAPTOP/outputs/v2_analysis/dense_ceiling/cell_ac_l1.parquet" \
    "sk3:$REPO_SK3/outputs/v2_analysis/dense_ceiling/"

# 3. run dense ceiling on ac_l1 only
ssh sk3 bash -lc "'
export HOME=/lfs/skampere3/0/alexspan
cd $REPO_SK3
CUDA_VISIBLE_DEVICES=7 /lfs/skampere3/0/alexspan/miniconda3/bin/python3 \
  scripts/dense_ceiling/run_dense_ceiling.py \
  --cells-dir outputs/v2_analysis/dense_ceiling \
  --out-dir outputs/v2_analysis/dense_ceiling/ac_l1_run \
  --cells ac_l1
'"

# 4. pull results
mkdir -p "$REPO_LAPTOP/outputs/v2_analysis/dense_ceiling/ac_l1_run"
scp -r "sk3:$REPO_SK3/outputs/v2_analysis/dense_ceiling/ac_l1_run/*.json" \
       "$REPO_LAPTOP/outputs/v2_analysis/dense_ceiling/ac_l1_run/"

echo "Done. Merge into report.md manually or re-run build_report.py with --include-l1."
