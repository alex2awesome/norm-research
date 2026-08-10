#!/bin/bash
# Waits for the S2 fetch (PID arg) to finish, then builds + bounds the full v2
# dataset, writes splits, and runs the bge upper bound on GPU 1 (if free).
# Runs under nohup; logs to auto_build.log.
export HOME=/lfs/skampere3/0/alexspan
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
cd /lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/best_papers/v2_build
FETCH_PID=$1
echo "[auto] waiting on S2 fetch PID $FETCH_PID ..."
while ps -p $FETCH_PID > /dev/null 2>&1; do sleep 120; done
echo "[auto] S2 fetch done at $(date); cache=$(wc -l < s2_cache.jsonl)"
echo "[auto] === build dataset ==="
$PY build_dataset.py --cache s2_cache.jsonl --input s2_input_prio.csv --out best_papers_v2_full.csv.gz
echo "[auto] === write splits ==="
$PY write_splits.py
echo "[auto] === deconfound + bounds ==="
$PY deconfound_bounds.py --data best_papers_v2_full.csv.gz
echo "[auto] === V-feature probe ==="
$PY v_feature_probe.py --data best_papers_v2_full.csv.gz
echo "[auto] === bge upper bound (GPU 1) ==="
g1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null)
if [ -n "$g1" ] && [ "$g1" -lt 20000 ]; then
  CUDA_VISIBLE_DEVICES=1 $PY bge_upper_bound.py --data best_papers_v2_full.csv.gz || echo "[auto] bge failed/skipped"
else
  echo "[auto] GPU 1 busy ($g1 MiB) -- skipping bge"
fi
echo "[auto] DONE at $(date)"
