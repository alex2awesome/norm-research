#!/bin/bash
# Watcher: for each cell, wait for R2 _expanded.json, convert via
# r2_to_r3_input.py, then run R3 meta-merge.
set -u
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
export HOME=/lfs/skampere3/0/alexspan

ROOT=/lfs/skampere3/0/alexspan/norm-research
LOGDIR=$ROOT/outputs/r3_logs
LOCKDIR=$ROOT/outputs/r3_locks
mkdir -p "$LOGDIR" "$LOCKDIR"
cd "$ROOT"

CELLS=(
  "code-review|general"
  "code-review|hyper_specific"
  "code-review|specific"
  "creative-writing|general"
  "grant-funding|general"
  "grant-funding|hyper_specific"
  "grant-funding|specific"
  "humor|general"
  "humor|hyper_specific"
  "humor|specific"
  "legal-outcome-prediction|general"
  "legal-outcome-prediction|hyper_specific"
  "legal-outcome-prediction|specific"
  "math-stackexchange|general"
  "math-stackexchange|hyper_specific"
  "math-stackexchange|specific"
  "news-homepages|general"
  "news-homepages|hyper_specific"
  "news-homepages|specific"
  "notice-and-comment|general"
  "notice-and-comment|hyper_specific"
  "notice-and-comment|specific"
  "patents|general"
  "patents|hyper_specific"
  "patents|specific"
  "peer-review|general"
  "peer-review|hyper_specific"
  "peer-review|specific"
  "press-releases|general"
  "press-releases|hyper_specific"
  "press-releases|specific"
)

run_r3_for_cell() {
  local task="$1" bucket="$2"
  local r2_expanded="outputs/hierarchy/${task}_${bucket}_r2_expanded.json"
  local r3_input="outputs/hierarchy/${task}_${bucket}_r2_as_r3_input.json"
  local r3_out="outputs/hierarchy/${task}_${bucket}_r3.json"
  local lock="$LOCKDIR/${task}_${bucket}.lock"
  local log="$LOGDIR/${task}_${bucket}.log"

  while [[ ! -f "$r2_expanded" ]]; do sleep 30; done
  if [[ -f "$r3_out" ]]; then
    echo "[skip] $task/$bucket R3 already exists"
    return 0
  fi

  touch "$lock"
  echo "[start R3] $task/$bucket"
  python scripts/r2_to_r3_input.py \
    --input "$r2_expanded" --output "$r3_input" >> "$log" 2>&1
  python scripts/meta_merge.py \
    --task "$task" --bucket "$bucket" \
    --input "$r3_input" --output "$r3_out" \
    --model gpt-5 --service-tier flex \
    --concurrency 200 --include-merges \
    >> "$log" 2>&1
  local rc=$?
  rm -f "$lock"
  if [[ $rc -eq 0 ]]; then
    echo "[done R3] $task/$bucket"
  else
    echo "[FAIL R3 rc=$rc] $task/$bucket (see $log)"
  fi
}

for cell in "${CELLS[@]}"; do
  IFS='|' read -r task bucket <<< "$cell"
  run_r3_for_cell "$task" "$bucket" &
done

wait
echo "ALL R3 COMPLETE"
