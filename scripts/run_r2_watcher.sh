#!/bin/bash
# Watcher: for each cell, wait for the refined R1 to appear, then run R2 meta-merge.
# Independent worker per cell, with a global cap of PARALLEL active R2 jobs.
set -u
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
export HOME=/lfs/skampere3/0/alexspan

ROOT=/lfs/skampere3/0/alexspan/norm-research
LOGDIR=$ROOT/outputs/r2_logs
LOCKDIR=$ROOT/outputs/r2_locks
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

PARALLEL=3

run_r2_for_cell() {
  local task="$1" bucket="$2"
  local refined="outputs/hierarchy/${task}_${bucket}_r1_refined.json"
  local r2_out="outputs/hierarchy/${task}_${bucket}_r2.json"
  local lock="$LOCKDIR/${task}_${bucket}.lock"
  local log="$LOGDIR/${task}_${bucket}.log"

  # Wait for refined to exist
  while [[ ! -f "$refined" ]]; do sleep 30; done

  # Skip if R2 already done
  if [[ -f "$r2_out" ]]; then
    echo "[skip] $task/$bucket R2 already exists"
    return 0
  fi

  # Global parallel cap via lockdir count
  while true; do
    local n_active
    n_active=$(ls "$LOCKDIR" 2>/dev/null | grep -c '\.lock$' || true)
    if (( n_active < PARALLEL )); then break; fi
    sleep 15
  done

  touch "$lock"
  echo "[start R2] $task/$bucket"
  python scripts/meta_merge.py \
    --task "$task" --bucket "$bucket" \
    --input "$refined" --output "$r2_out" \
    --model gpt-5 --service-tier flex \
    --concurrency 200 --include-merges \
    > "$log" 2>&1
  local rc=$?
  rm -f "$lock"
  if [[ $rc -eq 0 ]]; then
    echo "[done R2] $task/$bucket"
  else
    echo "[FAIL R2 rc=$rc] $task/$bucket (see $log)"
  fi
}

# Launch a watcher per cell (most will idle waiting for refined files).
for cell in "${CELLS[@]}"; do
  IFS='|' read -r task bucket <<< "$cell"
  run_r2_for_cell "$task" "$bucket" &
done

wait
echo "ALL R2 COMPLETE"
