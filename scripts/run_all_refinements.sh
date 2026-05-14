#!/bin/bash
# Run refine_parent.py on all R1 cells, 3 in parallel.
# Each cell uses --concurrency 200 --service-tier flex.
set -u
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
export HOME=/lfs/skampere3/0/alexspan

ROOT=/lfs/skampere3/0/alexspan/norm-research
LOGDIR=$ROOT/outputs/refine_logs
mkdir -p "$LOGDIR"

cd "$ROOT"

# (task bucket input_file) — patents/general already validated; skip.
CELLS=(
  "code-review|general|outputs/hierarchy/code-review_general_r1_expanded.json"
  "code-review|hyper_specific|outputs/hierarchy/code-review_hyper_specific_r1_expanded.json"
  "code-review|specific|outputs/hierarchy/code-review_specific_r1_expanded.json"
  "creative-writing|general|outputs/hierarchy/creative-writing_general_gpt5_k200_expanded.json"
  "grant-funding|general|outputs/hierarchy/grant-funding_general_r1_expanded.json"
  "grant-funding|hyper_specific|outputs/hierarchy/grant-funding_hyper_specific_r1_expanded.json"
  "grant-funding|specific|outputs/hierarchy/grant-funding_specific_r1_expanded.json"
  "humor|general|outputs/hierarchy/humor_general_r1_expanded.json"
  "humor|hyper_specific|outputs/hierarchy/humor_hyper_specific_r1_expanded.json"
  "humor|specific|outputs/hierarchy/humor_specific_r1_expanded.json"
  "legal-outcome-prediction|general|outputs/hierarchy/legal-outcome-prediction_general_r1_expanded.json"
  "legal-outcome-prediction|hyper_specific|outputs/hierarchy/legal-outcome-prediction_hyper_specific_r1_expanded.json"
  "legal-outcome-prediction|specific|outputs/hierarchy/legal-outcome-prediction_specific_r1_expanded.json"
  "math-stackexchange|general|outputs/hierarchy/math-stackexchange_general_r1_expanded.json"
  "math-stackexchange|hyper_specific|outputs/hierarchy/math-stackexchange_hyper_specific_r1_expanded.json"
  "math-stackexchange|specific|outputs/hierarchy/math-stackexchange_specific_r1_expanded.json"
  "news-homepages|general|outputs/hierarchy/news-homepages_general_r1_expanded.json"
  "news-homepages|hyper_specific|outputs/hierarchy/news-homepages_hyper_specific_r1_expanded.json"
  "news-homepages|specific|outputs/hierarchy/news-homepages_specific_r1_expanded.json"
  "notice-and-comment|general|outputs/hierarchy/notice-and-comment_general_r1_expanded.json"
  "notice-and-comment|hyper_specific|outputs/hierarchy/notice-and-comment_hyper_specific_r1_expanded.json"
  "notice-and-comment|specific|outputs/hierarchy/notice-and-comment_specific_r1_expanded.json"
  "patents|hyper_specific|outputs/hierarchy/patents_hyper_specific_r1_expanded.json"
  "patents|specific|outputs/hierarchy/patents_specific_r1_expanded.json"
  "peer-review|general|outputs/hierarchy/peer-review_general_r1_expanded.json"
  "peer-review|hyper_specific|outputs/hierarchy/peer-review_hyper_specific_r1_expanded.json"
  "peer-review|specific|outputs/hierarchy/peer-review_specific_r1_expanded.json"
  "press-releases|general|outputs/hierarchy/press-releases_general_r1_expanded.json"
  "press-releases|hyper_specific|outputs/hierarchy/press-releases_hyper_specific_r1_expanded.json"
  "press-releases|specific|outputs/hierarchy/press-releases_specific_r1_expanded.json"
)

PARALLEL=3
running=0

for cell in "${CELLS[@]}"; do
  IFS='|' read -r task bucket input <<< "$cell"
  output="outputs/hierarchy/${task}_${bucket}_r1_refined.json"
  log="$LOGDIR/${task}_${bucket}.log"

  if [[ -f "$output" ]]; then
    echo "[skip] $task/$bucket already done"; continue
  fi
  echo "[start] $task/$bucket → $output"
  (
    python scripts/refine_parent.py \
      --task "$task" --bucket "$bucket" \
      --input "$input" --output "$output" \
      --model gpt-5 --service-tier flex --concurrency 200 \
      > "$log" 2>&1
    echo "[done] $task/$bucket"
  ) &
  running=$((running + 1))
  if (( running >= PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "ALL REFINEMENTS COMPLETE"
