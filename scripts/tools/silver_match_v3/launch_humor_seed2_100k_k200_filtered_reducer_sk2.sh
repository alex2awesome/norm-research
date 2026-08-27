#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_binary_v1
CE_ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2
CODE=$CE_ROOT/code
PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python
MODEL=/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1
TRAIN_RUN=$ROOT/runs/final-joined-recipe-v1/seed-2026071502
CHECKPOINT=$TRAIN_RUN/checkpoints/exposure-000000100000
REPORT=$TRAIN_RUN/training_report.json
REPORT_SHA=38deb60ac112c9862dfcae2b65f07be04f551a08b4c4de0842733169819d02af
INPUTS=$ROOT/production_reducer/inputs_v1
PAIRS=$INPUTS/humor.k200.minus-joined-truth.pairs.jsonl
PAIRS_SHA=be6561860d9657490b365045374e84260038ff5f80bc8427afabd9640351057f
FILTER_REPORT=$INPUTS/humor.k200.minus-joined-truth.report.json
BASE=$CE_ROOT/runs/pilot_test_release_v1/BASE_MODEL_MANIFEST.json
BASE_SHA=4047fcc5c148a8522fd5a783dde7c68076b1a2548d5873cd58abd18feaf9577b
OUT=$ROOT/production_reducer/humor-k200-minus-joined22090.seed-2026071502.exposure100k.v1
CACHE=$ROOT/cache
GPUS=(0 1 2 3 5 7)
NUM_SHARDS=${#GPUS[@]}

test ! -e "$OUT"
test "$(sha256sum "$PAIRS" | awk '{print $1}')" = "$PAIRS_SHA"
test "$(sha256sum "$REPORT" | awk '{print $1}')" = "$REPORT_SHA"
test "$(sha256sum "$BASE" | awk '{print $1}')" = "$BASE_SHA"
test "$(jq -r '.status' "$FILTER_REPORT")" = COMPLETE_IMMUTABLE_FILTERED_PAIR_UNIVERSE
test "$(jq -r '.output.sha256' "$FILTER_REPORT")" = "$PAIRS_SHA"
test "$(jq -r '.output.rows' "$FILTER_REPORT")" = 11057600
test "$(jq -r '.output.norm_uids' "$FILTER_REPORT")" = 55288
test "$(jq -r '.joined_truth.norm_uids' "$FILTER_REPORT")" = 22090
test "$(jq -r '.selected_checkpoint.exposure_budget' "$REPORT")" = 100000
test "$(jq -r '.classification_mode' "$REPORT")" = binary

for gpu in "${GPUS[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  test "$used" -le 128
  test "$util" -eq 0
done

mkdir -p "$OUT/logs" "$OUT/pids"
cd "$CODE"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=${GPUS[$shard]}
  shard_name=$(printf 'shard-%03d-of-%03d' "$shard" "$NUM_SHARDS")
  output=$OUT/scores.$shard_name.jsonl
  log=$OUT/logs/$shard_name.log
  pid_file=$OUT/pids/$shard_name.pid
  env \
    HOME="$CACHE/home" \
    HF_HOME="$CACHE/huggingface" \
    HF_MODULES_CACHE="$CACHE/huggingface/modules" \
    TRANSFORMERS_CACHE="$CACHE/huggingface/transformers" \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="$gpu" \
    nohup "$PYTHON" -u -m scripts.tools.silver_match_v3.run_nemotron_ce score \
      --input-pairs "$PAIRS" \
      --output "$output" \
      --model "$MODEL" \
      --base-manifest "$BASE" \
      --base-manifest-sha256 "$BASE_SHA" \
      --checkpoint "$CHECKPOINT" \
      --training-report "$REPORT" \
      --training-report-sha256 "$REPORT_SHA" \
      --batch-size 32 \
      --max-length 1024 \
      --device 0 \
      --shard-id "$shard" \
      --num-shards "$NUM_SHARDS" \
      >"$log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf 'LAUNCHED shard=%s gpu=%s pid=%s output=%s\n' "$shard" "$gpu" "$pid" "$output"
done

