#!/usr/bin/env bash
set -euo pipefail

# Physical sk3 GPUs 1/2/3/4 are prohibited.  This complement uses only the
# currently allowed idle devices and never interrupts an existing process.
GPUS=(5 6 7)
NUM_SHARDS=${#GPUS[@]}
ROOT=/lfs/skampere3/0/alexspan/runtime/humor_ce_k85_complement_v1
CODE=$ROOT/code
PYTHON=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
PAIRS=$ROOT/humor.k85.exact-complement.minus-joined-truth.pairs.jsonl
BUILD_REPORT=$ROOT/BUILD_REPORT.json
STAGE=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1/dev_hybrid_v1/nemotron_dev_stage_v2
MODEL=$STAGE/source/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1
CHECKPOINT=$STAGE/source/runtime/humor_ce_binary_v1/runs/final-joined-recipe-v1/seed-2026071502/checkpoints/exposure-000000100000
REPORT=$STAGE/training_report.relocated.json
REPORT_SHA=31be7932392295fbb909c2dee0730f210165942fc884211916a7d3a6428b6c59
BASE=$STAGE/BASE_MODEL_MANIFEST.relocated.json
BASE_SHA=d1a13c104772dbf82cf95c08fc52dd88f93e9a48284aa5d8ba81f1c52ae406c8
OUT=$ROOT/scores.seed-2026071502.exposure100k.k85.v1
CACHE=$ROOT/cache

test -x "$PYTHON"
test -f "$BUILD_REPORT"
test "$(jq -r .status "$BUILD_REPORT")" = COMPLETE_EXACT_K85_COMPLEMENT
test "$(jq -r .output.rows "$BUILD_REPORT")" = 4699480
test "$(jq -r .output.norm_uids "$BUILD_REPORT")" = 55288
test "$(jq -r .output.k "$BUILD_REPORT")" = 85
test "$(sha256sum "$PAIRS" | awk '{print $1}')" = "$(jq -r .output.sha256 "$BUILD_REPORT")"
test "$(sha256sum "$REPORT" | awk '{print $1}')" = "$REPORT_SHA"
test "$(sha256sum "$BASE" | awk '{print $1}')" = "$BASE_SHA"
test ! -e "$OUT"

for gpu in "${GPUS[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  test "$used" -le 128
  test "$util" -eq 0
done

mkdir -p "$OUT/logs" "$OUT/pids" "$CACHE/home" "$CACHE/huggingface" \
  "$CACHE/modules" "$CACHE/torch_extensions" "$CACHE/torchinductor" \
  "$CACHE/triton" "$CACHE/xdg" "$ROOT/tmp"
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
    HF_MODULES_CACHE="$CACHE/modules" \
    TRANSFORMERS_CACHE="$CACHE/huggingface" \
    TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" \
    TRITON_CACHE_DIR="$CACHE/triton" \
    XDG_CACHE_HOME="$CACHE/xdg" \
    TMPDIR="$ROOT/tmp" \
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
      >"$log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf 'LAUNCHED shard=%s gpu=%s pid=%s output=%s\n' "$shard" "$gpu" "$pid" "$output"
done
