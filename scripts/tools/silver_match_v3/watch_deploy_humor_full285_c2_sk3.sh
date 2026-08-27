#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere3/0/alexspan/runtime/humor_ce_k85_complement_v1
K85_RUN=$ROOT/scores.seed-2026071502.exposure100k.k85.v1
K85=$K85_RUN/scores.merged.jsonl
K200=$ROOT/incoming_k200/scores.k200.merged.jsonl
K85_PAIRS=$ROOT/humor.k85.exact-complement.minus-joined-truth.pairs.jsonl
BANK=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks/humor.json
CODE=$ROOT/deploy_code
PYTHON=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
VLLM_PYTHON=/lfs/skampere3/0/alexspan/envs/vllm_latest/bin/python
DEPLOY=$ROOT/full285_c2_deployment_v1
EVENTS=$ROOT/full285_c2_deployment.watcher.events.log
MODEL=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1/model
MODEL_INVENTORY=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1/inputs/LLAMA_MODEL_INVENTORY.sk3.json
MODEL_INVENTORY_SHA=7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85
ADAPTER=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1/runs/llama31-8b-typed-compact2048-decision-balanced-v1/adapter.exposure-checkpoints/exposure_000000034944/adapter
GPUS=(5 6 7)
NUM_SHARDS=${#GPUS[@]}

test ! -e "$EVENTS"
test ! -e "$DEPLOY"
printf '%s WATCH_STARTED\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$EVENTS"
while ! test -s "$K85.meta.json" || ! test -s "$K200.meta.json"; do sleep 300; done
printf '%s BOTH_CE_SURFACES_READY\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$EVENTS"

cd "$CODE"
"$PYTHON" -u -m scripts.tools.silver_match_v3.merge_package_humor_full285_ce \
  --k200-scores "$K200" --k85-scores "$K85" --k85-pairs "$K85_PAIRS" \
  --bank "$BANK" --output-root "$DEPLOY" --ce-top 16 \
  >"$ROOT/full285.merge-package.log" 2>&1
test "$(jq -r .status "$DEPLOY/REPORT.json")" = COMPLETE_EXACT_FULL285_AND_PAIRED_PROMPTS
PROMPTS=$DEPLOY/paired_order.prompts.jsonl
PROMPTS_SHA=$(jq -r .outputs.paired_prompts.sha256 "$DEPLOY/REPORT.json")
test "$(sha256sum "$PROMPTS" | awk '{print $1}')" = "$PROMPTS_SHA"
printf '%s FULL285_AND_PROMPTS_COMPLETE full_sha=%s prompts_sha=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(jq -r .outputs.full_surface.sha256 "$DEPLOY/REPORT.json")" "$PROMPTS_SHA" >>"$EVENTS"

# K85 owned these exact allowed GPUs.  Wait for release and fail closed if a
# different process races onto any device.  Physical GPUs 1/2/3/4 are never queried for launch.
while :; do
  free=1
  for gpu in "${GPUS[@]}"; do
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if test "$used" -gt 128 || test "$util" -ne 0; then free=0; fi
  done
  test "$free" -eq 1 && break
  sleep 300
done

mkdir -p "$DEPLOY/typed/logs" "$DEPLOY/typed/pids"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=${GPUS[$shard]}
  name=$(printf 'shard-%03d-of-%03d' "$shard" "$NUM_SHARDS")
  OUT=$DEPLOY/typed/$name
  CACHE=$DEPLOY/typed/cache/$name
  mkdir -p "$CACHE/home/.cache/huggingface" "$CACHE/xdg" "$CACHE/torch_extensions" \
    "$CACHE/triton" "$CACHE/flashinfer_workspace_base" "$CACHE/flashinfer_jit" \
    "$CACHE/cuda" "$CACHE/vllm" "$CACHE/torchinductor" "$CACHE/tmp"
  env CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HOME="$CACHE/home" HF_HOME="$CACHE/home/.cache/huggingface" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" \
    TRITON_CACHE_DIR="$CACHE/triton" FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer_workspace_base" \
    FLASHINFER_JIT_DIR="$CACHE/flashinfer_jit" CUDA_CACHE_PATH="$CACHE/cuda" \
    VLLM_CACHE_ROOT="$CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_FLASHINFER_MOE_FP8=0 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 TMPDIR="$CACHE/tmp" PYTHONPATH="$CODE" \
    nohup "$VLLM_PYTHON" -u -m scripts.tools.silver_match_v3.run_humor_c2_production_paired_vllm \
      --prompts "$PROMPTS" --prompts-sha256 "$PROMPTS_SHA" \
      --model "$MODEL" --model-inventory "$MODEL_INVENTORY" \
      --model-inventory-sha256 "$MODEL_INVENTORY_SHA" --adapter "$ADAPTER" \
      --output-root "$OUT" --shard-id "$shard" --num-shards "$NUM_SHARDS" \
      --adapter-name humor_c2_full285_deployment --batch-size 128 --max-model-len 2048 \
      --max-tokens 192 --gpu-memory-utilization 0.88 --max-lora-rank 16 --seed 94137 \
      >"$DEPLOY/typed/logs/$name.log" 2>&1 < /dev/null &
  pid=$!; printf '%s\n' "$pid" >"$DEPLOY/typed/pids/$name.pid"
  printf '%s C2_SHARD_STARTED shard=%s gpu=%s pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$shard" "$gpu" "$pid" >>"$EVENTS"
done

failed=0
for p in "$DEPLOY"/typed/pids/*.pid; do
  pid=$(cat "$p")
  wait "$pid" || failed=1
done
test "$failed" -eq 0
typed_args=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  name=$(printf 'shard-%03d-of-%03d' "$shard" "$NUM_SHARDS")
  OUT=$DEPLOY/typed/$name
  test "$(jq -r .status "$OUT/INFERENCE_META.json")" = COMPLETE_C2_PRODUCTION_PAIRED_INFERENCE
  typed_args+=(--typed-root "$OUT")
done
printf '%s ALL_C2_SHARDS_COMPLETE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$EVENTS"

"$PYTHON" -u -m scripts.tools.silver_match_v3.finalize_humor_c2_full285_deployment \
  --candidate-package "$DEPLOY/candidates.top16-plus-positives.jsonl" \
  "${typed_args[@]}" \
  --output "$DEPLOY/production_predictions.normalized.55288.jsonl" \
  --report-output "$DEPLOY/FINALIZE_REPORT.json" \
  >"$DEPLOY/finalize.log" 2>&1
test "$(jq -r .status "$DEPLOY/FINALIZE_REPORT.json")" = COMPLETE_DEV_FROZEN_DEPLOYMENT_BLIND_P855
printf '%s DEPLOYMENT_NORMALIZED_COMPLETE predictions_sha=%s matches=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(jq -r .output.sha256 "$DEPLOY/FINALIZE_REPORT.json")" \
  "$(jq -r .coverage.match_rows "$DEPLOY/FINALIZE_REPORT.json")" >>"$EVENTS"

