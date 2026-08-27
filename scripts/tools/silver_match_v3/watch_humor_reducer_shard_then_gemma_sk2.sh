#!/usr/bin/env bash
set -euo pipefail

if test "$#" -ne 3; then
  echo "usage: $0 SHARD GPU REDUCER_PID" >&2
  exit 2
fi
SHARD=$1
GPU=$2
REDUCER_PID=$3
NUM_SHARDS=6

ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_binary_v1
CE_ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2
CODE=$CE_ROOT/code
PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python
OVERLAY=/lfs/skampere2/0/alexspan/envs/gemma4-lora-humor-v1-overlay
RUN=$ROOT/production_reducer/humor-k200.seed-2026071502.exposure100k.v1
INPUTS=$ROOT/production_reducer/inputs_v1
MANIFEST=$INPUTS/manifest.sk2.json
MODEL=/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/gemma-4-31b-it-3548789868c5356dbf307c98e6f609007b82b3eb-mirror-v1
MODEL_INVENTORY=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_gemma4_typed_v1/inputs/model_inventory.json
ADAPTER=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_gemma4_typed_v1/outputs/humor_gemma4_typed_v1_retry1
PROMPT=$CODE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md
PROMPT_ADDON=$CODE/scripts/tools/silver_match_v3/FINAL_DECISION_LABELING.md
EVENTS=$RUN/handoff.events.log
NAME=$(printf 'shard-%03d-of-%03d' "$SHARD" "$NUM_SHARDS")
SCORES=$RUN/scores.$NAME.jsonl
CANDIDATES=$RUN/candidate_shards/candidates.$NAME.jsonl
CANDIDATE_REPORT=$RUN/candidate_shards/candidates.$NAME.report.json
GEMMA_OUT=$RUN/typed_gemma/$NAME
GEMMA_LOG=$RUN/logs/gemma.$NAME.log
LORA_READY=$RUN/typed_gemma/HUMOR_TYPED_LORA_SELECTED_RELEASE_READY

event() {
  printf '%s %s shard=%s gpu=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$SHARD" "$GPU" >>"$EVENTS"
}

event HANDOFF_WATCH_STARTED
while kill -0 "$REDUCER_PID" 2>/dev/null; do
  sleep 300
done
if ! test -s "$SCORES.meta.json"; then
  event REDUCER_FAILED
  exit 3
fi
event REDUCER_COMPLETE

cd "$CODE"
"$PYTHON" -u -m scripts.tools.silver_match_v3.package_humor_ce_reducer_shard \
  --scores "$SCORES" \
  --manifest "$MANIFEST" \
  --output "$CANDIDATES" \
  --report-output "$CANDIDATE_REPORT" \
  --expected-pairs-sha256 f90c19bd3c06bcabffd52b165526aa88366ba1def1c21df3904af77edaf2b84a \
  --expected-checkpoint-metadata-sha256 76a58ba823fc3895a292b71d9cbee8a1e81314dfbf9762aa111ea3b4ea1d98d2 \
  >"$RUN/logs/package.$NAME.log" 2>&1
event CANDIDATES_PACKAGED

# Packaging is safe as soon as the reducer shard is complete, but production
# adjudication must not silently promote an adapter that was merely
# reload-verified.  The release controller creates this sentinel only after a
# Humor typed LoRA has passed its frozen selection checks.
if ! test -s "$LORA_READY"; then
  event WAITING_FOR_SELECTED_LORA
fi
while ! test -s "$LORA_READY"; do
  sleep 300
done
event SELECTED_LORA_READY

for _ in $(seq 1 60); do
  used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if test "$used" -le 128; then
    break
  fi
  sleep 10
done
test "$used" -le 128

CACHE=$RUN/typed_gemma/cache/$NAME
mkdir -p \
  "$CACHE/home/.cache/huggingface" \
  "$CACHE/xdg" \
  "$CACHE/torch_extensions" \
  "$CACHE/triton" \
  "$CACHE/flashinfer_workspace_base" \
  "$CACHE/flashinfer_jit" \
  "$CACHE/cuda" \
  "$CACHE/vllm" \
  "$CACHE/torchinductor" \
  "$CACHE/tmp"
event GEMMA_STARTED
env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HOME="$CACHE/home" \
  HF_HOME="$CACHE/home/.cache/huggingface" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  XDG_CACHE_HOME="$CACHE/xdg" \
  TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" \
  TRITON_CACHE_DIR="$CACHE/triton" \
  FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer_workspace_base" \
  FLASHINFER_JIT_DIR="$CACHE/flashinfer_jit" \
  CUDA_CACHE_PATH="$CACHE/cuda" \
  VLLM_CACHE_ROOT="$CACHE/vllm" \
  TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_USE_FLASHINFER_MOE_FP8=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  TMPDIR="$CACHE/tmp" \
  PYTHONPATH="$CODE:$OVERLAY" \
  "$PYTHON" -u -m scripts.tools.silver_match_v3.run_paired_gemma_lora_batch \
    --manifest "$MANIFEST" \
    --candidates "$CANDIDATES" \
    --prompt "$PROMPT" \
    --prompt-addon "$PROMPT_ADDON" \
    --model "$MODEL" \
    --model-inventory "$MODEL_INVENTORY" \
    --adapter "$ADAPTER" \
    --adapter-name humor_typed_candidate_reducer \
    --adapter-id 1 \
    --output-root "$GEMMA_OUT" \
    --max-candidates 16 \
    --batch-size 128 \
    --max-model-len 4096 \
    --max-tokens 160 \
    --gpu-memory-utilization 0.88 \
    --max-lora-rank 16 \
    --seed 17 \
    --keep-raw \
    --resume \
    >"$GEMMA_LOG" 2>&1
test -s "$GEMMA_OUT/paired_inference.meta.json"
event GEMMA_COMPLETE
