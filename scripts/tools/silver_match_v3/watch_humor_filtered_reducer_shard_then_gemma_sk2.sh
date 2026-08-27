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
RUN=$ROOT/production_reducer/humor-k200-minus-joined22090.seed-2026071502.exposure100k.v1
INPUTS=$ROOT/production_reducer/inputs_v1
MANIFEST=$INPUTS/manifest.sk2.json
TRUTH=$INPUTS/truth.joined.all.jsonl
TRUTH_SHA=1e579269c92f87de6896f1320cb868868b6be16a65c6bfa51530ded7da6d1906
PAIRS_SHA=be6561860d9657490b365045374e84260038ff5f80bc8427afabd9640351057f
MODEL=/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/gemma-4-31b-it-3548789868c5356dbf307c98e6f609007b82b3eb-mirror-v1
MODEL_INVENTORY=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_gemma4_typed_v1/inputs/model_inventory.json
PROMPT=$CODE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md
PROMPT_ADDON=$CODE/scripts/tools/silver_match_v3/FINAL_DECISION_LABELING.md
EVENTS=$RUN/handoff.events.log
NAME=$(printf 'shard-%03d-of-%03d' "$SHARD" "$NUM_SHARDS")
SCORES=$RUN/scores.$NAME.jsonl
CANDIDATES=$RUN/candidate_shards/candidates.$NAME.jsonl
CANDIDATE_REPORT=$RUN/candidate_shards/candidates.$NAME.report.json
GEMMA_OUT=$RUN/typed_gemma/$NAME
GEMMA_LOG=$RUN/logs/gemma.$NAME.log
LORA_SELECTION=$RUN/typed_gemma/HUMOR_TYPED_LORA_SELECTION.json

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
  --expected-pairs-sha256 "$PAIRS_SHA" \
  --expected-checkpoint-metadata-sha256 76a58ba823fc3895a292b71d9cbee8a1e81314dfbf9762aa111ea3b4ea1d98d2 \
  --excluded-truth "$TRUTH" \
  --expected-excluded-truth-sha256 "$TRUTH_SHA" \
  --expected-excluded-uids 22090 \
  >"$RUN/logs/package.$NAME.log" 2>&1
event CANDIDATES_PACKAGED

if ! test -s "$LORA_SELECTION"; then
  event WAITING_FOR_SELECTED_LORA
fi
while ! test -s "$LORA_SELECTION"; do
  sleep 300
done
if ! "$PYTHON" - "$LORA_SELECTION" <<'PY'
import hashlib
import json
import pathlib
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    selection.get("schema_version") != "silver-match-v3-humor-typed-lora-selection-v1"
    or selection.get("status") != "SELECTED_RELEASE_READY"
    or selection.get("task") != "humor"
    or selection.get("frozen_selection_passed") is not True
):
    raise SystemExit(1)
adapter = selection.get("adapter") or {}
root = pathlib.Path(str(adapter.get("path") or ""))
for name, field in (
    ("adapter_config.json", "adapter_config_sha256"),
    ("adapter_model.safetensors", "adapter_model_sha256"),
):
    path = root / name
    if not path.is_file():
        raise SystemExit(1)
    digest_state = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    if digest != adapter.get(field):
        raise SystemExit(1)
PY
then
  event INVALID_SELECTED_LORA_ATTESTATION
  exit 4
fi
ADAPTER=$(jq -r '.adapter.path' "$LORA_SELECTION")
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
