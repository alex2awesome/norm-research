#!/usr/bin/env bash
set -euo pipefail

# Wait for the already-launched dev-selected typed Llama run, then consume the
# frozen test+blind roles exactly once through the truth-firewalled evaluator.

TRAIN_PID=854488
BASE=/lfs/skampere2/0/alexspan/norm-research-silver-v3
CODE=$BASE/runtime/humor_ce_v2/code
RUN=$BASE/runtime/humor_ce_binary_v1/typed_llama_fallback_v1/runs/llama31-8b-typed-final-joined-v1-retry1-b4-ga4
ADAPTER=$RUN/adapter
REPORT=$RUN/TRAINING_REPORT.json
DATA=$BASE/runtime/humor_typed_llama8b_v1/data
EVAL=$BASE/runtime/humor_typed_llama8b_v1/eval/retry1_b4_ga4_selected_v1
MODEL=/lfs/skampere2/0/shared_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
INVENTORY=$BASE/runtime/humor_typed_llama8b_v1/identity/llama31_8b_instruct.base_manifest.json
INVENTORY_SHA=008cdbd6f737fced087cee5706ed99b33982ed3e020c076112268aaeecef77de
PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python

while [[ ! -f "$REPORT" ]]; do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "training exited without a final report" >&2
    exit 3
  fi
  sleep 60
done

"$PYTHON" - "$REPORT" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["status"] == "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED"
assert value["selection"]["status"] == "SELECTED_ON_DEV_ONLY"
assert value["selection"]["test_or_blind_data_read"] is False
assert value["adapter"]["inference_reload_verified"] is True
assert value["adapter"]["fresh_base_reload_verified"] is True

# The first exposure checkpoint is eligible for held-out consumption only
# under the predeclared high-precision gate.  If it does not pass, the
# trainer must finish and select a later checkpoint on dev; test/blind remain
# sealed throughout training.
selection = value["selection"]
if selection["chosen_cumulative_exposure"] == 17472:
    chosen = selection["chosen_dev_report"]
    gate = chosen["confidence_gate"]
    generation = chosen["generation"]
    assert gate["precision_wilson_gate_met"] is True
    assert gate["exact_precision"] >= 0.90
    assert gate["exact_precision_wilson_95_lower"] >= 0.85
    assert gate["predicted_exact_count"] >= 100
    assert 1.0 - generation["valid_predictions"] / generation["rows"] <= 0.01
PY

test ! -e "$EVAL"
mkdir -p "$EVAL"

export HOME=/lfs/skampere2/0/alexspan
export XDG_CACHE_HOME=$BASE/cache/typed-llama-eval-20260714/xdg
export TORCH_EXTENSIONS_DIR=$BASE/cache/typed-llama-eval-20260714/torch_extensions
export TRITON_CACHE_DIR=$BASE/cache/typed-llama-eval-20260714/triton
export FLASHINFER_WORKSPACE_BASE=$BASE/cache/typed-llama-eval-20260714/flashinfer_workspace_base
export FLASHINFER_JIT_DIR=$BASE/cache/typed-llama-eval-20260714/flashinfer_jit
export CUDA_CACHE_PATH=$BASE/cache/typed-llama-eval-20260714/cuda
export VLLM_CACHE_ROOT=$BASE/cache/typed-llama-eval-20260714/vllm
export TORCHINDUCTOR_CACHE_DIR=$BASE/cache/typed-llama-eval-20260714/torchinductor
export TMPDIR=$BASE/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=4
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$FLASHINFER_JIT_DIR" "$CUDA_CACHE_PATH" \
  "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"

cd "$CODE"
"$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval freeze \
  --test-dataset "$DATA/test.jsonl" \
  --test-sha256 767c8cb0a7a3fb3c1d8aa008c7879f0497f53d466f320b25a3eeb5913aeb0110 \
  --blind-dataset "$DATA/blind.jsonl" \
  --blind-sha256 0f883b77e91fec6a77b1fc847d18fca5ee531c65515657579051c1b4b9523111 \
  --adapter "$ADAPTER" \
  --training-report "$REPORT" \
  --output-root "$EVAL/frozen"

"$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval infer \
  --freeze "$EVAL/frozen/FREEZE.json" \
  --model "$MODEL" \
  --model-inventory "$INVENTORY" \
  --model-inventory-sha256 "$INVENTORY_SHA" \
  --adapter "$ADAPTER" \
  --output-root "$EVAL/inference" \
  --batch-size 128 \
  --max-model-len 8192 \
  --max-tokens 192 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 16 \
  --seed 94137

"$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval score \
  --freeze "$EVAL/frozen/FREEZE.json" \
  --inference-meta "$EVAL/inference/INFERENCE_META.json" \
  --output "$EVAL/SCORE.json"

echo "COMPLETE $EVAL/SCORE.json"
