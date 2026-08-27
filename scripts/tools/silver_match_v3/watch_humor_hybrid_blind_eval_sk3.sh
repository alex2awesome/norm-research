#!/usr/bin/env bash
set -euo pipefail

CODE=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/repo_snapshot
ROOT=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1
RUN=$ROOT/blind_hybrid_v1/strict_exp34944_v2
FREEZE=$RUN/freeze/FREEZE.json
TYPED=$RUN/typed_inference
CE=$RUN/ce_inference
SCORE=$RUN/SCORE.json
DEVCE=$ROOT/dev_hybrid_v1/nemotron_dev_stage_v2
PY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
OVERLAY=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/final_training_runtime_v1/gemma/typed-final/peft_overlay

test "$(sha256sum "$CODE/scripts/tools/silver_match_v3/run_hybrid_ce_typed_blind_eval.py" | cut -d' ' -f1)" = \
  926cf8394e27afa1ad12c5175cbe264b003793b28896dc2c66442b32fa495940
test "$(sha256sum "$FREEZE" | cut -d' ' -f1)" = \
  8378551ea0461cb048d6604e22530c859ec2bccf24deede2f8fa2658e487146c
test ! -e "$TYPED"
test ! -e "$CE"
test ! -e "$SCORE"
mkdir -p "$CE" "$RUN/cache/typed" "$RUN/cache/ce" "$RUN/tmp"

export PYTHONPATH=$OVERLAY${PYTHONPATH:+:$PYTHONPATH}
export HOME=/lfs/skampere3/0/alexspan
cd "$CODE"

(
  export CUDA_VISIBLE_DEVICES=7
  export XDG_CACHE_HOME=$RUN/cache/typed/xdg
  export TORCH_EXTENSIONS_DIR=$RUN/cache/typed/torch_extensions
  export TRITON_CACHE_DIR=$RUN/cache/typed/triton
  export VLLM_CACHE_ROOT=$RUN/cache/typed/vllm
  export TORCHINDUCTOR_CACHE_DIR=$RUN/cache/typed/torchinductor
  export TMPDIR=$RUN/tmp
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_USE_FLASHINFER_MOE_FP8=0
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
  exec "$PY" -u -m scripts.tools.silver_match_v3.run_hybrid_ce_typed_blind_eval infer-typed \
    --freeze "$FREEZE" \
    --model "$ROOT/model" \
    --model-inventory "$ROOT/inputs/LLAMA_MODEL_INVENTORY.sk3.json" \
    --model-inventory-sha256 7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85 \
    --adapter "$ROOT/runs/llama31-8b-typed-compact2048-decision-balanced-v1/adapter.exposure-checkpoints/exposure_000000034944/adapter" \
    --output-root "$TYPED" --batch-size 128 --max-model-len 2048 \
    --max-tokens 192 --gpu-memory-utilization 0.88 --max-lora-rank 16 --seed 94137
) >"$RUN/typed_inference.log" 2>&1 &
TYPED_PID=$!
echo "$TYPED_PID" >"$RUN/typed_inference.pid"

(
  export CUDA_VISIBLE_DEVICES=5
  export XDG_CACHE_HOME=$RUN/cache/ce/xdg
  export TORCH_EXTENSIONS_DIR=$RUN/cache/ce/torch_extensions
  export TRITON_CACHE_DIR=$RUN/cache/ce/triton
  export TORCHINDUCTOR_CACHE_DIR=$RUN/cache/ce/torchinductor
  export TMPDIR=$RUN/tmp
  mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR"
  exec "$PY" -u -m scripts.tools.silver_match_v3.run_nemotron_ce score \
    --input-pairs "$RUN/freeze/ce.pairs.truth_blind.jsonl" \
    --output "$CE/blind.scores.jsonl" \
    --model "$DEVCE/source/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1" \
    --base-manifest "$DEVCE/BASE_MODEL_MANIFEST.relocated.json" \
    --base-manifest-sha256 d1a13c104772dbf82cf95c08fc52dd88f93e9a48284aa5d8ba81f1c52ae406c8 \
    --checkpoint "$DEVCE/source/runtime/humor_ce_binary_v1/runs/final-joined-recipe-v1/seed-2026071502/checkpoints/exposure-000000100000" \
    --training-report "$DEVCE/training_report.relocated.json" \
    --training-report-sha256 31be7932392295fbb909c2dee0730f210165942fc884211916a7d3a6428b6c59 \
    --batch-size 8 --max-length 1024 --device 0 --attention eager
) >"$RUN/ce_inference.log" 2>&1 &
CE_PID=$!
echo "$CE_PID" >"$RUN/ce_inference.pid"

set +e
wait "$TYPED_PID"; TYPED_STATUS=$?
wait "$CE_PID"; CE_STATUS=$?
set -e
if [[ "$TYPED_STATUS" -ne 0 || "$CE_STATUS" -ne 0 ]]; then
  echo "INFERENCE_FAILED typed=$TYPED_STATUS ce=$CE_STATUS" >&2
  exit 10
fi
test -f "$TYPED/INFERENCE_META.json"
test -f "$CE/blind.scores.jsonl.meta.json"

# Only this command receives the blind gold-bearing source, after both inference
# streams have exited successfully and their metadata files are sealed.
"$PY" -u -m scripts.tools.silver_match_v3.run_hybrid_ce_typed_blind_eval score \
  --freeze "$FREEZE" \
  --typed-meta "$TYPED/INFERENCE_META.json" \
  --ce-scores "$CE/blind.scores.jsonl" \
  --ce-meta "$CE/blind.scores.jsonl.meta.json" \
  --blind-gold-source /lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/final_stack_handoff_v1/gemma/dataset/blind.jsonl \
  --output "$SCORE"

echo "COMPLETE $SCORE"
