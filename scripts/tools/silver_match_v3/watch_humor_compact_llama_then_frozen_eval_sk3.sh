#!/usr/bin/env bash
set -euo pipefail

# Bind the one-shot held-out evaluation to the corrected compact sk3 run.
# Test/blind bytes are not opened until the selected adapter has passed the
# trainer's dev-only selection and fresh-base reload verification.

TRAIN_PID=2918555
ROOT=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1
CODE=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/repo_snapshot
RUN=$ROOT/runs/llama31-8b-typed-compact2048-decision-balanced-v1
ADAPTER=$RUN/adapter
REPORT=$RUN/TRAINING_REPORT.json
CHECKPOINT=$RUN/adapter.exposure-checkpoints/exposure_000000017472/checkpoint.json
EARLY=$ROOT/runs/llama31-8b-typed-compact2048-decision-balanced-v1-early17472-selected
EVAL=$ROOT/eval/decision_balanced_v1_selected_one_shot_v1
MODEL=$ROOT/model
INVENTORY=$ROOT/inputs/LLAMA_MODEL_INVENTORY.sk3.json
PYTHON=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
OVERLAY=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/final_training_runtime_v1/gemma/typed-final/peft_overlay
TRAIN=$ROOT/compact_dataset_decision_balanced_v1/train.jsonl
DEV=$ROOT/compact_dataset/dev.jsonl
DATA_REPORT=$ROOT/compact_dataset_decision_balanced_v1/BALANCE_REPORT.json
TRAINER=$ROOT/code/train_gemma4_typed_lora.py
TEST=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/final_stack_handoff_v1/gemma/dataset/test.jsonl
BLIND=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/final_stack_handoff_v1/gemma/dataset/blind.jsonl

verify_sha() {
  local expected=$1 path=$2
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected"
}

# These are training/model identities only.  Do not hash or otherwise open
# TEST or BLIND before the selected report exists.
verify_sha 2941f87528bf23c98961b863dae9777f81919b614e8385ea98844d5ef227b698 "$TRAIN"
verify_sha 7ea33c8229e740a739db4ce0ed2dfe554460ebd32ef2033c5096c685ec63ac34 "$DEV"
verify_sha 5654d2f60780a9fb833f56218ea118fe2338a063d04dbab903545367a49a35bc "$DATA_REPORT"
verify_sha 7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85 "$INVENTORY"
verify_sha ec7716f304088feacf7b35a61471270eb6895d183e5e3f37470bedd42fb2f997 "$TRAINER"
verify_sha ca118069fe5f83cea9d702c63561bc08b7f8d6982f7d6a8f2984a2a41a5a01c2 \
  "$CODE/scripts/tools/silver_match_v3/build_compact_typed_llama_dataset.py"
verify_sha c507085cf0aecab55d1004f75ac023b7dada66fd35fe0835c998435896cd8a50 \
  "$CODE/scripts/tools/silver_match_v3/run_typed_lora_frozen_eval.py"
verify_sha 52c6715ef302cdddd1c75726faaeb3cc67868c9fe392e9b4b8012e4ad3329679 \
  "$CODE/scripts/tools/silver_match_v3/finalize_typed_lora_checkpoint.py"

while [[ ! -f "$CHECKPOINT" && ! -f "$REPORT" ]]; do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "compact training exited without a final report" >&2
    exit 3
  fi
  sleep 5
done

if [[ -f "$CHECKPOINT" && ! -f "$REPORT" ]]; then
  kill -STOP "$TRAIN_PID"
  # The adapter and JSON are closed before publication.  Durably sync the
  # exact checkpoint tree while the trainer is stopped before gate reading.
  "$PYTHON" - "$CHECKPOINT" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1]).resolve()
value = json.load(open(p, encoding="utf-8"))
adapter = Path(value["adapter"]["path"])
for path in [p, *(child for child in adapter.rglob("*") if child.is_file())]:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
for directory in {p.parent, adapter}:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
  if "$PYTHON" - "$CHECKPOINT" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
g = x["confidence_gate"]
generation = x["generation"]
invalid = 1.0 - generation["valid_predictions"] / generation["rows"]
assert x["cumulative_exposure"] == 17472
assert g["precision_wilson_gate_met"] is True
assert g["exact_precision"] >= .90
assert g["exact_precision_wilson_95_lower"] >= .85
assert g["predicted_exact_count"] >= 100
assert invalid <= .01
PY
  then
    kill -TERM "$TRAIN_PID"
    kill -CONT "$TRAIN_PID"
    for _ in $(seq 1 60); do
      kill -0 "$TRAIN_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "trainer did not stop after passing early gate" >&2
      exit 4
    fi
    export PYTHONPATH=$OVERLAY${PYTHONPATH:+:$PYTHONPATH}
    export CUDA_VISIBLE_DEVICES=5
    cd "$CODE"
    "$PYTHON" -u -m scripts.tools.silver_match_v3.finalize_typed_lora_checkpoint \
      --checkpoint "$CHECKPOINT" \
      --model "$MODEL" \
      --model-inventory "$INVENTORY" \
      --model-inventory-sha256 7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85 \
      --dataset "$TRAIN" \
      --dev-dataset "$DEV" \
      --trainer "$TRAINER" \
      --output-root "$EARLY" \
      --max-length 2048
    ADAPTER=$EARLY/adapter
    REPORT=$EARLY/TRAINING_REPORT.json
  else
    kill -CONT "$TRAIN_PID"
    echo "EARLY_DEV_GATE_FAILED_HELDOUT_REMAINS_SEALED" >&2
    exit 10
  fi
fi

"$PYTHON" - "$REPORT" "$TRAIN" "$DEV" "$INVENTORY" "$TRAINER" <<'PY'
import hashlib, json, sys

report_path, train, dev, inventory, trainer = sys.argv[1:]
value = json.load(open(report_path, encoding="utf-8"))
assert value["status"] == "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED"
assert value["dataset"]["path"] == train
assert value["dataset"]["sha256"] == "2941f87528bf23c98961b863dae9777f81919b614e8385ea98844d5ef227b698"
assert value["dev_dataset"]["path"] == dev
assert value["dev_dataset"]["sha256"] == "7ea33c8229e740a739db4ce0ed2dfe554460ebd32ef2033c5096c685ec63ac34"
assert value["model_inventory"]["path"] == inventory
assert value["model_inventory"]["sha256"] == "7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85"
assert value["trainer_script"]["path"] == trainer
assert value["trainer_script"]["sha256"] == "ec7716f304088feacf7b35a61471270eb6895d183e5e3f37470bedd42fb2f997"
assert value["selection"]["status"] == "SELECTED_ON_DEV_ONLY"
assert value["selection"]["test_or_blind_data_read"] is False
assert value["adapter"]["inference_reload_verified"] is True
assert value["adapter"]["fresh_base_reload_verified"] is True

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

export PYTHONPATH=$OVERLAY${PYTHONPATH:+:$PYTHONPATH}
export HOME=/lfs/skampere3/0/alexspan
export XDG_CACHE_HOME=$ROOT/cache/eval/xdg
export TORCH_EXTENSIONS_DIR=$ROOT/cache/eval/torch_extensions
export TRITON_CACHE_DIR=$ROOT/cache/eval/triton
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/eval/flashinfer_workspace_base
export FLASHINFER_JIT_DIR=$ROOT/cache/eval/flashinfer_jit
export CUDA_CACHE_PATH=$ROOT/cache/eval/cuda
export VLLM_CACHE_ROOT=$ROOT/cache/eval/vllm
export TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/eval/torchinductor
export TMPDIR=$ROOT/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=5
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$FLASHINFER_JIT_DIR" "$CUDA_CACHE_PATH" \
  "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"

cd "$CODE"
"$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval freeze \
  --test-dataset "$TEST" \
  --test-sha256 767c8cb0a7a3fb3c1d8aa008c7879f0497f53d466f320b25a3eeb5913aeb0110 \
  --blind-dataset "$BLIND" \
  --blind-sha256 0f883b77e91fec6a77b1fc847d18fca5ee531c65515657579051c1b4b9523111 \
  --adapter "$ADAPTER" \
  --training-report "$REPORT" \
  --compact-prompt \
  --model "$MODEL" \
  --compact-projector-sha256 ca118069fe5f83cea9d702c63561bc08b7f8d6982f7d6a8f2984a2a41a5a01c2 \
  --output-root "$EVAL/frozen"

"$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval infer \
  --freeze "$EVAL/frozen/FREEZE.json" \
  --model "$MODEL" \
  --model-inventory "$INVENTORY" \
  --model-inventory-sha256 7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85 \
  --adapter "$ADAPTER" \
  --output-root "$EVAL/inference" \
  --batch-size 128 \
  --max-model-len 2048 \
  --max-tokens 192 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 16 \
  --seed 94137

"$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval score \
  --freeze "$EVAL/frozen/FREEZE.json" \
  --inference-meta "$EVAL/inference/INFERENCE_META.json" \
  --output "$EVAL/SCORE.json"

echo "COMPLETE $EVAL/SCORE.json"
