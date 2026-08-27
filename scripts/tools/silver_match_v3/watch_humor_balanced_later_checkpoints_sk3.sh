#!/usr/bin/env bash
set -euo pipefail

# Sole fail-closed watcher for the already-running decision-balanced Humor run.
# The heldout paths are neither opened nor hashed until a dev checkpoint passes
# the complete predeclared precision gate and a fresh-base reload is verified.

TRAIN_PID=2918555
ROOT=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1
CODE=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/repo_snapshot
RUN=$ROOT/runs/llama31-8b-typed-compact2048-decision-balanced-v1
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
C2=$RUN/adapter.exposure-checkpoints/exposure_000000034944/checkpoint.json
C3=$RUN/adapter.exposure-checkpoints/exposure_000000069888/checkpoint.json
LOG=$ROOT/watch_balanced_later_checkpoints_v1.log

verify_sha() {
  local expected=$1 path=$2
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected"
}

# Training/model/code identities only: never inspect TEST/BLIND here.
verify_sha 2941f87528bf23c98961b863dae9777f81919b614e8385ea98844d5ef227b698 "$TRAIN"
verify_sha 7ea33c8229e740a739db4ce0ed2dfe554460ebd32ef2033c5096c685ec63ac34 "$DEV"
verify_sha 5654d2f60780a9fb833f56218ea118fe2338a063d04dbab903545367a49a35bc "$DATA_REPORT"
verify_sha 7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85 "$INVENTORY"
verify_sha ec7716f304088feacf7b35a61471270eb6895d183e5e3f37470bedd42fb2f997 "$TRAINER"
verify_sha ca118069fe5f83cea9d702c63561bc08b7f8d6982f7d6a8f2984a2a41a5a01c2 "$CODE/scripts/tools/silver_match_v3/build_compact_typed_llama_dataset.py"
verify_sha c507085cf0aecab55d1004f75ac023b7dada66fd35fe0835c998435896cd8a50 "$CODE/scripts/tools/silver_match_v3/run_typed_lora_frozen_eval.py"
verify_sha 1ff643ffb2d297533e2aa8921ec78fbf5fced8946cb1877ffd2df699fe944ee9 "$CODE/scripts/tools/silver_match_v3/finalize_typed_lora_checkpoint.py"

fsync_checkpoint() {
  local checkpoint=$1
  "$PYTHON" - "$checkpoint" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1]).resolve()
value = json.load(open(p, encoding="utf-8"))
adapter = Path(value["adapter"]["path"])
for path in [p, *(x for x in adapter.rglob("*") if x.is_file())]:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
for directory in {p.parent, adapter}:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
PY
}

stop_trainer() {
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
  kill -CONT "$TRAIN_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$TRAIN_PID" 2>/dev/null || return 0
    sleep 1
  done
  echo "TRAINER_DID_NOT_STOP pid=$TRAIN_PID" >&2
  return 1
}

metrics() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
g = x["confidence_gate"]
z = x["generation"]
print(json.dumps({
    "exposure": x["cumulative_exposure"],
    "exact_precision": g["exact_precision"],
    "wilson95_lower": g["exact_precision_wilson_95_lower"],
    "exact_recall": g["exact_recall"],
    "f0_5": g["exact_f_beta_0_5"],
    "accepted": g["predicted_exact_count"],
    "invalid_rate": 1.0 - z["valid_predictions"] / z["rows"],
}, sort_keys=True))
PY
}

full_gate() {
  local checkpoint=$1 exposure=$2
  "$PYTHON" - "$checkpoint" "$exposure" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
g = x["confidence_gate"]
z = x["generation"]
assert x["cumulative_exposure"] == int(sys.argv[2])
assert g["precision_wilson_gate_met"] is True
assert g["exact_precision"] >= .90
assert g["exact_precision_wilson_95_lower"] >= .85
assert g["predicted_exact_count"] >= 100
assert 1.0 - z["valid_predictions"] / z["rows"] <= .01
PY
}

precision_ge_half() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if x["confidence_gate"]["exact_precision"] >= .50 else 1)
PY
}

wait_for_checkpoint() {
  local checkpoint=$1
  while [[ ! -f "$checkpoint" ]]; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "TRAINER_EXITED_BEFORE_CHECKPOINT checkpoint=$checkpoint" >&2
      exit 3
    fi
    sleep 10
  done
  kill -STOP "$TRAIN_PID"
  fsync_checkpoint "$checkpoint"
  metrics "$checkpoint"
}

select_and_evaluate() {
  local checkpoint=$1 exposure=$2
  local selected=$ROOT/runs/llama31-8b-typed-compact2048-decision-balanced-v1-exposure${exposure}-selected
  local eval_root=$ROOT/eval/decision_balanced_v1_exposure${exposure}_selected_one_shot_v1
  stop_trainer
  export PYTHONPATH=$OVERLAY${PYTHONPATH:+:$PYTHONPATH}
  export CUDA_VISIBLE_DEVICES=5
  cd "$CODE"
  "$PYTHON" -u -m scripts.tools.silver_match_v3.finalize_typed_lora_checkpoint \
    --checkpoint "$checkpoint" --expected-exposure "$exposure" \
    --model "$MODEL" --model-inventory "$INVENTORY" \
    --model-inventory-sha256 7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85 \
    --dataset "$TRAIN" --dev-dataset "$DEV" --trainer "$TRAINER" \
    --output-root "$selected" --max-length 2048

  local adapter=$selected/adapter report=$selected/TRAINING_REPORT.json
  test ! -e "$eval_root"
  mkdir -p "$eval_root"
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
  mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
    "$FLASHINFER_WORKSPACE_BASE" "$FLASHINFER_JIT_DIR" "$CUDA_CACHE_PATH" \
    "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"

  # This is the first point at which sealed heldout bytes may be opened.
  "$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval freeze \
    --test-dataset "$TEST" --test-sha256 767c8cb0a7a3fb3c1d8aa008c7879f0497f53d466f320b25a3eeb5913aeb0110 \
    --blind-dataset "$BLIND" --blind-sha256 0f883b77e91fec6a77b1fc847d18fca5ee531c65515657579051c1b4b9523111 \
    --adapter "$adapter" --training-report "$report" --compact-prompt \
    --model "$MODEL" --compact-projector-sha256 ca118069fe5f83cea9d702c63561bc08b7f8d6982f7d6a8f2984a2a41a5a01c2 \
    --output-root "$eval_root/frozen"
  "$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval infer \
    --freeze "$eval_root/frozen/FREEZE.json" --model "$MODEL" \
    --model-inventory "$INVENTORY" \
    --model-inventory-sha256 7e7ae4ec5611ce96b7237d867da62f46f74db638718f87291391dae9b4097d85 \
    --adapter "$adapter" --output-root "$eval_root/inference" --batch-size 128 \
    --max-model-len 2048 --max-tokens 192 --gpu-memory-utilization 0.88 \
    --max-lora-rank 16 --seed 94137
  "$PYTHON" -u -m scripts.tools.silver_match_v3.run_typed_lora_frozen_eval score \
    --freeze "$eval_root/frozen/FREEZE.json" \
    --inference-meta "$eval_root/inference/INFERENCE_META.json" \
    --output "$eval_root/SCORE.json"
  echo "COMPLETE score=$eval_root/SCORE.json"
}

wait_for_checkpoint "$C2"
if ! precision_ge_half "$C2"; then
  stop_trainer
  echo "STOPPED_AT_34944_EXACT_PRECISION_BELOW_0.50_HELDOUT_SEALED"
  exit 20
fi
if full_gate "$C2" 34944; then
  select_and_evaluate "$C2" 34944
  exit 0
fi

echo "CONTINUE_TO_69888_EXACT_PRECISION_AT_LEAST_0.50_FULL_GATE_NOT_MET"
kill -CONT "$TRAIN_PID"
wait_for_checkpoint "$C3"
if full_gate "$C3" 69888; then
  select_and_evaluate "$C3" 69888
  exit 0
fi
stop_trainer
echo "STOPPED_AT_69888_FULL_GATE_FAILED_HELDOUT_SEALED"
exit 21
