#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_binary_v1
OLD=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2
CODE=$OLD/code
DATA=$OLD/data/existing_truth_compact400k_v2
PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python
MODEL=/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1
TRAIN=$CODE/outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/binary_ce_old_recipe_v1/binary.train.pairs.jsonl
SUBSET_REPORT=$CODE/outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/binary_ce_old_recipe_v1/REPORT.json
DEV=$CODE/outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/binary_ce_old_recipe_v1/binary.dev.pairs.jsonl
OUTPUT=$ROOT/runs/old-recipe-v1.binary.seed-2026071500
LOG=$ROOT/logs/old-recipe-v1.binary.seed-2026071500.log
RECEIPT=$ROOT/receipts/old-recipe-v1.binary.seed-2026071500.launch.json
GPU=0

for required in "$PYTHON" "$MODEL/config.json" "$TRAIN" "$DEV" \
  "$SUBSET_REPORT" "$CODE/scripts/tools/silver_match_v3/train_nemotron_cross_encoder.py"; do
  test -e "$required"
done
test ! -e "$OUTPUT"
test ! -e "$LOG"
test ! -e "$RECEIPT"
test "$(sha256sum "$TRAIN" | cut -d' ' -f1)" = 944dcd073e388473cbf36414a87bc57a91a37d9ad994ddc7a8e13365df795e1f
test "$(sha256sum "$DEV" | cut -d' ' -f1)" = f5d558c96d22cd6a991c65c26ac33df177fa811a3fa76c6744ebf8d9c33bb71b

used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
test "$used" -le 128

mkdir -p "$ROOT/logs" "$ROOT/runs" "$ROOT/receipts"
command=(
  "$PYTHON" -u -m scripts.tools.silver_match_v3.train_nemotron_cross_encoder
  --train-pairs "$TRAIN"
  --dev-pairs "$DEV"
  --model "$MODEL"
  --output "$OUTPUT"
  --classification-mode binary
  --binary-positive-fraction 0.5
  --exposure-budget 25000
  --exposure-budget 50000
  --exposure-budget 100000
  --exposure-budget 200000
  --exposure-budget 400000
  --max-length 1024
  --batch-size 8
  --eval-batch-size 16
  --gradient-accumulation-steps 4
  --lora-rank 32
  --lora-alpha 64
  --lora-learning-rate 5e-5
  --head-learning-rate 1e-3
  --lora-dropout 0.05
  --weight-decay 0.01
  --warmup-ratio 0.05
  --attention eager
  --seed 2026071500
  --min-exact-precision 0.90
  --min-wilson-lower 0.85
  --min-exact-predictions 100
)

cd "$CODE"
CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false \
  nohup "${command[@]}" >"$LOG" 2>&1 &
pid=$!

sleep 8
kill -0 "$pid"
test -f "$OUTPUT/events.jsonl"

export RECEIPT PID="$pid" GPU OUTPUT LOG TRAIN DEV MODEL PYTHON CODE SUBSET_REPORT
export TRAINER="$CODE/scripts/tools/silver_match_v3/train_nemotron_cross_encoder.py"
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def ref(name):
    path = Path(os.environ[name])
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}

payload = {
    "schema_version": "silver-match-v3-humor-binary-ce-launch-v1",
    "status": "LAUNCHED_STARTUP_VERIFIED",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "host": "sk2",
    "physical_gpu": int(os.environ["GPU"]),
    "pid": int(os.environ["PID"]),
    "classification": {"mode": "binary", "positive": "EXACT", "negative": ["FAMILY", "REJECT"], "positive_sampling_fraction": 0.5},
    "selection_role": "dev_only",
    "blind_or_test_tuning": False,
    "output": os.environ["OUTPUT"],
    "log": os.environ["LOG"],
    "inputs": {"train": ref("TRAIN"), "train_subset_report": ref("SUBSET_REPORT"), "dev": ref("DEV"), "trainer": ref("TRAINER")},
    "model": os.environ["MODEL"],
}
path = Path(os.environ["RECEIPT"])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
