#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1
REPO=$ROOT/repo_snapshot
FINAL=$ROOT/final_stack_handoff_v1
DATA=$FINAL/ce_binary_recipe_v1
TRAIN=$DATA/binary.train.pairs.jsonl
TRAIN_REPORT=$DATA/REPORT.json
DEV=$DATA/binary.dev.pairs.jsonl
PYTHON=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
MODEL=/lfs/skampere3/0/alexspan/.cache/huggingface/hub/models--nvidia--llama-embed-nemotron-8b/snapshots/aa3b43a495a9b280d1bdb716da37c54bb495d630
RUNS=$ROOT/final_training_runtime_v1/ce_binary_recipe_v1
RECEIPTS=$ROOT/final_training_runtime_v1/ce_binary_recipe_v1_receipts
TRAINER=$REPO/scripts/tools/silver_match_v3/train_nemotron_cross_encoder.py

for required in "$PYTHON" "$MODEL/config.json" "$TRAIN" "$TRAIN_REPORT" "$DEV" "$TRAINER"; do
  test -e "$required"
done
"$PYTHON" - "$TRAIN_REPORT" "$TRAIN" "$DEV" <<'PY'
import hashlib, json, sys
r=json.load(open(sys.argv[1]))
assert r["recipe"] == {
    "allows_zero_one_or_multiple_exact_metrics_per_norm": True,
    "dev_test_sampling": "NONE_NATURAL_FULL_SLATES",
    "easy_global": 7000,
    "hard_family": 3500,
    "hard_retrieval_top10": 3500,
    "retain_all_train_exact": True,
}
assert r["output_counts"]["train_negative_provenance"] == {
    "easy_global": 7000, "hard_family": 3500, "hard_retrieval_top10": 3500,
}
for role, path in (("train",sys.argv[2]),("dev",sys.argv[3])):
    assert r["outputs"][role]["sha256"] == hashlib.sha256(open(path,"rb").read()).hexdigest()
PY

mkdir -p "$RUNS/logs" "$RECEIPTS"
cd "$REPO"

launch() {
  local gpu=$1 seed=$2
  local output=$RUNS/seed-$seed
  local log=$RUNS/logs/seed-$seed.log
  local receipt=$RECEIPTS/seed-$seed.launch.json
  test ! -e "$output"; test ! -e "$log"; test ! -e "$receipt"
  local used
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  test "$used" -le 128
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false nohup \
    "$PYTHON" -u -m scripts.tools.silver_match_v3.train_nemotron_cross_encoder \
    --train-pairs "$TRAIN" --dev-pairs "$DEV" --model "$MODEL" --output "$output" \
    --classification-mode binary --binary-positive-fraction 0.5 \
    --exposure-budget 100000 --exposure-budget 200000 --exposure-budget 400000 \
    --max-length 1024 --batch-size 8 --eval-batch-size 16 \
    --gradient-accumulation-steps 4 --lora-rank 32 --lora-alpha 64 \
    --lora-learning-rate 5e-5 --head-learning-rate 1e-3 \
    --lora-dropout 0.05 --weight-decay 0.01 --warmup-ratio 0.05 \
    --attention eager --seed "$seed" --min-exact-precision 0.90 \
    --min-wilson-lower 0.85 --min-exact-predictions 100 >"$log" 2>&1 &
  local pid=$!
  sleep 8
  kill -0 "$pid"; test -f "$output/events.jsonl"
  GPU="$gpu" SEED="$seed" PID="$pid" OUTPUT="$output" LOG="$log" \
    TRAIN="$TRAIN" DEV="$DEV" TRAINER="$TRAINER" RECEIPT="$receipt" \
    "$PYTHON" - <<'PY'
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
def ref(k):
 p=Path(os.environ[k]); return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size}
d={"schema_version":"silver-match-v3-humor-final-binary-ce-launch-v1","status":"LAUNCHED_STARTUP_VERIFIED","created_at":datetime.now(timezone.utc).isoformat(),"host":"sk3","physical_gpu":int(os.environ["GPU"]),"seed":int(os.environ["SEED"]),"pid":int(os.environ["PID"]),"output":os.environ["OUTPUT"],"log":os.environ["LOG"],"classification":{"positive":"EXACT","negative":["FAMILY","REJECT"],"positive_sampling_fraction":0.5},"selection_role":"dev_only","blind_or_test_tuning":False,"inputs":{"train":ref("TRAIN"),"dev":ref("DEV"),"trainer":ref("TRAINER")}}
p=Path(os.environ["RECEIPT"]); p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n"); print(json.dumps(d,sort_keys=True))
PY
}

case "${ONLY_SEED:-both}" in
  both) launch 5 2026071501; launch 6 2026071502 ;;
  2026071501) launch 5 2026071501 ;;
  2026071502) launch 6 2026071502 ;;
  *) echo "unsupported ONLY_SEED=${ONLY_SEED}" >&2; exit 2 ;;
esac
