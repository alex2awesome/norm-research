#!/usr/bin/env bash
set -euo pipefail

# Launch the two predeclared Humor Nemotron CE pilot recipes on independent
# sk2 H200s.  The pair release is staged separately and is immutable.
ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2
CODE="$ROOT/code"
DATA="$ROOT/data/existing_truth_compact400k_v2"
RUNS="$ROOT/runs"
PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python
MODEL=/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1
TRAIN="$DATA/existing_truth.compact400k.v2.train.pairs.jsonl"
DEV="$DATA/existing_truth.compact400k.v2.dev.pairs.jsonl"

for required in "$PYTHON" "$TRAIN" "$DEV" "$MODEL/config.json"; do
  test -e "$required"
done
mkdir -p "$RUNS/logs"

launch() {
  local gpu="$1"
  local name="$2"
  local rank="$3"
  local alpha="$4"
  local learning_rate="$5"
  local output="$RUNS/$name"
  local log="$RUNS/logs/$name.log"
  test ! -e "$output"
  test ! -e "$log"
  (
    cd "$CODE"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -u -m \
      scripts.tools.silver_match_v3.train_nemotron_cross_encoder \
      --train-pairs "$TRAIN" \
      --dev-pairs "$DEV" \
      --model "$MODEL" \
      --output "$output" \
      --exposure-budget 10000 \
      --exposure-budget 25000 \
      --exposure-budget 50000 \
      --max-length 1024 \
      --batch-size 8 \
      --eval-batch-size 16 \
      --gradient-accumulation-steps 4 \
      --lora-rank "$rank" \
      --lora-alpha "$alpha" \
      --lora-learning-rate "$learning_rate" \
      --head-learning-rate 1e-3 \
      --lora-dropout 0.05 \
      --weight-decay 0.01 \
      --warmup-ratio 0.05 \
      --attention eager \
      --seed 20260713 \
      --min-exact-precision 0.90 \
      --min-wilson-lower 0.85 \
      --min-exact-predictions 100 \
      >"$log" 2>&1 &
    echo "$name gpu=$gpu pid=$! log=$log"
  )
}

launch 2 humor_ce_r16_a32_lr1e4_seed20260713_v2 16 32 1e-4
launch 3 humor_ce_r32_a64_lr5e5_seed20260713_v2 32 64 5e-5
