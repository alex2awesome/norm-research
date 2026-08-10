#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_binary_v1
OLD=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2
CODE=$OLD/code
PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python
MODEL=/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1
RUN=$ROOT/runs/old-recipe-v1.quick25k.seed-2026071599
REPORT=$RUN/training_report.json
TEST=$ROOT/test_inputs_old_recipe_v1/binary.test.pairs.jsonl
TRUTH=$OLD/data/existing_truth_compact400k_v2/truth.canonical.pair-eligible.v2.test-only.jsonl
BASE=$OLD/runs/pilot_test_release_v1/BASE_MODEL_MANIFEST.json
BASE_SHA=4047fcc5c148a8522fd5a783dde7c68076b1a2548d5873cd58abd18feaf9577b
OUT=$ROOT/test_evaluation/old-recipe-v1.quick25k.seed-2026071599
SCORES=$OUT/test.scores.jsonl
EVAL=$OUT/TEST_EVALUATION.json

test ! -e "$OUT"
mkdir -p "$OUT"
while test ! -f "$REPORT"; do
  if ! kill -0 1519606 2>/dev/null; then
    echo "quick25k trainer exited without report" >&2
    exit 3
  fi
  sleep 30
done

# The held-out input/truth are intentionally not opened until the dev-frozen
# training report exists.
test "$(sha256sum "$TEST" | cut -d' ' -f1)" = 63aadebc39a0fd405aaafc6a67aadc78a0942e55a0f1345fdcb390a42e6ab8ab
test "$(sha256sum "$BASE" | cut -d' ' -f1)" = "$BASE_SHA"
test -f "$TRUTH"
REPORT_SHA=$(sha256sum "$REPORT" | cut -d' ' -f1)
CKPT=$(jq -r '.selected_checkpoint.path' "$REPORT")
MAX_LENGTH=$(jq -r '.max_sequence_length // 1024' "$REPORT")
test -d "$CKPT"
test "$(jq -r '.classification_mode' "$REPORT")" = binary
used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
test "$used" -le 128

cd "$CODE"
CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false "$PYTHON" -u -m \
  scripts.tools.silver_match_v3.run_nemotron_ce score \
  --input-pairs "$TEST" --output "$SCORES" --model "$MODEL" \
  --base-manifest "$BASE" --base-manifest-sha256 "$BASE_SHA" \
  --checkpoint "$CKPT" --training-report "$REPORT" \
  --training-report-sha256 "$REPORT_SHA" --batch-size 16 \
  --max-length "$MAX_LENGTH" --device 0

"$PYTHON" -u -m scripts.tools.silver_match_v3.run_nemotron_ce evaluate \
  --scores "$SCORES" --truth "$TRUTH" --output "$EVAL"
jq -c '.' "$EVAL"
