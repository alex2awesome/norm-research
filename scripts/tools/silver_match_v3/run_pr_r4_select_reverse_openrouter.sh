#!/usr/bin/env bash
set -euo pipefail

# Execution-only completion of the globally predeclared reverse order.  The
# prompt, model, rendering, candidates, and per-order cap are byte-identical to
# original/hashed and were frozen before any R4 select score was opened.
ROOT=outputs/silver_match_v3/press-releases/gepa_clean_v2
PLAN="$ROOT/select_gepa_r4_v1/PLAN_SUPERSESSION_V2.json"
MANIFEST="$ROOT/select_gepa_r4_inputs_v1/local_inference_manifest.json"
PACK="$ROOT/label_workspaces/select_pass_a"
CANDIDATES="$PACK/candidates.frozen-k50.jsonl"
PROMPT="$ROOT/optimize_gepa_v1/r4_fresh_agent_handoff_v1/author_v4/prompt.frozen.txt"
OUTPUT="$ROOT/select_gepa_r4_v1/runs/reverse.jsonl"
LOG="$ROOT/select_gepa_r4_v1/runs/reverse.log"

verify_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(shasum -a 256 "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "hash mismatch: $path expected=$expected actual=$actual" >&2
    exit 2
  }
}

verify_sha "$PLAN" e8ffe5406c57e4b10e5c92eb52ac507f7065a6ffa674394ecab6167eacdc5b49
verify_sha "$MANIFEST" ef48d08d4a8279d68ec8fb8b0269176cd2cc23a60d23040cb2a193dc5a7ddcf4
verify_sha "$CANDIDATES" 84e76f9b3afe65ab6d9347a28ece4205d2631564690c48f19ba307e0c26f80b0
verify_sha "$PROMPT" 664f16dc6f459531fd1bcec98cd06130625ea28da6d4da2e042eb67e8db7d9c7
verify_sha scripts/tools/silver_match_v3/adjudicate_gemma_api.py 3ccb54d2e861725c991a96566c09e0d1c5ec0f0b94ccc7de78d7dbb17bca08a2
verify_sha scripts/tools/silver_match_v3/adjudicate_gemma.py 626764c142e6be0d48707cdc926bbae99683276c83f92904fd2220bb8fcea2fd
[[ ! -e "$OUTPUT" && ! -e "$OUTPUT.meta.json" ]] || {
  echo "refusing to overwrite or resume one-shot reverse output" >&2
  exit 2
}

python -u -m scripts.tools.silver_match_v3.adjudicate_gemma_api \
  --manifest "$MANIFEST" \
  --candidates "$CANDIDATES" \
  --output "$OUTPUT" \
  --split-role dev \
  --prompt "$PROMPT" \
  --api-base-url https://openrouter.ai/api/v1 \
  --api-key-file ~/.openrouter-api-key.txt \
  --max-api-requests 250 \
  --model google/gemma-4-31b-it \
  --max-candidates 50 \
  --context-chars 1200 \
  --description-chars 260 \
  --example-chars 80 \
  --max-examples 0 \
  --batch-size 64 \
  --concurrency 8 \
  --max-tokens 220 \
  --seed 17 \
  --request-timeout 180 \
  --transport-retries 0 \
  --order-mode reverse \
  --keep-raw >"$LOG" 2>&1
