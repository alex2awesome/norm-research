#!/usr/bin/env bash
set -euo pipefail

# One-shot execution of the predeclared PR R4 select plan.  This script may be
# rerun only while neither output exists; partial outputs fail closed and must
# be audited rather than silently resumed.
ROOT=outputs/silver_match_v3/press-releases/gepa_clean_v2
PLAN="$ROOT/select_gepa_r4_v1/PLAN.json"
MANIFEST="$ROOT/select_gepa_r4_inputs_v1/local_inference_manifest.json"
PACK="$ROOT/label_workspaces/select_pass_a"
CANDIDATES="$PACK/candidates.frozen-k50.jsonl"
PROMPT="$ROOT/optimize_gepa_v1/r4_fresh_agent_handoff_v1/author_v4/prompt.frozen.txt"
RUN_ROOT="$ROOT/select_gepa_r4_v1/runs"
RUNNER=scripts/tools/silver_match_v3/adjudicate_gemma_api.py
PARSER=scripts/tools/silver_match_v3/adjudicate_gemma.py

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

verify_sha "$PLAN" 80f8633739282d2a21175073e732cc786f00200e4eb2e35ce0da675751820080
verify_sha "$MANIFEST" ef48d08d4a8279d68ec8fb8b0269176cd2cc23a60d23040cb2a193dc5a7ddcf4
verify_sha "$CANDIDATES" 84e76f9b3afe65ab6d9347a28ece4205d2631564690c48f19ba307e0c26f80b0
verify_sha "$PACK/items.jsonl" a78832eafd939f534e8ecd58a8c771e9d3f7a436dc16372bb28cd9362967e1c5
verify_sha "$PACK/bank.json" 48f9075af972d6913a889735ddf606977363d836a30f28584917396dfa1ffa0b
verify_sha "$PACK/validation.json" 7fb8c2adff16f4e54624aacf3bd7a6527dfc535c9a31759a0a7801f3ab76d23f
verify_sha "$PROMPT" 664f16dc6f459531fd1bcec98cd06130625ea28da6d4da2e042eb67e8db7d9c7
verify_sha "$RUNNER" 3ccb54d2e861725c991a96566c09e0d1c5ec0f0b94ccc7de78d7dbb17bca08a2
verify_sha "$PARSER" 626764c142e6be0d48707cdc926bbae99683276c83f92904fd2220bb8fcea2fd

for order in original hashed; do
  [[ ! -e "$RUN_ROOT/$order.jsonl" && ! -e "$RUN_ROOT/$order.jsonl.meta.json" ]] || {
    echo "refusing to overwrite or resume one-shot R4 select output: $order" >&2
    exit 2
  }
done

run_order() {
  local order="$1"
  python -u -m scripts.tools.silver_match_v3.adjudicate_gemma_api \
    --manifest "$MANIFEST" \
    --candidates "$CANDIDATES" \
    --output "$RUN_ROOT/$order.jsonl" \
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
    --order-mode "$order" \
    --keep-raw
}

run_order original >"$RUN_ROOT/original.log" 2>&1 &
original_pid=$!
run_order hashed >"$RUN_ROOT/hashed.log" 2>&1 &
hashed_pid=$!

status=0
wait "$original_pid" || status=1
wait "$hashed_pid" || status=1
exit "$status"
