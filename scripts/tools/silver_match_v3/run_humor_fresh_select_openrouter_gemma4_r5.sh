#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/Users/spangher/Projects/stanford-research/norm-research
readonly ROOT="$REPO/outputs/silver_match_v3/humor/verifier_select_expansion_v3/openrouter_gemma4_r5"
readonly PACK="$REPO/outputs/silver_match_v3/humor/verifier_select_expansion_v3/labelpacks/pass_a"
readonly MANIFEST="$ROOT/gpu_inputs/local_inference_manifest.json"
readonly OUTPUT="$ROOT/run"
readonly LOGS="$ROOT/logs"
readonly KEY="$HOME/.openrouter-api-key.txt"
readonly MODEL=google/gemma-4-31b-it
readonly API=https://openrouter.ai/api/v1

test -s "$KEY"
test "$(shasum -a 256 "$PACK/validation.json" | awk '{print $1}')" = f612f98199b08b24f5d10db64ed55fd7d64926f2e2b4d7d883ee1927d49d169e
test "$(shasum -a 256 "$PACK/candidates.top50.jsonl" | awk '{print $1}')" = e869c09fb558b70927fb926ba92fdf860840a51e92417e97eff6a7a365ffbd2b
test "$(shasum -a 256 "$MANIFEST" | awk '{print $1}')" = 27b9a8939506f7c1f4519dae75899bc704800e8f657b89f130c330935c17af07

mkdir -p "$OUTPUT/adjudicator" "$OUTPUT/verifier_r5" "$LOGS"
cd "$REPO"

run_adjudicator() {
  local order=$1
  python -u -m scripts.tools.silver_match_v3.adjudicate_gemma_api \
    --manifest "$MANIFEST" \
    --candidates "$PACK/candidates.top50.jsonl" \
    --output "$OUTPUT/adjudicator/$order.jsonl" \
    --split-role dev \
    --prompt "$REPO/scripts/tools/silver_match_v3/prompts/gepa_round1_candidate.txt" \
    --prompt-addon "$REPO/scripts/tools/silver_match_v3/prompts/gepa_humor_k50_r1_cleantrain.txt" \
    --api-base-url "$API" \
    --api-key-file "$KEY" \
    --max-api-requests 330 \
    --model "$MODEL" \
    --max-candidates 50 \
    --context-chars 1200 \
    --description-chars 260 \
    --example-chars 80 \
    --max-examples 0 \
    --batch-size 32 \
    --concurrency 6 \
    --max-tokens 160 \
    --seed 17 \
    --request-timeout 180 \
    --transport-retries 0 \
    --order-mode "$order" \
    --resume \
    >"$LOGS/adjudicator.$order.log" 2>&1
}

run_adjudicator original &
adj_original_pid=$!
run_adjudicator hashed &
adj_hashed_pid=$!
wait "$adj_original_pid"
wait "$adj_hashed_pid"

python -u -m scripts.tools.silver_match_v3.build_two_order_consensus_proposals \
  --original "$OUTPUT/adjudicator/original.jsonl" \
  --hashed "$OUTPUT/adjudicator/hashed.jsonl" \
  --task humor \
  --output "$OUTPUT/adjudicator/exact_consensus.proposals.jsonl" \
  >"$LOGS/adjudicator.consensus.log" 2>&1

run_verifier() {
  local order=$1
  python -u -m scripts.tools.silver_match_v3.verify_gemma_api \
    --manifest "$MANIFEST" \
    --candidates "$PACK/candidates.top50.jsonl" \
    --primary "$OUTPUT/adjudicator/exact_consensus.proposals.jsonl" \
    --output "$OUTPUT/verifier_r5/$order.jsonl" \
    --split-role dev \
    --prompt "$REPO/scripts/tools/silver_match_v3/prompts/verify_match_v1.txt" \
    --prompt-addon "$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r3_exact_object.txt" \
    --prompt-addon "$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r4_speech_act_and_audio_owner.txt" \
    --prompt-addon "$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r5_criterion_nucleus.txt" \
    --api-base-url "$API" \
    --api-key-file "$KEY" \
    --max-api-requests 330 \
    --model "$MODEL" \
    --max-alternatives 49 \
    --batch-size 32 \
    --concurrency 4 \
    --max-model-len 8192 \
    --max-tokens 180 \
    --seed 29 \
    --request-timeout 180 \
    --transport-retries 0 \
    --order-mode "$order" \
    --context-chars 1200 \
    --description-chars 260 \
    --example-chars 80 \
    --max-examples 0 \
    --resume \
    >"$LOGS/verifier_r5.$order.log" 2>&1
}

run_verifier original &
ver_original_pid=$!
run_verifier hashed &
ver_hashed_pid=$!
run_verifier reverse &
ver_reverse_pid=$!
wait "$ver_original_pid"
wait "$ver_hashed_pid"
wait "$ver_reverse_pid"
