#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/Users/spangher/Projects/stanford-research/norm-research
readonly ROOT="$REPO/outputs/silver_match_v3/humor/verifier_select_expansion_v3/openrouter_gemma4_r5"
readonly PACK="$REPO/outputs/silver_match_v3/humor/verifier_select_expansion_v3/labelpacks/pass_a"
readonly MANIFEST="$ROOT/gpu_inputs/local_inference_manifest.json"
readonly PRIMARY="$ROOT/run/adjudicator/exact_consensus.proposals.jsonl"
readonly OUTPUT="$ROOT/run/verifier_r5/reverse.jsonl"
readonly LOG="$ROOT/logs/verifier_r5.reverse.resume1.log"
readonly KEY="$HOME/.openrouter-api-key.txt"

test -s "$KEY"
test "$(shasum -a 256 "$PACK/candidates.top50.jsonl" | awk '{print $1}')" = e869c09fb558b70927fb926ba92fdf860840a51e92417e97eff6a7a365ffbd2b
test "$(shasum -a 256 "$MANIFEST" | awk '{print $1}')" = 27b9a8939506f7c1f4519dae75899bc704800e8f657b89f130c330935c17af07
test "$(shasum -a 256 "$PRIMARY" | awk '{print $1}')" = 51dfb8af972b4566b24f013a01646f45f6357878cc3b6368385278cb6ed8d378
test "$(wc -l < "$OUTPUT" | tr -d ' ')" = 96

cd "$REPO"
python -u -m scripts.tools.silver_match_v3.verify_gemma_api \
  --manifest "$MANIFEST" \
  --candidates "$PACK/candidates.top50.jsonl" \
  --primary "$PRIMARY" \
  --output "$OUTPUT" \
  --split-role dev \
  --prompt "$REPO/scripts/tools/silver_match_v3/prompts/verify_match_v1.txt" \
  --prompt-addon "$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r3_exact_object.txt" \
  --prompt-addon "$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r4_speech_act_and_audio_owner.txt" \
  --prompt-addon "$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r5_criterion_nucleus.txt" \
  --api-base-url https://openrouter.ai/api/v1 \
  --api-key-file "$KEY" \
  --max-api-requests 80 \
  --model google/gemma-4-31b-it \
  --max-alternatives 49 \
  --batch-size 32 \
  --concurrency 4 \
  --max-model-len 8192 \
  --max-tokens 180 \
  --seed 29 \
  --request-timeout 180 \
  --transport-retries 0 \
  --order-mode reverse \
  --context-chars 1200 \
  --description-chars 260 \
  --example-chars 80 \
  --max-examples 0 \
  --resume \
  >"$LOG" 2>&1
