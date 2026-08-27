#!/usr/bin/env bash
set -euo pipefail

ROOT=outputs/silver_match_v3/press-releases/gepa_clean_v2/optimize_gepa_v1
LOCK="$ROOT/ROUND0_R3_INFERENCE_LOCK.json"
MANIFEST="$ROOT/local_inference_manifest.json"
CANDIDATES="$ROOT/optimize.candidates.frozen-k50.jsonl"
BASE=scripts/tools/silver_match_v3/prompts/gepa_round2_candidate.txt
ADDON=scripts/tools/silver_match_v3/prompts/gepa_press_releases_k50_r3_consolidated_cleantrain.txt
RUNNER=scripts/tools/silver_match_v3/adjudicate_gemma_api.py

verify_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "hash mismatch: $path expected=$expected observed=$observed" >&2
    exit 2
  fi
}

verify_sha "$LOCK" 7af5a1b120df196c7fe4522f7cc28660b818c3a9f742bc5c810fd3dc74d75a12
verify_sha "$MANIFEST" 46cb4389936887c76049735dafafa97188cfbd98cf1ac610e9111fa3f6881490
verify_sha "$CANDIDATES" 03a08adbb822be340d36c7fe0047b383fc770a517ee3be05eb53cae756acc08f
verify_sha "$BASE" 068ae4f55fe74375bd062108d978b124066df006834b8b78b6d1dafaa4d74056
verify_sha "$ADDON" 3a8cd2c94559c22f639fe9a207e0ebba029f66508d019b6eddd14c54c04b48e9
verify_sha "$RUNNER" 3ccb54d2e861725c991a96566c09e0d1c5ec0f0b94ccc7de78d7dbb17bca08a2

mkdir -p "$ROOT/runs/r3"
for ORDER in original hashed; do
  python3 -u -m scripts.tools.silver_match_v3.adjudicate_gemma_api \
    --manifest "$MANIFEST" \
    --candidates "$CANDIDATES" \
    --output "$ROOT/runs/r3/$ORDER.jsonl" \
    --split-role train \
    --prompt "$BASE" \
    --prompt-addon "$ADDON" \
    --api-base-url https://openrouter.ai/api/v1 \
    --api-key-file ~/.openrouter-api-key.txt \
    --max-api-requests 240 \
    --model google/gemma-4-31b-it \
    --max-candidates 50 \
    --context-chars 1200 \
    --description-chars 260 \
    --example-chars 80 \
    --max-examples 0 \
    --batch-size 64 \
    --concurrency 8 \
    --max-tokens 160 \
    --seed 17 \
    --transport-retries 0 \
    --order-mode "$ORDER" \
    --resume
done
