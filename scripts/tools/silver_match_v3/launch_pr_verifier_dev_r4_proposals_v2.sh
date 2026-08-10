#!/usr/bin/env bash
set -euo pipefail

REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
MODEL_ROOT=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
PR_ROOT="$MODEL_ROOT/adjudicator_k50/press-releases/gepa_clean_v2"
V1_ROOT="$PR_ROOT/verifier_dev_v2/r4_three_order_proposals_v1"
RUN_ROOT="$PR_ROOT/verifier_dev_v2/r4_three_order_proposals_v2"
PYTHON=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
MODEL=/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb
EXPECTED_PLAN_SHA=b3f2c427b319b4ae873c073b5e46129251b0e295bf015b834b69ef6cdc42a5e1

cd "$REPO"
test "$(sha256sum "$RUN_ROOT/PLAN.json" | cut -d' ' -f1)" = "$EXPECTED_PLAN_SHA"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

for order in original hashed reverse; do
  output="$RUN_ROOT/runs/$order.jsonl"
  test ! -e "$output"
  "$PYTHON" -u -m scripts.tools.silver_match_v3.adjudicate_gemma \
    --manifest "$DATA/manifest.json" \
    --candidates "$V1_ROOT/candidate_source/candidates.top50.jsonl" \
    --output "$output" \
    --prompt "$RUN_ROOT/interface_repair/prompt.frozen.txt" \
    --model "$MODEL" \
    --max-candidates 50 \
    --context-chars 1200 \
    --description-chars 260 \
    --example-chars 80 \
    --max-examples 0 \
    --batch-size 32 \
    --max-model-len 8192 \
    --max-tokens 512 \
    --gpu-memory-utilization 0.88 \
    --seed 2026071321 \
    --order-mode "$order" \
    --keep-raw
done
