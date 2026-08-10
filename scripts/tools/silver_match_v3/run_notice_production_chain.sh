#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_PID="${1:?usage: run_notice_production_chain.sh ADJUDICATOR_PID}"
REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
MODEL=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
PROD="$DATA/production_v1"
PLAN="$PROD/plans/notice-and-comment.FROZEN.rendering-bound.v2.json"
CANDIDATES="$PROD/candidates/notice-and-comment.all-corpora.primary.nemotron_adapter.jsonl"
PRIMARY="$PROD/adjudicator/notice-and-comment.primary.original.jsonl"
ORDER="$PROD/adjudicator/notice-and-comment.primary.hashed.jsonl"
ADJ_AUDIT="$PROD/adjudicator/notice-and-comment.two-order.audit.json"
VERIFY_DIR="$PROD/verifier"
VERIFY_ORIGINAL="$VERIFY_DIR/notice-and-comment.primary.verify.original.jsonl"
VERIFY_HASHED="$VERIFY_DIR/notice-and-comment.primary.verify.hashed.jsonl"
VERIFY_COMBINED="$VERIFY_DIR/notice-and-comment.primary.verify.strict-combined.jsonl"
ADJ_SELECTION="$MODEL/adjudicator_k50/notice-and-comment.gepa-dev-selection.json"
VERIFY_SELECTION="$MODEL/adjudicator_k50/verifier_calibration_v1/notice-and-comment/notice-and-comment.verifier-selection.dev-only.json"
VERIFY_POLICY="$MODEL/adjudicator_k50/verifier_calibration_v1/notice-and-comment/notice-and-comment.verifier-production-policy.dev-supported.json"
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
GPY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
PINNED_VERIFY_SHA=797e6ade28dba5c3493e28c6fb5c0123d9877e6354649e595f9690860b3afc7e

export CUDA_VISIBLE_DEVICES=0
export HOME=/lfs/skampere3/0/alexspan
export XDG_CACHE_HOME=/lfs/skampere3/0/alexspan/.cache
export TORCHINDUCTOR_CACHE_DIR=/lfs/skampere3/0/alexspan/.cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/lfs/skampere3/0/alexspan
export VLLM_NO_USAGE_STATS=1
export PYTHONPATH=.
cd "$REPO"
mkdir -p "$VERIFY_DIR" "$PROD/subsets" "$PROD/final_pre_rescue"

while kill -0 "$UPSTREAM_PID" 2>/dev/null; do
  sleep 60
done
test -f "$PRIMARY.meta.json"
test -f "$ORDER.meta.json"

$PY -m scripts.tools.silver_match_v3.audit_production_adjudications \
  --plan "$PLAN" \
  --original "$PRIMARY" \
  --hashed "$ORDER" \
  --output "$ADJ_AUDIT"

test "$(sha256sum scripts/tools/silver_match_v3/verify_gemma.py | cut -d' ' -f1)" = "$PINNED_VERIFY_SHA"
for order in original hashed; do
  output="$VERIFY_DIR/notice-and-comment.primary.verify.$order.jsonl"
  $GPY -u -m scripts.tools.silver_match_v3.verify_gemma \
    --manifest "$DATA/manifest.json" \
    --candidates "$CANDIDATES" \
    --primary "$PRIMARY" \
    --output "$output" \
    --prompt scripts/tools/silver_match_v3/prompts/verify_match_v1.txt \
    --prompt-addon scripts/tools/silver_match_v3/prompts/verify_notice_shepherded_v2.txt \
    --order-mode "$order" \
    --max-alternatives 49 \
    --context-chars 1200 \
    --description-chars 260 \
    --example-chars 180 \
    --max-examples 0 \
    --batch-size 128 \
    --gpu-memory-utilization .55 \
    --max-model-len 8192 \
    --max-tokens 180 \
    --seed 29 \
    --resume
  test "$(sha256sum scripts/tools/silver_match_v3/verify_gemma.py | cut -d' ' -f1)" = "$PINNED_VERIFY_SHA"
done

$PY -m scripts.tools.silver_match_v3.combine_two_order_verifications \
  --primary "$PRIMARY" \
  --original "$VERIFY_ORIGINAL" \
  --hashed "$VERIFY_HASHED" \
  --selection "$VERIFY_SELECTION" \
  --policy "$VERIFY_POLICY" \
  --plan "$PLAN" \
  --output "$VERIFY_COMBINED"

for corpus in notice_and_comment nc_public_comments; do
  primary_subset="$PROD/subsets/$corpus.primary.original.jsonl"
  order_subset="$PROD/subsets/$corpus.primary.hashed.jsonl"
  verification_subset="$PROD/subsets/$corpus.verify.strict-combined.jsonl"
  $PY -m scripts.tools.silver_match_v3.filter_labels \
    --input "$PRIMARY" --output "$primary_subset" --where "corpus=$corpus"
  $PY -m scripts.tools.silver_match_v3.filter_labels \
    --input "$ORDER" --output "$order_subset" --where "corpus=$corpus"
  $PY -m scripts.tools.silver_match_v3.filter_labels \
    --input "$VERIFY_COMBINED" --output "$verification_subset" --where "corpus=$corpus"
  $PY -m scripts.tools.silver_match_v3.finalize_adjudications \
    --manifest "$DATA/manifest.json" \
    --corpus "$corpus" \
    --primary "$primary_subset" \
    --order-check "$order_subset" \
    --verification "$verification_subset" \
    --adjudicator-selection "$ADJ_SELECTION" \
    --verifier-selection "$VERIFY_SELECTION" \
    --verifier-policy "$VERIFY_POLICY" \
    --production-plan "$PLAN" \
    --strict-production \
    --output "$PROD/final_pre_rescue/$corpus.jsonl"
done
