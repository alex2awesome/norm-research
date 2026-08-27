#!/usr/bin/env bash
set -euo pipefail

# Resume-safe parallel form of run_notice_production_chain.sh.  The two Gemma
# verifier orders are scientifically independent and write disjoint artifacts,
# so they can run concurrently after both adjudicator orders have sealed.  All
# prompts, rendering parameters, implementation pins, combination policy, and
# finalization gates are identical to the sequential chain.

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
PINNED_ADJUDICATE_SHA=66e5bd7f2785a2597fe550be3c10d40eaac51f1e8e3e15a0b16612306fe17208
PINNED_VERIFY_SHA=797e6ade28dba5c3493e28c6fb5c0123d9877e6354649e595f9690860b3afc7e
VERIFY_ORIGINAL_GPU="${NOTICE_VERIFY_ORIGINAL_GPU:-0}"
VERIFY_HASHED_GPU="${NOTICE_VERIFY_HASHED_GPU:-5}"

export HOME=/lfs/skampere3/0/alexspan
export XDG_CACHE_HOME=/lfs/skampere3/0/alexspan/.cache
export TORCHINDUCTOR_CACHE_DIR=/lfs/skampere3/0/alexspan/.cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/lfs/skampere3/0/alexspan
export VLLM_NO_USAGE_STATS=1
export PYTHONPATH=.
cd "$REPO"
mkdir -p "$VERIFY_DIR" "$PROD/subsets" "$PROD/final_pre_rescue"

check_sources() {
  test "$(sha256sum scripts/tools/silver_match_v3/adjudicate_gemma.py | cut -d' ' -f1)" = "$PINNED_ADJUDICATE_SHA"
  test "$(sha256sum scripts/tools/silver_match_v3/verify_gemma.py | cut -d' ' -f1)" = "$PINNED_VERIFY_SHA"
}

while test ! -f "$PRIMARY.meta.json" || test ! -f "$ORDER.meta.json"; do
  sleep 30
done
# A metadata file is written only at successful completion, but wait for both
# vLLM parents to release their GPUs before starting the verifier engines.
while pgrep -f 'adjudicate_gemma.*notice-and-comment.primary' >/dev/null; do
  sleep 10
done
check_sources

if test ! -f "$ADJ_AUDIT"; then
  "$PY" -m scripts.tools.silver_match_v3.audit_production_adjudications \
    --plan "$PLAN" \
    --original "$PRIMARY" \
    --hashed "$ORDER" \
    --output "$ADJ_AUDIT"
fi

run_verify() {
  local gpu="$1"
  local order="$2"
  local output="$VERIFY_DIR/notice-and-comment.primary.verify.$order.jsonl"
  check_sources
  CUDA_VISIBLE_DEVICES="$gpu" "$GPY" -u -m scripts.tools.silver_match_v3.verify_gemma \
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
    --gpu-memory-utilization .88 \
    --max-model-len 8192 \
    --max-tokens 180 \
    --seed 29 \
    --resume
  check_sources
}

verify_pids=()
if test ! -f "$VERIFY_ORIGINAL.meta.json"; then
  run_verify "$VERIFY_ORIGINAL_GPU" original >>"$VERIFY_DIR/notice-and-comment.primary.verify.original.log" 2>&1 &
  verify_pids+=("$!")
fi
if test ! -f "$VERIFY_HASHED.meta.json"; then
  run_verify "$VERIFY_HASHED_GPU" hashed >>"$VERIFY_DIR/notice-and-comment.primary.verify.hashed.log" 2>&1 &
  verify_pids+=("$!")
fi
for verify_pid in "${verify_pids[@]}"; do
  wait "$verify_pid"
done
test -f "$VERIFY_ORIGINAL.meta.json"
test -f "$VERIFY_HASHED.meta.json"
check_sources

if test ! -f "$VERIFY_COMBINED"; then
  "$PY" -m scripts.tools.silver_match_v3.combine_two_order_verifications \
    --primary "$PRIMARY" \
    --original "$VERIFY_ORIGINAL" \
    --hashed "$VERIFY_HASHED" \
    --selection "$VERIFY_SELECTION" \
    --policy "$VERIFY_POLICY" \
    --plan "$PLAN" \
    --output "$VERIFY_COMBINED"
fi

for corpus in notice_and_comment nc_public_comments; do
  primary_subset="$PROD/subsets/$corpus.primary.original.jsonl"
  order_subset="$PROD/subsets/$corpus.primary.hashed.jsonl"
  verification_subset="$PROD/subsets/$corpus.verify.strict-combined.jsonl"
  final="$PROD/final_pre_rescue/$corpus.jsonl"
  if test ! -f "$primary_subset"; then
    "$PY" -m scripts.tools.silver_match_v3.filter_labels \
      --input "$PRIMARY" --output "$primary_subset" --where "corpus=$corpus"
  fi
  if test ! -f "$order_subset"; then
    "$PY" -m scripts.tools.silver_match_v3.filter_labels \
      --input "$ORDER" --output "$order_subset" --where "corpus=$corpus"
  fi
  if test ! -f "$verification_subset"; then
    "$PY" -m scripts.tools.silver_match_v3.filter_labels \
      --input "$VERIFY_COMBINED" --output "$verification_subset" --where "corpus=$corpus"
  fi
  if test ! -f "$final.report.json"; then
    "$PY" -m scripts.tools.silver_match_v3.finalize_adjudications \
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
      --output "$final"
  fi
done
