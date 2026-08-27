#!/usr/bin/env bash
set -euo pipefail

# Resume-safe parallel form of run_notice_rescue_chain.sh.  Only disjoint
# frozen inference artifacts run concurrently.  The existing capture,
# aggregate, strict two-order combines, merge, coverage audit, and blind-audit
# preparation gates are unchanged.

REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
MODEL=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
PROD="$DATA/production_v1"
RESCUE="$PROD/rescue/notice-and-comment"
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
GPY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
ADJ_SELECTION="$MODEL/adjudicator_k50/notice-and-comment.gepa-dev-selection.json"
VERIFY_SELECTION="$MODEL/adjudicator_k50/verifier_calibration_v1/notice-and-comment/notice-and-comment.verifier-selection.dev-only.json"
VERIFY_POLICY="$MODEL/adjudicator_k50/verifier_calibration_v1/notice-and-comment/notice-and-comment.verifier-production-policy.dev-supported.json"
PINNED_ADJUDICATE_SHA=66e5bd7f2785a2597fe550be3c10d40eaac51f1e8e3e15a0b16612306fe17208
PINNED_VERIFY_SHA=797e6ade28dba5c3493e28c6fb5c0123d9877e6354649e595f9690860b3afc7e
RESCUE_GPU_A="${NOTICE_RESCUE_GPU_A:-5}"
RESCUE_GPU_B="${NOTICE_RESCUE_GPU_B:-7}"
MIN_FREE_MIB="${NOTICE_RESCUE_MIN_FREE_MIB:-170000}"

export HOME=/lfs/skampere3/0/alexspan
export XDG_CACHE_HOME=/lfs/skampere3/0/alexspan/.cache
export TORCHINDUCTOR_CACHE_DIR=/lfs/skampere3/0/alexspan/.cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/lfs/skampere3/0/alexspan
export VLLM_NO_USAGE_STATS=1
export PYTHONPATH=.
cd "$REPO"

check_sources() {
  test "$(sha256sum scripts/tools/silver_match_v3/adjudicate_gemma.py | cut -d' ' -f1)" = "$PINNED_ADJUDICATE_SHA"
  test "$(sha256sum scripts/tools/silver_match_v3/verify_gemma.py | cut -d' ' -f1)" = "$PINNED_VERIFY_SHA"
}

wait_gpu_available() {
  local gpu="$1"
  local free_mib
  while true; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if test "$free_mib" -ge "$MIN_FREE_MIB"; then
      return 0
    fi
    sleep 30
  done
}

while test ! -f "$PROD/final_pre_rescue/notice_and_comment.jsonl.report.json" \
  || test ! -f "$PROD/final_pre_rescue/nc_public_comments.jsonl.report.json"; do
  sleep 30
done
check_sources
for system in nemotron_adapter nemotron_base bge; do
  for corpus in notice_and_comment nc_public_comments; do
    test -f "$PROD/full_bank/$corpus.full88.$system.audit.json"
  done
done

mkdir -p "$RESCUE"
if test ! -f "$RESCUE/captures/rescue_manifest.json"; then
  "$PY" -m scripts.tools.silver_match_v3.build_abstention_rescue \
    --manifest "$DATA/manifest.json" \
    --candidates "$PROD/full_bank/notice_and_comment.full88.nemotron_adapter.jsonl" \
    --candidates "$PROD/full_bank/nc_public_comments.full88.nemotron_adapter.jsonl" \
    --candidates "$PROD/full_bank/notice_and_comment.full88.nemotron_base.jsonl" \
    --candidates "$PROD/full_bank/nc_public_comments.full88.nemotron_base.jsonl" \
    --candidates "$PROD/full_bank/notice_and_comment.full88.bge.jsonl" \
    --candidates "$PROD/full_bank/nc_public_comments.full88.bge.jsonl" \
    --primary "$PROD/final_pre_rescue/notice_and_comment.jsonl" \
    --primary "$PROD/final_pre_rescue/nc_public_comments.jsonl" \
    --output-root "$RESCUE/captures" \
    --block-size 50 \
    --all-abstentions \
    --coverage-repeats 2 \
    --reinclude-primary
fi

trial_paths=("$RESCUE"/captures/trial-*.jsonl)
test -e "${trial_paths[0]}"
mkdir -p "$RESCUE/trial_adjudications"
run_trial_worker() {
  local gpu="$1"
  local offset="$2"
  local index trial_path name output
  for ((index=offset; index<${#trial_paths[@]}; index+=2)); do
    trial_path="${trial_paths[index]}"
    name="$(basename "$trial_path" .jsonl)"
    output="$RESCUE/trial_adjudications/$name.original.jsonl"
    wait_gpu_available "$gpu"
    check_sources
    CUDA_VISIBLE_DEVICES="$gpu" "$GPY" -u -m scripts.tools.silver_match_v3.adjudicate_gemma \
      --manifest "$DATA/manifest.json" \
      --candidates "$trial_path" \
      --output "$output" \
      --prompt scripts/tools/silver_match_v3/prompts/gepa_round2_candidate.txt \
      --prompt-addon scripts/tools/silver_match_v3/prompts/gepa_notice_k50_shepherded_v1.txt \
      --max-candidates 50 \
      --context-chars 1200 \
      --description-chars 260 \
      --example-chars 80 \
      --max-examples 0 \
      --batch-size 128 \
      --gpu-memory-utilization .88 \
      --max-model-len 8192 \
      --max-tokens 160 \
      --seed 17 \
      --order-mode original \
      --resume >>"$RESCUE/trial_adjudications/$name.original.log" 2>&1
    check_sources
  done
}

run_trial_worker "$RESCUE_GPU_A" 0 &
trial_worker_0=$!
run_trial_worker "$RESCUE_GPU_B" 1 &
trial_worker_5=$!
wait "$trial_worker_0"
wait "$trial_worker_5"
check_sources

trial_args=()
for trial_path in "${trial_paths[@]}"; do
  name="$(basename "$trial_path" .jsonl)"
  output="$RESCUE/trial_adjudications/$name.original.jsonl"
  test -f "$output.meta.json"
  trial_args+=(--adjudication "$output")
done

if test ! -f "$RESCUE/aggregate/match_finalists.jsonl"; then
  "$PY" -m scripts.tools.silver_match_v3.aggregate_abstention_rescue \
    --manifest "$DATA/manifest.json" \
    --rescue-manifest "$RESCUE/captures/rescue_manifest.json" \
    --primary "$PROD/final_pre_rescue/notice_and_comment.jsonl" \
    --primary "$PROD/final_pre_rescue/nc_public_comments.jsonl" \
    "${trial_args[@]}" \
    --output-root "$RESCUE/aggregate" \
    --max-finalists 16
fi

FINALISTS="$RESCUE/aggregate/match_finalists.jsonl"
NO_MATCH="$RESCUE/aggregate/no_match_provisional.jsonl"
test -s "$FINALISTS"
test -s "$NO_MATCH"

mkdir -p "$RESCUE/finalists"
run_finalist_adjudication() {
  local gpu="$1"
  local order="$2"
  local output="$RESCUE/finalists/adjudicate.$order.jsonl"
  wait_gpu_available "$gpu"
  check_sources
  CUDA_VISIBLE_DEVICES="$gpu" "$GPY" -u -m scripts.tools.silver_match_v3.adjudicate_gemma \
    --manifest "$DATA/manifest.json" \
    --candidates "$FINALISTS" \
    --output "$output" \
    --prompt scripts/tools/silver_match_v3/prompts/gepa_round2_candidate.txt \
    --prompt-addon scripts/tools/silver_match_v3/prompts/gepa_notice_k50_shepherded_v1.txt \
    --max-candidates 16 \
    --context-chars 1200 \
    --description-chars 260 \
    --example-chars 80 \
    --max-examples 0 \
    --batch-size 128 \
    --gpu-memory-utilization .88 \
    --max-model-len 8192 \
    --max-tokens 160 \
    --seed 17 \
    --order-mode "$order" \
    --resume >>"$RESCUE/finalists/adjudicate.$order.log" 2>&1
  check_sources
}

run_finalist_adjudication "$RESCUE_GPU_A" original &
finalist_adj_0=$!
run_finalist_adjudication "$RESCUE_GPU_B" hashed &
finalist_adj_5=$!
wait "$finalist_adj_0"
wait "$finalist_adj_5"
test -f "$RESCUE/finalists/adjudicate.original.jsonl.meta.json"
test -f "$RESCUE/finalists/adjudicate.hashed.jsonl.meta.json"

run_finalist_verifier() {
  local gpu="$1"
  local order="$2"
  local output="$RESCUE/finalists/verify.$order.jsonl"
  wait_gpu_available "$gpu"
  check_sources
  CUDA_VISIBLE_DEVICES="$gpu" "$GPY" -u -m scripts.tools.silver_match_v3.verify_gemma \
    --manifest "$DATA/manifest.json" \
    --candidates "$FINALISTS" \
    --primary "$RESCUE/finalists/adjudicate.original.jsonl" \
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
    --resume >>"$RESCUE/finalists/verify.$order.log" 2>&1
  check_sources
}

run_finalist_verifier "$RESCUE_GPU_A" original &
finalist_verify_0=$!
run_finalist_verifier "$RESCUE_GPU_B" hashed &
finalist_verify_5=$!
wait "$finalist_verify_0"
wait "$finalist_verify_5"
test -f "$RESCUE/finalists/verify.original.jsonl.meta.json"
test -f "$RESCUE/finalists/verify.hashed.jsonl.meta.json"
check_sources

if test ! -f "$RESCUE/finalists/verify.strict-combined.jsonl"; then
  "$PY" -m scripts.tools.silver_match_v3.combine_two_order_verifications \
    --primary "$RESCUE/finalists/adjudicate.original.jsonl" \
    --original "$RESCUE/finalists/verify.original.jsonl" \
    --hashed "$RESCUE/finalists/verify.hashed.jsonl" \
    --selection "$VERIFY_SELECTION" \
    --policy "$VERIFY_POLICY" \
    --output "$RESCUE/finalists/verify.strict-combined.jsonl"
fi

mkdir -p "$RESCUE/typed_abstentions"
run_abstention_verifier() {
  local gpu="$1"
  local order="$2"
  local output="$RESCUE/typed_abstentions/verify.$order.jsonl"
  wait_gpu_available "$gpu"
  check_sources
  CUDA_VISIBLE_DEVICES="$gpu" "$GPY" -u -m scripts.tools.silver_match_v3.verify_abstention_gemma \
    --manifest "$DATA/manifest.json" \
    --audits "$NO_MATCH" \
    --output "$output" \
    --prompt scripts/tools/silver_match_v3/prompts/verify_abstention_v1.txt \
    --order-mode "$order" \
    --batch-size 128 \
    --gpu-memory-utilization .88 \
    --max-model-len 8192 \
    --max-tokens 180 \
    --seed 43 \
    --resume >>"$RESCUE/typed_abstentions/verify.$order.log" 2>&1
  check_sources
}

run_abstention_verifier "$RESCUE_GPU_A" original &
abstention_verify_0=$!
run_abstention_verifier "$RESCUE_GPU_B" hashed &
abstention_verify_5=$!
wait "$abstention_verify_0"
wait "$abstention_verify_5"
test -f "$RESCUE/typed_abstentions/verify.original.jsonl.meta.json"
test -f "$RESCUE/typed_abstentions/verify.hashed.jsonl.meta.json"
check_sources

if test ! -f "$RESCUE/typed_abstentions/verify.strict-combined.jsonl"; then
  "$PY" -m scripts.tools.silver_match_v3.combine_two_order_abstention_verifications \
    --audits "$NO_MATCH" \
    --original "$RESCUE/typed_abstentions/verify.original.jsonl" \
    --hashed "$RESCUE/typed_abstentions/verify.hashed.jsonl" \
    --output "$RESCUE/typed_abstentions/verify.strict-combined.jsonl"
fi

manual_unresolved_args=()
manual_unresolved_labels="${NOTICE_MANUAL_UNRESOLVED_LABELS:-$RESCUE/unresolved_blind_pack/notice-and-comment/label_pack/labels.validated.jsonl}"
manual_unresolved_validation="${NOTICE_MANUAL_UNRESOLVED_VALIDATION:-$RESCUE/unresolved_blind_pack/notice-and-comment/label_pack/labels.validation.json}"
if test -f "$manual_unresolved_labels" && test -f "$manual_unresolved_validation"; then
  manual_unresolved_args=(
    --manual-unresolved-labels "$manual_unresolved_labels"
    --manual-unresolved-validation "$manual_unresolved_validation"
  )
fi

if test ! -f "$RESCUE/final.all-corpora.jsonl"; then
  if ! "$PY" -m scripts.tools.silver_match_v3.merge_rescue_decisions \
    --manifest "$DATA/manifest.json" \
    --primary "$PROD/final_pre_rescue/notice_and_comment.jsonl" \
    --primary "$PROD/final_pre_rescue/nc_public_comments.jsonl" \
    --finalist-candidates "$FINALISTS" \
    --finalist-adjudications "$RESCUE/finalists/adjudicate.original.jsonl" \
    --finalist-order-check "$RESCUE/finalists/adjudicate.hashed.jsonl" \
    --finalist-verification "$RESCUE/finalists/verify.strict-combined.jsonl" \
    --no-match-audits "$NO_MATCH" \
    --abstention-verifications "$RESCUE/typed_abstentions/verify.strict-combined.jsonl" \
    --adjudicator-selection "$ADJ_SELECTION" \
    --verifier-selection "$VERIFY_SELECTION" \
    --verifier-policy "$VERIFY_POLICY" \
    --unresolved-output "$RESCUE/unresolved.jsonl" \
    "${manual_unresolved_args[@]}" \
    --strict-production \
    --output "$RESCUE/final.all-corpora.jsonl"; then
    test -s "$RESCUE/unresolved.jsonl"
    if test ! -f "$RESCUE/unresolved_blind_pack/validation.json"; then
      "$PY" -m scripts.tools.silver_match_v3.prepare_unresolved_decision_pack \
        --manifest "$DATA/manifest.json" \
        --unresolved "$RESCUE/unresolved.jsonl" \
        --output-root "$RESCUE/unresolved_blind_pack" \
        --chunk-size 25 \
        --seed 161803
    fi
    exit 1
  fi
fi

mkdir -p "$RESCUE/final_by_corpus"
for corpus in notice_and_comment nc_public_comments; do
  if test ! -f "$RESCUE/final_by_corpus/$corpus.jsonl"; then
    "$PY" -m scripts.tools.silver_match_v3.filter_labels \
      --input "$RESCUE/final.all-corpora.jsonl" \
      --output "$RESCUE/final_by_corpus/$corpus.jsonl" \
      --where "corpus=$corpus"
  fi
done

if test ! -f "$RESCUE/final.audit.json"; then
  "$PY" -m scripts.tools.silver_match_v3.audit_final_outputs \
    --manifest "$DATA/manifest.json" \
    --task notice-and-comment \
    --final "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
    --final "$RESCUE/final_by_corpus/nc_public_comments.jsonl" \
    --output "$RESCUE/final.audit.json"
fi

if test ! -f "$RESCUE/blind_audit_match/sample_report.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_audit \
    --manifest "$DATA/manifest.json" \
    --final "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
    --final "$RESCUE/final_by_corpus/nc_public_comments.jsonl" \
    --output-root "$RESCUE/blind_audit_match" \
    --global-n 300 \
    --per-task-n 200 \
    --seed 271828 \
    --sample-kind match
fi

if test ! -f "$RESCUE/blind_audit_abstention/sample_report.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_audit \
    --manifest "$DATA/manifest.json" \
    --final "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
    --final "$RESCUE/final_by_corpus/nc_public_comments.jsonl" \
    --output-root "$RESCUE/blind_audit_abstention" \
    --global-n 300 \
    --per-task-n 200 \
    --seed 314159 \
    --sample-kind abstention
fi
