#!/usr/bin/env bash
set -euo pipefail

# Freeze and analyze N&C only after the leakage-safe blind audit labels exist.

REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
MODEL=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
PROD="$DATA/production_v1"
RESCUE="$PROD/rescue/notice-and-comment"
RELEASE="$RESCUE/analysis_release"
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python

TRAIN="$MODEL/teacher_sets/notice-and-comment.train-only.jsonl"
DEV="$MODEL/teacher_sets_common_manual/notice-and-comment/external_dev.jsonl"
TEST="$MODEL/teacher_sets_common_manual/notice-and-comment/external_test.jsonl"
VERIFIER="$MODEL/adjudicator_k50/verifier_calibration_v1/notice-and-comment/dev.truth.jsonl"
EXCLUSIONS=("$TRAIN" "$DEV" "$TEST" "$VERIFIER")

MATCH_LABELS="$RESCUE/analysis_blind_audit_match/task_label_pack/labels.validated.jsonl"
MATCH_VALIDATION="$RESCUE/analysis_blind_audit_match/task_label_pack/labels.validation.json"
ABSTENTION_LABELS="$RESCUE/analysis_blind_audit_abstention/task_label_pack/labels.validated.jsonl"
ABSTENTION_VALIDATION="$RESCUE/analysis_blind_audit_abstention/task_label_pack/labels.validation.json"
RISK="$RELEASE/notice-and-comment.analysis-blind-risk.json"
TASK_RELEASE="$RELEASE/notice-and-comment.json"
MI="$RELEASE/notice-and-comment.mi-validation-v3.json"

verify_sha() {
  local path=$1
  local expected=$2
  test -f "$path"
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected"
}

verify_sources() {
  verify_sha scripts/tools/silver_match_v3/audit_false_abstentions.py 99eaed7d941e6d379194c6e5051b0a65958476efadcb9b88cd12fc020afd62cd
  verify_sha scripts/tools/silver_match_v3/freeze_task_analysis_release.py 5aeff47b16a091da5b3213981c3f73b16b6a60e06106001cf1c1b50db5d13c30
  verify_sha scripts/tools/silver_match_v3/silver_mi_validation_v3.py 363716f9674f0820d6a6aa2f6d9fd870c5fea94279e1043bc8a2fc5eb27535ee
}

verify_sha "$TRAIN" 428f1389d88f54c35449cf2dd3a7f139b096da19495668c2d8e72ab845804f1e
verify_sha "$DEV" ae0782368a6974274804d837a84e4bafd90c8f0aefa1bc68bb6b2aaf230ec8bf
verify_sha "$TEST" b13ae83b4fd4c110e4c34606c5a50e0e76574c7de990d4e1eaf586a7abf6ac58
verify_sha "$VERIFIER" 49b283cdb7b347bb3f1d6c1c343712700d439799bd5903e22b75014254c6a185

cd "$REPO"
verify_sources
while test ! -f "$MATCH_LABELS" || test ! -f "$MATCH_VALIDATION" \
  || test ! -f "$ABSTENTION_LABELS" || test ! -f "$ABSTENTION_VALIDATION"; do
  sleep 30
done

jq -e '.complete == true and .task == "notice-and-comment"' "$MATCH_VALIDATION" >/dev/null
jq -e '.complete == true and .task == "notice-and-comment"' "$ABSTENTION_VALIDATION" >/dev/null
verify_sources
mkdir -p "$RELEASE"

ANALYSIS_EXCLUSION_ARGS=()
for path in "${EXCLUSIONS[@]}"; do
  ANALYSIS_EXCLUSION_ARGS+=(--analysis-exclusion "$path")
done

if test ! -f "$RISK"; then
  "$PY" -m scripts.tools.silver_match_v3.audit_false_abstentions \
    --gold "$MATCH_LABELS" \
    --gold "$ABSTENTION_LABELS" \
    --predictions "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
    --predictions "$RESCUE/final_by_corpus/nc_public_comments.jsonl" \
    --output "$RISK" \
    "${ANALYSIS_EXCLUSION_ARGS[@]}"
fi
jq -e '.analysis_exclusions.count == 373 and .missing_prediction_rows == 0' "$RISK" >/dev/null
verify_sources

if test ! -f "$TASK_RELEASE"; then
  "$PY" -m scripts.tools.silver_match_v3.freeze_task_analysis_release \
    --manifest "$DATA/manifest.json" \
    --task notice-and-comment \
    --plan "$PROD/plans/notice-and-comment.FROZEN.rendering-bound.v2.json" \
    --final-audit "$RESCUE/final.audit.json" \
    --final "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
    --final "$RESCUE/final_by_corpus/nc_public_comments.jsonl" \
    --blind-risk-audit "$RISK" \
    "${ANALYSIS_EXCLUSION_ARGS[@]}" \
    --output "$TASK_RELEASE"
fi
jq -e '.status == "TASK_FROZEN_ANALYSIS_READY" and .analysis_exclusions.count == 373' "$TASK_RELEASE" >/dev/null
verify_sources

if test ! -f "$MI"; then
  "$PY" -m scripts.tools.silver_match_v3.silver_mi_validation_v3 \
    --release "$TASK_RELEASE" \
    --certificate "$DATA/analysis_inputs/mi_certificates/notice-and-comment.json" \
    --n-permutations 2000 \
    --n-bootstrap 1000 \
    --seed 1729 \
    --output "$MI"
fi

sha256sum "$RISK" "$TASK_RELEASE" "$MI"
