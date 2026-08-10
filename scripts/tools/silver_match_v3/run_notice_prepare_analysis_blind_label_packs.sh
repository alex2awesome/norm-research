#!/usr/bin/env bash
set -euo pipefail

# Prepare the N&C release-audit packs from rows that were never used to train,
# select, shepherd, or calibrate any selected matching component.  These packs
# are separate from the all-row blind packs used for dataset-wide quality claims.

REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
MODEL=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
RESCUE="$DATA/production_v1/rescue/notice-and-comment"
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python

TRAIN="$MODEL/teacher_sets/notice-and-comment.train-only.jsonl"
DEV="$MODEL/teacher_sets_common_manual/notice-and-comment/external_dev.jsonl"
TEST="$MODEL/teacher_sets_common_manual/notice-and-comment/external_test.jsonl"
VERIFIER="$MODEL/adjudicator_k50/verifier_calibration_v1/notice-and-comment/dev.truth.jsonl"
EXCLUSIONS=("$TRAIN" "$DEV" "$TEST" "$VERIFIER")

verify_sha() {
  local path=$1
  local expected=$2
  test -f "$path"
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected"
}

verify_sha "$TRAIN" 428f1389d88f54c35449cf2dd3a7f139b096da19495668c2d8e72ab845804f1e
verify_sha "$DEV" ae0782368a6974274804d837a84e4bafd90c8f0aefa1bc68bb6b2aaf230ec8bf
verify_sha "$TEST" b13ae83b4fd4c110e4c34606c5a50e0e76574c7de990d4e1eaf586a7abf6ac58
verify_sha "$VERIFIER" 49b283cdb7b347bb3f1d6c1c343712700d439799bd5903e22b75014254c6a185

cd "$REPO"
while test ! -f "$RESCUE/final.audit.json" \
  || test ! -f "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
  || test ! -f "$RESCUE/final_by_corpus/nc_public_comments.jsonl"; do
  sleep 30
done

EXCLUDE_ARGS=()
for path in "${EXCLUSIONS[@]}"; do
  EXCLUDE_ARGS+=(--exclude "$path")
done

MATCH="$RESCUE/analysis_blind_audit_match"
ABSTENTION="$RESCUE/analysis_blind_audit_abstention"

if test ! -f "$MATCH/sample_report.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_audit \
    --manifest "$DATA/manifest.json" \
    --final "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
    --final "$RESCUE/final_by_corpus/nc_public_comments.jsonl" \
    --output-root "$MATCH" \
    --global-n 300 \
    --per-task-n 200 \
    --seed 271829 \
    --sample-kind match \
    "${EXCLUDE_ARGS[@]}"
fi

if test ! -f "$ABSTENTION/sample_report.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_audit \
    --manifest "$DATA/manifest.json" \
    --final "$RESCUE/final_by_corpus/notice_and_comment.jsonl" \
    --final "$RESCUE/final_by_corpus/nc_public_comments.jsonl" \
    --output-root "$ABSTENTION" \
    --global-n 300 \
    --per-task-n 200 \
    --seed 314160 \
    --sample-kind abstention \
    "${EXCLUDE_ARGS[@]}"
fi

for report in "$MATCH/sample_report.json" "$ABSTENTION/sample_report.json"; do
  jq -e \
    '.analysis_exclusions.count == 373 and
     .analysis_exclusions.excluded_final_rows_seen == 373 and
     (.outputs["task:notice-and-comment"].sample_n > 0)' \
    "$report" >/dev/null
done

if test ! -f "$MATCH/task_label_pack/validation.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_label_pack \
    --sample-report "$MATCH/sample_report.json" \
    --scope task:notice-and-comment \
    --output-root "$MATCH/task_label_pack" \
    --chunk-size 25 \
    --seed 57722
fi

if test ! -f "$ABSTENTION/task_label_pack/validation.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_label_pack \
    --sample-report "$ABSTENTION/sample_report.json" \
    --scope task:notice-and-comment \
    --output-root "$ABSTENTION/task_label_pack" \
    --chunk-size 25 \
    --seed 91578
fi

sha256sum \
  "$MATCH/sample_report.json" \
  "$MATCH/task_label_pack/validation.json" \
  "$ABSTENTION/sample_report.json" \
  "$ABSTENTION/task_label_pack/validation.json"
