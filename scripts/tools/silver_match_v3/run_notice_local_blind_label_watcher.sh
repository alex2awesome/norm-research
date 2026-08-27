#!/usr/bin/env bash
set -euo pipefail

# Local orchestration for N&C's independent blind-label closures.  Codex sees
# only each immutable label pack; validation and final joins happen remotely.

REMOTE=sk3
REMOTE_REPO=/lfs/skampere3/0/alexspan/norm-research
REMOTE_PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
PROD="$DATA/production_v1"
RESCUE="$PROD/rescue/notice-and-comment"
LOCAL_REPO=/Users/spangher/Projects/stanford-research/norm-research
LOCAL_ROOT="$LOCAL_REPO/outputs/silver_match_v3_notice_blind_labels_20260712"
NOTICE_CODEX_CONCURRENCY=${NOTICE_CODEX_CONCURRENCY:-6}

remote_file() {
  ssh "$REMOTE" "test -f '$1'"
}

wait_for_remote_pack() {
  local pack=$1
  while ! remote_file "$pack/validation.json"; do
    sleep 30
  done
}

label_pack() {
  local name=$1
  local remote_pack=$2
  local pass_name=$3
  local local_pack="$LOCAL_ROOT/$name"
  local schema=scripts/tools/silver_match_v3/schemas/independent_labels_25.schema.json
  local chunk count

  wait_for_remote_pack "$remote_pack"
  if remote_file "$remote_pack/labels.validation.json"; then
    return 0
  fi

  mkdir -p "$local_pack"
  rsync -a "$REMOTE:$remote_pack/" "$local_pack/"
  for chunk in "$local_pack"/chunks/part-*.jsonl; do
    count="$(wc -l < "$chunk" | tr -d ' ')"
    if test "$count" -ne 25; then
      schema=scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json
      break
    fi
  done

  cd "$LOCAL_REPO"
  for attempt in 1 2 3; do
    if python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
      --pack-root "$local_pack" \
      --task notice-and-comment \
      --pass-name "$pass_name" \
      --boundary-guide scripts/tools/silver_match_v3/FINAL_DECISION_LABELING.md \
      --concurrency "$NOTICE_CODEX_CONCURRENCY" \
      --model gpt-5.6-sol \
      --reasoning-effort high \
      --timeout-seconds 1200 \
      --chunk-attempts 3 \
      --output-schema "$schema"; then
      break
    fi
    if test "$attempt" -eq 3; then
      return 1
    fi
    sleep 30
  done

  if test ! -f "$local_pack/transcript.audit.json"; then
    python -m scripts.tools.silver_match_v3.audit_isolated_labeler_transcripts \
      --pack-root "$local_pack" \
      --guide scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md \
      --guide scripts/tools/silver_match_v3/FINAL_DECISION_LABELING.md \
      --repo "$LOCAL_REPO" \
      --output "$local_pack/transcript.audit.json"
  fi
  jq -e '.complete == true and .status == "PASS" and (.violations | length) == 0' \
    "$local_pack/transcript.audit.json" >/dev/null

  rsync -a "$local_pack/raw_labels/" "$REMOTE:$remote_pack/raw_labels/"
  rsync -a "$local_pack/logs/" "$REMOTE:$remote_pack/logs/"
  rsync -a "$local_pack/transcript.audit.json" "$REMOTE:$remote_pack/transcript.audit.json"
  ssh "$REMOTE" \
    "cd '$REMOTE_REPO' && '$REMOTE_PY' -m scripts.tools.silver_match_v3.validate_independent_teacher_labels \
      --pack-root '$remote_pack' \
      --raw-label-dir '$remote_pack/raw_labels' \
      --transcript-audit '$remote_pack/transcript.audit.json' \
      --annotator codex-gpt-5.6-sol-high-blind \
      --output '$remote_pack/labels.validated.jsonl' \
      --report '$remote_pack/labels.validation.json'"
  ssh "$REMOTE" \
    "jq -e '.complete == true and .task == \"notice-and-comment\"' '$remote_pack/labels.validation.json' >/dev/null"
}

restart_rescue() {
  ssh "$REMOTE" \
    "cd '$REMOTE_REPO'; nohup env NOTICE_RESCUE_GPU_A=5 NOTICE_RESCUE_GPU_B=7 \
      bash scripts/tools/silver_match_v3/run_notice_rescue_chain_parallel.sh \
      >> '$PROD/notice_rescue_parallel.manual-closure.log' 2>&1 < /dev/null & echo \$!"
}

mkdir -p "$LOCAL_ROOT"
unresolved_resumed=0
UNRESOLVED="$RESCUE/unresolved_blind_pack/notice-and-comment/label_pack"
while ! remote_file "$RESCUE/final.audit.json"; do
  if remote_file "$UNRESOLVED/validation.json" && test "$unresolved_resumed" -eq 0; then
    label_pack unresolved "$UNRESOLVED" notice-unresolved-blind-closure
    ssh "$REMOTE" \
      "jq -e '.complete == true and ((.confidence_counts | keys) == [\"high\"])' \
        '$UNRESOLVED/labels.validation.json' >/dev/null"
    restart_rescue
    unresolved_resumed=1
  fi
  sleep 30
done

# Release-critical leakage-safe labels run first.
label_pack \
  analysis_match \
  "$RESCUE/analysis_blind_audit_match/task_label_pack" \
  notice-analysis-match-blind-audit
label_pack \
  analysis_abstention \
  "$RESCUE/analysis_blind_audit_abstention/task_label_pack" \
  notice-analysis-abstention-blind-audit

# Separate all-row samples support dataset-wide quality statements.
label_pack \
  all_rows_match \
  "$RESCUE/blind_audit_match/task_label_pack" \
  notice-all-rows-match-blind-audit
label_pack \
  all_rows_abstention \
  "$RESCUE/blind_audit_abstention/task_label_pack" \
  notice-all-rows-abstention-blind-audit

ALL_ROW_RISK="$RESCUE/blind_audit_all_rows.risk.json"
if ! remote_file "$ALL_ROW_RISK"; then
  ssh "$REMOTE" \
    "cd '$REMOTE_REPO' && '$REMOTE_PY' -m scripts.tools.silver_match_v3.audit_false_abstentions \
      --gold '$RESCUE/blind_audit_match/task_label_pack/labels.validated.jsonl' \
      --gold '$RESCUE/blind_audit_abstention/task_label_pack/labels.validated.jsonl' \
      --predictions '$RESCUE/final_by_corpus/notice_and_comment.jsonl' \
      --predictions '$RESCUE/final_by_corpus/nc_public_comments.jsonl' \
      --output '$ALL_ROW_RISK'"
fi

ssh "$REMOTE" "sha256sum \
  '$RESCUE/analysis_blind_audit_match/task_label_pack/labels.validation.json' \
  '$RESCUE/analysis_blind_audit_abstention/task_label_pack/labels.validation.json' \
  '$RESCUE/blind_audit_match/task_label_pack/labels.validation.json' \
  '$RESCUE/blind_audit_abstention/task_label_pack/labels.validation.json' \
  '$ALL_ROW_RISK'"
