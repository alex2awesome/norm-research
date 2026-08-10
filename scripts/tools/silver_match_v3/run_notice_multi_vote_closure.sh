#!/usr/bin/env bash
set -euo pipefail

# Close N&C's 7,888-row blind unresolved set without force-promoting a single
# labeler.  The frozen base Codex pass is transcript-audited and repaired only
# by violation-selected chunks; Gemma original+hashed count as one vote; any
# remaining disagreement receives additional isolated, truth-hidden Codex
# passes until the unique two-vote gate closes.

REMOTE=sk3
REMOTE_REPO=/lfs/skampere3/0/alexspan/norm-research
REMOTE_PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
PROD="$DATA/production_v1"
RESCUE="$PROD/rescue/notice-and-comment"
PACK="$RESCUE/unresolved_blind_pack/notice-and-comment/label_pack"
GEMMA="$PACK/gemma_full_bank_v1"
LOCAL_REPO=/Users/spangher/Projects/stanford-research/norm-research
LOCAL_BASE="$LOCAL_REPO/outputs/silver_match_v3_notice_blind_labels_20260712/unresolved"
LOCAL_WORK="$LOCAL_REPO/outputs/silver_match_v3_notice_multi_vote_closure_20260713"
CODEX_CONCURRENCY=${NOTICE_RESOLVER_CODEX_CONCURRENCY:-8}
MAX_ROUNDS=${NOTICE_MAX_CONSENSUS_ROUNDS:-8}

remote_file() {
  ssh "$REMOTE" "test -f '$1'"
}

remote_exec() {
  local command
  printf -v command '%q ' "$@"
  ssh "$REMOTE" "cd '$REMOTE_REPO' && $command"
}

wait_file() {
  local path=$1
  while ! remote_file "$path"; do
    sleep 30
  done
}

wait_local_base() {
  local expected observed
  expected="$(find "$LOCAL_BASE/chunks" -type f -name 'part-*.jsonl' | wc -l | tr -d ' ')"
  while true; do
    observed="$(find "$LOCAL_BASE/raw_labels" -type f -name 'part-*.json' 2>/dev/null | wc -l | tr -d ' ')"
    if test "$observed" -eq "$expected"; then
      if ! pgrep -f "run_codex_pack_labels .*silver_match_v3_notice_blind_labels_20260712/unresolved" >/dev/null; then
        return 0
      fi
    fi
    sleep 30
  done
}

audit_local_pack() {
  local pack=$1
  local output=$2
  if test ! -f "$output"; then
    set +e
    python -m scripts.tools.silver_match_v3.audit_isolated_labeler_transcripts \
      --pack-root "$pack" \
      --guide scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md \
      --guide scripts/tools/silver_match_v3/FINAL_DECISION_LABELING.md \
      --guide scripts/tools/silver_match_v3/ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md \
      --repo "$LOCAL_REPO" \
      --output "$output"
    set -e
  fi
}

run_repair_or_resolver_pack() {
  local local_pack=$1
  local pass_name=$2
  local audit=$3
  local schema=scripts/tools/silver_match_v3/schemas/independent_labels_25.schema.json
  local count chunk
  for chunk in "$local_pack"/chunks/part-*.jsonl; do
    count="$(wc -l < "$chunk" | tr -d ' ')"
    if test "$count" -ne 25; then
      schema=scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json
      break
    fi
  done
  python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
    --pack-root "$local_pack" \
    --task notice-and-comment \
    --pass-name "$pass_name" \
    --boundary-guide scripts/tools/silver_match_v3/FINAL_DECISION_LABELING.md \
    --boundary-guide scripts/tools/silver_match_v3/ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md \
    --concurrency "$CODEX_CONCURRENCY" \
    --model gpt-5.6-sol \
    --reasoning-effort high \
    --timeout-seconds 1200 \
    --chunk-attempts 3 \
    --output-schema "$schema"
  audit_local_pack "$local_pack" "$audit"
  jq -e '.complete == true and .status == "PASS" and (.violations | length) == 0' \
    "$audit" >/dev/null
}

sync_and_validate_pack() {
  local local_pack=$1
  local remote_pack=$2
  local audit=$3
  local labels=$4
  local validation=$5
  rsync -a "$local_pack/raw_labels/" "$REMOTE:$remote_pack/raw_labels/"
  rsync -a "$local_pack/logs/" "$REMOTE:$remote_pack/logs/"
  rsync -a "$audit" "$REMOTE:$remote_pack/transcript.audit.json"
  if ! remote_file "$validation"; then
    remote_exec "$REMOTE_PY" -m scripts.tools.silver_match_v3.validate_independent_teacher_labels \
      --pack-root "$remote_pack" \
      --raw-label-dir "$remote_pack/raw_labels" \
      --transcript-audit "$remote_pack/transcript.audit.json" \
      --annotator codex-gpt-5.6-sol-high-blind \
      --output "$labels" \
      --report "$validation"
  fi
}

mkdir -p "$LOCAL_WORK"
cd "$LOCAL_REPO"

# The base pass was frozen before the no-discovery operational guide existed.
# Audit it once, then replace all and only violating chunks.
wait_local_base
BASE_AUDIT="$LOCAL_WORK/base.transcript.audit.json"
if test ! -f "$BASE_AUDIT"; then
  set +e
  python -m scripts.tools.silver_match_v3.audit_isolated_labeler_transcripts \
    --pack-root "$LOCAL_BASE" \
    --guide scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md \
    --guide scripts/tools/silver_match_v3/FINAL_DECISION_LABELING.md \
    --repo "$LOCAL_REPO" \
    --output "$BASE_AUDIT"
  set -e
fi
rsync -a "$LOCAL_BASE/raw_labels/" "$REMOTE:$PACK/raw_labels/"
rsync -a "$LOCAL_BASE/logs/" "$REMOTE:$PACK/logs/"
rsync -a "$BASE_AUDIT" "$REMOTE:$PACK/transcript.base.audit.json"

# Allow the original watcher to seal its schema validator; if it has exited,
# do the same validation here.  This base artifact is never promoted directly
# when the transcript audit failed.
for _ in $(seq 1 20); do
  remote_file "$PACK/labels.validation.json" && break
  sleep 30
done
if ! remote_file "$PACK/labels.validation.json"; then
  remote_exec "$REMOTE_PY" -m scripts.tools.silver_match_v3.validate_independent_teacher_labels \
    --pack-root "$PACK" --raw-label-dir "$PACK/raw_labels" \
    --annotator codex-gpt-5.6-sol-high-blind \
    --output "$PACK/labels.validated.jsonl" --report "$PACK/labels.validation.json"
fi

BASE_CLEAN_LABELS="$PACK/transcript_clean/pass_01.labels.jsonl"
BASE_CLEAN_VALIDATION="$PACK/transcript_clean/pass_01.validation.json"
base_status="$(jq -r '.status' "$BASE_AUDIT")"
if test "$base_status" = PASS; then
  if ! remote_file "$BASE_CLEAN_VALIDATION"; then
    remote_exec "$REMOTE_PY" -m scripts.tools.silver_match_v3.validate_independent_teacher_labels \
      --pack-root "$PACK" --raw-label-dir "$PACK/raw_labels" \
      --transcript-audit "$PACK/transcript.base.audit.json" \
      --annotator codex-gpt-5.6-sol-high-blind \
      --output "$BASE_CLEAN_LABELS" --report "$BASE_CLEAN_VALIDATION"
  fi
elif test "$base_status" = FAIL; then
  REPAIR_REMOTE="$PACK/transcript_clean/repair_01"
  REPAIR_LOCAL="$LOCAL_WORK/base_repair_01"
  if ! remote_file "$REPAIR_REMOTE/validation.json"; then
    remote_exec "$REMOTE_PY" -m scripts.tools.silver_match_v3.prepare_transcript_isolation_repair_pack \
      --source-pack "$PACK" --failed-audit "$PACK/transcript.base.audit.json" \
      --output-root "$REPAIR_REMOTE"
  fi
  mkdir -p "$REPAIR_LOCAL"
  rsync -a "$REMOTE:$REPAIR_REMOTE/" "$REPAIR_LOCAL/"
  REPAIR_AUDIT="$REPAIR_LOCAL/transcript.audit.json"
  run_repair_or_resolver_pack "$REPAIR_LOCAL" notice-transcript-isolation-repair-01 "$REPAIR_AUDIT"
  sync_and_validate_pack "$REPAIR_LOCAL" "$REPAIR_REMOTE" "$REPAIR_AUDIT" \
    "$REPAIR_REMOTE/labels.validated.jsonl" "$REPAIR_REMOTE/labels.validation.json"
  if ! remote_file "$BASE_CLEAN_VALIDATION"; then
    remote_exec "$REMOTE_PY" -m scripts.tools.silver_match_v3.promote_transcript_isolation_repairs \
      --source-pack "$PACK" \
      --base-labels "$PACK/labels.validated.jsonl" \
      --base-validation "$PACK/labels.validation.json" \
      --failed-audit "$PACK/transcript.base.audit.json" \
      --repair-pack "$REPAIR_REMOTE" \
      --repair-labels "$REPAIR_REMOTE/labels.validated.jsonl" \
      --repair-validation "$REPAIR_REMOTE/labels.validation.json" \
      --repair-audit "$REPAIR_REMOTE/transcript.audit.json" \
      --output "$BASE_CLEAN_LABELS" --report "$BASE_CLEAN_VALIDATION"
  fi
else
  echo "base transcript audit did not produce PASS or FAIL" >&2
  exit 2
fi

# Both Gemma order runs must be complete and hash-bound before they count as a
# single secondary vote.
wait_file "$GEMMA/gemma.original.v2.jsonl.meta.json"
wait_file "$GEMMA/gemma.hashed.v2.jsonl.meta.json"

codex_packs=("$PACK")
codex_labels=("$BASE_CLEAN_LABELS")
codex_validations=("$BASE_CLEAN_VALIDATION")
FINAL_LABELS=
FINAL_VALIDATION=

for round in $(seq 1 "$MAX_ROUNDS"); do
  printf -v round_name 'round-%02d' "$round"
  output="$PACK/consensus/$round_name"
  if ! remote_file "$output/validation.json"; then
    command=(
      "$REMOTE_PY" -m scripts.tools.silver_match_v3.finalize_full_bank_multi_vote_consensus
      --pack-root "$PACK"
      --candidates "$GEMMA/candidates.full_bank.jsonl"
      --candidate-freeze "$GEMMA/FREEZE.json"
      --gemma-freeze "$GEMMA/REMOTE_INFERENCE_FREEZE.json"
      --gemma-retry-freeze "$GEMMA/RETRY_V2_FREEZE.json"
      --gemma-original "$GEMMA/gemma.original.v2.jsonl"
      --gemma-hashed "$GEMMA/gemma.hashed.v2.jsonl"
      --min-gemma-confidence medium
      --output-root "$output"
    )
    for index in "${!codex_packs[@]}"; do
      command+=(
        --codex-pack-root "${codex_packs[index]}"
        --codex-labels "${codex_labels[index]}"
        --codex-validation "${codex_validations[index]}"
      )
    done
    remote_exec "${command[@]}"
  fi
  unresolved="$(ssh "$REMOTE" "jq -r '.unresolved_count' '$output/validation.json'")"
  if test "$unresolved" -eq 0; then
    FINAL_LABELS="$output/labels.jsonl"
    FINAL_VALIDATION="$output/validation.json"
    break
  fi
  if test "$round" -eq "$MAX_ROUNDS"; then
    echo "N&C consensus remains unresolved after $MAX_ROUNDS rounds" >&2
    exit 3
  fi

  next=$((round + 1))
  printf -v next_name 'resolver-%02d' "$next"
  resolver_remote="$PACK/consensus/$next_name/label_pack"
  resolver_local="$LOCAL_WORK/$next_name"
  if ! remote_file "$resolver_remote/validation.json"; then
    remote_exec "$REMOTE_PY" -m scripts.tools.silver_match_v3.prepare_exact_unresolved_resolver_pack \
      --pack-root "$PACK" --unresolved "$output/unresolved.jsonl" \
      --output-root "$resolver_remote" --seed "$((161803 + next))" --chunk-size 25
  fi
  mkdir -p "$resolver_local"
  rsync -a "$REMOTE:$resolver_remote/" "$resolver_local/"
  resolver_audit="$resolver_local/transcript.audit.json"
  run_repair_or_resolver_pack "$resolver_local" "notice-multi-vote-$next_name" "$resolver_audit"
  resolver_labels="$resolver_remote/labels.validated.jsonl"
  resolver_validation="$resolver_remote/labels.validation.json"
  sync_and_validate_pack "$resolver_local" "$resolver_remote" "$resolver_audit" \
    "$resolver_labels" "$resolver_validation"
  codex_packs+=("$resolver_remote")
  codex_labels+=("$resolver_labels")
  codex_validations+=("$resolver_validation")
done

test -n "$FINAL_LABELS"
test -n "$FINAL_VALIDATION"

# The rescue chain is resume-safe; all GPU stages are already sealed, so this
# invocation performs the consensus-bound merge, exact audit, and blind-packet
# preparation without reopening model selection.
ssh "$REMOTE" \
  "cd '$REMOTE_REPO'; nohup env \
    NOTICE_MANUAL_UNRESOLVED_LABELS='$FINAL_LABELS' \
    NOTICE_MANUAL_UNRESOLVED_VALIDATION='$FINAL_VALIDATION' \
    bash scripts/tools/silver_match_v3/run_notice_rescue_chain_parallel.sh \
    >> '$PROD/notice_rescue_parallel.multi-vote-closure.log' 2>&1 < /dev/null & echo \$!"
wait_file "$RESCUE/final.audit.json"

echo "N&C multi-vote unresolved closure and exact final audit complete"
