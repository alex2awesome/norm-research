#!/usr/bin/env bash
# Run one task/pass queue in a minimal staging tree so pass A/B predictions are
# never present in the other pass's working directory. Each pack is copied in
# before inference and its raw outputs are synced back append-only afterwards.
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TASK PASS" >&2
  exit 2
fi

TASK=$1
PASS=$2
if [[ "$TASK" != "code-review" && "$TASK" != "math-stackexchange" ]]; then
  echo "unsupported task: $TASK" >&2
  exit 2
fi
if [[ "$PASS" != "A" && "$PASS" != "B" ]]; then
  echo "PASS must be A or B" >&2
  exit 2
fi

REPO=/Users/spangher/Projects/stanford-research/norm-research
ROOT="$REPO/outputs/silver_match_v3/task_local_gepa_clean_v1/$TASK"
STAGING_ROOT="/private/tmp/silver_match_v3_clean_gepa_codex/$TASK/pass_$PASS"
SCHEMA="$REPO/scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
GUIDE="$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"
FAILURES=0
echo "queue_pid=$$ task=$TASK pass=$PASS planned_items=160 planned_chunks=8"

for PANEL in optimize100 select60; do
  SOURCE="$ROOT/$PANEL/independent_pass_$PASS"
  STAGE="$STAGING_ROOT/$PANEL"
  if [[ ! -f "$SOURCE/validation.json" || ! -f "$SOURCE/items.jsonl" || ! -f "$SOURCE/bank.json" ]]; then
    echo "missing frozen source pack: $SOURCE" >&2
    FAILURES=$((FAILURES + 1))
    continue
  fi
  mkdir -p "$STAGE/pack" "$STAGE/scripts/tools/silver_match_v3" "$SOURCE/runtime"
  rsync -az "$SOURCE/" "$STAGE/pack/"
  rsync -az "$GUIDE" "$STAGE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

  PASS_NAME="gpt-5.6-sol-high-clean-gepa-${TASK}-${PANEL}-pass-${PASS}"
  echo "starting task=$TASK pass=$PASS panel=$PANEL"
  (
    cd "$STAGE"
    PYTHONPATH="$REPO" python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
      --pack-root pack \
      --task "$TASK" \
      --pass-name "$PASS_NAME" \
      --concurrency 1 \
      --model gpt-5.6-sol \
      --reasoning-effort high \
      --timeout-seconds 1800 \
      --chunk-attempts 3 \
      --output-schema "$SCHEMA"
  ) >"$STAGE/runner.log" 2>&1
  STATUS=$?

  if [[ -d "$STAGE/pack/raw_labels" ]]; then
    mkdir -p "$SOURCE/raw_labels"
    rsync -az "$STAGE/pack/raw_labels/" "$SOURCE/raw_labels/"
  fi
  if [[ -d "$STAGE/pack/logs" ]]; then
    mkdir -p "$SOURCE/logs"
    rsync -az "$STAGE/pack/logs/" "$SOURCE/logs/"
  fi
  if [[ -d "$STAGE/pack/invalid_raw_labels" ]]; then
    mkdir -p "$SOURCE/invalid_raw_labels"
    rsync -az "$STAGE/pack/invalid_raw_labels/" "$SOURCE/invalid_raw_labels/"
  fi
  rsync -az "$STAGE/runner.log" "$SOURCE/runtime/runner.log"
  printf '%s\n' "$STATUS" >"$SOURCE/runtime/exit_code"
  printf '%s\n' "$PASS_NAME" >"$SOURCE/runtime/pass_name"
  echo "finished task=$TASK pass=$PASS panel=$PANEL status=$STATUS"
  if [[ $STATUS -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
  fi
done

exit "$FAILURES"
