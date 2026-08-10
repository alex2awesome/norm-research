#!/usr/bin/env bash
# Label a predeclared next-round exact-unresolved pack in an isolated staging
# tree. The source pack contains original items only, never prior votes.
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TASK PANEL" >&2
  exit 2
fi

TASK=$1
PANEL=$2
if [[ "$TASK" != "code-review" && "$TASK" != "math-stackexchange" ]]; then
  echo "unsupported task: $TASK" >&2
  exit 2
fi
if [[ "$PANEL" != "optimize100" && "$PANEL" != "select60" ]]; then
  echo "unsupported panel: $PANEL" >&2
  exit 2
fi

REPO=/Users/spangher/Projects/stanford-research/norm-research
SOURCE="$REPO/outputs/silver_match_v3/task_local_gepa_clean_v1/$TASK/$PANEL/exact_unresolved_resolver_round2_v1"
STAGE="/private/tmp/silver_match_v3_clean_gepa_codex/$TASK/resolver_round2_$PANEL"
SCHEMA="$REPO/scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
GUIDE="$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

if [[ ! -f "$SOURCE/validation.json" || ! -f "$SOURCE/items.jsonl" || ! -f "$SOURCE/bank.json" ]]; then
  echo "missing frozen exact-unresolved round-2 pack: $SOURCE" >&2
  exit 2
fi
if [[ -d "$SOURCE/raw_labels" ]] && find "$SOURCE/raw_labels" -name 'part-*.json' -type f -print -quit | grep -q .; then
  echo "refusing to relaunch round-2 resolver with existing raw labels: $SOURCE" >&2
  exit 2
fi

mkdir -p "$STAGE/pack" "$STAGE/scripts/tools/silver_match_v3" "$SOURCE/runtime"
rsync -az "$SOURCE/" "$STAGE/pack/"
rsync -az "$GUIDE" "$STAGE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

PASS_NAME="gpt-5.6-sol-high-clean-gepa-${TASK}-${PANEL}-exact-resolver-round2-v1"
echo "resolver_pid=$$ task=$TASK panel=$PANEL round=2"
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

for NAME in raw_labels logs invalid_raw_labels; do
  if [[ -d "$STAGE/pack/$NAME" ]]; then
    mkdir -p "$SOURCE/$NAME"
    rsync -az "$STAGE/pack/$NAME/" "$SOURCE/$NAME/"
  fi
done
rsync -az "$STAGE/runner.log" "$SOURCE/runtime/runner.log"
printf '%s\n' "$STATUS" >"$SOURCE/runtime/exit_code"
printf '%s\n' "$PASS_NAME" >"$SOURCE/runtime/pass_name"
echo "finished task=$TASK panel=$PANEL round=2 status=$STATUS"
exit "$STATUS"
