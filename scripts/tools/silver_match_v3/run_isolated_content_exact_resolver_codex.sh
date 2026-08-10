#!/usr/bin/env bash
# Label one content-task exact-consensus resolver pack from a minimal staging
# tree.  The staged worker can see the frozen items, a permuted full bank, and
# the independent-labeling guide, but no earlier votes or resolved truth.
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TASK ROLE PASS_ORDINAL" >&2
  exit 2
fi

TASK=$1
ROLE=$2
ORDINAL=$3
case "$TASK" in
  creative-writing|legal-outcome-prediction|peer-review|press-releases) ;;
  *) echo "unsupported content task: $TASK" >&2; exit 2 ;;
esac
case "$ROLE" in
  optimize|select) ;;
  *) echo "unsupported panel role: $ROLE" >&2; exit 2 ;;
esac
case "$ORDINAL" in
  3) LETTER=c ;;
  4) LETTER=d ;;
  5) LETTER=e ;;
  6) LETTER=f ;;
  7) LETTER=g ;;
  8) LETTER=h ;;
  *) echo "resolver pass ordinal must be in [3, 8]" >&2; exit 2 ;;
esac

REPO=/Users/spangher/Projects/stanford-research/norm-research
if [[ "$TASK" == "press-releases" ]]; then
  GEPA_DIR=gepa_clean_v2
else
  GEPA_DIR=gepa_clean_v1
fi
SOURCE="$REPO/outputs/silver_match_v3/$TASK/$GEPA_DIR/${ROLE}_resolution_v1/resolver_pass_${LETTER}"
STAGE="/private/tmp/silver_match_v3_content_exact_resolver/$TASK/$ROLE/pass_${LETTER}"
SCHEMA="$REPO/scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
GUIDE="$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

for NAME in validation.json items.jsonl bank.json; do
  [[ -f "$SOURCE/$NAME" ]] || {
    echo "missing frozen resolver artifact: $SOURCE/$NAME" >&2
    exit 2
  }
done

EXPECTED=$(find "$SOURCE/chunks" -maxdepth 1 -name 'part-*.jsonl' -type f | wc -l | tr -d ' ')
OBSERVED=$(find "$SOURCE/raw_labels" -maxdepth 1 -name 'part-*.json' -type f 2>/dev/null | wc -l | tr -d ' ')
if [[ "$EXPECTED" == "0" || "$OBSERVED" -gt "$EXPECTED" ]]; then
  echo "invalid resolver chunk state: expected=$EXPECTED observed=$OBSERVED" >&2
  exit 2
fi
if [[ "$OBSERVED" == "$EXPECTED" ]]; then
  echo "refusing to relaunch completed resolver: $SOURCE" >&2
  exit 2
fi

mkdir -p "$STAGE/pack" "$STAGE/scripts/tools/silver_match_v3" "$SOURCE/runtime"
if [[ -f "$STAGE/pack/validation.json" ]]; then
  SOURCE_SHA=$(shasum -a 256 "$SOURCE/validation.json" | awk '{print $1}')
  STAGE_SHA=$(shasum -a 256 "$STAGE/pack/validation.json" | awk '{print $1}')
  [[ "$SOURCE_SHA" == "$STAGE_SHA" ]] || {
    echo "staging validation differs from frozen source pack" >&2
    exit 2
  }
fi
rsync -az "$SOURCE/" "$STAGE/pack/"
rsync -az "$GUIDE" "$STAGE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

PASS_NAME="gpt-5.6-sol-high-${TASK}-${ROLE}-exact-resolver-pass-${LETTER}-v1"
echo "resolver_pid=$$ task=$TASK role=$ROLE pass=$LETTER expected_chunks=$EXPECTED observed_chunks=$OBSERVED"
(
  cd "$STAGE" || exit 2
  PYTHONPATH="$REPO" python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
    --pack-root pack \
    --task "$TASK" \
    --pass-name "$PASS_NAME" \
    --concurrency 2 \
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
echo "finished task=$TASK role=$ROLE pass=$LETTER status=$STATUS"
exit "$STATUS"
