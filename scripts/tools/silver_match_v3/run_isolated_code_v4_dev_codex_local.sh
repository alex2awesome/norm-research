#!/usr/bin/env bash
# Local fallback for Code V4 reference labels when the sk3 Codex credential is
# unavailable.  Immutable inputs are copied into pass-specific minimal staging
# trees; only raw outputs/transcripts are synced back to the remote view.
set -uo pipefail

if [[ $# -ne 1 || ( "$1" != "A" && "$1" != "B" ) ]]; then
  echo "usage: $0 A|B" >&2
  exit 2
fi

PASS=$1
LETTER=$(printf '%s' "$PASS" | tr '[:upper:]' '[:lower:]')
REPO=/Users/spangher/Projects/stanford-research/norm-research
REMOTE_ROOT=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context/cross_encoder_v4/code-review/fresh_dev300_stratified
REMOTE_SOURCE="$REMOTE_ROOT/label_views/pass_$LETTER"
STAGE="/private/tmp/silver_match_v3_code_v4_dev_local/pass_$PASS"
SCHEMA="$REPO/scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
GUIDE="$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

mkdir -p "$STAGE/pack" "$STAGE/scripts/tools/silver_match_v3"
rsync -a \
  --exclude raw_labels \
  --exclude logs \
  --exclude invalid_raw_labels \
  --exclude runtime \
  "sk3:$REMOTE_SOURCE/" "$STAGE/pack/"
rsync -a "$GUIDE" "$STAGE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"
for path in "$STAGE/pack/validation.json" "$STAGE/pack/items.jsonl" "$STAGE/pack/bank.json" "$SCHEMA"; do
  [[ -f "$path" ]] || { echo "missing staged frozen input: $path" >&2; exit 2; }
done
EXPECTED=$(find "$STAGE/pack/chunks" -maxdepth 1 -name 'part-*.jsonl' -type f | wc -l | tr -d ' ')
[[ "$EXPECTED" == "12" ]] || { echo "expected 12 chunks, found $EXPECTED" >&2; exit 2; }

PASS_NAME="gpt-5.6-sol-high-code-v4-dev-reference-pass-$PASS-local-fallback"
echo "runner_pid=$$ pass=$PASS expected_chunks=$EXPECTED"
(
  cd "$STAGE" || exit 2
  PYTHONPATH="$REPO" python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
    --pack-root pack \
    --task code-review \
    --pass-name "$PASS_NAME" \
    --concurrency 2 \
    --model gpt-5.6-sol \
    --reasoning-effort high \
    --timeout-seconds 1800 \
    --chunk-attempts 3 \
    --output-schema "$SCHEMA"
) >"$STAGE/runner.log" 2>&1
STATUS=$?

ssh sk3 "mkdir -p '$REMOTE_SOURCE/raw_labels' '$REMOTE_SOURCE/logs' '$REMOTE_SOURCE/invalid_raw_labels' '$REMOTE_SOURCE/runtime'"
for name in raw_labels logs invalid_raw_labels; do
  if [[ -d "$STAGE/pack/$name" ]]; then
    rsync -a --ignore-existing "$STAGE/pack/$name/" "sk3:$REMOTE_SOURCE/$name/"
  fi
done
rsync -a "$STAGE/runner.log" "sk3:$REMOTE_SOURCE/runtime/local_fallback.runner.log"
printf '%s\n' "$STATUS" >"$STAGE/exit_code"
printf '%s\n' "$PASS_NAME" >"$STAGE/pass_name"
rsync -a "$STAGE/exit_code" "sk3:$REMOTE_SOURCE/runtime/local_fallback.exit_code"
rsync -a "$STAGE/pass_name" "sk3:$REMOTE_SOURCE/runtime/local_fallback.pass_name"
echo "finished pass=$PASS status=$STATUS"
exit "$STATUS"
