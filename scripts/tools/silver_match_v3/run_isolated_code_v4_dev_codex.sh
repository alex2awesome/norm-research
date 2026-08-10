#!/usr/bin/env bash
# Run one Code V4 development-label pass in a minimal, mutually hidden staging
# tree.  The annotator sees only its permuted bank, assigned item chunk, and the
# independent-labeling guide.  Pass A and B never share a working directory.
set -uo pipefail

if [[ $# -ne 1 || ( "$1" != "A" && "$1" != "B" ) ]]; then
  echo "usage: $0 A|B" >&2
  exit 2
fi

PASS=$1
LETTER=$(printf '%s' "$PASS" | tr '[:upper:]' '[:lower:]')
REPO=/lfs/skampere3/0/alexspan/norm-research
ROOT=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context/cross_encoder_v4/code-review/fresh_dev300_stratified
SOURCE="$ROOT/label_views/pass_$LETTER"
STAGE="/lfs/skampere3/0/alexspan/staging/silver_match_v3_code_v4_dev/pass_$PASS"
SCHEMA="$REPO/scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
GUIDE="$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"
PYTHON=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python

for path in "$SOURCE/validation.json" "$SOURCE/items.jsonl" "$SOURCE/bank.json" "$SCHEMA" "$GUIDE"; do
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
done
EXPECTED=$(find "$SOURCE/chunks" -maxdepth 1 -name 'part-*.jsonl' -type f | wc -l | tr -d ' ')
[[ "$EXPECTED" -gt 0 ]] || { echo "source pack has no chunks" >&2; exit 2; }

mkdir -p "$STAGE/pack" "$STAGE/scripts/tools/silver_match_v3" "$SOURCE/runtime"
if [[ -f "$STAGE/pack/validation.json" ]]; then
  SOURCE_SHA=$(sha256sum "$SOURCE/validation.json" | awk '{print $1}')
  STAGE_SHA=$(sha256sum "$STAGE/pack/validation.json" | awk '{print $1}')
  [[ "$SOURCE_SHA" == "$STAGE_SHA" ]] || {
    echo "staged validation differs from the frozen source view" >&2
    exit 2
  }
fi
rsync -a "$SOURCE/" "$STAGE/pack/"
rsync -a "$GUIDE" "$STAGE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

PASS_NAME="gpt-5.6-sol-high-code-v4-dev-reference-pass-$PASS"
echo "runner_pid=$$ pass=$PASS expected_chunks=$EXPECTED"
(
  cd "$STAGE" || exit 2
  export PATH="/lfs/skampere3/0/alexspan/.npm-global/bin:$PATH"
  PYTHONPATH="$REPO" "$PYTHON" -m scripts.tools.silver_match_v3.run_codex_pack_labels \
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

for name in raw_labels logs invalid_raw_labels; do
  if [[ -d "$STAGE/pack/$name" ]]; then
    mkdir -p "$SOURCE/$name"
    rsync -a --ignore-existing "$STAGE/pack/$name/" "$SOURCE/$name/"
  fi
done
rsync -a "$STAGE/runner.log" "$SOURCE/runtime/runner.log"
printf '%s\n' "$STATUS" >"$SOURCE/runtime/exit_code"
printf '%s\n' "$PASS_NAME" >"$SOURCE/runtime/pass_name"
echo "finished pass=$PASS status=$STATUS"
exit "$STATUS"
