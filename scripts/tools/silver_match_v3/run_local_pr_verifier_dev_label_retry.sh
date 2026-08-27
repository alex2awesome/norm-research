#!/usr/bin/env bash
# Isolated local retry for the PR verifier-dev truth passes after the sk3
# Codex credential failed closed. The two calls use disjoint staged views.
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PASS_NAME STAGE REMOTE_VIEW" >&2
  exit 2
fi

PASS_NAME=$1
STAGE=$2
REMOTE_VIEW=$3
REPO=/Users/spangher/Projects/stanford-research/norm-research

cd "$STAGE" || exit 2
PYTHONPATH="$REPO" python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
  --pack-root pack \
  --task press-releases \
  --pass-name "$PASS_NAME" \
  --concurrency 1 \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --timeout-seconds 1800 \
  --chunk-attempts 2 \
  --output-schema scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json
status=$?

for name in raw_labels logs invalid_raw_labels; do
  if [[ -d "pack/$name" ]]; then
    rsync -rlt --no-perms --ignore-existing "pack/$name/" "sk3:$REMOTE_VIEW/$name/"
  fi
done
ssh sk3 "mkdir -p '$REMOTE_VIEW/runtime' && printf '%s\n' '$status' >'$REMOTE_VIEW/runtime/local_retry_exit_code' && printf '%s\n' '$PASS_NAME' >'$REMOTE_VIEW/runtime/local_retry_pass_name'"
exit "$status"
