#!/usr/bin/env bash
set -uo pipefail

# Label the frozen PR optimize exact-disagreement pack from a minimal staging
# tree. Earlier predictions, consensus truth, select labels, MI, and outcomes
# are absent from the staging directory.

REPO=/Users/spangher/Projects/stanford-research/norm-research
SOURCE="$REPO/outputs/silver_match_v3/press-releases/gepa_clean_v2/optimize_resolution_v1/resolver_pass_c"
STAGE=/private/tmp/silver_match_v3_pr_optimize_resolver_v1_escalated_retry
RUNNER="$REPO/scripts/tools/silver_match_v3/run_codex_pack_labels.py"
SCHEMA="$REPO/scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
GUIDE="$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

verify_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "hash mismatch: $path expected=$expected observed=$observed" >&2
    exit 2
  fi
}

verify_sha "$SOURCE/validation.json" 01db00ecd027629d0ec71d4a87dea20d7843cf89e5a2572322ca2a1cf1daf295
verify_sha "$SOURCE/items.jsonl" 7fc203f51ef21232cc73e641ab440677bbac068c598f45d48d2ca1234a466d66
verify_sha "$SOURCE/bank.json" fcd31a484a7f35ed4cabb357b702328eba2c4508bd3736f383663b216432d275
verify_sha "$RUNNER" cb49d940baf498a813d2631ccd3f6099c1aba41bc3df1140c1f40e71f47639ab
verify_sha "$SCHEMA" 9a67fd26e6a2c498bb591d76049a0eea02ea5bf96d41a5b37ae30b92e7e5c496
verify_sha "$GUIDE" 03e95ac5e072a9c79e2c88375753502fa82748d7152b1fad32ca0bffad4b19ad

if [[ -d "$SOURCE/raw_labels" ]] && find "$SOURCE/raw_labels" -name 'part-*.json' -type f -print -quit | grep -q .; then
  echo "refusing to relaunch resolver with existing raw labels: $SOURCE" >&2
  exit 2
fi
if [[ -e "$STAGE" ]]; then
  echo "refusing to reuse nonempty staging path: $STAGE" >&2
  exit 2
fi

mkdir -p "$STAGE/pack" "$STAGE/scripts/tools/silver_match_v3" "$SOURCE/runtime"
rsync -az "$SOURCE/" "$STAGE/pack/"
rsync -az "$GUIDE" "$STAGE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

PASS_NAME=gpt-5.6-sol-high-pr-clean-optimize-exact-resolver-v1
(
  cd "$STAGE" || exit 2
  PYTHONPATH="$REPO" python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
    --pack-root pack \
    --task press-releases \
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
exit "$STATUS"
