#!/usr/bin/env bash
set -uo pipefail

# Fifth independent pass over the final two PR optimize rows. No previous
# votes, select truth, MI, or outcomes are staged.
REPO=/Users/spangher/Projects/stanford-research/norm-research
SOURCE="$REPO/outputs/silver_match_v3/press-releases/gepa_clean_v2/optimize_resolution_v1/resolver_pass_e"
STAGE=/private/tmp/silver_match_v3_pr_optimize_unresolved_resolver_v2
RUNNER="$REPO/scripts/tools/silver_match_v3/run_codex_pack_labels.py"
SCHEMA="$REPO/scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
GUIDE="$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

verify_sha() {
  local path="$1" expected="$2" observed
  observed="$(shasum -a 256 "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || { echo "hash mismatch: $path" >&2; exit 2; }
}

verify_sha "$SOURCE/validation.json" de61f91c38afc6f2af694a26e9a854e0b23c2ee761288ae7dd33de637a57e5aa
verify_sha "$SOURCE/items.jsonl" f3c175cf21296a83e508c4886d8acfabc4e368da53a32fdc465657fee2219180
verify_sha "$SOURCE/bank.json" 1f8eae914ff7fb9b4bc48d28920318aa2579bdb91c5a13facb32dbae4f6fa458
verify_sha "$RUNNER" cb49d940baf498a813d2631ccd3f6099c1aba41bc3df1140c1f40e71f47639ab
verify_sha "$SCHEMA" 9a67fd26e6a2c498bb591d76049a0eea02ea5bf96d41a5b37ae30b92e7e5c496
verify_sha "$GUIDE" 03e95ac5e072a9c79e2c88375753502fa82748d7152b1fad32ca0bffad4b19ad

if [[ -d "$SOURCE/raw_labels" ]] && find "$SOURCE/raw_labels" -name 'part-*.json' -type f -print -quit | grep -q .; then
  echo "refusing to relaunch completed resolver" >&2
  exit 2
fi
[[ ! -e "$STAGE" ]] || { echo "refusing to reuse staging path" >&2; exit 2; }
mkdir -p "$STAGE/pack" "$STAGE/scripts/tools/silver_match_v3" "$SOURCE/runtime"
rsync -az "$SOURCE/" "$STAGE/pack/"
rsync -az "$GUIDE" "$STAGE/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"

PASS_NAME=gpt-5.6-sol-high-pr-clean-optimize-exact-unresolved-resolver-v2
(
  cd "$STAGE" || exit 2
  PYTHONPATH="$REPO" python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
    --pack-root pack --task press-releases --pass-name "$PASS_NAME" \
    --concurrency 1 --model gpt-5.6-sol --reasoning-effort high \
    --timeout-seconds 1800 --chunk-attempts 3 --output-schema "$SCHEMA"
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
