#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere3/0/alexspan/runtime/humor_ce_k85_complement_v1
DEPLOY=$ROOT/full285_c2_deployment_v1
CODE=$ROOT/deploy_code
PYTHON=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
BANK=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks/humor.json
OUTPUT=$DEPLOY/early_manual_audit_first_sealed_v1
EVENTS=$ROOT/early_manual_audit.watcher.events.log

test ! -e "$EVENTS"
test ! -e "$OUTPUT"
printf '%s EARLY_AUDIT_WATCH_STARTED\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$EVENTS"

meta=
while test -z "$meta"; do
  if test -d "$DEPLOY/typed"; then
    meta=$(find "$DEPLOY/typed" -mindepth 2 -maxdepth 2 -name INFERENCE_META.json \
      -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n 1 | cut -d' ' -f2- || true)
  fi
  test -n "$meta" || sleep 30
done
typed_root=$(dirname "$meta")
printf '%s FIRST_C2_SHARD_SEALED typed_root=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$typed_root" >>"$EVENTS"

cd "$CODE"
nice -n 10 ionice -c 3 "$PYTHON" -u -m scripts.tools.silver_match_v3.build_humor_c2_early_manual_audit \
  --candidate-package "$DEPLOY/candidates.top16-plus-positives.jsonl" \
  --prompts "$DEPLOY/paired_order.prompts.jsonl" \
  --bank "$BANK" --typed-root "$typed_root" --output-root "$OUTPUT" \
  >"$ROOT/early_manual_audit.build.log" 2>&1
test "$(jq -r .status "$OUTPUT/REPORT.json")" = COMPLETE_TRUTH_BLIND_FIRST_SEALED_SHARD_AUDIT_PACKET
printf '%s EARLY_AUDIT_PACKET_COMPLETE packet=%s sha=%s rows=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUTPUT/EARLY_MANUAL_AUDIT_PACKET.jsonl" \
  "$(jq -r .output.sha256 "$OUTPUT/REPORT.json")" "$(jq -r .selected_total "$OUTPUT/REPORT.json")" >>"$EVENTS"
