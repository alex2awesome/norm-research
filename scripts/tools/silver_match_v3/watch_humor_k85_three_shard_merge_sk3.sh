#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere3/0/alexspan/runtime/humor_ce_k85_complement_v1
RUN=$ROOT/scores.seed-2026071502.exposure100k.k85.v1
CODE=$ROOT/code
PYTHON=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
EVENTS=$RUN/merge.watcher.events.log
MERGED=$RUN/scores.merged.jsonl
EXPECTED_SHARDS=3

test -d "$RUN"
test ! -e "$EVENTS"
test ! -e "$MERGED"
printf '%s WATCH_STARTED expected_shards=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPECTED_SHARDS" >>"$EVENTS"

while :; do
  alive=0
  for pid_file in "$RUN"/pids/*.pid; do
    pid=$(tr -d '[:space:]' <"$pid_file")
    kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)) || true
  done
  test "$alive" -eq 0 && break
  sleep 300
done

inputs=()
for shard in $(seq 0 $((EXPECTED_SHARDS - 1))); do
  name=$(printf 'scores.shard-%03d-of-%03d.jsonl' "$shard" "$EXPECTED_SHARDS")
  path=$RUN/$name
  test -s "$path"
  test -s "$path.meta.json"
  test "$(jq -r .row_count "$path.meta.json")" -gt 0
  test "$(jq -r .num_shards "$path.meta.json")" = "$EXPECTED_SHARDS"
  test "$(jq -r .shard_id "$path.meta.json")" = "$shard"
  inputs+=("$path")
done
printf '%s SHARDS_COMPLETE count=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPECTED_SHARDS" >>"$EVENTS"

cd "$CODE"
"$PYTHON" -u -m scripts.tools.silver_match_v3.run_nemotron_ce merge \
  --inputs "${inputs[@]}" --output "$MERGED" >>"$RUN/merge.log" 2>&1
test -s "$MERGED"
test -s "$MERGED.meta.json"
test "$(jq -r .row_count "$MERGED.meta.json")" = 4699480
test "$(jq -r .norm_group_count "$MERGED.meta.json")" = 55288
printf '%s MERGE_COMPLETE scores_sha256=%s meta_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(sha256sum "$MERGED" | awk '{print $1}')" \
  "$(sha256sum "$MERGED.meta.json" | awk '{print $1}')" >>"$EVENTS"

