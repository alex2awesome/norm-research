#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_binary_v1
CODE=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2/code
PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python
RUN=$ROOT/production_reducer/humor-k200-minus-joined22090.seed-2026071502.exposure100k.v1
EVENTS=$RUN/watcher.events.log
MERGED=$RUN/scores.merged.jsonl
EXPECTED_SHARDS=6

test -d "$RUN"
test ! -e "$EVENTS"
test ! -e "$MERGED"
printf '%s WATCH_STARTED expected_shards=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPECTED_SHARDS" >>"$EVENTS"

while :; do
  alive=0
  for pid_file in "$RUN"/pids/*.pid; do
    pid=$(tr -d '[:space:]' <"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  if test "$alive" -eq 0; then
    break
  fi
  sleep 300
done

inputs=()
for shard in $(seq 0 $((EXPECTED_SHARDS - 1))); do
  name=$(printf 'scores.shard-%03d-of-%03d.jsonl' "$shard" "$EXPECTED_SHARDS")
  path=$RUN/$name
  test -s "$path"
  test -s "$path.meta.json"
  inputs+=("$path")
done
printf '%s SHARDS_COMPLETE count=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPECTED_SHARDS" >>"$EVENTS"

cd "$CODE"
"$PYTHON" -u -m scripts.tools.silver_match_v3.run_nemotron_ce merge \
  --inputs "${inputs[@]}" \
  --output "$MERGED" \
  >>"$RUN/merge.log" 2>&1
test -s "$MERGED"
test -s "$MERGED.meta.json"
printf '%s MERGE_COMPLETE scores_sha256=%s meta_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(sha256sum "$MERGED" | awk '{print $1}')" \
  "$(sha256sum "$MERGED.meta.json" | awk '{print $1}')" \
  >>"$EVENTS"

