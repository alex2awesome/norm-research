#!/usr/bin/env bash
set -euo pipefail

RUN=/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_binary_v1/production_reducer/humor-k200-minus-joined22090.seed-2026071502.exposure100k.v1
SOURCE=$RUN/scores.merged.jsonl
SOURCE_META=$SOURCE.meta.json
EVENTS=$RUN/full285-transfer.events.log
REMOTE_HOST=skampere3
REMOTE_ROOT=/lfs/skampere3/0/alexspan/runtime/humor_ce_k85_complement_v1/incoming_k200
REMOTE_SCORES=$REMOTE_ROOT/scores.k200.merged.jsonl
REMOTE_META=$REMOTE_SCORES.meta.json

test ! -e "$EVENTS"
printf '%s WATCH_STARTED\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$EVENTS"
while ! test -s "$SOURCE_META"; do sleep 300; done
test -s "$SOURCE"
test "$(jq -r .row_count "$SOURCE_META")" = 11057600
test "$(jq -r .norm_group_count "$SOURCE_META")" = 55288
test "$(jq -r .classification_mode "$SOURCE_META")" = binary
test "$(jq -r .checkpoint_contract.checkpoint_metadata_sha256 "$SOURCE_META")" = 76a58ba823fc3895a292b71d9cbee8a1e81314dfbf9762aa111ea3b4ea1d98d2
test "$(jq -r .input_pairs_sha256 "$SOURCE_META")" = be6561860d9657490b365045374e84260038ff5f80bc8427afabd9640351057f
SOURCE_SHA=$(sha256sum "$SOURCE" | awk '{print $1}')
META_SHA=$(sha256sum "$SOURCE_META" | awk '{print $1}')
test "$SOURCE_SHA" = "$(jq -r .output_sha256 "$SOURCE_META")"
ssh "$REMOTE_HOST" "set -eu; mkdir -p '$REMOTE_ROOT'; test ! -e '$REMOTE_SCORES'; test ! -e '$REMOTE_META'; test ! -e '$REMOTE_SCORES.partial'; test ! -e '$REMOTE_META.partial'"
printf '%s TRANSFER_STARTED scores_sha256=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_SHA" >>"$EVENTS"
rsync -a --partial "$SOURCE" "$REMOTE_HOST:$REMOTE_SCORES.partial"
rsync -a --partial "$SOURCE_META" "$REMOTE_HOST:$REMOTE_META.partial"
ssh "$REMOTE_HOST" "set -eu; test \"\$(sha256sum '$REMOTE_SCORES.partial' | cut -d' ' -f1)\" = '$SOURCE_SHA'; test \"\$(sha256sum '$REMOTE_META.partial' | cut -d' ' -f1)\" = '$META_SHA'; mv '$REMOTE_SCORES.partial' '$REMOTE_SCORES'; mv '$REMOTE_META.partial' '$REMOTE_META'"
printf '%s TRANSFER_COMPLETE scores_sha256=%s meta_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_SHA" "$META_SHA" >>"$EVENTS"
