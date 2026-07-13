#!/usr/bin/env bash
set -euo pipefail

# Resume the two Tier B shards evicted from prohibited sk3 GPUs 3 and 4.
# They wait for the normal Tier B work on allowed GPUs 6 and 7, then reuse
# those devices and their existing content-addressed caches.
ROOT=${V13_ROOT:-/lfs/skampere3/0/alexspan/cr3-v13.1}
POLL_SECONDS=${V13_POLL_SECONDS:-30}

resume_after() {
    local predecessor=$1
    local shard=$2
    local device=$3
    local predecessor_manifest="$ROOT/outputs/tier_b/lanes/llama33_70b_shard_$predecessor/campaign_manifest.json"

    until [[ -f "$predecessor_manifest" ]]; do
        sleep "$POLL_SECONDS"
    done

    V13_SHARD_IDS="$shard" V13_DEVICE_OVERRIDE="$device" \
        "$ROOT/scripts/run_v13_llama70_shards_sk3.sh"
}

resume_after 4 1 6 &
resume_after 5 2 7 &
wait
