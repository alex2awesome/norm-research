#!/bin/bash
# LANE-PINNED GPU runner for the jokes_community closure campaign.
#
# The full-sweep queue (notes/2026-08-09__full_sweep_queue.md) assigns this campaign
# LANE A = GPU 5.  gpu_stack_runner.sh picks whichever card has the most free memory,
# which would wander out of the lane and into another lane's card, so this variant pins
# the device, waits for it to have the required headroom, claims it in the shared ledger
# naming any co-tenant it stacks behind, runs, and releases.
#
# It never signals, kills, or otherwise touches a process it did not start.
#
# Usage: ./gpu_lane_runner.sh <tag> <logfile> <gpu_index> <need_MiB> <python> <script> [args...]
set -u
TAG="$1"; LOG="$2"; GPU="$3"; NEED="$4"; shift 4
LEDGER=/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt
LOCK=/tmp/alexspan_gpu_claim.lock
export HOME=/lfs/skampere3/0/alexspan
MAX_WAIT_MIN=${MAX_WAIT_MIN:-720}

free_on() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

t0=$(date +%s)
while :; do
  FREE=$(free_on "$GPU")
  if [ "${FREE:-0}" -ge "$NEED" ]; then
    exec 9>"$LOCK"; flock -w 60 9 || { sleep 30; continue; }
    FREE=$(free_on "$GPU")                # re-read under the lock
    if [ "${FREE:-0}" -lt "$NEED" ]; then flock -u 9; sleep 60; continue; fi
    CO=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" | tr '\n' ';')
    printf "%s | cell=jokes_community layer3 closure (LANE A) | GPU=%s | agent=claude-jokes-closure | job=%s | CLAIM (lane-pinned; free %s MiB >= need %s; co-tenants left untouched: %s)\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU" "$TAG" "$FREE" "$NEED" "${CO:-none}" >> "$LEDGER"
    flock -u 9
    echo "[gpu_lane_runner:$TAG] GPU $GPU, ${FREE} MiB free, launching $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "$@" >> "$LOG" 2>&1
    rc=$?
    printf "%s | GPU=%s | agent=claude-jokes-closure | job=%s | RELEASE rc=%s\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU" "$TAG" "$rc" >> "$LEDGER"
    exit $rc
  fi
  if [ $(( ($(date +%s)-t0)/60 )) -ge "$MAX_WAIT_MIN" ]; then
    echo "[gpu_lane_runner:$TAG] GPU $GPU never had ${NEED} MiB free within ${MAX_WAIT_MIN} min" | tee -a "$LOG"
    exit 2
  fi
  sleep 120
done
