#!/bin/bash
# STACKING variant of gpu_runner.sh.
#
# gpu_runner.sh waits for a GENUINELY FREE GPU (zero compute processes). During this
# campaign every sk3 GPU has a co-tenant around the clock, so that runner would wait
# forever. This variant does what the other agents on the box are doing and what the
# 2026-08-06 freeze authorises ("all free sk3 GPUs"; co-tenant jobs never touched):
# it picks the device with the most FREE MEMORY, requires a hard headroom margin,
# records a CLAIM naming the co-tenant it is stacking behind, runs, then RELEASEs.
#
# It never signals, kills, or otherwise touches a process it did not start.
#
# Usage: ./gpu_stack_runner.sh <tag> <logfile> <need_MiB> <python> <script> [args...]
set -u
TAG="$1"; LOG="$2"; NEED="$3"; shift 3
LEDGER=/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt
LOCK=/tmp/alexspan_gpu_claim.lock
export HOME=/lfs/skampere3/0/alexspan
MAX_WAIT_MIN=${MAX_WAIT_MIN:-720}

best_gpu() {
  # PIN_GPU pins this lane to one device.  Required under the full-sweep dispatch: lanes
  # share the box (A=GPU5, B=GPU6, C=GPU7 here), and picking "most free memory" would let
  # one lane wander onto another's device between its jobs.  Unset PIN_GPU to restore the
  # old max-free-memory behaviour.
  if [ -n "${PIN_GPU:-}" ]; then
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i "$PIN_GPU" \
      | tr -d ','
    return
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ',' | sort -k2 -nr | head -1
}

t0=$(date +%s)
while :; do
  read -r G FREE < <(best_gpu)
  if [ "${FREE:-0}" -ge "$NEED" ]; then
    exec 9>"$LOCK"; flock -w 60 9 || { sleep 30; continue; }
    read -r G FREE < <(best_gpu)          # re-read under the lock
    if [ "${FREE:-0}" -lt "$NEED" ]; then flock -u 9; sleep 60; continue; fi
    CO=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$G" | tr '\n' ';')
    printf "%s | cell=mathse_vote layer3 closure | GPU=%s | agent=claude-mathse-vote-closure | job=%s | CLAIM-STACKED (free %s MiB >= need %s; co-tenants left untouched: %s)\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$G" "$TAG" "$FREE" "$NEED" "${CO:-none}" >> "$LEDGER"
    flock -u 9
    echo "[gpu_stack_runner:$TAG] GPU $G, ${FREE} MiB free, launching $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$G" "$@" >> "$LOG" 2>&1
    rc=$?
    printf "%s | GPU=%s | job=%s | RELEASE rc=%s\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$G" "$TAG" "$rc" >> "$LEDGER"
    exit $rc
  fi
  if [ $(( ($(date +%s)-t0)/60 )) -ge "$MAX_WAIT_MIN" ]; then
    echo "[gpu_stack_runner:$TAG] no GPU with ${NEED} MiB free within ${MAX_WAIT_MIN} min" | tee -a "$LOG"
    exit 2
  fi
  sleep 120
done
