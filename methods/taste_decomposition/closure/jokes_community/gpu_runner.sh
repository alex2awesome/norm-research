#!/bin/bash
# Wait for a genuinely FREE sk3 GPU (zero compute processes), claim it in the
# shared ledger, run the given command pinned to it, then release the claim.
#
# Written after the 2026-08-07 03:57Z incident: this cell's Gemma job was
# SIGTERMed on GPU 6 mid-run (another tenant took the device 20 minutes later),
# and the box then had no free GPU at all.  Polling + claim + re-verify is the
# only safe way to proceed under the "never touch a GPU with a co-tenant" rule.
#
# Usage: ./gpu_runner.sh <tag> <logfile> <python> <script> [args...]
set -u
TAG="$1"; LOG="$2"; shift 2
LEDGER=/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt
LOCK=/tmp/alexspan_gpu_claim.lock
export HOME=/lfs/skampere3/0/alexspan
export NR_REPO=/lfs/skampere3/0/alexspan/norm-research
MAX_WAIT_MIN=${MAX_WAIT_MIN:-720}
RETRIES=${RETRIES:-30}

free_gpu() {
  local busy idx
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  while read -r idx uuid; do
    idx=${idx%,}
    if ! grep -q "$uuid" <<<"$busy"; then echo "$idx"; return 0; fi
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ',')
  return 1
}

claim() {
  local g
  exec 9>"$LOCK"; flock -w 60 9 || return 1
  g=$(free_gpu) || { flock -u 9; return 1; }
  printf "%s | cell=jokes_community layer3 closure | GPU=%s | agent=claude-jokes-closure | job=%s | CLAIM\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$g" "$TAG" >> "$LEDGER"
  sleep 5
  # re-verify nobody else grabbed it while we wrote
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | \
     grep -q "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")"; then
    printf "%s | GPU=%s | job=%s | CLAIM-ABANDONED (co-tenant appeared)\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$g" "$TAG" >> "$LEDGER"
    flock -u 9; return 1
  fi
  flock -u 9
  echo "$g"
}

t0=$(date +%s)
attempt=0
while [ $attempt -lt "$RETRIES" ]; do
  G=$(claim) || G=""
  if [ -n "$G" ]; then
    echo "[gpu_runner:$TAG] claimed GPU $G at $(date -u +%H:%M:%SZ), launching" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$G" "$@" >> "$LOG" 2>&1
    rc=$?
    printf "%s | GPU=%s | job=%s | RELEASE rc=%s\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$G" "$TAG" "$rc" >> "$LEDGER"
    if [ $rc -eq 0 ]; then echo "[gpu_runner:$TAG] DONE rc=0"; exit 0; fi
    echo "[gpu_runner:$TAG] rc=$rc -- job died (checkpoints keep the work); retrying" | tee -a "$LOG"
    attempt=$((attempt+1))
    sleep 60
  else
    if [ $(( ($(date +%s)-t0)/60 )) -ge "$MAX_WAIT_MIN" ]; then
      echo "[gpu_runner:$TAG] gave up waiting for a free GPU" | tee -a "$LOG"; exit 2
    fi
    sleep 60
  fi
done
echo "[gpu_runner:$TAG] exhausted $RETRIES attempts" | tee -a "$LOG"
exit 3
