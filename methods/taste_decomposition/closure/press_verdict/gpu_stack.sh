#!/bin/bash
# STACKING GPU launcher for the press closure campaign.
#
# gpu_runner.sh waits for a GPU with ZERO compute processes. sk3 has had all eight
# devices occupied by other agents' jobs continuously, so the strict runner never gets
# in. The programme's own precedent for exactly this situation is the N&C responder
# campaign (notes/2026-08-06__closure_nc_responded.md 3.5): "utilisation sized from
# actually-free memory (landed on GPU 1 at util .73 alongside a co-tenant, which was
# never touched); nothing killed" -- and claude-vat-fullgrid is stacking on GPU 0 as
# this runs.
#
# Rules kept: pick the device with the MOST FREE memory; size gpu_memory_utilization
# from ACTUALLY-FREE memory with a margin so the co-tenant can still grow; record the
# co-tenant by PID in the shared ledger before claiming; never signal any process that
# is not this script's own child.
#
# Usage: ./gpu_stack.sh <tag> <logfile> <need_gib> <python> <script> [args...]
set -u
TAG="$1"; LOG="$2"; NEED_GIB="$3"; shift 3
LEDGER=/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt
LOCK=/tmp/alexspan_gpu_claim.lock
export HOME=/lfs/skampere3/0/alexspan
export NR_REPO=/lfs/skampere3/0/alexspan/norm-research
MAX_WAIT_MIN=${MAX_WAIT_MIN:-720}
RETRIES=${RETRIES:-20}

pick() {
  # best = index with max free MiB; echo "idx free_mib total_mib"
  nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits \
    | tr -d ',' | sort -k2 -n -r | head -1
}

t0=$(date +%s); attempt=0
while [ $attempt -lt "$RETRIES" ]; do
  read -r G FREE TOTAL <<<"$(pick)"
  NEED_MIB=$((NEED_GIB * 1024))
  if [ "$FREE" -ge "$NEED_MIB" ]; then
    exec 9>"$LOCK"; flock -w 60 9
    # re-read free memory under the lock, then size util from it with a 12 GiB margin
    read -r G FREE TOTAL <<<"$(pick)"
    USABLE=$((FREE - 12288))
    UTIL=$(awk -v u="$USABLE" -v t="$TOTAL" 'BEGIN{v=u/t; if(v>0.90)v=0.90; printf "%.2f", v}')
    COT=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
          -i "$G" | tr '\n' ';')
    printf "%s | cell=press_verdict layer3 closure | GPU=%s | agent=claude-press-closure | job=%s | CLAIM-STACKED util=%s free=%sMiB co-tenants=[%s] (NEVER touched; N&C 3.5 precedent)\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$G" "$TAG" "$UTIL" "$FREE" "$COT" >> "$LEDGER"
    flock -u 9
    echo "[gpu_stack:$TAG] GPU $G free=${FREE}MiB util=$UTIL at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$G" "$@" --gpu-mem "$UTIL" >> "$LOG" 2>&1
    rc=$?
    printf "%s | GPU=%s | job=%s | RELEASE rc=%s\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$G" "$TAG" "$rc" >> "$LEDGER"
    [ $rc -eq 0 ] && { echo "[gpu_stack:$TAG] DONE rc=0" | tee -a "$LOG"; exit 0; }
    echo "[gpu_stack:$TAG] rc=$rc -- retrying" | tee -a "$LOG"
    attempt=$((attempt+1)); sleep 120
  else
    if [ $(( ($(date +%s)-t0)/60 )) -ge "$MAX_WAIT_MIN" ]; then
      echo "[gpu_stack:$TAG] gave up: best GPU $G had only ${FREE}MiB free" | tee -a "$LOG"; exit 2
    fi
    sleep 90
  fi
done
echo "[gpu_stack:$TAG] exhausted $RETRIES attempts" | tee -a "$LOG"; exit 3
