#!/bin/bash
# Claim a free GPU in gpu_ledger.txt, run the RoyalRoad 5-fold cross-fit dense on
# it, release on exit. Strict check: 0 MiB / 0% util and no un-released ledger
# claim before claiming. Own PIDs only; co-tenants never touched.
set -u
export HOME=/lfs/skampere3/0/alexspan
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
AGENT=claude-cw-expert-rebuild
JOB=rr_crossfit_dense
LOGS=$NR/logs/cw_expert
mkdir -p "$LOGS"
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }

ledger_free () {
  awk -v g="$1" '
    $0 ~ ("GPU=" g "[^0-9]") || $0 ~ ("GPU=" g "$") {
      if ($0 ~ /RELEASE/) c = 0; else if ($0 ~ /CLAIM/) c = 1 }
    END { exit (c ? 1 : 0) }' "$LEDGER"
}

GPU=""
for i in $(seq 1 8640); do
  while read -r idx used util; do
    idx=${idx%,}; used=$(echo "${used%,}" | tr -dc '0-9'); util=$(echo "$util" | tr -dc '0-9')
    [ -z "$used" ] && continue
    # <=8 MiB is an idle card (nvidia-smi reports a few MiB of context on some boxes)
    if [ "$used" -le 8 ] && [ "$util" -eq 0 ] && ledger_free "$idx"; then GPU=$idx; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
  [ -n "$GPU" ] && break
  sleep 10
done
[ -z "$GPU" ] && { echo "[poll] no free GPU $(ts)"; exit 2; }

echo "$(ts) | cell=cw_royalroad_verdict 5-FOLD CROSS-FIT dense (power fix for the 141-row-eval T-at-chance; honest set n=651, selection-free) | GPU=$GPU | agent=$AGENT | job=$JOB | CLAIM (co-tenant check: nvidia-smi <=8 MiB / 0% util immediately before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"

GPU=$GPU bash "$NR/methods/dense/run_royalroad_crossfit.sh" > "$LOGS/rr_crossfit.log" 2>&1
rc=$?

mine=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | tr -d ' ' | tr '\n' ' ')
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=$JOB | RELEASE rc=$rc (remaining compute PIDs: ${mine:-none} -- not mine, never touched)" >> "$LEDGER"
echo "[release] GPU=$GPU rc=$rc $(ts)"
echo "RR_CROSSFIT_LAUNCHER_DONE rc=$rc"
