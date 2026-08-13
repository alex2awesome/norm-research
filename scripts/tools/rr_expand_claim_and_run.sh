#!/bin/bash
# Claim a free GPU, run the RoyalRoad re-matched expansion sweep (bge-large
# clustering of 2,367 + topic x era matching sweep), release on exit.
set -u
export HOME=/lfs/skampere3/0/alexspan
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
AGENT=claude-cw-expert-rebuild
JOB=rr_expansion_sweep
LOGS=$NR/logs/cw_expert
mkdir -p "$LOGS"
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
ledger_free () {
  awk -v g="$1" '$0 ~ ("GPU=" g "[^0-9]") || $0 ~ ("GPU=" g "$") {
      if ($0 ~ /RELEASE/) c=0; else if ($0 ~ /CLAIM/) c=1 } END { exit (c?1:0) }' "$LEDGER"
}
GPU=""
for i in $(seq 1 4320); do
  while read -r idx used util; do
    idx=${idx%,}; used=$(echo "${used%,}" | tr -dc '0-9'); util=$(echo "$util" | tr -dc '0-9')
    [ -z "$used" ] && continue
    if [ "$used" -le 8 ] && [ "$util" -eq 0 ] && ledger_free "$idx"; then GPU=$idx; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
  [ -n "$GPU" ] && break
  sleep 10
done
[ -z "$GPU" ] && { echo "[poll] no free GPU $(ts)"; exit 2; }
echo "$(ts) | cell=cw_royalroad_verdict EXPANSION sweep (bge-large clustering of the 2,367 usable pool + topic x era matching sweep for the largest lexical<.58 subsample) | GPU=$GPU | agent=$AGENT | job=$JOB | CLAIM (co-tenant check: <=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"
CUDA_DEVICE_ORDER=PCI_BUS_ID /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python \
  "$NR/datasets/creative-writing/build_royalroad_expanded.py" --gpu "$GPU" \
  > "$LOGS/rr_expansion.log" 2>&1
rc=$?
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=$JOB | RELEASE rc=$rc" >> "$LEDGER"
echo "[release] GPU=$GPU rc=$rc $(ts)"
echo "RR_EXPANSION_LAUNCHER_DONE rc=$rc"
