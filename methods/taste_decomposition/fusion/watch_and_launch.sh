#!/usr/bin/env bash
# Waits for a fully-free GPU (0 MiB / 0% util), claims it in the ledger
# (race-safe: re-checks for a competing later CLAIM and retracts if lost),
# then launches run_fusion_dense_chain.sh detached. Max wait ~12h.
set -u
export HOME=/lfs/skampere3/0/alexspan
NR=$HOME/norm-research
FUS=$NR/methods/taste_decomposition/fusion
LEDGER=$NR/gpu_ledger.txt
AGENT=claude-vat-fusion
LOG=$FUS/watcher.log

echo "[watcher] start $(date)" >> "$LOG"
for i in $(seq 1 360); do
  free_gpu=""
  while IFS=, read -r idx mem util; do
    mem=$(echo "$mem" | tr -dc '0-9'); util=$(echo "$util" | tr -dc '0-9')
    if [ "${mem:-9999}" -lt 100 ] && [ "${util:-99}" -eq 0 ]; then
      free_gpu=$(echo "$idx" | tr -dc '0-9'); break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)

  if [ -n "$free_gpu" ]; then
    ts=$(date -u +%FT%TZ)
    echo "$ts | cell=cap fusion directions 2+3 | GPU=$free_gpu | agent=$AGENT | job=fusion_dense_chain (6x llama8b lora seed42 + scoring) | CLAIM" >> "$LEDGER"
    sleep 20
    # lost-race check: any later CLAIM for this GPU by another agent?
    later=$(awk -v g="GPU=$free_gpu" -v a="$AGENT" -v t="$ts" '
      index($0,g) && index($0,"CLAIM") && !index($0,"RETRACT") {
        line_ts=substr($0,1,20);
        if (line_ts >= t && !index($0,a)) print $0 }' "$LEDGER")
    mem_now=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$free_gpu" | tr -dc '0-9')
    if [ -n "$later" ] || [ "${mem_now:-9999}" -ge 1000 ]; then
      echo "$(date -u +%FT%TZ) | GPU=$free_gpu | agent=$AGENT | RETRACT claim (lost race or GPU no longer free)" >> "$LEDGER"
      echo "[watcher] lost race on GPU $free_gpu, keep waiting" >> "$LOG"
      sleep 60
      continue
    fi
    echo "[watcher] claimed GPU $free_gpu, launching chain $(date)" >> "$LOG"
    nohup env GPU=$free_gpu bash "$FUS/run_fusion_dense_chain.sh" > "$FUS/fusion_chain.log" 2>&1 < /dev/null &
    echo "[watcher] chain pid $!" >> "$LOG"
    disown
    exit 0
  fi
  sleep 120
done
echo "[watcher] TIMED OUT after 12h without a free GPU $(date)" >> "$LOG"
