#!/bin/bash
# Health-check supervisor for the full-corpus Tier-3 run. Watches an EXACT PID
# (no pgrep pattern matching — pilot watcher self-deadlocked on its own cmdline).
# Every 30 min appends {rows_done, repos_done, error_rate, disk_usage} to the
# progress log; flags error rate >5% or clone-tmp dir >50 GB. No auto-restarts.
MAIN_PID=$1
BASE=/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis
SHARDS=$BASE/pr_tier3_shards
CLONES=/lfs/skampere3/0/alexspan/tmp_t3_clones
LOG=$BASE/pr_t3_full_progress.log

echo "[$(date '+%F %T')] supervisor up, watching PID $MAIN_PID" >> "$LOG"
while kill -0 "$MAIN_PID" 2>/dev/null; do
    rows=$(find "$SHARDS" -name '*.jsonl' -print0 | xargs -0 cat 2>/dev/null | wc -l)
    okrows=$(find "$SHARDS" -name '*.jsonl' -print0 | xargs -0 cat 2>/dev/null \
             | grep -cE '"t3_status": "(ok|no_src_files)"')
    repos=$(find "$SHARDS" -name '*.jsonl' | wc -l)
    disk_kb=$(du -sk "$CLONES" 2>/dev/null | cut -f1)
    disk_gb=$(( ${disk_kb:-0} / 1048576 ))
    if [ "$rows" -gt 0 ]; then
        errpct=$(awk -v r="$rows" -v o="$okrows" 'BEGIN{printf "%.2f", 100*(r-o)/r}')
    else
        errpct="0.00"
    fi
    line="[$(date '+%F %T')] rows_done=$rows repos_done=$repos error_rate=${errpct}% disk=${disk_gb}GB"
    flag=""
    awk -v e="$errpct" 'BEGIN{exit !(e>5)}' && flag="$flag FLAG:ERROR_RATE>5%"
    [ "$disk_gb" -gt 50 ] && flag="$flag FLAG:DISK>50GB"
    echo "$line$flag" >> "$LOG"
    sleep 1800
done
echo "[$(date '+%F %T')] main PID $MAIN_PID gone — supervisor exiting" >> "$LOG"
