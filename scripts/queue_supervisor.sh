#!/bin/bash
# Idempotent supervisor for the dense-sweep queue workers.
# Run from cron (~every 30 min). Relaunches any dead worker, e.g. after a reboot.
# A slice with a queue_<slice>.done marker is considered finished and skipped.
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
export HOME=/lfs/skampere3/0/alexspan
cd /lfs/skampere3/0/alexspan/norm-research || exit 1

DRIVERS_DIR=logs/sweep_drivers
mkdir -p "$DRIVERS_DIR"

worker_alive() {
  local pid="$1" slice="$2"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # confirm the PID is actually our worker (guards against PID reuse)
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "queue_gpu1.sh ${slice}" || return 1
  return 0
}

for slice in math humor codereview; do
  donefile="${DRIVERS_DIR}/queue_${slice}.done"
  if [[ -f "$donefile" ]]; then
    echo "[$(date '+%F %T')] ${slice}: complete (done marker present), skipping"
    continue
  fi
  pidfile="${DRIVERS_DIR}/queue_${slice}.pid"
  pid=""
  [[ -f "$pidfile" ]] && pid=$(cat "$pidfile" 2>/dev/null)
  if worker_alive "$pid" "$slice"; then
    echo "[$(date '+%F %T')] ${slice}: worker alive (PID $pid)"
    continue
  fi
  ts=$(date +%Y%m%d_%H%M%S)
  log="${DRIVERS_DIR}/queue_${slice}_${ts}.log"
  echo "[$(date '+%F %T')] ${slice}: (re)launching -> ${log}"
  nohup ./scripts/queue_gpu1.sh "$slice" > "$log" 2>&1 < /dev/null &
  disown
done
