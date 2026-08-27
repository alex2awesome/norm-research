#!/bin/bash
# Poll sk3 for a round's Gemma score file, fetch it plus its report, and run the
# local readout under the PINNED sklearn-1.9.0 venv (cells.sklearn_guard asserts it).
#
# Usage: ./after_scoring.sh <round>
set -u
R="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
SK3=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/press_verdict
PY=/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/sk19/bin/python
cd "$HERE" || exit 1

while true; do
  if ssh sk3 "test -f $SK3/press_verdict_r${R}_scores.npz" 2>/dev/null; then
    sleep 20   # let the score_report write finish
    scp -q "sk3:$SK3/press_verdict_r${R}_scores.npz" . && \
    scp -q "sk3:$SK3/press_verdict_r${R}_score_report.json" . && break
  fi
  sleep 120
done
echo "[after_scoring r${R}] scores fetched $(date -u +%H:%M:%SZ)"
$PY readout.py --cell press_verdict --round "$R" > "readout_r${R}.log" 2>&1
echo "[after_scoring r${R}] readout rc=$? $(date -u +%H:%M:%SZ)"
