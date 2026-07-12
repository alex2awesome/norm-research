#!/bin/bash
# E7 pilot continuation (2026-06-12): weak tier 1B -> gemma-3-4b-it after the 1B judge
# proved degenerate (blanket "applicable: false" on every task). Waits for the orphaned
# CW-scorecard python (from run 1) to finish, then runs the three focal grids + triads.
set -u
cd "$(dirname "$0")/../.."
LOG=outputs/metric_implementer/e7_pilot2_$(date +%Y%m%d_%H%M).log
exec >>"$LOG" 2>&1

TIERS="google/gemma-3-4b-it,meta-llama/llama-3.1-8b-instruct"
ANCHOR="anthropic/claude-sonnet-4.5"
KEY=$(cat ~/.openrouter-api-key.txt)
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "waiting for prior scorecard run (pid $WAIT_PID) to finish..."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 10; done
  echo "pid $WAIT_PID done."
fi

check_credits() {
  USAGE=$(curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $KEY" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['total_usage'])")
  echo "=== [$(date +%H:%M:%S)] phase '$1' done; total_usage=\$$USAGE ==="
  python3 -c "import sys; sys.exit(0 if float('$USAGE') < 44.40 else 1)" || {
    echo "=== BUDGET GUARD TRIPPED at \$$USAGE — stopping before next phase ==="; exit 0; }
}

run() { echo "--- $* ---"; python -m methods.metric_implementer.run_trial "$@"; }

check_credits "start"

run scaling --task law --metrics element_mapping --kinds prompt \
  --token-caps 120,1000 --rounds-caps 1 --judge-models "$TIERS" \
  --n-pool 48 --oracle-items 8
check_credits "law-grid" || exit 0

run scaling --task creative-writing --metrics distinctive_voice --kinds prompt \
  --token-caps 120,1000 --rounds-caps 1 --judge-models "$TIERS" \
  --n-pool 48 --oracle-items 8
check_credits "cw-grid" || exit 0

run scaling --task code-review --metrics edge_case_handling --kinds prompt \
  --token-caps 120,1000 --rounds-caps 1 --judge-models "$TIERS" \
  --n-pool 48 --oracle-items 8
check_credits "code-grid" || exit 0

run triad --task law --metric element_mapping --judge-models "$TIERS" \
  --passes 2 --anchor "$ANCHOR" --n-pool 24
run triad --task creative-writing --metric distinctive_voice --judge-models "$TIERS" \
  --passes 2 --anchor "$ANCHOR" --n-pool 24
run triad --task code-review --metric edge_case_handling --judge-models "$TIERS" \
  --passes 2 --anchor "$ANCHOR" --n-pool 24
check_credits "triads"

echo "=== E7 PILOT (continuation) COMPLETE ==="
