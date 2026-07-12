#!/bin/bash
# E7 credit-tight pilot (2026-06-12): bounded-articulability brackets on 3 domains.
# Budget guard: stops when OpenRouter total_usage exceeds $44.40 (account cap $45).
# Grid per focal metric: token-caps {120,1000} x tiers {1B,8B} x 1 round, prompt kind.
# Plus seed scorecards for the full thin->mid->thick ladder on law + CW, and triad
# panels (F_hi components) on each focal metric.
set -u
cd "$(dirname "$0")/../.."          # repo root
LOG=outputs/metric_implementer/e7_pilot_$(date +%Y%m%d_%H%M).log
mkdir -p outputs/metric_implementer
exec >>"$LOG" 2>&1

TIERS="meta-llama/llama-3.2-1b-instruct,meta-llama/llama-3.1-8b-instruct"
ANCHOR="anthropic/claude-sonnet-4.5"
KEY=$(cat ~/.openrouter-api-key.txt)

check_credits() {
  USAGE=$(curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $KEY" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['total_usage'])")
  echo "=== [$(date +%H:%M:%S)] phase '$1' done; total_usage=\$$USAGE ==="
  python3 -c "import sys; sys.exit(0 if float('$USAGE') < 44.40 else 1)" || {
    echo "=== BUDGET GUARD TRIPPED at \$$USAGE — stopping before next phase ==="; exit 0; }
}

run() { echo "--- $* ---"; python -m methods.metric_implementer.run_trial "$@"; }

check_credits "start"

# Phase 1: law live smoke (validates the new preset end-to-end, ~2c)
run smoke --task law
check_credits "law-smoke" || exit 0

# Phase 2: law ladder seed scorecards (3 metrics x both kinds + convergence)
run scorecard --task law --n-pool 48 --oracle-items 8
check_credits "law-scorecard" || exit 0

# Phase 3: law focal grid (thick: element_mapping)
run scaling --task law --metrics element_mapping --kinds prompt \
  --token-caps 120,1000 --rounds-caps 1 --judge-models "$TIERS" \
  --n-pool 48 --oracle-items 8
check_credits "law-grid" || exit 0

# Phase 4: CW ladder seed scorecards
run scorecard --task creative-writing --n-pool 48 --oracle-items 8
check_credits "cw-scorecard" || exit 0

# Phase 5: CW focal grid (thick: distinctive_voice)
run scaling --task creative-writing --metrics distinctive_voice --kinds prompt \
  --token-caps 120,1000 --rounds-caps 1 --judge-models "$TIERS" \
  --n-pool 48 --oracle-items 8
check_credits "cw-grid" || exit 0

# Phase 6: code focal grid (edge_case_handling; ladder scorecards exist from 06-11)
run scaling --task code-review --metrics edge_case_handling --kinds prompt \
  --token-caps 120,1000 --rounds-caps 1 --judge-models "$TIERS" \
  --n-pool 48 --oracle-items 8
check_credits "code-grid" || exit 0

# Phase 7: triad panels on each focal metric (F_hi reliability components + anchor)
run triad --task law --metric element_mapping --judge-models "$TIERS" \
  --passes 2 --anchor "$ANCHOR" --n-pool 24
run triad --task creative-writing --metric distinctive_voice --judge-models "$TIERS" \
  --passes 2 --anchor "$ANCHOR" --n-pool 24
run triad --task code-review --metric edge_case_handling --judge-models "$TIERS" \
  --passes 2 --anchor "$ANCHOR" --n-pool 24
check_credits "triads"

echo "=== E7 PILOT COMPLETE ==="
