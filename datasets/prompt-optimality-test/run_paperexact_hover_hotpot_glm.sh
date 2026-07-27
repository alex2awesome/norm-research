#!/usr/bin/env bash
# T3 — paper-exact hover + hotpot, GLM-5.2 column (API, 0-GPU), arms in dependency order.
set -uo pipefail
cd "$(dirname "$0")"
LM="anthropic/glm-5.2"
BASE="https://api.z.ai/api/anthropic"
KEY="$HOME/.z-ai-api-key-spangher.txt"
for BENCH in hover hotpot; do
  LOGDIR="runs_paperexact/$BENCH/glm-5.2"
  mkdir -p "$LOGDIR"
  for ARM in official inhouse unitrecomb; do
    echo "=== $BENCH $ARM start $(date -u +%FT%TZ) ==="
    .venv/bin/python paperexact_arms.py "$BENCH" --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
      --api-key-file "$KEY" --temperature 0.6 --top-p 0.95 --max-tokens 32000 \
      > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "$BENCH $ARM FAILED"; exit 1; }
  done
done
echo PAPEREXACT_HOVER_HOTPOT_GLM_DONE
