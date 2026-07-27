#!/usr/bin/env bash
# T3 — paper-exact AIME, Qwen3-8B column (paper-exact task LM), arms in dependency order.
# Needs: sk2 Qwen3-8B server on :8077 + local tunnel (see plan note T3).
set -uo pipefail
cd "$(dirname "$0")"
LM="openai/Qwen3-8B"
BASE="http://127.0.0.1:8077/v1"
LOGDIR="runs_paperexact/aime/Qwen3-8B"
mkdir -p "$LOGDIR"
for ARM in official inhouse unitrecomb; do
  echo "=== $ARM start $(date -u +%FT%TZ) ==="
  .venv/bin/python paperexact_arms.py aime --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
    --temperature 0.6 --top-p 0.95 --max-tokens 8000 \
    > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "ARM $ARM FAILED (see $LOGDIR/${ARM}_run.log)"; exit 1; }
  echo "=== $ARM done $(date -u +%FT%TZ) ==="
done
echo PAPEREXACT_AIME_QWEN_DONE
