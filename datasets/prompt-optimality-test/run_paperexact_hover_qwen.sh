#!/usr/bin/env bash
# T3 — paper-exact hover, Qwen3-8B column. Needs sk2 Qwen3-8B on :8077 + tunnel.
# LAUNCH ONLY after the AIME rescore finishes (shares the same vLLM server).
set -uo pipefail
cd "$(dirname "$0")"
LM="openai/Qwen3-8B"
BASE="http://127.0.0.1:8077/v1"
LOGDIR="runs_paperexact/hover/Qwen3-8B"
mkdir -p "$LOGDIR"
for ARM in official inhouse unitrecomb; do
  echo "=== $ARM start $(date -u +%FT%TZ) ==="
  .venv/bin/python paperexact_arms.py hover --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
    --temperature 0.6 --top-p 0.95 --max-tokens 8000 \
    > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "ARM $ARM FAILED (see $LOGDIR/${ARM}_run.log)"; exit 1; }
  echo "=== $ARM done $(date -u +%FT%TZ) ==="
done
echo PAPEREXACT_HOVER_QWEN_DONE
