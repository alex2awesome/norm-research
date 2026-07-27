#!/usr/bin/env bash
# Track B — GLM-5.2 column on the three NEW benchmarks (key A; concurrent with hover/hotpot GLM
# chain on key B). ifbench -> livebench -> pupa, arms in dependency order.
set -uo pipefail
cd "$(dirname "$0")"
export ZAI_KEY_FILE="$HOME/.z-ai-api-key-alexander-spangher.txt"
LM="anthropic/glm-5.2"
BASE="https://api.z.ai/api/anthropic"
for BENCH in ifbench livebench pupa; do
  LOGDIR="runs_paperexact/$BENCH/glm-5.2"
  mkdir -p "$LOGDIR"
  for ARM in official inhouse unitrecomb; do
    echo "=== $BENCH $ARM start $(date -u +%FT%TZ) ==="
    .venv/bin/python paperexact_arms.py "$BENCH" --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
      --api-key-file "$ZAI_KEY_FILE" --temperature 0.6 --top-p 0.95 --max-tokens 32000 \
      > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "$BENCH $ARM FAILED"; exit 1; }
  done
done
echo PAPEREXACT_NEWBENCH_GLM_DONE
