#!/usr/bin/env bash
# T3 — paper-exact AIME, GLM-5.2 column (our-best task LM; API, 0-GPU), arms in dependency order.
set -uo pipefail
cd "$(dirname "$0")"
LM="anthropic/glm-5.2"
BASE="https://api.z.ai/api/anthropic"
KEY="$HOME/.z-ai-api-key-spangher.txt"
LOGDIR="runs_paperexact/aime/glm-5.2"
mkdir -p "$LOGDIR"
for ARM in official inhouse unitrecomb; do
  echo "=== $ARM start $(date -u +%FT%TZ) ==="
  .venv/bin/python paperexact_arms.py aime --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
    --api-key-file "$KEY" --temperature 0.6 --top-p 0.95 --max-tokens 32000 \
    --robust-answer-extract \
    > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "ARM $ARM FAILED (see $LOGDIR/${ARM}_run.log)"; exit 1; }
  echo "=== $ARM done $(date -u +%FT%TZ) ==="
done
echo PAPEREXACT_AIME_GLM_DONE
