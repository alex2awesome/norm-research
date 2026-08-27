#!/usr/bin/env bash
# T3/T2b chain: hover rescore (300-item EVT input) -> hotpot arms -> hotpot rescore.
set -uo pipefail
cd "$(dirname "$0")"
LM="openai/Qwen3-8B"
BASE="http://127.0.0.1:8077/v1"
echo "=== hover rescore start $(date -u +%FT%TZ) ==="
.venv/bin/python paperexact_rescore.py hover --lm-tag Qwen3-8B --task-lm "$LM" \
  --api-base "$BASE" || { echo "HOVER RESCORE FAILED"; exit 1; }
LOGDIR="runs_paperexact/hotpot/Qwen3-8B"
mkdir -p "$LOGDIR"
for ARM in official inhouse unitrecomb; do
  echo "=== hotpot $ARM start $(date -u +%FT%TZ) ==="
  .venv/bin/python paperexact_arms.py hotpot --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
    --temperature 0.6 --top-p 0.95 --max-tokens 8000 \
    > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "ARM $ARM FAILED"; exit 1; }
done
echo "=== hotpot rescore start $(date -u +%FT%TZ) ==="
.venv/bin/python paperexact_rescore.py hotpot --lm-tag Qwen3-8B --task-lm "$LM" \
  --api-base "$BASE" || { echo "HOTPOT RESCORE FAILED"; exit 1; }
echo PAPEREXACT_HOTPOT_AND_RESCORES_DONE
