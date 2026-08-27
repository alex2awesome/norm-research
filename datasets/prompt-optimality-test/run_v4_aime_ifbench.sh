#!/usr/bin/env bash
# v4 push (user directive): beat GEPA on aime + ifbench (Qwen). 96-unit pools, 3 LLM framings,
# val-augmented confirm slices for guard power. Launch AFTER the Track-A chain frees the server.
set -uo pipefail
cd "$(dirname "$0")"
LM="openai/Qwen3-8B"; BASE="http://127.0.0.1:8077/v1"
for B in aime; do
  LOGDIR="runs_paperexact/$B/Qwen3-8B"; mkdir -p "$LOGDIR"
  EXTRA=""; [ "$B" = "aime" ] && EXTRA="--robust-answer-extract"
  echo "=== v4 $B unitrecomb start $(date -u +%FT%TZ) ==="
  .venv/bin/python paperexact_arms.py "$B" --arm unitrecomb --task-lm "$LM" --api-base "$BASE" \
    --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 8000 --max-units 96 --confirm-add-val \
    --budget-calls 24000 $EXTRA \
    > "$LOGDIR/unitrecomb_v4_run.log" 2>&1 || { echo "v4 $B FAILED"; exit 1; }
done
.venv/bin/python paperexact_rescore.py aime --lm-tag Qwen3-8B --task-lm "$LM" --api-base "$BASE" || true
.venv/bin/python paperexact_rescore.py ifbench --lm-tag Qwen3-8B --task-lm "$LM" --api-base "$BASE" || true
echo V4_AIME_IFBENCH_DONE
