#!/usr/bin/env bash
# Recovery: livebench Qwen arms after the dead-tunnel incident (official was quarantined).
set -uo pipefail
cd "$(dirname "$0")"
LM="openai/Qwen3-8B"; BASE="http://127.0.0.1:8077/v1"
LOGDIR="runs_paperexact/livebench/Qwen3-8B"; mkdir -p "$LOGDIR"
for ARM in official inhouse unitrecomb; do
  echo "=== livebench $ARM start $(date -u +%FT%TZ) ==="
  .venv/bin/python paperexact_arms.py livebench --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
    --temperature 0.6 --top-p 0.95 --max-tokens 8000 \
    > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "livebench $ARM FAILED"; exit 1; }
done
.venv/bin/python paperexact_rescore.py livebench --lm-tag Qwen3-8B --task-lm "$LM" --api-base "$BASE" || true
echo LIVEBENCH_QWEN_RECOVERY_DONE
