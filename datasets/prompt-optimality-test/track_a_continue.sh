#!/usr/bin/env bash
# Track A continuation — Qwen3-8B serial lane after the hotpot chain frees the server.
# Priority: new-bench arms first (wins need official baselines), then v3 re-runs, then rescores.
set -uo pipefail
cd "$(dirname "$0")"
LM="openai/Qwen3-8B"; BASE="http://127.0.0.1:8077/v1"
arms() {  # arms <bench> [extra flags...]
  local BENCH=$1; shift
  local LOGDIR="runs_paperexact/$BENCH/Qwen3-8B"; mkdir -p "$LOGDIR"
  for ARM in official inhouse unitrecomb; do
    if [ -f "$LOGDIR/../Qwen3-8B/$ARM/result.json" ] && [ "$ARM" != "unitrecomb" ]; then
      echo "=== $BENCH $ARM exists, skip ==="; continue
    fi
    echo "=== $BENCH $ARM start $(date -u +%FT%TZ) ==="
    .venv/bin/python paperexact_arms.py "$BENCH" --arm "$ARM" --task-lm "$LM" --api-base "$BASE" \
      --temperature 0.6 --top-p 0.95 --max-tokens 8000 "$@" \
      > "$LOGDIR/${ARM}_run.log" 2>&1 || { echo "$BENCH $ARM FAILED"; return 1; }
  done
}
arms ifbench || true
arms livebench || true
arms aime --robust-answer-extract || true      # v3 unitrecomb rerun (official/inhouse skipped)
arms hover || true                             # v3 unitrecomb rerun
arms pupa || true
for BENCH in ifbench livebench aime hover pupa; do
  echo "=== rescore $BENCH $(date -u +%FT%TZ) ==="
  .venv/bin/python paperexact_rescore.py "$BENCH" --lm-tag Qwen3-8B --task-lm "$LM" \
    --api-base "$BASE" || echo "RESCORE $BENCH FAILED"
done
python3 analyze_unit_missing_mass.py > runs/unit_missing_mass_latest.log 2>&1 || true
for BENCH in aime hover hotpot ifbench livebench pupa; do
  python3 analyze_bounds_evt.py --paperexact "$BENCH" --lm-tag Qwen3-8B >/dev/null 2>&1 || true
done
echo TRACK_A_CONTINUE_DONE
