#!/bin/bash
# One-shot dispatcher for a round's non-Claude fleet legs.
# Claude legs are launched separately via the Agent tool (sealed subagents).
# Usage: ./run_round_fleet.sh r1
set -u
TAG="$1"
cd "$(dirname "$0")"
echo "=== codex (gpt-5.6-luna) leg: ${TAG}A + ${TAG}B ==="
python3 run_codex_cw.py --tags "${TAG}A,${TAG}B" --ids codex_luna_a,codex_luna_b || true
echo "=== glm leg (skipped automatically if rate-limited) ==="
python3 run_glm_cw.py --tags "${TAG}A,${TAG}B" --ids glm_a,glm_b || true
