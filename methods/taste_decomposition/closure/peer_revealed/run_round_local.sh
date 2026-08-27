#!/bin/bash
# Local (laptop, CPU-only) half of one closure round for a peer cell:
#   slice -> sealed prompts (P=6 fleet) -> decomposer brief -> codex + GLM legs.
# The two Claude proposer slots, the decomposer, the blind auditor and the arbiter are
# dispatched as sealed subagents by the coordinating agent; this script does the rest.
#
#   bash run_round_local.sh <cell> <round> [--decompose <upto>]
set -eu
CELL=$1; R=$2; shift 2
DECOMP=""; UPTO=""
if [ "${1:-}" = "--decompose" ]; then DECOMP=yes; UPTO=$2; fi
cd "$(dirname "$0")"

python3 stage1_slice.py --cell "$CELL" --round "$R" 2>&1 | grep -v "RuntimeWarning\|nanmean" | tail -4
python3 harness_maps.py build --cell "$CELL" --round "$R"
if [ -n "$DECOMP" ]; then
  python3 mixed_parents.py --cell "$CELL" --round "$R" --upto "$UPTO" --n 3 | tail -12
fi
nohup python3 run_fleet.py codex --tags "${CELL}_r${R}" --tracks A,B > "codex_r${R}.log" 2>&1 &
nohup python3 run_fleet.py glm   --tags "${CELL}_r${R}" --tracks A,B > "glm_r${R}.log" 2>&1 &
echo "[${CELL} r${R}] codex + glm legs launched; dispatch the 2 Claude proposer slots"
