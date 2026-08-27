#!/bin/bash
# Wait for <cell>_r<round>_scores.npz to appear on sk3, fetch it, run the readout,
# and (for round 1) immediately build the round-2 slice, sealed prompts and codex leg.
# Claude proposer + auditor + arbiter steps are dispatched by the coordinating agent.
#
# Usage: ./pipeline_after_score.sh <cell> <round> [next_round_yes]
set -u
CELL=$1; R=$2; NEXT=${3:-yes}
SSHOPT="-o ControlPath=/tmp/ssh_sk3_%r@%h:%p"
D=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/maps_batch1
L=/Users/spangher/Projects/stanford-research/norm-research/methods/taste_decomposition/closure/maps_batch1
cd "$L"

until ssh $SSHOPT sk3 "ls $D/${CELL}_r${R}_scores.npz" >/dev/null 2>&1; do sleep 45; done
scp $SSHOPT "sk3:$D/${CELL}_r${R}_scores.npz" "sk3:$D/${CELL}_r${R}_score_report.json" "$L/" || exit 1
echo "[pipeline] fetched ${CELL}_r${R}"

python3 readout.py --cell "$CELL" --round "$R" > "readout_${CELL}_r${R}.log" 2>&1
echo "[pipeline] readout ${CELL}_r${R} rc=$?"

if [ "$R" = "1" ] && [ "$NEXT" = "yes" ]; then
  python3 stage1_slice.py --cell "$CELL" --round 2 >> "r2_${CELL}.log" 2>&1 \
    && python3 harness_maps.py build --cell "$CELL" --round 2 >> "r2_${CELL}.log" 2>&1 \
    && python3 run_fleet.py codex --tags "${CELL}_r2" --tracks A,B >> "r2_${CELL}.log" 2>&1
  echo "[pipeline] round-2 prompts + codex leg done for $CELL"
fi
