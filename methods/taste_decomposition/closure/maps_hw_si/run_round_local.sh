#!/bin/bash
# Local (laptop) half of one map-focused round: slice -> sealed prompts -> codex leg.
# The Claude proposer slots and the blind auditor are dispatched as sealed subagents by
# the coordinator; this script does everything else.
#
#   bash run_round_local.sh 2            # both cells, round 2
set -eu
R=${1:-2}
cd "$(dirname "$0")"
for C in hashtagwars_verdict style_inv_toptier; do
  python stage1_slice.py --cell "$C" --round "$R" 2>&1 | grep -v "RuntimeWarning\|nanmean" | tail -3
  python harness_maps.py build --cell "$C" --round "$R" 2>&1 | grep -v "RuntimeWarning\|nanmean"
done
nohup python run_fleet.py codex \
  --tags "hashtagwars_verdict_r${R},style_inv_toptier_r${R}" --tracks A,B \
  > "codex_r${R}.log" 2>&1 &
echo "codex leg launched for round ${R}; dispatch the two Claude proposer slots per cell now"
