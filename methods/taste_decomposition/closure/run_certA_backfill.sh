#!/bin/bash
# Track-A certificate backfill (2026-08-11): judge + merge the six runnable terminal rounds.
# Two judge legs per packet (gpt-5.6-sol, gpt-5.6-luna), then the strict merge, which
# writes a NEW <tag>_species_strictA.json and never touches the tau-era species file.
# Resume-by-output-file throughout, so a rerun repeats no finished judging.
set -u
CL="$(cd "$(dirname "$0")" && pwd)"
cd "$CL"

CELLS=(
  "peer_curation_ext:peer_curation:5"
  "peer_revealed:peer_revealed:5"
  "maps_hw_si:hashtagwars_verdict:4"
  "press_verdict:press_verdict:2"
  "mathse_vote:mathse_vote:3"
  "mathse_accepted:mathse_accepted:2"
)

# judge legs first, two at a time (one per model, same packet) so a slow leg never blocks
# a different cell's other leg
for spec in "${CELLS[@]}"; do
  d="${spec%%:*}"; rest="${spec#*:}"; c="${rest%%:*}"; r="${rest##*:}"
  pk="$d/${c}_r${r}_bmergeA_packet.json"
  echo "=== $c r$r judges $(date) ==="
  python3 run_bmerge_judges.py --packet "$pk" --model gpt-5.6-sol \
      --out "$d/${c}_r${r}_bmergeA_judge_sol.json" &
  p1=$!
  python3 run_bmerge_judges.py --packet "$pk" --model gpt-5.6-luna \
      --out "$d/${c}_r${r}_bmergeA_judge_luna.json" &
  p2=$!
  wait $p1 $p2
done

for spec in "${CELLS[@]}"; do
  d="${spec%%:*}"; rest="${spec#*:}"; c="${rest%%:*}"; r="${rest##*:}"
  echo "=== $c r$r merge $(date) ==="
  ( cd "$d" && python3 species_merge.py apply --cell "$c" --round "$r" --track A \
      --verdicts "${c}_r${r}_bmergeA_judge_sol.json,${c}_r${r}_bmergeA_judge_luna.json" \
      | head -6 )
done

echo "CERTA_BACKFILL_JUDGE_MERGE_DONE $(date)"
