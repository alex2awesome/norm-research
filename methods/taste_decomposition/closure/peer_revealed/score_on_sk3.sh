#!/bin/bash
# Drive one Gemma-4-31B corpus-wide scoring wave for the two peer cells from the laptop.
#
#   bash score_on_sk3.sh <round>            # scores peer_revealed_r<round> and
#                                           # peer_curation_r<round> in ONE vLLM process
#
# Both cells share the model load; the scorer skips any tag whose *_scores.npz already
# exists, so a rerun repeats no work.  GPU claim/release goes through gpu_runner.sh,
# which polls for a genuinely free GPU (zero compute processes), writes the shared
# ledger, re-verifies, and never touches a device with a co-tenant.
set -eu
R=$1
SSHOPT="-o ControlPath=/tmp/ssh_sk3_%r@%h:%p"
REMOTE=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

# push the round's inputs (species file = the 25 selected criteria; population csv)
for pair in "peer_revealed:peer_revealed" "peer_curation_ext:peer_curation"; do
  D=${pair%%:*}; C=${pair##*:}
  [ -f "$LOCAL/$D/${C}_r${R}_species.json" ] || { echo "missing $D/${C}_r${R}_species.json"; exit 1; }
  rsync -az -e "ssh $SSHOPT" \
    "$LOCAL/$D/${C}_r${R}_species.json" "$LOCAL/$D/${C}_population.csv" \
    "$LOCAL/$D/score_gemma_maps.py" "$LOCAL/$D/gpu_runner.sh" \
    "sk3:$REMOTE/$D/"
done

ssh $SSHOPT sk3 "cd $REMOTE/peer_revealed && chmod +x gpu_runner.sh && \
  HOME=/lfs/skampere3/0/alexspan nohup ./gpu_runner.sh peer_r${R}_score score_r${R}.log \
  /lfs/skampere3/0/alexspan/envs/gemma4/bin/python score_gemma_maps.py \
  --jobs peer_revealed_r${R} > runner_r${R}.log 2>&1 &"
ssh $SSHOPT sk3 "cd $REMOTE/peer_curation_ext && chmod +x gpu_runner.sh && \
  HOME=/lfs/skampere3/0/alexspan nohup ./gpu_runner.sh peercur_r${R}_score score_r${R}.log \
  /lfs/skampere3/0/alexspan/envs/gemma4/bin/python score_gemma_maps.py \
  --jobs peer_curation_r${R} > runner_r${R}.log 2>&1 &"
echo "launched round-$R scoring on sk3 for both peer cells"
