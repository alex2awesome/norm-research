#!/bin/bash
# Sync a round's routing + criteria to sk3 and launch the corpus-wide Gemma-4-31B pass
# under the stacking GPU claim.  Run from the LOCAL campaign dir.
#
#   bash launch_score.sh 1            # round tag jokes_community_r1
#
# The population is 16,000 rows; 25 criteria + a 150-text anchor battery is ~404k prompts,
# but the items are SHORT (median 77 chars) so throughput is far higher than the math cell.
# Prefix caching carries the shared item prefix across a round's criteria.  Budget ~1-2 h.
set -eu
R="$1"
GPU="${2:-5}"        # LANE A = GPU 5 (notes/2026-08-09__full_sweep_queue.md)
TAG="jokes_community_r${R}"
LOCAL="$(cd "$(dirname "$0")" && pwd)"
REMOTE=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/jokes_community

rsync -a "$LOCAL/${TAG}_species.json" "$LOCAL/${TAG}_routing_final.json" \
      "$LOCAL/jokes_community_population.csv" "$LOCAL/score_gemma_maps.py" \
      "$LOCAL/gpu_lane_runner.sh" sk3:"$REMOTE/" 2>/dev/null || \
rsync -a "$LOCAL/${TAG}_species.json" "$LOCAL/jokes_community_population.csv" \
      "$LOCAL/score_gemma_maps.py" "$LOCAL/gpu_lane_runner.sh" sk3:"$REMOTE/"

ssh sk3 "export HOME=/lfs/skampere3/0/alexspan; cd $REMOTE && chmod +x gpu_lane_runner.sh && \
  setsid nohup ./gpu_lane_runner.sh $TAG \$HOME/norm-research/logs/${TAG}_gemma.log ${GPU:-5} 100000 \
    \$HOME/envs/gemma4/bin/python score_gemma_maps.py --jobs $TAG --gpu-mem 0.60 --max-model-len 4096 \
    > \$HOME/norm-research/logs/${TAG}_runner.log 2>&1 & echo LAUNCHED \$!"
sleep 20
ssh sk3 "export HOME=/lfs/skampere3/0/alexspan; tail -5 \$HOME/norm-research/logs/${TAG}_runner.log; \
  pgrep -af score_gemma_maps | head -2"
