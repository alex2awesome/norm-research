#!/bin/bash
# Sync a round's routing + criteria to sk3 and launch the corpus-wide Gemma-4-31B pass
# under the stacking GPU claim.  Run from the LOCAL campaign dir.
#
#   bash launch_score.sh 1            # round tag mathse_vote_r1
#
# The population is 11,629 rows; 25 criteria + a 150-text anchor battery is ~295k prompts.
# Prefix caching carries the shared item prefix across a round's criteria.  Budget ~1.5-2.5 h.
set -eu
R="$1"
TAG="mathse_vote_r${R}"
LOCAL="$(cd "$(dirname "$0")" && pwd)"
REMOTE=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/mathse_vote

rsync -a "$LOCAL/${TAG}_species.json" "$LOCAL/${TAG}_routing_final.json" \
      "$LOCAL/mathse_vote_population.csv" "$LOCAL/score_gemma_maps.py" \
      "$LOCAL/gpu_stack_runner.sh" sk3:"$REMOTE/" 2>/dev/null || \
rsync -a "$LOCAL/${TAG}_species.json" "$LOCAL/mathse_vote_population.csv" \
      "$LOCAL/score_gemma_maps.py" "$LOCAL/gpu_stack_runner.sh" sk3:"$REMOTE/"

ssh sk3 "export HOME=/lfs/skampere3/0/alexspan; export PIN_GPU=${PIN_GPU:-7}; cd $REMOTE && chmod +x gpu_stack_runner.sh && \
  setsid nohup ./gpu_stack_runner.sh $TAG \$HOME/norm-research/logs/${TAG}_gemma.log 100000 \
    \$HOME/envs/gemma4/bin/python score_gemma_maps.py --jobs $TAG --gpu-mem 0.60 --max-model-len 8192 \
    > \$HOME/norm-research/logs/${TAG}_runner.log 2>&1 & echo LAUNCHED \$!"
sleep 20
ssh sk3 "export HOME=/lfs/skampere3/0/alexspan; tail -5 \$HOME/norm-research/logs/${TAG}_runner.log; \
  pgrep -af score_gemma_maps | head -2"
