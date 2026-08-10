#!/bin/zsh
# Overnight arm-comparison runs, sequential (z.ai rate limit): CW then peer-review.
cd /Users/spangher/Projects/stanford-research/norm-research
export ANTHROPIC_API_KEY=$(cat ~/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic

python scripts/tools/run_arm_comparison.py --task creative-writing \
  --rubrics-dir datasets/creative-writing/medoid-bank \
  --n 400 --max-metrics 40 --max-rounds 4 \
  --arms residual,unconditional,label_contrast \
  --out outputs/ctree/arm_comparison/creative-writing \
  > outputs/ctree/arm_comparison/cw.log 2>&1

python scripts/tools/run_arm_comparison.py --task peer-review \
  --rubrics-dir datasets/peer-review/medoid-bank \
  --n 400 --max-metrics 40 --max-rounds 4 \
  --arms residual,unconditional,label_contrast \
  --out outputs/ctree/arm_comparison/peer-review \
  > outputs/ctree/arm_comparison/pr.log 2>&1

echo "ALL_RUNS_DONE"
