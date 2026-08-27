#!/bin/zsh
# Rerun arm comparisons with the paired-CV acceptance gate (calibrated: gate > CV-SE, not > split-SE).
# Same items/banks as the overnight run -> bank materialization is fully cached.
cd /Users/spangher/Projects/stanford-research/norm-research
export ANTHROPIC_API_KEY=$(cat ~/.z-ai-api-key.txt)
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic

python scripts/tools/run_arm_comparison.py --task creative-writing \
  --rubrics-dir datasets/creative-writing/medoid-bank \
  --n 400 --max-metrics 40 --max-rounds 6 \
  --acceptance-eval cv --min-auc-gain 0.01 --min-bits-gain 0.005 \
  --arms residual,unconditional,label_contrast \
  --out outputs/ctree/arm_comparison/creative-writing-cv \
  > outputs/ctree/arm_comparison/cw_cv.log 2>&1

python scripts/tools/run_arm_comparison.py --task peer-review \
  --rubrics-dir datasets/peer-review/medoid-bank \
  --n 400 --max-metrics 40 --max-rounds 6 \
  --acceptance-eval cv --min-auc-gain 0.01 --min-bits-gain 0.005 \
  --arms residual,unconditional,label_contrast \
  --out outputs/ctree/arm_comparison/peer-review-cv \
  > outputs/ctree/arm_comparison/pr_cv.log 2>&1

echo "CV_RUNS_DONE"
