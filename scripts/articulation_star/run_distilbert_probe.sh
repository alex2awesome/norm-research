#!/bin/bash
# DistilBERT leakage probe for an articulation-STaR run.
#
# Trains DistilBERT on iter-0 kept rationales (frozen baseline), then scores
# all 4 test-eval stages (base, iter00, iter01, iter02). The classifier's
# accuracy ON THE RATIONALES IS the leakage rate.
#
# Per [[feedback_gpu_usage]]: single GPU.
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

TASK="${TASK:-creative_writing}"
RUN_NAME="${RUN_NAME:-v1_overnight_logprob}"
GPU="${GPU:-1}"

LOG_DIR="logs/articulation_star/${RUN_NAME}/distilbert_probe"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] === train DistilBERT on iter-0 kept rationales ==="
CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.distilbert_leakage \
  --task "$TASK" --run_name "$RUN_NAME" --mode train --train_iter 0 --n_epochs 3 \
  > "$LOG_DIR/train.log" 2>&1
tail -8 "$LOG_DIR/train.log"

for STAGE in base iter00 iter01 iter02; do
  echo "[$(date '+%F %T')] === score test stage ${STAGE} ==="
  CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.distilbert_leakage \
    --task "$TASK" --run_name "$RUN_NAME" --mode score --target test --stage "$STAGE" \
    > "$LOG_DIR/score_${STAGE}.log" 2>&1
  tail -4 "$LOG_DIR/score_${STAGE}.log"
done

echo
echo "[$(date '+%F %T')] === summary ==="
python -m methods.articulation_star.distilbert_leakage \
  --task "$TASK" --run_name "$RUN_NAME" --mode summarize \
  2>&1 | tee "$LOG_DIR/summary.log"
