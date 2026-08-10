#!/bin/bash
# JOB 2 of the 2026-08-08 corrections, split out to run STACKED on the same GPU as job 1.
# The mathlib train-big arm turned out to be ~6.5 h/seed (29,324 long Lean diffs), so
# running homepage behind it in one chain would delay this cell by most of a day for no
# reason: the card has 183 GB and job 1 uses 28 GB. Same GPU, second process, per the
# one-GPU/stack-processes rule. run_corrections_mathlib_homepage.sh's own homepage block
# is a no-op afterwards -- it skips on the RUN_DONE sentinels these runs drop.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
ROOT=/lfs/skampere3/0/alexspan/norm-research
TR=$ROOT/methods/dense/train_reward_model.py
SCORE=$ROOT/methods/dense/score_eval_dense_v4.py
DIR=$ROOT/datasets/news-homepages/va/dense_standard_storygrouped

for seed in 42 1 2; do
  out=$DIR/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[hp] seed$seed already done, skip $(date)"; continue; fi
  echo "[hp] === homepage_storygrouped seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
  $PY "$TR" \
    --data_path "$DIR/data.csv" --split_dir "$DIR/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval \
    --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  echo "[hp] === seed$seed EXIT $rc $(date) ==="
  [ $rc -eq 0 ] && touch "$out/RUN_DONE"
done

echo "[hp] === scoring $(date) ==="
$PY "$SCORE" --dir "$DIR" --name homepage_storygrouped > "$DIR/score.log" 2>&1
echo "[hp] === scoring EXIT $? $(date) ==="
echo "HOMEPAGE_STORYGROUPED_ALL_DONE $(date)"
