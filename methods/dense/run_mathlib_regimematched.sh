#!/bin/bash
# mathlib REGIME-MATCHED arm (secondary to job 1 of the 2026-08-08 corrections).
#
# The TF-IDF ablation (outputs/tfidf_ablation_mathlib_bigtrain.json) shows the whole
# train->eval distribution-shift cost of training big is carried by ONE stratum: rows that
# fail the audit's regime filter (conv_prefix=='feat' AND year>=2025). Dropping just those
# recovers the canonical arm's AUC (.675/.791 vs .680/.788) while keeping 2.1x its
# negatives (780 vs 363). This arm asks whether the dense reader agrees -- i.e. whether the
# starved canonical run is fixed by negatives alone, without paying for the shift.
#
# Same canonical eval/test folds, same recipe, class-weighted, seed 42.
# Waits for the homepage lane to finish so it inherits that lane rather than adding a
# third concurrent process to GPU 5.
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
DIR=$ROOT/datasets/math/mathlib/dense_standard_regimematched

echo "[regime] waiting for the homepage lane to free up $(date)"
until grep -q "HOMEPAGE_STORYGROUPED_ALL_DONE" "$ROOT/logs/homepage_storygrouped.log" 2>/dev/null; do
  sleep 120
done
echo "[regime] homepage lane free $(date)"

for seed in 42 1 2; do
  out=$DIR/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[regime] seed$seed already done, skip"; continue; fi
  echo "[regime] === seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
  $PY "$TR" \
    --data_path "$DIR/data.csv" --split_dir "$DIR/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --class_weight_auto --selection_split eval \
    --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  echo "[regime] === seed$seed EXIT $rc $(date) ==="
  [ $rc -eq 0 ] && touch "$out/RUN_DONE"
  echo "[regime] === scoring after seed$seed $(date) ==="
  $PY "$SCORE" --dir "$DIR" --name mathlib_regimematched > "$DIR/score.log" 2>&1
done

echo "MATHLIB_REGIMEMATCHED_ALL_DONE $(date)"
