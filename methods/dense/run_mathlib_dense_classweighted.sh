#!/bin/bash
# Corrected mathlib_verdict dense rerun (task D7 job 1 FOLLOW-UP, 2026-08-08): the first
# area-grouped rerun (run_dense_standard_scaleupA.sh) COLLAPSED -- val Precision/Recall at
# every checkpoint across all 3 seeds matched the base rate almost exactly (Recall .977-1.000,
# Precision .944 == pos_rate .9428), i.e. near-constant majority-class prediction, because the
# frozen dense-standard recipe does NOT class-weight the loss and mathlib's population is
# 94.3% positive. An independent class-weighted linear TF-IDF check on the IDENTICAL split
# scored eval=.677/test=.786 (vs the collapsed LoRA's eval mean .580/test mean .473), proving
# the population/split are fine and the gap is a training-recipe artifact.
#
# Fix: --class_weight_auto (train_reward_model.py's own BCEWithLogitsLoss pos_weight =
# num_neg/num_pos flag; already used by other imbalanced dense-standard cells, e.g.
# code-review/dense_standard_v3). SAME data/split/seeds/everything else, only this one flag
# added. GPU=3 (claimed on ledger; GPU=5 was this job's original GPU but has an unresolved
# claim from another agent as of 2026-08-08, so this follow-up uses GPU=3 instead per
# verified-free nvidia-smi + ledger RELEASE at 06:53:25Z, no co-tenant since).
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-3}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
TR=/lfs/skampere3/0/alexspan/norm-research/methods/dense/train_reward_model.py
SCORE=/lfs/skampere3/0/alexspan/norm-research/methods/dense/score_eval_dense_v4.py
MATHLIB=/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/dense_standard_cw

run () {
  name=$1; dir=$2; seed=$3
  out=$dir/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[mathlib-cw] $name seed$seed already done, skip $(date)"; return; fi
  echo "[mathlib-cw] === $name seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
  $PY "$TR" \
    --data_path "$dir/data.csv" --split_dir "$dir/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --class_weight_auto --selection_split eval \
    --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  echo "[mathlib-cw] === $name seed$seed EXIT $rc $(date) ==="
  if [ $rc -eq 0 ]; then touch "$out/RUN_DONE"; else echo "[mathlib-cw] $name seed$seed FAILED rc=$rc"; fi
}

score () {
  name=$1; dir=$2
  echo "[mathlib-cw] === scoring $name $(date) ==="
  $PY "$SCORE" --dir "$dir" --name "$name" > "$dir/score.log" 2>&1
  rc=$?
  echo "[mathlib-cw] === scoring $name EXIT $rc $(date) ==="
}

for seed in 42 1 2; do run mathlib_verdict_cw "$MATHLIB" $seed; done
score mathlib_verdict_cw "$MATHLIB"

echo "MATHLIB_CW_ALL_DONE $(date)"
