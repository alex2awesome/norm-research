#!/bin/bash
# Scale-up wave A (task D7, 2026-08-07): mathlib_verdict area-grouped rerun +
# press_verdict_k3 company-grouped rerun. EXACT dense-standard recipe, no
# deviation: Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs,
# gradient-checkpointing, select-on-eval. 3 seeds each (42,1,2; small n).
#
# Chained smallest-first on ONE GPU (set via GPU env var before launch), same
# RUN_DONE-sentinel / resumable pattern as run_dense_standard_v4.sh.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-1}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
TR=/lfs/skampere3/0/alexspan/norm-research/methods/dense/train_reward_model.py
SCORE=/lfs/skampere3/0/alexspan/norm-research/methods/dense/score_eval_dense_v4.py
MATHLIB=/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/dense_standard
PRESS=/lfs/skampere3/0/alexspan/norm-research/datasets/press-releases/dense_standard_k3

run () {
  name=$1; dir=$2; seed=$3
  out=$dir/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[scaleupA] $name seed$seed already done, skip $(date)"; return; fi
  echo "[scaleupA] === $name seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
  $PY "$TR" \
    --data_path "$dir/data.csv" --split_dir "$dir/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval \
    --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  echo "[scaleupA] === $name seed$seed EXIT $rc $(date) ==="
  if [ $rc -eq 0 ]; then touch "$out/RUN_DONE"; else echo "[scaleupA] $name seed$seed FAILED rc=$rc"; fi
}

score () {
  name=$1; dir=$2
  echo "[scaleupA] === scoring $name $(date) ==="
  $PY "$SCORE" --dir "$dir" --name "$name" > "$dir/score.log" 2>&1
  rc=$?
  echo "[scaleupA] === scoring $name EXIT $rc $(date) ==="
}

for seed in 42 1 2; do run mathlib_verdict "$MATHLIB" $seed; done
score mathlib_verdict "$MATHLIB"

for seed in 42 1 2; do run press_verdict_k3 "$PRESS" $seed; done
score press_verdict_k3 "$PRESS"

echo "DENSE_STANDARD_SCALEUPA_ALL_DONE $(date)"
