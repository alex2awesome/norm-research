#!/bin/bash
# Dense-standard arms for the scale-up wave C instrument builds (task D7).
# EXACT dense-standard recipe, no deviation: Llama-3.1-8B LoRA r16/a32, lr5e-5,
# batch16, max_len1024, 2 epochs, gradient-checkpointing, select-on-eval.
# NO class_weight_auto (PR-task-specific flag, not part of the frozen recipe).
#
# Chained on ONE GPU (GPU env var), RUN_DONE-sentinel resumable, same pattern as
# run_dense_standard_v4.sh. Cells are passed as CELLS="name:dir name:dir ...".
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-1}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
NR=/lfs/skampere3/0/alexspan/norm-research
TR=$NR/methods/dense/train_reward_model.py
SCORE=$NR/methods/dense/score_eval_dense_v4.py
SEEDS=${SEEDS:-"42 1 2"}

run () {
  name=$1; dir=$2; seed=$3
  out=$dir/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[C] $name seed$seed already done, skip $(date)"; return; fi
  mkdir -p "$out"
  echo "[C] === $name seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
  $PY "$TR" \
    --data_path "$dir/data.csv" --split_dir "$dir/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval \
    --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  echo "[C] === $name seed$seed EXIT $rc $(date) ==="
  if [ $rc -eq 0 ]; then touch "$out/RUN_DONE"; else echo "[C] $name seed$seed FAILED rc=$rc"; fi
}

score () {
  name=$1; dir=$2
  echo "[C] === scoring $name $(date) ==="
  $PY "$SCORE" --dir "$dir" --name "$name" > "$dir/score.log" 2>&1
  echo "[C] === scoring $name EXIT $? $(date) ==="
}

for spec in $CELLS; do
  name=${spec%%:*}; dir=${spec#*:}
  for seed in $SEEDS; do run "$name" "$dir" $seed; done
  score "$name" "$dir"
done

echo "DENSE_STANDARD_SCALEUPC_DONE $(date)"
