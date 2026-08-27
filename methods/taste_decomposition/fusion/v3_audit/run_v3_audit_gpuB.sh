#!/usr/bin/env bash
# V3 audit chain B (CW mirror, ONE GPU via env GPU=N): matched-n raw control +
# aug arm on the CW community honest population. Same frozen LoRA recipe;
# selection_split=eval (MONITOR) -- fresh cell, generic V4 convention.
# CW splits are 68/16/16 so ALL runs go through the relaxed-gate launcher.
# Resumable: RUN_DONE sentinel per run dir.
set -uo pipefail

export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
NR=$HOME/norm-research
FUS=$NR/methods/taste_decomposition/fusion
DATA=$FUS/dense_data
DENSE=$NR/methods/dense
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
GPU=${GPU:?set GPU=N}
export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
LEDGER=$NR/gpu_ledger.txt

common_args() {
  local d=$1
  echo "--data_path $d/data.csv --split_dir $d/split --output_dir $d/rm_out_seed42 \
--model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
--epochs 2 --batch_size 16 --eval_batch_size 32 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --weight_decay 0.01 --warmup_ratio 0.1 --max_length 1024 \
--seed 42 --selection_split eval --gradient-checkpointing"
}

run_one() {
  local name=$1
  local d=$DATA/$name
  if [ -f "$d/rm_out_seed42/RUN_DONE" ]; then
    echo "[chainB] $name already RUN_DONE, skipping"
    return 0
  fi
  echo "[chainB] TRAIN $name START $(date)"
  $PY "$FUS/train_grown_split.py" $(common_args "$d") > "$d/train_seed42.log" 2>&1
  local rc=$?
  echo "[chainB] TRAIN $name EXIT $rc $(date)"
  if [ $rc -eq 0 ]; then
    touch "$d/rm_out_seed42/RUN_DONE"
    echo "[chainB] SCORE $name START $(date)"
    $PY "$DENSE/score_eval_dense_v4.py" --dir "$d" --name "$name" > "$d/score.log" 2>&1
    echo "[chainB] SCORE $name EXIT $? $(date)"
  else
    echo "[chainB] $name FAILED rc=$rc -- continuing chain" >&2
  fi
}

for name in "$@"; do
  run_one "$name"
done
echo "V3_AUDIT_CHAIN_B_BATCH_DONE ($*) $(date)"
