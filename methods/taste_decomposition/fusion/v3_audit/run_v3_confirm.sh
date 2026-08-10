#!/usr/bin/env bash
# V3 audit confirm runs: usage GPU=N run_v3_confirm.sh <arm_dir> <sel_split> [stock|grown]
# e.g. GPU=5 run_v3_confirm.sh cap_finalist_k40 test stock
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

name=$1; sel=$2; mode=${3:-stock}
d=$DATA/$name
trainer=$DENSE/train_reward_model.py
[ "$mode" = "grown" ] && trainer=$FUS/train_grown_split.py
if [ -f "$d/rm_out_seed42/RUN_DONE" ]; then echo "[confirm] $name RUN_DONE, skip"; exit 0; fi
echo "[confirm] TRAIN $name START $(date)"
$PY "$trainer" --data_path $d/data.csv --split_dir $d/split --output_dir $d/rm_out_seed42 \
  --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --epochs 2 --batch_size 16 --eval_batch_size 32 --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 --weight_decay 0.01 --warmup_ratio 0.1 --max_length 1024 \
  --seed 42 --selection_split $sel --gradient-checkpointing > "$d/train_seed42.log" 2>&1
rc=$?
echo "[confirm] TRAIN $name EXIT $rc $(date)"
if [ $rc -eq 0 ]; then
  touch "$d/rm_out_seed42/RUN_DONE"
  $PY "$DENSE/score_eval_dense_v4.py" --dir "$d" --name "$name" > "$d/score.log" 2>&1
  echo "[confirm] SCORE $name EXIT $? $(date)"
fi
echo "V3_CONFIRM_DONE_$name"
