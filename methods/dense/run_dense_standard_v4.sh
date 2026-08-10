#!/bin/bash
# Dense-standard V4 remaining cells (2026-08-06): HashtagWars humor verdict,
# Style Invitational top-tier, patents claim-fell. EXACT dense-standard recipe,
# no deviation: Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs,
# gradient-checkpointing, select-on-eval. NO class_weight_auto (not part of the
# frozen recipe -- that flag is PR-task-specific in run_pr_dense_v2.sh).
#
# Chained smallest-first on ONE GPU (set via GPU env var before launch), same
# RUN_DONE-sentinel / resumable pattern as run_dense_standard.sh /
# run_pr_dense_v2.sh. Small cells (hashtagwars, style_inv) get seeds 42,1,2;
# patents claim-fell (59,937 rows, single honest run) gets seed 42 only.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-1}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
TR=/lfs/skampere3/0/alexspan/norm-research/methods/dense/train_reward_model.py
SCORE=/lfs/skampere3/0/alexspan/norm-research/methods/dense/score_eval_dense_v4.py
HW=/lfs/skampere3/0/alexspan/norm-research/datasets/humor/hashtagwars/dense_standard
SI=/lfs/skampere3/0/alexspan/norm-research/datasets/humor/style_invitational/dense_standard
PAT=/lfs/skampere3/0/alexspan/norm-research/datasets/patents/dense_standard

run () {
  name=$1; dir=$2; seed=$3
  out=$dir/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[v4] $name seed$seed already done, skip $(date)"; return; fi
  echo "[v4] === $name seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
  $PY "$TR" \
    --data_path "$dir/data.csv" --split_dir "$dir/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval \
    --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  echo "[v4] === $name seed$seed EXIT $rc $(date) ==="
  if [ $rc -eq 0 ]; then touch "$out/RUN_DONE"; else echo "[v4] $name seed$seed FAILED rc=$rc"; fi
}

score () {
  name=$1; dir=$2
  echo "[v4] === scoring $name $(date) ==="
  $PY "$SCORE" --dir "$dir" --name "$name" > "$dir/score.log" 2>&1
  rc=$?
  echo "[v4] === scoring $name EXIT $rc $(date) ==="
}

for seed in 42 1 2; do run hashtagwars_verdict "$HW" $seed; done
score hashtagwars_verdict "$HW"

for seed in 42 1 2; do run style_inv_toptier   "$SI" $seed; done
score style_inv_toptier "$SI"

run patents_claimfell "$PAT" 42
score patents_claimfell "$PAT"

echo "DENSE_STANDARD_V4_ALL_DONE $(date)"
