#!/bin/bash
# STANDARDIZED dense arm (decision 2026-07-27): Llama-3.1-8B LoRA r16/a32, lr 5e-5,
# batch 16, max_len 1024, 2 epochs, single full-data run, grouped 80/10/10 splits.
# Six runs chained smallest-first on ONE GPU. Per-run log + final DENSE_STANDARD_DONE.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
TR=/lfs/skampere3/0/alexspan/norm-research/methods/dense/train_reward_model.py
PEER=/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/vat_3y/dense_llama
NC=/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/v4/dense_llama

run () {
  name=$1; dir=$2
  if [ -f "$dir/rm_out/RUN_DONE" ]; then echo "[dense-std] $name already done, skip"; return; fi
  echo "[dense-std] === $name START $(date) ==="
  $PY "$TR" \
    --data_path "$dir/data.csv" \
    --split_dir "$dir/split" \
    --model_name meta-llama/Llama-3.1-8B \
    --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 --epochs 2 --gradient-checkpointing \
    --seed 42 \
    --output_dir "$dir/rm_out" > "$dir/train.log" 2>&1
  rc=$?
  echo "[dense-std] === $name EXIT $rc $(date) ==="
  if [ $rc -eq 0 ]; then touch "$dir/rm_out/RUN_DONE"; else echo "[dense-std] $name FAILED rc=$rc"; fi
}

run peer_revealed  "$PEER/revealed"
run nc_agree       "$NC/agree"
run nc_outcome     "$NC/outcome"
run nc_responded   "$NC/responded"
run peer_curation  "$PEER/curation"
run peer_verdict   "$PEER/verdict"
echo "DENSE_STANDARD_DONE"
