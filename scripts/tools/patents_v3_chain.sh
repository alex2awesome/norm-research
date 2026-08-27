#!/bin/bash
# Patents claim-only rebuild, phase 1 (user request 2026-08-13): arm_t (honest T,
# claim text only) then arm_a (V3 fused arm, V_claim+STRUCT block + claim text).
# Construct: examiner rejected this claim element (any ground) — references DROPPED.
# Invoked THROUGH gpu_stack_runner (which sets CUDA_VISIBLE_DEVICES + claims the log).
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
NR=/lfs/skampere3/0/alexspan/norm-research
BASE=$NR/datasets/patents/v3_claimonly
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
LOGS=$NR/logs/patents_v3; mkdir -p "$LOGS"
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
for arm in t a; do
  d=$BASE/arm_$arm; out=$d/rm_out_seed42
  [ -f "$out/RUN_DONE" ] && continue
  mkdir -p "$out"; echo "[patents_v3] arm_$arm START $(ts)"
  $PY "$NR/methods/dense/train_reward_model.py" \
    --data_path "$d/data.csv" --split_dir "$d/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 768 --epochs 2 \
    --gradient-checkpointing --selection_split eval --class_weight_auto \
    --seed 42 --output_dir "$out" > "$LOGS/arm_$arm.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[patents_v3] arm_$arm FAILED rc=$rc $(ts)"; exit 1; }
  touch "$out/RUN_DONE"; echo "[patents_v3] arm_$arm DONE $(ts)"
done
echo "PATENTS_V3_CHAIN_DONE $(ts)"
