#!/bin/bash
# so_accepted_qtrunc dense (3 seeds) STACKED on sk3 GPU0 (user: "double up on sk3").
# Frozen scaleupC recipe. Co-tenant untouched; ~35GB footprint in ~106GB free.
set -u
export HOME=/lfs/skampere3/0/alexspan
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export TOKENIZERS_PARALLELISM=false
NR=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
D=$NR/datasets/stackoverflow-votes/so_accepted/dense_standard_so_accepted_qtrunc
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(ts) | cell=so_accepted_qtrunc dense | GPU=${GPU:-0} | agent=claude-main | job=soacc_dense_sk3 | CLAIM (mode=STACKED-ok, user directive double-up)" >> "$NR/gpu_ledger.txt"
for s in 42 1 2; do
  out=$D/rm_out_seed$s
  [ -f "$out/RUN_DONE" ] && { echo "[soacc] seed$s already done"; continue; }
  mkdir -p "$out"
  echo "[soacc] seed$s START $(ts)"
  $PY "$NR/methods/dense/train_reward_model.py" \
    --data_path "$D/data.csv" --split_dir "$D/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval \
    --seed $s --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[soacc] seed$s FAILED rc=$rc $(ts)"; break; }
  touch "$out/RUN_DONE"; echo "[soacc] seed$s DONE $(ts)"
done
$PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$D" --name so_accepted_qtrunc > "$D/score.log" 2>&1 \
  && echo "[soacc] scored $(ts)" || echo "[soacc] SCORING FAILED $(ts)"
echo "$(ts) | GPU=${GPU:-0} | job=soacc_dense_sk3 | RELEASE" >> "$NR/gpu_ledger.txt"
echo "SOACC_DENSE_SK3_DONE $(ts)"
