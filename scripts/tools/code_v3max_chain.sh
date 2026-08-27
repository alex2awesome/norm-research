#!/bin/bash
# Direction-3 / V3 feature-augmented dense on CODE COMPETITIONS (label: v3_aug).
# User request 2026-08-12. Arm a, 5 folds, seed 42. ESTIMAND: fused V+A+T arm,
# max-of-variants column only. Never an honest T.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export DENSE_SPLIT_FRACTION_ATOL=0.15
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt; LOGS=$NR/logs/code_v3max
BASE=$NR/methods/taste_decomposition/code_competitions/dense_crossfit_v3max
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
mkdir -p "$LOGS"; ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
ledger_free () { awk -v g="$1" '$0 ~ ("GPU=" g "[^0-9]") || $0 ~ ("GPU=" g "$") {
  if ($0 ~ /RELEASE/) c=0; else if ($0 ~ /CLAIM/) c=1 } END { exit (c?1:0) }' "$LEDGER"; }
GPU=""
for i in $(seq 1 4320); do
  while read -r idx used util; do
    idx=${idx%,}; used=$(echo "${used%,}" | tr -dc '0-9'); util=$(echo "$util" | tr -dc '0-9')
    [ -z "$used" ] && continue
    if [ "$used" -le 8 ] && [ "$util" -eq 0 ] && ledger_free "$idx"; then GPU=$idx; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
  [ -n "$GPU" ] && break; sleep 10
done
[ -z "$GPU" ] && { echo "[poll] no free GPU $(ts)"; exit 2; }
export CUDA_VISIBLE_DEVICES=$GPU
echo "$(ts) | cell=code_competitions V3 feature-augmented dense (v3max arm a, 5 folds, FULL 166-line VA block + code, 6,353-row four-platform pool, max_len 6144) | GPU=$GPU | agent=claude-main | job=code_v3max | CLAIM (<=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"
for k in 0 1 2 3 4; do
  d=$BASE/arm_a/fold$k; out=$d/rm_out_seed42
  [ -f "$out/RUN_DONE" ] && continue
  mkdir -p "$out"; echo "[v3a] fold$k START $(ts)"
  $PY "$NR/methods/dense/train_reward_model.py" \
    --data_path "$d/data.csv" --split_dir "$d/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 2 --eval_batch_size 4 \
    --gradient_accumulation_steps 8 --max_length 6144 --epochs 2 \
    --gradient-checkpointing --selection_split eval --class_weight_auto \
    --seed 42 --output_dir "$out" > "$LOGS/fold$k.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[v3a] fold$k FAILED rc=$rc $(ts)"; break; }
  touch "$out/RUN_DONE"; echo "[v3a] fold$k DONE $(ts)"
done
echo "$(ts) | cell=code_competitions v3_aug arm a chain end | GPU=$GPU | agent=claude-main | job=code_v3max | RELEASE" >> "$LEDGER"
echo "CODE_V3MAX_CHAIN_DONE $(ts)"
