#!/bin/bash
# Direction-3 / V3 feature-augmented dense on RoyalRoad (label: v3_aug).
# Lane position after the Wigleaf cross-fit. Smoke = arm a, seed 42, all 5 folds;
# arm b and seeds 1/2 only run if arm a MOVES (>= +.03 over the head+tail T .5846).
# ESTIMAND: fused V+A+T arm, max-of-variants column only. Never an honest T.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export DENSE_SPLIT_FRACTION_ATOL=0.03
export DENSE_SCORE_MAXLEN=16384
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt; LOGS=$NR/logs/cw_expert
AGENT=claude-cw-expert-rebuild
BASE=$NR/datasets/creative-writing/royalroad_stubs/dense_crossfit_v3aug_fulltext
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
echo "$(ts) | cell=cw_royalroad_verdict V3 FULLTEXT feature-augmented dense (v3_aug_fulltext arms a+b, COMPLETE chapter + VA block, max_len 16384, batch 1 x grad-accum 16 = effective 16, per-fold train-only importance) | GPU=$GPU | agent=$AGENT | job=rr_v3aug | CLAIM (<=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"
for arm in a b; do
  for k in 0 1 2 3 4; do
    d=$BASE/arm_$arm/fold$k; out=$d/rm_out_seed42
    [ -f "$out/RUN_DONE" ] && continue
    mkdir -p "$out"; echo "[v3$arm] fold$k START $(ts)"
    $PY "$NR/methods/dense/train_reward_model.py" \
      --data_path "$d/data.csv" --split_dir "$d/split" \
      --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
      --learning_rate 5e-5 --batch_size 1 --eval_batch_size 2 \
      --gradient_accumulation_steps 16 --max_length 16384 --epochs 2 \
      --gradient-checkpointing --selection_split eval --class_weight_auto \
      --seed 42 --output_dir "$out" > "$out.train.log" 2>&1
    rc=$?; echo "[v3$arm] fold$k EXIT $rc $(ts)"; [ $rc -eq 0 ] && touch "$out/RUN_DONE"
    $PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$d" --name "v3aug_${arm}_f$k" > "$d/score.log" 2>&1
  done
done
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=rr_v3aug | RELEASE rc=0" >> "$LEDGER"
echo "RR_V3AUG_CHAIN_DONE $(ts)"
