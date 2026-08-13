#!/bin/bash
# cw_transfer_v1 STAGE A (pilot): LoRA pretrain on pooled CW preference data.
# THIRD in the lane queue -- waits for BOTH the expansion chain and the head+tail
# audit arm before claiming a card. Stage B is deliberately NOT in this script:
# which view Stage B fine-tunes on depends on the head+tail result, so it is
# launched separately once that verdict is in (validate-before-scaling).
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export DENSE_SPLIT_FRACTION_ATOL=0.15   # Stage A is a 90/10 pretrain split, not 80/10/10
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
LOGS=$NR/logs/cw_expert
AGENT=claude-cw-expert-rebuild
SA=$NR/datasets/creative-writing/cw_transfer_v1/stageA
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
mkdir -p "$LOGS"
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[wait] queue position 4: waiting for expansion + head+tail + wigleaf cross-fit $(ts)"
for i in $(seq 1 1440); do
  a=0; b=0; c=0
  grep -q "RR_EXPANDED_CHAIN_DONE" "$LOGS/rr_expanded_launcher.log" 2>/dev/null && a=1
  grep -q "RR_JUDGEVIEW_CHAIN_DONE" "$LOGS/rr_judgeview_launcher.log" 2>/dev/null && b=1
  grep -q "WIGLEAF_CROSSFIT_CHAIN_DONE" "$LOGS/wigleaf_crossfit_launcher.log" 2>/dev/null && c=1
  [ $a -eq 1 ] && [ $b -eq 1 ] && [ $c -eq 1 ] && break
  sleep 30
done
echo "[wait] predecessors done $(ts)"

ledger_free () {
  awk -v g="$1" '$0 ~ ("GPU=" g "[^0-9]") || $0 ~ ("GPU=" g "$") {
      if ($0 ~ /RELEASE/) c=0; else if ($0 ~ /CLAIM/) c=1 } END { exit (c?1:0) }' "$LEDGER"
}
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
echo "$(ts) | cell=cw_transfer_v1 STAGE A pilot (LoRA pretrain on pooled WP + LitBench, 24k rows; NO RoyalRoad/Wigleaf rows, leakage guard verified 0 collisions) | GPU=$GPU | agent=$AGENT | job=cw_transfer_stageA | CLAIM (<=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"

out=$SA/rm_out_pilot
if [ ! -f "$out/RUN_DONE" ]; then
  mkdir -p "$out"
  echo "[A] === stage A pilot START $(ts) ==="
  $PY "$NR/methods/dense/train_reward_model.py" \
    --data_path "$SA/data.csv" --split_dir "$SA/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 1 \
    --gradient-checkpointing --selection_split eval \
    --seed 42 --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?; echo "[A] === stage A pilot EXIT $rc $(ts) ==="
  [ $rc -eq 0 ] && touch "$out/RUN_DONE"
fi
$PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$SA" --name cw_transfer_stageA \
  > "$SA/score.log" 2>&1
echo "[A] scored rc=$? $(ts)"
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=cw_transfer_stageA | RELEASE rc=0" >> "$LEDGER"
echo "CW_TRANSFER_STAGEA_DONE $(ts)"
