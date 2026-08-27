#!/bin/bash
# RoyalRoad A-JUDGE-VIEW dense arm (mandated bank>dense audit).
# WAITS for the expansion chain to finish first (never preempt our own job),
# then claims ONE card, trains the 5 cross-fit folds at the judge's exact view
# (head 960 + tail 640 tokens, max_length 1600), scores at the same budget.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export DENSE_SPLIT_FRACTION_ATOL=0.03
export DENSE_SCORE_MAXLEN=1600
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
LOGS=$NR/logs/cw_expert
AGENT=claude-cw-expert-rebuild
BASE=$NR/datasets/creative-writing/royalroad_stubs/dense_crossfit_judgeview
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
mkdir -p "$LOGS"
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[wait] holding for RR_EXPANDED_CHAIN_DONE $(ts)"
for i in $(seq 1 720); do
  grep -q "RR_EXPANDED_CHAIN_DONE" "$LOGS/rr_expanded_launcher.log" 2>/dev/null && break
  sleep 30
done
echo "[wait] expansion chain done (or timed out) $(ts)"

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
echo "$(ts) | cell=cw_royalroad_verdict A-JUDGE-VIEW dense audit (head 960 + tail 640 tokens, max_len 1600, same folds/seed as cross-fit) | GPU=$GPU | agent=$AGENT | job=rr_judgeview | CLAIM (<=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"

for k in 0 1 2 3 4; do
  d=$BASE/fold$k; out=$d/rm_out_seed42
  [ -f "$out/RUN_DONE" ] && { echo "[jv] fold$k done, skip"; continue; }
  mkdir -p "$out"
  echo "[jv] === fold$k START $(ts) ==="
  $PY "$NR/methods/dense/train_reward_model.py" \
    --data_path "$d/data.csv" --split_dir "$d/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1600 --epochs 2 \
    --gradient-checkpointing --selection_split eval --class_weight_auto \
    --seed 42 --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?; echo "[jv] === fold$k EXIT $rc $(ts) ==="
  [ $rc -eq 0 ] && touch "$out/RUN_DONE"
  $PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$d" --name "rr_jv_fold$k" \
    > "$d/score.log" 2>&1
  echo "[jv] scored fold$k rc=$? $(ts)"
done
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=rr_judgeview | RELEASE rc=0" >> "$LEDGER"
echo "RR_JUDGEVIEW_CHAIN_DONE $(ts)"
