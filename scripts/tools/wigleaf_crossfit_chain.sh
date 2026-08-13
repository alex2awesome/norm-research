#!/bin/bash
# Wigleaf 5-fold cross-fitted dense arm. Lane position 3 (after expansion and the
# head+tail audit, before cw_transfer Stage A). Fixes the select-on-eval item-level
# T AND supplies out-of-sample scores for T_pair.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export DENSE_SPLIT_FRACTION_ATOL=0.03
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
LOGS=$NR/logs/cw_expert
AGENT=claude-cw-expert-rebuild
BASE=$NR/datasets/creative-writing/wigleaf/dense_crossfit
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
mkdir -p "$LOGS"
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[wait] lane position 3: waiting for expansion + head+tail $(ts)"
for i in $(seq 1 1440); do
  a=0; b=0
  grep -q "RR_EXPANDED_CHAIN_DONE" "$LOGS/rr_expanded_launcher.log" 2>/dev/null && a=1
  grep -q "RR_JUDGEVIEW_CHAIN_DONE" "$LOGS/rr_judgeview_launcher.log" 2>/dev/null && b=1
  [ $a -eq 1 ] && [ $b -eq 1 ] && break
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
echo "$(ts) | cell=cw_wigleaf_curation 5-FOLD CROSS-FIT dense (fixes select-on-eval T and supplies out-of-sample scores for T_pair; honest set n=747) | GPU=$GPU | agent=$AGENT | job=wigleaf_crossfit | CLAIM (<=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"

for k in 0 1 2 3 4; do
  d=$BASE/fold$k; out=$d/rm_out_seed42
  [ -f "$out/RUN_DONE" ] && { echo "[wcf] fold$k done, skip"; continue; }
  mkdir -p "$out"
  echo "[wcf] === fold$k START $(ts) ==="
  $PY "$NR/methods/dense/train_reward_model.py" \
    --data_path "$d/data.csv" --split_dir "$d/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval --class_weight_auto \
    --seed 42 --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?; echo "[wcf] === fold$k EXIT $rc $(ts) ==="
  [ $rc -eq 0 ] && touch "$out/RUN_DONE"
  $PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$d" --name "wig_cf_fold$k" \
    > "$d/score.log" 2>&1
  echo "[wcf] scored fold$k rc=$? $(ts)"
done
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=wigleaf_crossfit | RELEASE rc=0" >> "$LEDGER"
echo "WIGLEAF_CROSSFIT_CHAIN_DONE $(ts)"
