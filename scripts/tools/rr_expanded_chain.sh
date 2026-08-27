#!/bin/bash
# Expansion chain: score the 719 NEW rows with the Gemma bank (K=50 battery +
# distribution check), then retrain dense at n=1,742 (3 seeds, frozen recipe,
# class weighting), then score. Claims one GPU, releases on exit.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
AGENT=claude-cw-expert-rebuild
LOGS=$NR/logs/cw_expert
mkdir -p "$LOGS"
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
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
echo "$(ts) | cell=cw_royalroad_verdict EXPANDED n=1742 (bank rescore on 719 NEW rows + dense 3 seeds) | GPU=$GPU | agent=$AGENT | job=rr_expanded_chain | CLAIM (<=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"

EXP=$NR/datasets/creative-writing/royalroad_stubs/dense_expanded
if [ ! -f "$LOGS/EXP_STAGE1_DONE" ]; then
  echo "[1] Gemma bank on 719 NEW rows $(ts)"
  VA_OUT_CWX=$NR/outputs/va_gemma_banks_cw_expert_expanded \
  /lfs/skampere3/0/alexspan/envs/gemma4/bin/python \
    "$NR/datasets/va_gemma_banks/score_cw_expert_banks.py" \
    --tasks cw_royalroad_expanded_newrows --util 0.93 --auto-util --min-gib 80 \
    --max-model-len 4096 --battery 50 >> "$LOGS/exp_stage1.log" 2>&1
  rc=$?; echo "[1] EXIT $rc $(ts)"
  if [ $rc -ne 0 ]; then
    echo "$(ts) | GPU=$GPU | agent=$AGENT | job=rr_expanded_chain | RELEASE rc=$rc (stage1 failed)" >> "$LEDGER"
    exit 1
  fi
  touch "$LOGS/EXP_STAGE1_DONE"
fi

for seed in 42 1 2; do
  out=$EXP/rm_out_seed$seed
  [ -f "$out/RUN_DONE" ] && { echo "[2] seed$seed done, skip"; continue; }
  mkdir -p "$out"
  echo "[2] === expanded seed$seed START $(ts) ==="
  /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python "$NR/methods/dense/train_reward_model.py" \
    --data_path "$EXP/data.csv" --split_dir "$EXP/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval --class_weight_auto \
    --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?; echo "[2] === expanded seed$seed EXIT $rc $(ts) ==="
  [ $rc -eq 0 ] && touch "$out/RUN_DONE"
done
/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python "$NR/methods/dense/score_eval_dense_v4.py" \
  --dir "$EXP" --name cw_royalroad_expanded > "$EXP/score.log" 2>&1
echo "[3] scoring EXIT $? $(ts)"
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=rr_expanded_chain | RELEASE rc=0" >> "$LEDGER"
echo "RR_EXPANDED_CHAIN_DONE $(ts)"
