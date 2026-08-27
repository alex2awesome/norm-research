#!/bin/bash
# U3/U4 dense chains on sk1 (skampere1, A100-80GB): mathse_bounty + so_accepted_qtrunc,
# 3 seeds each, FROZEN scaleupC recipe (lora 16/32, lr 5e-5, batch 16, max_len 1024,
# 2 epochs, select-on-eval, NO class_weight_auto). GPU pinned by env GPU (default 6 —
# verified free 2026-08-17). Scoring pass (preds eval+test) after each cell.
set -u
export HOME=/lfs/skampere1/0/alexspan
export HF_HUB_CACHE=/lfs/skampere1/0/shared_hf_cache  # hub-layout dir (models--* at top level); HF_HOME misresolves it
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-6}
export TOKENIZERS_PARALLELISM=false
NR=/lfs/skampere1/0/alexspan/norm-research
PY=/lfs/skampere1/0/alexspan/envs/unified_v1/bin/python
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i ${GPU:-6} | tr -dc '0-9')
[ "$used" -gt 100 ] && { echo "GPU ${GPU:-6} not free (${used} MiB) — abort"; exit 2; }
echo "$(ts) | cell=U3/U4 unified dense chains | GPU=${GPU:-6} | agent=claude-main | job=unified_dense_sk1 | CLAIM" >> "$NR/gpu_ledger.txt"

run_cell () {
  local name=$1 d=$2
  for s in 42 1 2; do
    out=$d/rm_out_seed$s
    [ -f "$out/RUN_DONE" ] && { echo "[$name] seed$s already done"; continue; }
    mkdir -p "$out"
    echo "[$name] seed$s START $(ts)"
    $PY "$NR/methods/dense/train_reward_model.py" \
      --data_path "$d/data.csv" --split_dir "$d/split" \
      --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
      --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
      --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
      --gradient-checkpointing --selection_split eval \
      --seed $s --output_dir "$out" > "$out.train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[$name] seed$s FAILED rc=$rc $(ts)"; return 1; }
    touch "$out/RUN_DONE"
    echo "[$name] seed$s DONE $(ts)"
  done
  $PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$d" --name "$name" > "$d/score.log" 2>&1 \
    && echo "[$name] scored $(ts)" || echo "[$name] SCORING FAILED $(ts)"
}

run_cell mathse_bounty "$NR/datasets/math-se/mathse_bounty/dense_standard_mathse_bounty"
run_cell so_accepted_qtrunc "$NR/datasets/stackoverflow-votes/so_accepted/dense_standard_so_accepted_qtrunc"
echo "$(ts) | GPU=${GPU:-6} | job=unified_dense_sk1 | RELEASE" >> "$NR/gpu_ledger.txt"
echo "UNIFIED_DENSE_SK1_DONE $(ts)"
