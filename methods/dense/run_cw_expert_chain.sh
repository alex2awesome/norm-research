#!/bin/bash
# CW EXPERT rebuild chain (RoyalRoad market VERDICT + Wigleaf editorial CURATION),
# both cells stacked on ONE GPU, run sequentially so the Gemma-4-31B judge and the
# LoRA trainer never contend for the same card at the same time.
#
#   stage 1  Gemma-4-31B A-bank scoring, OFFLINE BATCH vLLM (never an HTTP server),
#            both cells in one engine, GPU_MEM_UTIL .93, K=50/class anchor battery,
#            judge score-distribution check.
#   stage 2  dense T, EXACT dense-standard recipe: Llama-3.1-8B LoRA r16/a32,
#            lr5e-5, batch16, max_len1024, 2 epochs, --gradient-checkpointing,
#            select-on-eval, seeds 42 then 1, 2.
#            Wigleaf ONLY adds --class_weight_auto (404 positives / 1,164 negatives).
#   stage 3  clean eval+test scoring pass for every seed.
#
# RUN_DONE-sentinel resumable at seed granularity; shard-checkpointed in stage 1.
# Usage:  GPU=5 nohup bash methods/dense/run_cw_expert_chain.sh > logs/... 2>&1 &
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

NR=/lfs/skampere3/0/alexspan/norm-research
PY_GEMMA=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
PY_TRAIN=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
TR=$NR/methods/dense/train_reward_model.py
SCORE=$NR/methods/dense/score_eval_dense_v4.py
LOGS=$NR/logs/cw_expert
SEEDS=${SEEDS:-"42 1 2"}
mkdir -p "$LOGS"

echo "=== CW EXPERT CHAIN START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="

# ---------------------------------------------------------------- stage 1 ----
if [ ! -f "$LOGS/STAGE1_DONE" ]; then
  echo "[1] Gemma-4-31B A-bank scoring (offline batch) $(date)"
  $PY_GEMMA "$NR/datasets/va_gemma_banks/score_cw_expert_banks.py" \
      --tasks cw_royalroad_verdict,cw_wigleaf_curation \
      --util 0.93 --auto-util --min-gib 80 --max-model-len 4096 --battery 50 \
      >> "$LOGS/stage1_gemma_score.log" 2>&1
  rc=$?
  echo "[1] EXIT $rc $(date)"
  if [ $rc -ne 0 ]; then echo "[1] FAILED rc=$rc — stopping chain"; exit 1; fi
  grep -q CW_EXPERT_SCORE_DONE "$LOGS/stage1_gemma_score.log" || {
      echo "[1] sentinel missing — stopping chain"; exit 1; }
  touch "$LOGS/STAGE1_DONE"
else
  echo "[1] already done, skip"
fi

# ---------------------------------------------------------------- stage 2 ----
train () {
  name=$1; dir=$2; seed=$3; shift 3
  out=$dir/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[2] $name seed$seed done, skip"; return; fi
  mkdir -p "$out"
  echo "[2] === $name seed$seed START $(date) ==="
  $PY_TRAIN "$TR" \
    --data_path "$dir/data.csv" --split_dir "$dir/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval \
    --seed "$seed" --output_dir "$out" "$@" > "$out.train.log" 2>&1
  rc=$?
  echo "[2] === $name seed$seed EXIT $rc $(date) ==="
  [ $rc -eq 0 ] && touch "$out/RUN_DONE" || echo "[2] $name seed$seed FAILED rc=$rc"
}

RR_DIR=$NR/datasets/creative-writing/royalroad_stubs/dense_standard
WG_DIR=$NR/datasets/creative-writing/wigleaf/dense_standard

# RoyalRoad's canonical split is a fiction-grouped STABLE HASH
# (md5("split::"+fiction_id)%1000), so its realised fractions are 77.8/11.1/11.1
# rather than exactly 80/10/10. Widen the on-disk-split ratio guard for this cell
# ONLY -- reshuffling to hit 80/10/10 would violate feedback_stable_hash_splits.
# Wigleaf (79.5/10.8/9.7) clears the default guard and is left untouched.
export DENSE_SPLIT_FRACTION_ATOL=0.03
for seed in $SEEDS; do
  train cw_royalroad_verdict "$RR_DIR" "$seed"
done
unset DENSE_SPLIT_FRACTION_ATOL
for seed in $SEEDS; do
  # CLASS-WEIGHTED: 404 absolute positives vs 1,164 negatives
  train cw_wigleaf_curation "$WG_DIR" "$seed" --class_weight_auto
done

# ---------------------------------------------------------------- stage 3 ----
for spec in "cw_royalroad_verdict:$RR_DIR" "cw_wigleaf_curation:$WG_DIR"; do
  name=${spec%%:*}; dir=${spec#*:}
  echo "[3] === scoring $name $(date) ==="
  $PY_TRAIN "$SCORE" --dir "$dir" --name "$name" > "$dir/score.log" 2>&1
  echo "[3] === scoring $name EXIT $? $(date) ==="
done

echo "CW_EXPERT_CHAIN_DONE $(date)"
