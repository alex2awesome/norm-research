#!/bin/bash
# 5-fold cross-fitted dense for cw_royalroad_verdict, all folds on ONE GPU.
# FROZEN dense-standard recipe, no deviation: Llama-3.1-8B LoRA r16/a32, lr5e-5,
# batch16, max_len1024, 2 epochs, --gradient-checkpointing, select-on-eval.
# Plus --class_weight_auto per the coordinator's instruction.
# Seed 42 first (SEEDS overridable to add 1 2 later).
#
# Per-fold RUN_DONE sentinel -> resumable. Scoring reuses score_eval_dense_v4.py
# unchanged; its preds_test.csv carries the `group` column, which for this cell
# IS fiction_id, so honest-set rows re-key to the A/V matrices without a join hack.
#
#   GPU=5 nohup bash methods/dense/run_royalroad_crossfit.sh > logs/... 2>&1 &
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
export TOKENIZERS_PARALLELISM=false
# fold splits are exact tenths (8/1/1) but per-tenth row counts vary a little
export DENSE_SPLIT_FRACTION_ATOL=0.03

NR=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
TR=$NR/methods/dense/train_reward_model.py
SCORE=$NR/methods/dense/score_eval_dense_v4.py
BASE=$NR/datasets/creative-writing/royalroad_stubs/dense_crossfit
SEEDS=${SEEDS:-42}
FOLDS=${FOLDS:-"0 1 2 3 4"}

echo "=== ROYALROAD CROSSFIT START $(date) GPU=$CUDA_VISIBLE_DEVICES seeds=$SEEDS ==="

for k in $FOLDS; do
  d=$BASE/fold$k
  for seed in $SEEDS; do
    out=$d/rm_out_seed$seed
    if [ -f "$out/RUN_DONE" ]; then echo "[cf] fold$k seed$seed done, skip"; continue; fi
    mkdir -p "$out"
    echo "[cf] === fold$k seed$seed START $(date) ==="
    $PY "$TR" \
      --data_path "$d/data.csv" --split_dir "$d/split" \
      --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
      --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
      --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
      --gradient-checkpointing --selection_split eval --class_weight_auto \
      --seed "$seed" --output_dir "$out" > "$out.train.log" 2>&1
    rc=$?
    echo "[cf] === fold$k seed$seed EXIT $rc $(date) ==="
    [ $rc -eq 0 ] && touch "$out/RUN_DONE" || echo "[cf] fold$k seed$seed FAILED rc=$rc"
  done
  echo "[cf] === scoring fold$k $(date) ==="
  $PY "$SCORE" --dir "$d" --name "rr_crossfit_fold$k" > "$d/score.log" 2>&1
  echo "[cf] === scoring fold$k EXIT $? $(date) ==="
done

echo "ROYALROAD_CROSSFIT_DONE $(date)"
