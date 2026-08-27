#!/bin/bash
# Tail of JOB 1 (mathlib TRAIN-BIG / EVAL-CANONICAL), replacing the mathlib blocks of
# run_corrections_mathlib_homepage.sh.
#
# Why the replacement: job 2 was split out to run stacked in its own lane once job 1
# measured out at ~3 h/seed, and the original chain would have reached its own (now
# redundant) homepage block while the dedicated homepage lane was still on seed 1 or 2 --
# two trainings writing the same rm_out_seed* directory. The original chain's wrapper
# shell was killed (its in-flight seed-42 python was deliberately left running and
# adopted by init; a bash script cannot be edited safely while bash is reading it).
# This script waits for that adopted seed-42 process to exit, scores it, then runs
# seeds 1 and 2 with a scoring pass after each so partial results are always on disk.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
ROOT=/lfs/skampere3/0/alexspan/norm-research
BIGTRAIN=$ROOT/methods/dense/run_bigtrain_eval_canonical.py
SCORE=$ROOT/methods/dense/score_eval_dense_v4.py
DIR=$ROOT/datasets/math/mathlib/dense_standard_bigtrain
INFLIGHT_PID=${INFLIGHT_PID:-0}

if [ "$INFLIGHT_PID" != "0" ]; then
  echo "[mltail] waiting for adopted seed-42 PID $INFLIGHT_PID $(date)"
  while kill -0 "$INFLIGHT_PID" 2>/dev/null; do sleep 120; done
  echo "[mltail] seed-42 PID $INFLIGHT_PID exited $(date)"
  # the adopted run had no wrapper left to drop its sentinel
  [ -d "$DIR/rm_out_seed42/best_model" ] && touch "$DIR/rm_out_seed42/RUN_DONE"
fi

for seed in 42 1 2; do
  out=$DIR/rm_out_seed$seed
  if [ ! -f "$out/RUN_DONE" ]; then
    echo "[mltail] === seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
    $PY "$BIGTRAIN" \
      --data_path "$DIR/data.csv" --split_dir "$DIR/split" \
      --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
      --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
      --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
      --gradient-checkpointing --class_weight_auto --selection_split eval \
      --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
    rc=$?
    echo "[mltail] === seed$seed EXIT $rc $(date) ==="
    [ $rc -eq 0 ] && touch "$out/RUN_DONE"
  else
    echo "[mltail] seed$seed already done, skip $(date)"
  fi
  echo "[mltail] === scoring after seed$seed $(date) ==="
  $PY "$SCORE" --dir "$DIR" --name mathlib_bigtrain > "$DIR/score.log" 2>&1
  echo "[mltail] === scoring EXIT $? $(date) ==="
done

echo "MATHLIB_BIGTRAIN_ALL_DONE $(date)"
