#!/bin/bash
# Two corrective dense runs (user-directed correction 2026-08-08; registry entry
# "MATHLIB AND HOMEPAGE TERMINAL VERDICTS RETRACTED").
#
#  JOB 1  mathlib TRAIN-BIG / EVAL-CANONICAL
#         train on the 29,324-row pre-audit population minus the canonical eval/test area
#         groups, evaluate on the canonical de-confounded rows verbatim. Class-weighted.
#         Run through run_bigtrain_eval_canonical.py because the on-disk split is
#         .944/.030/.026 by design and the frozen trainer hard-asserts 80/10/10 +-2pp.
#  JOB 2  homepage STORY-GROUPED
#         snapshot(date-block)-grouped, packed within outlet, article-de-duplicated
#         9,737/1,313/1,318 split of the same 12,998-row scale-up-C scored population.
#         Balanced (pos .50) so no class weighting.
#
# ONE GPU, both jobs chained. Interleaved seed order so BOTH cells have a usable number
# early: mathlib s42 -> homepage s42/s1/s2 -> mathlib s1/s2, with a scoring pass after
# each block so partial results are always on disk.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
ROOT=/lfs/skampere3/0/alexspan/norm-research
BIGTRAIN=$ROOT/methods/dense/run_bigtrain_eval_canonical.py
TR=$ROOT/methods/dense/train_reward_model.py
SCORE=$ROOT/methods/dense/score_eval_dense_v4.py
MATHLIB=$ROOT/datasets/math/mathlib/dense_standard_bigtrain
HOMEPAGE=$ROOT/datasets/news-homepages/va/dense_standard_storygrouped

run () {
  tag=$1; entry=$2; dir=$3; seed=$4; shift 4
  out=$dir/rm_out_seed$seed
  if [ -f "$out/RUN_DONE" ]; then echo "[corr] $tag seed$seed already done, skip $(date)"; return; fi
  echo "[corr] === $tag seed$seed START $(date) GPU=$CUDA_VISIBLE_DEVICES ==="
  $PY "$entry" \
    --data_path "$dir/data.csv" --split_dir "$dir/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
    --gradient-checkpointing --selection_split eval \
    "$@" --seed $seed --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?
  echo "[corr] === $tag seed$seed EXIT $rc $(date) ==="
  if [ $rc -eq 0 ]; then touch "$out/RUN_DONE"; else echo "[corr] $tag seed$seed FAILED rc=$rc"; fi
}

score () {
  tag=$1; dir=$2
  echo "[corr] === scoring $tag $(date) ==="
  $PY "$SCORE" --dir "$dir" --name "$tag" > "$dir/score.log" 2>&1
  echo "[corr] === scoring $tag EXIT $? $(date) ==="
}

run mathlib_bigtrain  "$BIGTRAIN" "$MATHLIB"  42 --class_weight_auto
score mathlib_bigtrain "$MATHLIB"

for seed in 42 1 2; do run homepage_storygrouped "$TR" "$HOMEPAGE" $seed; done
score homepage_storygrouped "$HOMEPAGE"

for seed in 1 2; do run mathlib_bigtrain "$BIGTRAIN" "$MATHLIB" $seed --class_weight_auto; done
score mathlib_bigtrain "$MATHLIB"

echo "CORRECTIONS_ALL_DONE $(date)"
