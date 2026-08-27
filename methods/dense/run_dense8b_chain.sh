#!/bin/bash
# Sequential Llama-3.1-8B LoRA dense runs, three claim-matcher domains, one GPU.
# v2 (post-Codex-audit): checkpoint selection on EVAL (--selection_split eval); test scored
# EXACTLY ONCE post hoc (eval_test_once.py). Fails fast per lane.
export HOME=/lfs/skampere3/0/alexspan
export HF_HUB_OFFLINE=1
cd $HOME/norm-research/methods/dense
PY=$HOME/envs/ai_usage/bin/python
B=$HOME/norm-research/outputs/dense8b

run () {  # name max_len bs accum epochs
  local name=$1 ml=$2 bs=$3 ac=$4 ep=$5
  echo "=== LANE $name START $(date) ==="
  $PY train_reward_model.py --data_path "$B/$name/all.csv" --split_dir "$B/$name/splits" \
      --output_dir "$B/$name/run" --selection_split eval \
      --max_length "$ml" --batch_size "$bs" --gradient_accumulation_steps "$ac" --epochs "$ep" \
      || { echo "LANE_${name}_FAILED"; exit 1; }
  $PY eval_test_once.py --model_dir "$B/$name/run/best_model" \
      --test_csv "$B/$name/splits/test.csv" --max_length "$ml" \
      || { echo "LANE_${name}_TESTEVAL_FAILED"; exit 1; }
  echo "=== LANE $name DONE $(date) ==="
}

run peer          512  16 2 3
run news_pooled   1024  8 4 3
run news_loo_latimes  1024 8 4 3
run news_loo_guardian 1024 8 4 3
run patents       640  16 2 2

$PY eval_patents_within_claim.py --model_dir $B/patents/run/best_model \
    --test_csv $B/patents/splits/test.csv --max_length 640 \
    || echo "PATENTS_WITHIN_CLAIM_FAILED"
echo "DENSE8B_CHAIN_DONE"
