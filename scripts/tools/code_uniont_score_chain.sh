#!/bin/bash
# UNION plain-text dense (code_uniont) SCORING pass — 5 trained folds, eval-split
# preds via score_eval_dense_v4 (DENSE_SCORE_MAXLEN=4096, same as v3max scoring).
# One GPU, ledger-claimed; launched after the bbc_t0 job releases.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export DENSE_SCORE_MAXLEN=4096
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
BASE=$NR/methods/taste_decomposition/code_competitions/dense_crossfit_uniont
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
ledger_free () { awk -v g="$1" '$0 ~ ("GPU=" g "[^0-9]") || $0 ~ ("GPU=" g "$") {
  if ($0 ~ /RELEASE/) c=0; else if ($0 ~ /CLAIM/) c=1 } END { exit (c?1:0) }' "$LEDGER"; }
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
echo "$(ts) | cell=code_competitions union plain-text dense SCORING (5 folds, maxlen 4096) | GPU=$GPU | agent=claude-main | job=code_uniont_score | CLAIM" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"
for k in 0 1 2 3 4; do
  d=$BASE/arm_t/fold$k
  [ -f "$d/rm_out_seed42/preds_eval.csv" ] && { echo "[score] fold$k already scored"; continue; }
  echo "[score] fold$k START $(ts)"
  $PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$d" --name "uniont_f$k" > "$d/score.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[score] fold$k FAILED rc=$rc $(ts)"; break; }
  echo "[score] fold$k DONE $(ts)"
done
echo "$(ts) | GPU=$GPU | agent=claude-main | job=code_uniont_score | RELEASE" >> "$LEDGER"
echo "CODE_UNIONT_SCORE_DONE $(ts)"
