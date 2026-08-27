#!/usr/bin/env bash
# Single-cell V3-grid train + score wrapper.
#
#   bash train_v3_cell.sh <slug> <cuda_device>
#
#   <slug>          the dataset dir suffix, i.e. dense_data/v3grid_<slug>/
#                   (e.g. nc_agree, peer_curation, hashtagwars_verdict_k10)
#   <cuda_device>   physical GPU index, exported as CUDA_VISIBLE_DEVICES only
#                   (nothing here hard-codes a device; score_eval_dense_v4.py's
#                   "cuda:0" therefore resolves to the device you pass)
#
# Recipe is the FROZEN dense-standard one (Llama-3.1-8B LoRA r16/a32, lr 5e-5,
# bs16, eval_bs 32, grad-accum 1, max_len 1024, 2 epochs, grad-checkpointing,
# seed 42).  The two things that vary per cell are read OUT OF THE MANIFEST the
# builder wrote, so this script never guesses:
#     .selection_split   -- matches that cell's ORIGINAL dense chain
#     .trainer_entry     -- methods/dense/train_reward_model.py, or
#                           fusion/train_grown_split.py when the cell's own
#                           split fractions fall outside the stock 80/10/10
#                           +-2% gate (that launcher relaxes the GATE ONLY)
#
# Outputs:  <dir>/rm_out_seed42/{best_model,final_model,preds_eval.csv,preds_test.csv}
#           <dir>/eval_pass_results.json
#           <dir>/train_v3_<slug>.log      (this wrapper's own log)
#           <dir>/rm_out_seed42/RUN_DONE   (resume sentinel)
#
# Safe to run TWO AT A TIME on one B200: separate output dirs, separate logs,
# no pkill/killall anywhere, no ledger writes, no device assumptions.
set -uo pipefail

SLUG=${1:?usage: train_v3_cell.sh <slug> <cuda_device>}
DEV=${2:?usage: train_v3_cell.sh <slug> <cuda_device>}

export HOME=${V3_HOME:-/lfs/skampere3/0/alexspan}
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$DEV
export TOKENIZERS_PARALLELISM=false

NR=${V3_NR:-$HOME/norm-research}
FUS=$NR/methods/taste_decomposition/fusion
DENSE=$NR/methods/dense
DIR=$FUS/dense_data/v3grid_$SLUG
PY=${V3_PY:-/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python}
LOG=$DIR/train_v3_$SLUG.log

if [ ! -f "$DIR/manifest.json" ]; then
  echo "[v3grid:$SLUG] NO manifest.json at $DIR -- cell not built yet, refusing" >&2
  exit 2
fi

read -r SEL ENTRY < <("$PY" - "$DIR/manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["selection_split"], m["trainer_entry"])
PYEOF
)
if [ -z "${SEL:-}" ] || [ -z "${ENTRY:-}" ]; then
  echo "[v3grid:$SLUG] could not read selection_split/trainer_entry from manifest" >&2
  exit 3
fi
TRAINER=$NR/$ENTRY
if [ ! -f "$TRAINER" ]; then
  echo "[v3grid:$SLUG] trainer entry not found: $TRAINER" >&2
  exit 4
fi

OUT=$DIR/rm_out_seed42
mkdir -p "$OUT"

{
  echo "[v3grid:$SLUG] START $(date -u +%FT%TZ)"
  echo "[v3grid:$SLUG] dir=$DIR gpu=$CUDA_VISIBLE_DEVICES selection_split=$SEL trainer=$TRAINER"

  if [ -f "$OUT/RUN_DONE" ]; then
    echo "[v3grid:$SLUG] RUN_DONE already present -- skipping training"
    rc=0
  else
    "$PY" "$TRAINER" \
      --data_path "$DIR/data.csv" \
      --split_dir "$DIR/split" \
      --output_dir "$OUT" \
      --model_name meta-llama/Llama-3.1-8B \
      --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
      --learning_rate 5e-5 --weight_decay 0.01 --warmup_ratio 0.1 \
      --batch_size 16 --eval_batch_size 32 --gradient_accumulation_steps 1 \
      --max_length 1024 --epochs 2 --gradient-checkpointing \
      --seed 42 --selection_split "$SEL"
    rc=$?
    echo "[v3grid:$SLUG] TRAIN EXIT $rc $(date -u +%FT%TZ)"
    [ $rc -eq 0 ] && touch "$OUT/RUN_DONE"
  fi

  if [ $rc -eq 0 ]; then
    # writes rm_out_seed*/preds_{eval,test}.csv + <dir>/eval_pass_results.json
    "$PY" "$DENSE/score_eval_dense_v4.py" --dir "$DIR" --name "v3grid_$SLUG"
    src=$?
    echo "[v3grid:$SLUG] SCORE EXIT $src $(date -u +%FT%TZ)"
    if [ -f "$OUT/preds_eval.csv" ] && [ -f "$OUT/preds_test.csv" ]; then
      echo "[v3grid:$SLUG] PREDS OK $(wc -l < "$OUT/preds_eval.csv") eval / $(wc -l < "$OUT/preds_test.csv") test rows"
    else
      echo "[v3grid:$SLUG] PREDS MISSING after scoring" >&2
    fi
  fi
  echo "[v3grid:$SLUG] DONE rc=$rc $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$LOG"

exit ${PIPESTATUS[0]}
