#!/usr/bin/env bash
# Single-cell RAW-MATCHED displacement-control train + score wrapper.
#
#   bash train_v3_rawmatched_cell.sh <slug> <cuda_device>
#
#   <slug>          the CELL slug, e.g. nc_agree.  The arm dir it drives is
#                   dense_data/v3grid_<slug>_rawmatched/ (built by
#                   build_v3_rawmatched.py).
#   <cuda_device>   physical GPU index, exported as CUDA_VISIBLE_DEVICES only.
#
# WHY THIS IS A SEPARATE FILE FROM train_v3_cell.sh
# -------------------------------------------------
# It is train_v3_cell.sh with exactly ONE behavioural addition: max_length is
# read from the manifest (`.max_length`, default 1024) instead of being pinned
# to 1024, and the same value is handed to the scorer through
# DENSE_SCORE_MAXLEN.  That is the whole experiment -- the control arm trains
# and scores at the reduced document budget the V3 block left behind.  It is a
# NEW FILE rather than an edit because bash re-reads a script from disk while
# executing it, and train_v3_cell.sh is live under other V3-grid jobs.
#
# Everything else is byte-identical to the frozen recipe: Llama-3.1-8B LoRA
# r16/a32, lr 5e-5, bs16, eval_bs 32, grad-accum 1, 2 epochs,
# gradient-checkpointing, seed 42, the cell's own --selection_split, and NO
# --class_weight_auto.
#
# Outputs:  <dir>/rm_out_seed42/{best_model,final_model,preds_eval.csv,preds_test.csv}
#           <dir>/eval_pass_results.json
#           <dir>/train_rawmatched_<slug>.log
#           <dir>/rm_out_seed42/RUN_DONE      (resume sentinel)
set -uo pipefail

SLUG=${1:?usage: train_v3_rawmatched_cell.sh <slug> <cuda_device>}
DEV=${2:?usage: train_v3_rawmatched_cell.sh <slug> <cuda_device>}

export HOME=${V3_HOME:-/lfs/skampere3/0/alexspan}
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$DEV
export TOKENIZERS_PARALLELISM=false

NR=${V3_NR:-$HOME/norm-research}
FUS=$NR/methods/taste_decomposition/fusion
DENSE=$NR/methods/dense
DIR=$FUS/dense_data/v3grid_${SLUG}_rawmatched
PY=${V3_PY:-/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python}
LOG=$DIR/train_rawmatched_$SLUG.log

if [ ! -f "$DIR/manifest.json" ]; then
  echo "[rawmatched:$SLUG] NO manifest.json at $DIR -- arm not built yet, refusing" >&2
  exit 2
fi

read -r SEL ENTRY MAXLEN < <("$PY" - "$DIR/manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["selection_split"], m["trainer_entry"], int(m.get("max_length", 1024)))
PYEOF
)
if [ -z "${SEL:-}" ] || [ -z "${ENTRY:-}" ] || [ -z "${MAXLEN:-}" ]; then
  echo "[rawmatched:$SLUG] could not read selection_split/trainer_entry/max_length" >&2
  exit 3
fi
TRAINER=$NR/$ENTRY
if [ ! -f "$TRAINER" ]; then
  echo "[rawmatched:$SLUG] trainer entry not found: $TRAINER" >&2
  exit 4
fi

# the scorer must read the SAME budget or train/score would disagree
export DENSE_SCORE_MAXLEN=$MAXLEN

OUT=$DIR/rm_out_seed42
mkdir -p "$OUT"

{
  echo "[rawmatched:$SLUG] START $(date -u +%FT%TZ)"
  echo "[rawmatched:$SLUG] dir=$DIR gpu=$CUDA_VISIBLE_DEVICES selection_split=$SEL max_length=$MAXLEN trainer=$TRAINER"

  if [ -f "$OUT/RUN_DONE" ]; then
    echo "[rawmatched:$SLUG] RUN_DONE already present -- skipping training"
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
      --max_length "$MAXLEN" --epochs 2 --gradient-checkpointing \
      --seed 42 --selection_split "$SEL"
    rc=$?
    echo "[rawmatched:$SLUG] TRAIN EXIT $rc $(date -u +%FT%TZ)"
    [ $rc -eq 0 ] && touch "$OUT/RUN_DONE"
  fi

  if [ $rc -eq 0 ]; then
    "$PY" "$DENSE/score_eval_dense_v4.py" --dir "$DIR" --name "v3grid_${SLUG}_rawmatched"
    src=$?
    echo "[rawmatched:$SLUG] SCORE EXIT $src (DENSE_SCORE_MAXLEN=$DENSE_SCORE_MAXLEN) $(date -u +%FT%TZ)"
    if [ -f "$OUT/preds_eval.csv" ] && [ -f "$OUT/preds_test.csv" ]; then
      echo "[rawmatched:$SLUG] PREDS OK $(wc -l < "$OUT/preds_eval.csv") eval / $(wc -l < "$OUT/preds_test.csv") test rows"
      echo "[rawmatched:$SLUG] RAWMATCHED_CELL_DONE"
    else
      echo "[rawmatched:$SLUG] PREDS MISSING after scoring" >&2
    fi
  fi
  echo "[rawmatched:$SLUG] DONE rc=$rc $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$LOG"

exit ${PIPESTATUS[0]}
