#!/bin/bash
# RESIDUAL SCALING LAW — does the deconfounded dense residual grow with training
# data?  Train T on nested train-set fractions {12.5, 25, 50}% (100% already
# exists), same eval/test rows throughout, frozen recipe.  Fractions are nested
# by stable hash of the GROUP id (question), so 12.5% ⊂ 25% ⊂ 50% ⊂ 100% and no
# group straddles the boundary.  Readout (CPU, later): (d)-(c) per fraction with
# the declared-channel deconf frame — the scaling curve.
# Usage: CELL=so_bounty DDIR=<dense_standard dir> GPU=6 SEEDS="42 1" bash residual_scaling_chain.sh
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:?}
export TOKENIZERS_PARALLELISM=false
# fraction layouts break the 80/10/10 ratio guard by design (train shrinks,
# eval/test stay full) — widen the tolerance, the known 5-fold precedent
export DENSE_SPLIT_FRACTION_ATOL=0.7
CELL=${CELL:?}; DDIR=${DDIR:?}; PY=${PY:?}; NR=${NR:?}
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(ts) | cell=$CELL residual-scaling ladder | GPU=$GPU | agent=claude-main | job=scaling_$CELL | CLAIM" >> "$NR/gpu_ledger.txt"

$PY - <<PYEOF
import hashlib, pandas as pd
from pathlib import Path
D = Path("$DDIR")
tr = pd.read_csv(D / "split/train.csv")
def bucket(g):
    return int(hashlib.sha256(f"scaling|$CELL|{g}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
gb = {g: bucket(g) for g in tr.group.astype(str).unique()}
for frac, name in ((.125, "f125"), (.25, "f25"), (.5, "f50")):
    keep = tr[tr.group.astype(str).map(gb) < frac]
    out = D / f"scaling_{name}"
    (out / "split").mkdir(parents=True, exist_ok=True)
    keep.to_csv(out / "split/train.csv", index=False)
    for s in ("eval", "test"):
        pd.read_csv(D / f"split/{s}.csv").to_csv(out / f"split/{s}.csv", index=False)
    pd.concat([keep, pd.read_csv(D / "split/eval.csv"), pd.read_csv(D / "split/test.csv")]).to_csv(out / "data.csv", index=False)
    print(f"{name}: train {len(keep)} rows ({keep.group.nunique()} groups)")
PYEOF

for frac in f125 f25 f50; do
  d=$DDIR/scaling_$frac
  for s in $SEEDS; do
    out=$d/rm_out_seed$s
    [ -f "$out/RUN_DONE" ] && { echo "[$CELL/$frac] seed$s done, skip"; continue; }
    mkdir -p "$out"
    echo "[$CELL/$frac] seed$s START $(ts)"
    $PY "$NR/methods/dense/train_reward_model.py" \
      --data_path "$d/data.csv" --split_dir "$d/split" \
      --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
      --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
      --gradient_accumulation_steps 1 --max_length 1024 --epochs 2 \
      --gradient-checkpointing --selection_split eval \
      --seed $s --output_dir "$out" > "$out.train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[$CELL/$frac] seed$s FAILED rc=$rc $(ts)"; break 2; }
    touch "$out/RUN_DONE"; echo "[$CELL/$frac] seed$s DONE $(ts)"
  done
  $PY "$NR/methods/dense/score_eval_dense_v4.py" --dir "$d" --name "${CELL}_$frac" > "$d/score.log" 2>&1 \
    && echo "[$CELL/$frac] scored $(ts)"
done
echo "$(ts) | GPU=$GPU | job=scaling_$CELL | RELEASE" >> "$NR/gpu_ledger.txt"
echo "SCALING_CHAIN_DONE_$CELL $(ts)"
