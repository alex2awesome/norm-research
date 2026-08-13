#!/bin/bash
# Phase 2, queued behind the V3 arm-a chain. One card, sequential:
#   (i)   V3 arm b (scores + importance weights)
#   (ii)  decomposition arm: head 614 + tail 410 @ max_len 1024 (VIEW isolated from BUDGET)
#   (iii) Stage-A gate + Stage-B-free ZERO-SHOT of the Stage-A adapter on both CW
#         honest sets (RoyalRoad n=651, Wigleaf n=747), standard 1024 view -- the view
#         Stage A was trained on, so this reads what the pretrain alone knows.
set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export DENSE_SPLIT_FRACTION_ATOL=0.15
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt; LOGS=$NR/logs/cw_expert
AGENT=claude-cw-expert-rebuild
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
TR=$NR/methods/dense/train_reward_model.py
SC=$NR/methods/dense/score_eval_dense_v4.py
RR=$NR/datasets/creative-writing/royalroad_stubs
SA=$NR/datasets/creative-writing/cw_transfer_v1/stageA
mkdir -p "$LOGS"; ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[wait] phase 2 waiting for V3 arm-a $(ts)"
for i in $(seq 1 1440); do
  grep -q "RR_V3AUG_CHAIN_DONE" "$LOGS/rr_v3aug_launcher.log" 2>/dev/null && break; sleep 30
done
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
echo "$(ts) | cell=cw_royalroad_verdict phase2 (V3 arm b -> decomp1024 view-isolation arm -> Stage-A gate + zero-shot) | GPU=$GPU | agent=$AGENT | job=rr_phase2 | CLAIM (<=8 MiB / 0% util before claim)" >> "$LEDGER"
echo "[claim] GPU=$GPU $(ts)"

train () { # dir maxlen tag
  local d=$1 ml=$2 tag=$3 out=$1/rm_out_seed42
  [ -f "$out/RUN_DONE" ] && { echo "[$tag] done, skip"; return; }
  mkdir -p "$out"; echo "[$tag] START $(ts)"
  DENSE_SPLIT_FRACTION_ATOL=0.03 $PY "$TR" --data_path "$d/data.csv" --split_dir "$d/split" \
    --model_name meta-llama/Llama-3.1-8B --lora_r 16 --lora_alpha 32 \
    --learning_rate 5e-5 --batch_size 16 --eval_batch_size 32 \
    --gradient_accumulation_steps 1 --max_length "$ml" --epochs 2 \
    --gradient-checkpointing --selection_split eval --class_weight_auto \
    --seed 42 --output_dir "$out" > "$out.train.log" 2>&1
  rc=$?; echo "[$tag] EXIT $rc $(ts)"; [ $rc -eq 0 ] && touch "$out/RUN_DONE"
  DENSE_SCORE_MAXLEN=$ml $PY "$SC" --dir "$d" --name "$tag" > "$d/score.log" 2>&1
}

for k in 0 1 2 3 4; do train "$RR/dense_crossfit_v3aug/arm_b/fold$k" 1600 "v3aug_b_f$k"; done
for k in 0 1 2 3 4; do train "$RR/dense_crossfit_decomp1024/fold$k" 1024 "decomp1024_f$k"; done

# ---- Stage-A gate ----
DENSE_SCORE_MAXLEN=1024 $PY "$SC" --dir "$SA" --name cw_transfer_stageA > "$SA/score.log" 2>&1
echo "[A] stage-A gate scored rc=$? $(ts)"

# ---- Stage-B-free zero-shot on both CW honest sets ----
$PY - <<'PYX'
import pandas as pd, json
from pathlib import Path
R = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing")
for tag, cf, pop in (("royalroad", R/"royalroad_stubs/dense_crossfit",
                      R/"royalroad_stubs/va/population.csv.gz"),
                     ("wigleaf", R/"wigleaf/dense_crossfit", R/"wigleaf/va/population.csv.gz")):
    ids = []
    for k in range(5):
        ids += pd.read_csv(cf/f"fold{k}/split/test.csv").row_id.astype(str).tolist()
    p = pd.read_csv(pop); p["row_id"] = p.row_id.astype(str)
    h = p[p.row_id.isin(set(ids))][["text","judgement","group","row_id"]]
    d = R/"cw_transfer_v1"/f"zeroshot_{tag}"; (d/"split").mkdir(parents=True, exist_ok=True)
    h.to_csv(d/"data.csv", index=False)
    for sp in ("eval","test"): h.to_csv(d/f"split/{sp}.csv", index=False)
    print(tag, "honest rows", len(h), "pos", int(h.judgement.sum()))
PYX
for tag in royalroad wigleaf; do
  Z=$NR/datasets/creative-writing/cw_transfer_v1/zeroshot_$tag
  ln -sfn "$SA/rm_out_seed42" "$Z/rm_out_seed42"
  DENSE_SCORE_MAXLEN=1024 $PY "$SC" --dir "$Z" --name "zeroshot_$tag" > "$Z/score.log" 2>&1
  echo "[Z] zero-shot $tag rc=$? $(ts)"
done
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=rr_phase2 | RELEASE rc=0" >> "$LEDGER"
echo "RR_PHASE2_CHAIN_DONE $(ts)"
