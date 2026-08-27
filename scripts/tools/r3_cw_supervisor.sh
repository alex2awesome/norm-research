#!/usr/bin/env bash
# R3 creative-writing B_E scaling run — PERSISTENT (nohup; survives laptop close).
# Proposer = GLM-4.7 (3 lists, zai_anthropic). Executors = llama-8B (Phase A, generates Ω) then
# rescore the SAME Ω with llama-3B / llama-70B-FP8 / Qwen-122B (Phase B, apples-to-apples scaling).
# Phase C = be_report (B_E, recovery, unsupervised, scaling, bge signal-matching accuracy).
# Waits for bge_pertask (train_cross_encoder) to free GPUs first. Crash-resilient via --skip-existing.
set -u
REPO=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
OUT=/lfs/skampere3/0/alexspan/outputs/r3_cw
LOG=$OUT/_log; mkdir -p "$LOG"
SUPLOG=$OUT/supervisor.log
GIJSON=/lfs/skampere3/0/alexspan/outputs/crc_scaling/r3_55_gi.json   # 57 distinct R3 gi
FAMJSON=$OUT/families_3glm.json
FP8=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
ZAI_KEY=/lfs/skampere3/0/alexspan/.z-ai-api-key.txt
COMMON="HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache VLLM_GPU_MEM_UTIL=0.93 VLLM_MAX_MODEL_LEN=12288 TOKENIZERS_PARALLELISM=false"
GI=$($PY -c "import json;print(','.join(map(str,json.load(open('$GIJSON'))['gi'])))")
N=$($PY -c "import json;print(json.load(open('$GIJSON'))['n'])")
A_DIR=$OUT/llama8b_glm
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$SUPLOG"; }
log "R3 supervisor start. metrics=$N families=3xglm-4.7 executors=llama8b/3b/70b-fp8/qwen122b"

# ---- wait for bge_pertask to free the GPUs ----
log "waiting for bge_pertask (train_cross_encoder) to finish..."
while pgrep -f train_cross_encoder.py >/dev/null 2>&1; do sleep 120; done
log "bge_pertask finished; GPUs freeing."
sleep 20
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$SUPLOG" 2>&1

# ---- Phase A: GLM-4.7 proposer + llama-8B, 57 R3 gi, GPU1 ----
log "Phase A: run_alpha_probe (llama8b + 3xglm-4.7) GPU1, $N metrics -> $A_DIR"
( cd "$REPO" && env $COMMON ZAI_KEY_FILE=$ZAI_KEY CUDA_VISIBLE_DEVICES=1 \
    $PY -m methods.metric_implementer.experiments.run_alpha_probe \
    --task creative-writing --level R3 --gi-list "$GI" --n-metrics 0 --skip-existing \
    --target-model meta-llama/Llama-3.1-8B-Instruct --M-freegen 600 --n-probes 300 \
    --cmi-thresh 0.15 --form-invariance-n 12 --families "$(cat $FAMJSON)" \
    --out-dir "$A_DIR" > "$LOG/phaseA.log" 2>&1 ) &
PIDA=$!
log "Phase A pid=$PIDA"

# ---- Phase B: rescore loops — poll A_DIR, rescore each new ckpt (--skip-existing), GPU2/3/7 ----
rescore_loop(){
  local gpu=$1 model=$2 suffix=$3 extra=$4
  local RD="$OUT/$suffix"
  mkdir -p "$RD"
  while true; do
    local nd=$(ls "$RD"/*_sigs.npz 2>/dev/null | wc -l)
    if [ "$nd" -ge "$N" ]; then log "$suffix rescore complete ($nd/$N)"; break; fi
    ( cd "$REPO" && env $COMMON $extra CUDA_VISIBLE_DEVICES=$gpu \
        $PY -m methods.metric_implementer.experiments.rescore_executor \
        --src-dir "$A_DIR" --target-model "$model" --out-dir "$RD" \
        --task creative-writing --skip-existing > "$LOG/$suffix.log" 2>&1 )
    sleep 600
  done
}
log "Phase B: rescore loops launching (GPU2=3b, GPU3=70b-fp8, GPU7=qwen)"
rescore_loop 2 meta-llama/Llama-3.2-3B-Instruct       llama8b_to_llama3b     "" &
rescore_loop 3 "$FP8"                                  llama8b_to_llama70b_fp8 "FLASHINFER_DISABLE_VERSION_CHECK=1" &
rescore_loop 7 Qwen/Qwen3.5-122B-A10B-FP8              llama8b_to_qwen122b    "VLLM_USE_FLASHINFER_MOE_FP8=0" &

wait $PIDA; log "Phase A (llama8b+GLM) complete."
wait;        log "All Phase B rescoring complete."

# ---- Phase C: report ----
log "Phase C: be_report"
( cd "$REPO" && PYTHONPATH="$REPO" $PY -m methods.metric_implementer.experiments.be_report \
    --executors llama8b:"$A_DIR",llama3b:$OUT/llama8b_to_llama3b,llama70b_fp8:$OUT/llama8b_to_llama70b_fp8,qwen122b:$OUT/llama8b_to_qwen122b \
    --json "$OUT/be_report.json" > "$LOG/be_report.log" 2>&1 )
log "DONE. report -> $OUT/be_report.json ; tables -> $LOG/be_report.log"
