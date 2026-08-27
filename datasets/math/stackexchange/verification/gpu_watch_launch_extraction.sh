#!/bin/bash
# Watch for a GPU with enough free memory for Qwen3.5-122B-FP8 (~127G weights),
# then launch the Math.SE sympy-claim extraction on the pool eval split.
# Retries if the launch dies early (e.g. lost a race for the GPU).
# Logs to gpu_watch.log next to this script.

NEED_MIB=145000
VDIR=/lfs/skampere3/0/alexspan/norm-research/datasets/math/stackexchange/verification
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python3.11
LOG=$VDIR/gpu_watch.log
export HOME=/lfs/skampere3/0/alexspan

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "watcher started (need ${NEED_MIB}MiB free)"
while true; do
  GPU=""
  while read -r idx used total; do
    free=$((total - used))
    if [ "$free" -ge "$NEED_MIB" ]; then GPU=$idx; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
           --format=csv,noheader,nounits | tr -d ',')
  if [ -n "$GPU" ]; then
    log "GPU $GPU has >= ${NEED_MIB}MiB free — launching extraction"
    cd "$VDIR" || exit 1
    CUDA_VISIBLE_DEVICES=$GPU \
      VLLM_USE_FLASHINFER_MOE_FP8=0 FLASHINFER_DISABLE_VERSION_CHECK=1 \
      TOKENIZERS_PARALLELISM=false \
      nohup "$PY" run_extraction_sk3.py \
        --data ../math_se_v3_1_pool.csv.gz --split eval \
        > extraction_eval.log 2>&1 &
    PID=$!
    log "launched pid $PID on GPU $GPU; verifying survival (6 min)"
    sleep 360
    if kill -0 "$PID" 2>/dev/null && grep -qi "init engine\|Engine.*init\|Loading model\|llm_engine" extraction_eval.log; then
      log "extraction alive after 6 min (pid $PID, GPU $GPU) — watcher standing down"
      exit 0
    fi
    log "launch died or no engine-init line (lost GPU race?); resuming watch"
    tail -5 extraction_eval.log >> "$LOG"
  fi
  sleep 45
done
