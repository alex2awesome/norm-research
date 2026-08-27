#!/bin/bash
# Sequential single-GPU job queue, v2: per-job scorer + per-job python env.
# Job file line: "<prompts_path> <out_path> [scorer_py] [python_bin]"
#   scorer default = gemma_score_v1.py (gemma4 env); llama jobs pass llama_score_sk3.py
#   with the base miniconda python (vLLM 0.17 handles Llama BF16; gemma4 env handles Gemma).
# Same semantics as gpu_queue.sh otherwise: watches QDIR2, one job at a time on $GPU,
# done/ on rc=0, .failed on error, exits when STOP present and queue empty. Idempotent.
export HOME=/lfs/skampere3/0/alexspan
QDIR=/lfs/skampere3/0/alexspan/norm-research/outputs/metric_seam_pilot/queue2
GEMMA_PY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
BASE_PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
GEMMA_SCORER=/lfs/skampere3/0/alexspan/norm-research/outputs/metric_seam_pilot/v1/gemma_score_v1.py
GPU=${GPU:-1}
mkdir -p "$QDIR/done"
echo "queue2 start $(date)" >> "$QDIR/queue.log"
while true; do
  job=$(ls "$QDIR"/*.job 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then
    if [ -f "$QDIR/STOP" ]; then echo "queue2 stop $(date)" >> "$QDIR/queue.log"; exit 0; fi
    sleep 30; continue
  fi
  read -r prompts out scorer pybin < "$job"
  scorer=${scorer:-$GEMMA_SCORER}
  pybin=${pybin:-$GEMMA_PY}
  case "$scorer" in *llama*) pybin=$BASE_PY;; esac
  echo "job $(basename "$job") start ($scorer) $(date)" >> "$QDIR/queue.log"
  CUDA_VISIBLE_DEVICES=$GPU "$pybin" "$scorer" --prompts "$prompts" --out "$out" \
      --max-model-len 10240 >> "${out%.jsonl}.log" 2>&1
  rc=$?
  echo "job $(basename "$job") rc=$rc $(date)" >> "$QDIR/queue.log"
  if [ $rc -eq 0 ]; then mv "$job" "$QDIR/done/"; else mv "$job" "$job.failed"; fi
done
