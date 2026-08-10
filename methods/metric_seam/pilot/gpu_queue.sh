#!/bin/bash
# Sequential single-GPU job queue for seam-survey batches on sk3.
# Watches QDIR for *.job files (each: "<prompts_path> <out_path>"), runs them one at a
# time through gemma_score_v1.py on CUDA_VISIBLE_DEVICES=$GPU, moves finished jobs to
# done/. Exits when a STOP file appears and no jobs remain. Idempotent (scorer resumes).
export HOME=/lfs/skampere3/0/alexspan
QDIR=/lfs/skampere3/0/alexspan/norm-research/outputs/metric_seam_pilot/queue
PY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
SCORER=/lfs/skampere3/0/alexspan/norm-research/outputs/metric_seam_pilot/v1/gemma_score_v1.py
GPU=${GPU:-1}
mkdir -p "$QDIR/done"
echo "queue start $(date)" >> "$QDIR/queue.log"
while true; do
  job=$(ls "$QDIR"/*.job 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then
    if [ -f "$QDIR/STOP" ]; then echo "queue stop $(date)" >> "$QDIR/queue.log"; exit 0; fi
    sleep 30; continue
  fi
  read -r prompts out < "$job"
  echo "job $(basename "$job") start $(date)" >> "$QDIR/queue.log"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCORER" --prompts "$prompts" --out "$out" \
      --max-model-len 10240 >> "${out%.jsonl}.log" 2>&1
  rc=$?
  echo "job $(basename "$job") rc=$rc $(date)" >> "$QDIR/queue.log"
  if [ $rc -eq 0 ]; then mv "$job" "$QDIR/done/"; else mv "$job" "$job.failed"; fi
done
