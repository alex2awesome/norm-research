#!/usr/bin/env bash
set -euo pipefail

bundle=/tmp/metric-seam-family-scale-sk3-0704955.tgz
wait_seconds=${WAIT_SECONDS:-900}
sleep "$wait_seconds"

ssh sk3 'set -euo pipefail
ROOT=/lfs/skampere3/0/alexspan/metric-seam-family-scale-replication-0704955
mkdir -p "$ROOT"
tar -xzf - -C "$ROOT"

if [ -f "$ROOT/results_llama70b/manifest.json" ]; then
  echo "ALREADY_COMPLETE root=$ROOT"
  exit 0
fi
if [ -f "$ROOT/submission.receipt" ]; then
  old_pid=$(awk -F= '\''$1=="pid" {print $2}'\'' "$ROOT/submission.receipt")
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ALREADY_RUNNING pid=$old_pid root=$ROOT"
    exit 0
  fi
fi

mine=0
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d " " | sort -u); do
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d " " || true)
  if [ "$owner" = alexspan ]; then mine=$((mine + 1)); fi
done
if [ "$mine" -ge 4 ]; then
  echo "NOT_LAUNCHED active_user_gpu_processes=$mine cap=4"
  exit 75
fi

gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '\''
  {gsub(/ /,"",$1); gsub(/ /,"",$2)}
  ($1==0||$1==5||$1==6||$1==7) && $2<1000 {print $1; exit}
'\'')
if [ -z "$gpu" ]; then
  echo "NOT_LAUNCHED no_free_allowed_gpu active_user_gpu_processes=$mine"
  exit 75
fi

MODEL=/lfs/skampere3/0/alexspan/merged_models/Llama-3.3-70B-FP8-with-tokenizer
if [ ! -d "$MODEL" ]; then
  MODEL=$(find /lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
fi
test -d "$MODEL"
REQ="$ROOT/outputs/metric_seam_pilot/family_scale_v1/semantic_alignment_complete/requests"
OUT="$ROOT/results_llama70b"
LOG="$ROOT/llama70b.log"
LOCK=/tmp/metric-seam-family-scale-gpu-${gpu}.lock
mkdir -p "$OUT"

nohup flock -n "$LOCK" env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  HOME=/lfs/skampere3/0/alexspan \
  HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface \
  VLLM_CACHE_ROOT=/lfs/skampere3/0/alexspan/.cache/vllm \
  TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/.cache/triton \
  XDG_CACHE_HOME=/lfs/skampere3/0/alexspan/.cache \
  /lfs/skampere3/0/alexspan/miniconda3/bin/python \
  "$ROOT/methods/metric_seam/family_scale/semantic_alignment_vllm.py" \
  --requests-dir "$REQ" --output-dir "$OUT" --model "$MODEL" \
  --gpu-memory-utilization 0.90 --max-model-len 16384 --max-tokens 3000 \
  >"$LOG" 2>&1 &
pid=$!
printf "pid=%s\ngpu=%s\nactive_before=%s\nroot=%s\nmodel=%s\nlog=%s\n" \
  "$pid" "$gpu" "$mine" "$ROOT" "$MODEL" "$LOG" > "$ROOT/submission.receipt"
sleep 5
if kill -0 "$pid" 2>/dev/null; then
  echo "SUBMITTED pid=$pid gpu=$gpu active_before=$mine root=$ROOT"
  sed -n "1,30p" "$LOG"
else
  echo "LAUNCH_FAILED pid=$pid gpu=$gpu"
  tail -100 "$LOG"
  exit 1
fi
' < "$bundle"
