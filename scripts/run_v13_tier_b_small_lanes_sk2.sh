#!/usr/bin/env bash
set -euo pipefail

ROOT=${V13_ROOT:-/lfs/skampere2/0/alexspan/cr3-v13.1}
PY=${V13_PYTHON:-/lfs/skampere2/0/alexspan/cr3-v12/envs/ai_usage_v12/bin/python}
HOME_ROOT=${METRIC_IMPLEMENTER_LFS_HOME:-/lfs/skampere2/0/alexspan}
MANIFEST=${V13_MANIFEST:-$ROOT/assets/tier_b/tier_b_metrics.json}
CODE=${V13_CODE:-$ROOT/code}
OUT=${V13_OUT:-$ROOT/outputs/tier_b}
LOGS=${V13_LOGS:-$ROOT/logs/tier_b}

mkdir -p "$OUT/lanes" "$LOGS"
test -f "$MANIFEST"

launch_lane() {
    local lane=$1
    local device=$2
    local model=$3
    local pid_file="$LOGS/$lane.pid"
    local log_file="$LOGS/$lane.log"
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
        echo "$lane already running as PID $(<"$pid_file")"
        return
    fi
    HOME="$HOME_ROOT" \
    HF_HOME="$HOME_ROOT/.cache/huggingface" \
    METRIC_IMPLEMENTER_LFS_HOME="$HOME_ROOT" \
    CUDA_VISIBLE_DEVICES="$device" \
    VLLM_GPU_MEM_UTIL=0.82 \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_TP_SIZE=1 \
    VLLM_EXECUTOR_TP_SIZE=1 \
    PYTHONPATH="$CODE" \
    nohup "$PY" -u -m methods.metric_implementer.experiments.run_v13_value_campaign \
        --channels mcq behavioral \
        --constructor-models "$model" \
        --metrics-manifest "$MANIFEST" \
        --tier B \
        --out-root "$OUT/lanes/$lane" \
        --query-batch-size 2048 \
        --disable-auto-upgrade \
        >"$log_file" 2>&1 &
    echo $! >"$pid_file"
    echo "$lane launched as PID $(<"$pid_file") on CUDA device $device"
}

launch_lane llama31_8b 4 meta-llama/Llama-3.1-8B-Instruct
launch_lane qwen25_14b 5 Qwen/Qwen2.5-14B-Instruct
launch_lane phi4 7 microsoft/phi-4
