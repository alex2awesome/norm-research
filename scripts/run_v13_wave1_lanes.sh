#!/usr/bin/env bash
set -euo pipefail

ROOT=${V13_ROOT:-/lfs/skampere2/0/alexspan/cr3-v13.1}
PY=${V13_PYTHON:-/lfs/skampere2/0/alexspan/cr3-v12/envs/ai_usage_v12/bin/python}
MANIFEST=${V13_MANIFEST:-$ROOT/manifests/wave1.json}
CODE=${V13_CODE:-$ROOT/code}
OUT=${V13_OUT:-$ROOT/outputs/wave1}
LOGS=${V13_LOGS:-$ROOT/logs/wave1}
HOME_ROOT=${METRIC_IMPLEMENTER_LFS_HOME:-/lfs/skampere2/0/alexspan}

mkdir -p "$OUT/lanes" "$LOGS"

launch_lane() {
    local lane=$1
    local devices=$2
    local tp=$3
    local executor_tp=$4
    local model=$5
    local pid_file="$LOGS/$lane.pid"
    local log_file="$LOGS/$lane.log"
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
        echo "$lane already running as PID $(<"$pid_file")"
        return
    fi
    HOME="$HOME_ROOT" \
    HF_HOME="$HOME_ROOT/.cache/huggingface" \
    METRIC_IMPLEMENTER_LFS_HOME="$HOME_ROOT" \
    CUDA_VISIBLE_DEVICES="$devices" \
    VLLM_GPU_MEM_UTIL=0.82 \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_TP_SIZE="$tp" \
    VLLM_EXECUTOR_TP_SIZE="$executor_tp" \
    PYTHONPATH="$CODE" \
    nohup "$PY" -u -m methods.metric_implementer.experiments.run_v13_value_campaign \
        --channels mcq behavioral \
        --constructor-models "$model" \
        --metrics-manifest "$MANIFEST" \
        --tier A \
        --out-root "$OUT/lanes/$lane" \
        --query-batch-size 2048 \
        >"$log_file" 2>&1 &
    echo $! >"$pid_file"
    echo "$lane launched as PID $(<"$pid_file") on CUDA devices $devices"
}

launch_lane llama31_8b 1 1 1 meta-llama/Llama-3.1-8B-Instruct
launch_lane qwen25_14b 2 1 1 Qwen/Qwen2.5-14B-Instruct
launch_lane phi4 3 1 1 microsoft/phi-4
launch_lane llama33_70b 4,5 2 1 meta-llama/Llama-3.3-70B-Instruct
