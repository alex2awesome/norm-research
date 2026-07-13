#!/usr/bin/env bash
set -euo pipefail

ROOT=${V13_ROOT:-/lfs/skampere3/0/alexspan/cr3-v13.1}
PY=${V13_PYTHON:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
HOME_ROOT=${METRIC_IMPLEMENTER_LFS_HOME:-/lfs/skampere3/0/alexspan}
MODEL_PATH=${V13_LLAMA70_PATH:-/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}
SOURCE_PREFIX=/lfs/skampere2/0/alexspan/cr3-v12/consolidation_v1/sk2_runs/cr3_mcq_v12t8c_humor_R3_sentinel6_af623e0261far2
ASSET_PREFIX=$ROOT/assets/wave1
OUT=$ROOT/outputs/wave1/lanes
LOGS=$ROOT/logs/wave1

mkdir -p "$OUT" "$LOGS"

launch_shard() {
    local lane=$1
    local device=$2
    local output_name=$3
    shift 3
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
    VLLM_GPU_MEM_UTIL=0.90 \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_TP_SIZE=1 \
    VLLM_EXECUTOR_TP_SIZE=1 \
    V13_MODEL_PATH_OVERRIDES_JSON="{\"meta-llama/Llama-3.3-70B-Instruct\":\"$MODEL_PATH\"}" \
    V13_PATH_REWRITE_JSON="{\"$SOURCE_PREFIX\":\"$ASSET_PREFIX\"}" \
    PYTHONPATH="$ROOT/code" \
    nohup "$PY" -u -m methods.metric_implementer.experiments.run_v13_value_campaign \
        --channels mcq behavioral \
        --constructor-models meta-llama/Llama-3.3-70B-Instruct \
        --metrics-manifest "$ROOT/manifests/wave1.json" \
        --metric-keys "$@" \
        --tier A \
        --out-root "$OUT/$output_name" \
        --query-batch-size 2048 \
        >"$log_file" 2>&1 &
    echo $! >"$pid_file"
    echo "$lane launched as PID $(<"$pid_file") on CUDA device $device for $*"
}

# The first shard deliberately reuses the original root and its partial metric-0 cache.
launch_shard llama33_70b_shard_0_10 0 llama33_70b \
    humor_R3_metric0 humor_R3_metric10
launch_shard llama33_70b_shard_11_12 5 llama33_70b_shard_11_12 \
    humor_R3_metric11 humor_R3_metric12
launch_shard llama33_70b_shard_34 6 llama33_70b_shard_34 \
    humor_R3_metric34
launch_shard llama33_70b_shard_50 7 llama33_70b_shard_50 \
    humor_R3_metric50
