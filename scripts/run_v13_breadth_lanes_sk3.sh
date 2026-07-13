#!/usr/bin/env bash
set -euo pipefail

# Launch independently resumable v13.1 breadth/upgrade lanes on sk3.  Invoke once
# with V13_LANES="llama31_8b qwen25_14b phi4", then add llama33_70b when GPU 0 is free.
ROOT=${V13_ROOT:-/lfs/skampere3/0/alexspan/cr3-v13.1}
PY=${V13_PYTHON:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
HOME_ROOT=${METRIC_IMPLEMENTER_LFS_HOME:-/lfs/skampere3/0/alexspan}
CODE=${V13_CODE:-$ROOT/code}
CAMPAIGN=${V13_CAMPAIGN:-tier_b}
TIER=${V13_TIER:-B}
MANIFEST=${V13_MANIFEST:-$ROOT/manifests/tier_b.json}
OUT=${V13_OUT:-$ROOT/outputs/$CAMPAIGN}
LOGS=${V13_LOGS:-$ROOT/logs/$CAMPAIGN}
LANES=${V13_LANES:-"llama31_8b qwen25_14b phi4 llama33_70b"}
LLAMA70_PATH=${V13_LLAMA70_PATH:-/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}
SK2_V13_ASSETS=/lfs/skampere2/0/alexspan/cr3-v13.1/assets/tier_b
SK3_V13_ASSETS=$ROOT/assets/tier_b
SK2_LEGACY_INPUTS=/lfs/skampere2/0/alexspan/cr3-v12/inputs
SK3_LEGACY_INPUTS=$ROOT/assets/tier_b_inputs
PATH_REWRITES="{\"$SK2_V13_ASSETS\":\"$SK3_V13_ASSETS\",\"$SK2_LEGACY_INPUTS\":\"$SK3_LEGACY_INPUTS\"}"

mkdir -p "$OUT/lanes" "$LOGS"

launch_lane() {
    local lane=$1
    local devices=$2
    local model=$3
    local overrides=$4
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
    VLLM_GPU_MEM_UTIL=0.90 \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_TP_SIZE=1 \
    VLLM_EXECUTOR_TP_SIZE=1 \
    V13_MODEL_PATH_OVERRIDES_JSON="$overrides" \
    V13_PATH_REWRITE_JSON="$PATH_REWRITES" \
    PYTHONPATH="$CODE" \
    nohup "$PY" -u -m methods.metric_implementer.experiments.run_v13_value_campaign \
        --channels mcq behavioral \
        --constructor-models "$model" \
        --metrics-manifest "$MANIFEST" \
        --tier "$TIER" \
        --out-root "$OUT/lanes/$lane" \
        --query-batch-size 2048 \
        --disable-auto-upgrade \
        >"$log_file" 2>&1 &
    echo $! >"$pid_file"
    echo "$lane launched as PID $(<"$pid_file") on CUDA device $devices"
}

for lane in $LANES; do
    case "$lane" in
        llama31_8b)
            launch_lane "$lane" 5 meta-llama/Llama-3.1-8B-Instruct '{}'
            ;;
        qwen25_14b)
            launch_lane "$lane" 6 Qwen/Qwen2.5-14B-Instruct '{}'
            ;;
        phi4)
            launch_lane "$lane" 7 microsoft/phi-4 '{}'
            ;;
        llama33_70b)
            launch_lane "$lane" 0 meta-llama/Llama-3.3-70B-Instruct \
                "{\"meta-llama/Llama-3.3-70B-Instruct\":\"$LLAMA70_PATH\"}"
            ;;
        *)
            echo "unknown lane: $lane" >&2
            exit 2
            ;;
    esac
done
