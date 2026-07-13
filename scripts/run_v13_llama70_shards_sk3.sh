#!/usr/bin/env bash
set -euo pipefail

# Generic exact-70B sharder for Tier B and the later shared Tier-A upgrades.
ROOT=${V13_ROOT:-/lfs/skampere3/0/alexspan/cr3-v13.1}
PY=${V13_PYTHON:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
HOME_ROOT=${METRIC_IMPLEMENTER_LFS_HOME:-/lfs/skampere3/0/alexspan}
MODEL_PATH=${V13_LLAMA70_PATH:-/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}
CAMPAIGN=${V13_CAMPAIGN:-tier_b}
TIER=${V13_TIER:-B}
MANIFEST=${V13_MANIFEST:-$ROOT/manifests/tier_b.json}
OUT=${V13_OUT:-$ROOT/outputs/$CAMPAIGN}
LOGS=${V13_LOGS:-$ROOT/logs/$CAMPAIGN}
SHARD_IDS=${V13_SHARD_IDS:-"0 1 2 3 4 5"}
DEVICES=(0 3 4 5 6 7)
SK2_V13_ASSETS=/lfs/skampere2/0/alexspan/cr3-v13.1/assets/tier_b
SK3_V13_ASSETS=$ROOT/assets/tier_b
SK2_LEGACY_INPUTS=/lfs/skampere2/0/alexspan/cr3-v12/inputs
SK3_LEGACY_INPUTS=$ROOT/assets/tier_b_inputs
PATH_REWRITES="{\"$SK2_V13_ASSETS\":\"$SK3_V13_ASSETS\",\"$SK2_LEGACY_INPUTS\":\"$SK3_LEGACY_INPUTS\"}"

mkdir -p "$OUT/lanes" "$LOGS"
test -f "$MANIFEST"

mapfile -t METRIC_KEYS < <(
    HOME="$HOME_ROOT" \
    V13_PATH_REWRITE_JSON="$PATH_REWRITES" \
    PYTHONPATH="$ROOT/code" \
    "$PY" - "$MANIFEST" <<'PY'
import sys
from methods.metric_implementer.experiments.run_v13_value_campaign import (
    load_metrics_manifest,
    select_metric_entries,
)
manifest, base = load_metrics_manifest(sys.argv[1])
for entry in select_metric_entries(manifest, base):
    print(entry["metric_key"])
PY
)

if [[ ${#METRIC_KEYS[@]} -eq 0 ]]; then
    echo "no metrics selected from $MANIFEST" >&2
    exit 2
fi

launch_shard() {
    local shard=$1
    local device=${DEVICES[$shard]}
    local lane="llama33_70b_shard_$shard"
    local output="$OUT/lanes/$lane"
    local pid_file="$LOGS/$lane.pid"
    local log_file="$LOGS/$lane.log"
    local keys=()
    local index
    for index in "${!METRIC_KEYS[@]}"; do
        if (( index % ${#DEVICES[@]} == shard )); then
            keys+=("${METRIC_KEYS[$index]}")
        fi
    done
    if [[ ${#keys[@]} -eq 0 ]]; then
        echo "$lane has no assigned metrics"
        return
    fi
    if [[ -f "$output/campaign_manifest.json" ]]; then
        echo "$lane already complete"
        return
    fi
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
    V13_PATH_REWRITE_JSON="$PATH_REWRITES" \
    PYTHONPATH="$ROOT/code" \
    nohup "$PY" -u -m methods.metric_implementer.experiments.run_v13_value_campaign \
        --channels mcq behavioral \
        --constructor-models meta-llama/Llama-3.3-70B-Instruct \
        --metrics-manifest "$MANIFEST" \
        --metric-keys "${keys[@]}" \
        --tier "$TIER" \
        --out-root "$output" \
        --query-batch-size 2048 \
        --disable-auto-upgrade \
        >"$log_file" 2>&1 &
    echo $! >"$pid_file"
    echo "$lane launched as PID $(<"$pid_file") on GPU $device for ${#keys[@]} metrics"
}

for shard in $SHARD_IDS; do
    if (( shard < 0 || shard >= ${#DEVICES[@]} )); then
        echo "invalid shard id: $shard" >&2
        exit 2
    fi
    launch_shard "$shard"
done
