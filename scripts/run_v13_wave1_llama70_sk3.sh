#!/usr/bin/env bash
set -euo pipefail

ROOT=${V13_ROOT:-/lfs/skampere3/0/alexspan/cr3-v13.1}
PY=${V13_PYTHON:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
HOME_ROOT=${METRIC_IMPLEMENTER_LFS_HOME:-/lfs/skampere3/0/alexspan}
MODEL_PATH=${V13_LLAMA70_PATH:-/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}
SOURCE_PREFIX=/lfs/skampere2/0/alexspan/cr3-v12/consolidation_v1/sk2_runs/cr3_mcq_v12t8c_humor_R3_sentinel6_af623e0261far2
ASSET_PREFIX=$ROOT/assets/wave1
OUT=$ROOT/outputs/wave1/lanes/llama33_70b
LOG=$ROOT/logs/wave1/llama33_70b.log
PID_FILE=$ROOT/logs/wave1/llama33_70b.pid

mkdir -p "$(dirname "$LOG")" "$OUT"
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
    echo "llama33_70b already running as PID $(<"$PID_FILE")"
    exit 0
fi

HOME="$HOME_ROOT" \
HF_HOME="$HOME_ROOT/.cache/huggingface" \
METRIC_IMPLEMENTER_LFS_HOME="$HOME_ROOT" \
CUDA_VISIBLE_DEVICES=${V13_CUDA_DEVICES:-0} \
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
    --tier A \
    --out-root "$OUT" \
    --query-batch-size 2048 \
    >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "llama33_70b launched as PID $(<"$PID_FILE")"
