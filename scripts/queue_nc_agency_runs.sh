#!/bin/bash
# Wait for a PID to exit, then launch sequential local_explanations runs
# for the 5 smallest notice-and-comment agencies on GPU 7.
#
# Usage: queue_nc_agency_runs.sh <PID_to_wait_on>

# NOTE: do NOT use `set -u` here — conda's nvcc activate script references
# NVCC_PREPEND_FLAGS without setting it first and trips on unbound-variable.
# Also avoid `set -e` — one failed agency run shouldn't kill the rest of the queue.
WAIT_PID="${1:?usage: $0 <pid-to-wait-on>}"

# Require the PID to actually exist right now; otherwise fail loudly.
if ! kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "[$(date)] ERROR: target PID $WAIT_PID is not running. refusing to start queue."
    exit 1
fi

REPO=/lfs/skampere3/0/alexspan/norm-research
QWEN=/lfs/skampere3/0/shared_hf_cache/hub/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/fb53b9f3bdaab287c597d4e943783153ec527e06
LOGDIR="$REPO/outputs/nc_agency_queue_logs"
mkdir -p "$LOGDIR"

# 5 smallest agencies (ascending)
AGENCIES=(fsis fs irs blm osha)

# One-time env setup
export HOME=/lfs/skampere3/0/alexspan
cd "$REPO"
source /lfs/skampere3/0/alexspan/miniconda3/etc/profile.d/conda.sh && conda activate base
export PATH=/usr/local/cuda-12.6/bin:/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
export LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$(ls -d /lfs/skampere3/0/alexspan/miniconda3/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=7
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export OPENAI_API_KEY=$(cat /lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt)

echo "[$(date)] waiting for PID $WAIT_PID to exit..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] PID $WAIT_PID exited. starting agency queue: ${AGENCIES[*]}"

for AGENCY in "${AGENCIES[@]}"; do
    LOG="$LOGDIR/${AGENCY}.log"
    echo "[$(date)] === launching $AGENCY ==="
    python3 scripts/run_local_explanations.py \
        --dataset notice-and-comment \
        --agency "$AGENCY" \
        --approach star_local \
        --model "$QWEN" \
        --proposer-model openai/gpt-5-mini \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.95 \
        --n-canonical-features 30 \
        --max-text-tokens 512 \
        --output-dir "outputs/local_explanations/nc_${AGENCY}_star_local_qwen35" \
        > "$LOG" 2>&1
    EXIT=$?
    echo "[$(date)] === $AGENCY finished with exit $EXIT ==="
done

echo "[$(date)] all ${#AGENCIES[@]} agencies done."
