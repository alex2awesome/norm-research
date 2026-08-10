#!/usr/bin/env bash
# Polls GPU 7 every 30s; when memory.used drops below 2 GB, launches a third
# classifier worker on it. Workers coordinate via .processing markers in
# outputs/classifier_chunks_FULL. Exits after launching.

set -uo pipefail
cd /lfs/skampere3/0/alexspan/norm-research

GPU=7
THRESHOLD_MB=2000
POLL_INTERVAL_SEC=30

TS=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG="logs/batch_runs/classifier_FULL_AUTO_gpu${GPU}_${TS}.log"
WAIT_LOG="logs/batch_runs/wait_for_gpu${GPU}_classifier_${TS}.log"

echo "[wait_gpu${GPU}] start; polling GPU ${GPU} (threshold <= ${THRESHOLD_MB} MB)" | tee -a $WAIT_LOG

while true; do
    mem=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$mem" ] && [ "$mem" -lt "$THRESHOLD_MB" ]; then
        echo "[wait_gpu${GPU}] FOUND idle (used=${mem} MiB) -- launching classifier" | tee -a $WAIT_LOG

        nohup env HOME=/lfs/skampere3/0/alexspan \
                 HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface \
                 PATH=/usr/local/cuda-12.6/bin:/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH \
                 LD_LIBRARY_PATH=$(ls -d /lfs/skampere3/0/alexspan/miniconda3/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs \
                 CUDA_VISIBLE_DEVICES=$GPU \
                 /lfs/skampere3/0/alexspan/miniconda3/bin/python3 -u scripts/batch_classify_vllm.py \
                   --chunk-size 1024 \
                   --gpu-memory-utilization 0.95 \
                   --output outputs/classifier_llama_FULL.parquet \
                   --chunks-dir outputs/classifier_chunks_FULL \
                   > $LAUNCH_LOG 2>&1 &

        PID=$!
        disown
        echo "[wait_gpu${GPU}] launched PID $PID on GPU $GPU -- log: $LAUNCH_LOG" | tee -a $WAIT_LOG
        exit 0
    fi
    echo "[wait_gpu${GPU}] $(date +%H:%M:%S): GPU $GPU used=${mem:-?} MiB" >> $WAIT_LOG
    sleep $POLL_INTERVAL_SEC
done
