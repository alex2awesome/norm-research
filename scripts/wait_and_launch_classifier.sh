#!/usr/bin/env bash
# Polls nvidia-smi every 60s; when a GPU (except 4, where the primary
# classifier worker is running) drops below 2 GB used, launches an
# additional batch_classify_vllm.py worker on that GPU. Workers share
# outputs/classifier_chunks_FULL via O_CREAT|O_EXCL chunk_*.processing
# files — they won't step on each other. Exits after launching.

set -uo pipefail
cd /lfs/skampere3/0/alexspan/norm-research

# Skip GPU 4 (primary classifier worker is running there). Try the rest.
CANDIDATE_GPUS="0 1 2 3 5 6 7"
THRESHOLD_MB=2000
POLL_INTERVAL_SEC=60

TS=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG="logs/batch_runs/classifier_FULL_AUTO_${TS}.log"
WAIT_LOG="logs/batch_runs/wait_for_gpu_classifier_${TS}.log"

echo "[wait_and_launch_classifier] start; polling for free GPU (threshold ≤${THRESHOLD_MB} MB, skipping GPU 4)" | tee -a $WAIT_LOG

while true; do
    for gpu in $CANDIDATE_GPUS; do
        mem=$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -z "$mem" ]; then continue; fi
        if [ "$mem" -lt "$THRESHOLD_MB" ]; then
            echo "[wait_and_launch_classifier] FOUND idle GPU $gpu (used=${mem} MiB) — launching classifier worker" | tee -a $WAIT_LOG

            nohup env HOME=/lfs/skampere3/0/alexspan \
                     HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface \
                     PATH=/usr/local/cuda-12.6/bin:/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH \
                     LD_LIBRARY_PATH=$(ls -d /lfs/skampere3/0/alexspan/miniconda3/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs \
                     CUDA_VISIBLE_DEVICES=$gpu \
                     /lfs/skampere3/0/alexspan/miniconda3/bin/python3 -u scripts/batch_classify_vllm.py \
                       --chunk-size 1024 \
                       --gpu-memory-utilization 0.95 \
                       --output outputs/classifier_llama_FULL.parquet \
                       --chunks-dir outputs/classifier_chunks_FULL \
                       > $LAUNCH_LOG 2>&1 &

            PID=$!
            disown
            echo "[wait_and_launch_classifier] launched PID $PID on GPU $gpu — log: $LAUNCH_LOG" | tee -a $WAIT_LOG
            exit 0
        fi
    done
    summary=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd' | ' -)
    echo "[wait_and_launch_classifier] all busy at $(date +%H:%M:%S): $summary" >> $WAIT_LOG
    sleep $POLL_INTERVAL_SEC
done
