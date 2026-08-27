#!/bin/bash
# Chained pipeline for §102 anticipation retrieval training.
#  Step 1: extract pairs from OARD+JSONL
#  Step 2: fine-tune BGE-M3 on those pairs (GPU)
#  Step 3: build retriv index over the full claim corpus
#
# Run as:
#   nohup bash scripts/run_anticipation_pipeline.sh > logs/anticipation_pipeline.log 2>&1 &
set -e
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
mkdir -p logs

echo "=== $(date) STEP 1: Extract training pairs ==="
python3 scripts/extract_anticipation_training_pairs.py
echo "=== $(date) STEP 1 done ==="

# Pick GPU 6 (currently empty after we move things around)
# But first verify it's still free
echo "=== $(date) Checking GPU state ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
TARGET_GPU=6

echo "=== $(date) STEP 2: Fine-tune BGE-M3 ==="
CUDA_VISIBLE_DEVICES=$TARGET_GPU python3 scripts/finetune_bge_m3_anticipation.py \
    --base_model BAAI/bge-m3 \
    --batch_size 32 \
    --epochs 2 \
    --max_seq_len 512 \
    --lr 2e-5
echo "=== $(date) STEP 2 done ==="

echo "=== $(date) STEP 3: Build retriv index ==="
python3 scripts/build_retriv_claim_index.py
echo "=== $(date) STEP 3 done ==="

echo "=== $(date) PIPELINE COMPLETE ==="
