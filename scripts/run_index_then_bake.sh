#!/bin/bash
# Chain: build FAISS index over 4.7M claims, then bake top-K into the patents file.
set -e
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
mkdir -p logs indexes

echo "=== $(date) STEP 1: FAISS index ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/build_faiss_index_direct.py

echo "=== $(date) STEP 2: Bake top-K retrievals ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/bake_top_k_retrievals.py

echo "=== $(date) DONE ==="
