#!/bin/bash
# v2 fine-tune (FAST): 1M sampled pairs, 1 epoch, seq_len 256, batch 64.
# Target wall time: 2-4 hours on a single B200.
set -e
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
mkdir -p logs

echo "=== $(date) STEP A: v2 fine-tune (fast settings) ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/finetune_bge_m3_anticipation.py \
    --base_model BAAI/bge-m3 \
    --pairs /lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/anticipation_training_pairs_v2.jsonl.gz \
    --batch_size 64 \
    --epochs 1 \
    --max_seq_len 256 \
    --lr 2e-5 \
    --max_pairs 1000000 \
    --out /lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2

echo "=== $(date) STEP B: retriv index ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/build_retriv_claim_index.py \
    --collection patent_claims_v2 \
    --model /lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2

echo "=== $(date) DONE ==="
