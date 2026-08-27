#!/bin/bash
# v2 anticipation pipeline: waits for granted parquet to land, then runs
# v2 pair extraction → re-fine-tune BGE-M3 → re-build retriv index.
set -e
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
mkdir -p logs

GRANTED=/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/granted_patents_claim1.parquet

# Wait until granted parquet exists.
echo "=== $(date) waiting for granted parquet ==="
until [ -f "$GRANTED" ]; do sleep 60; done
echo "=== $(date) granted parquet ready ==="

# Wait a beat for the writer to flush.
sleep 30

echo "=== $(date) STEP V2-1: Extract v2 training pairs ==="
python3 scripts/extract_anticipation_training_pairs_v2.py
echo "=== $(date) STEP V2-1 done ==="

# Wait for v1 fine-tune to finish before launching v2 fine-tune (same GPU)
echo "=== $(date) waiting for v1 fine-tune to clear GPU ==="
until ! pgrep -f finetune_bge_m3_anticipation.py > /dev/null; do sleep 60; done
echo "=== $(date) GPU available ==="

echo "=== $(date) STEP V2-2: Fine-tune BGE-M3 on v2 pairs ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/finetune_bge_m3_anticipation.py \
    --base_model BAAI/bge-m3 \
    --batch_size 32 \
    --epochs 2 \
    --max_seq_len 512 \
    --lr 2e-5 \
    --out /lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2
echo "=== $(date) STEP V2-2 done ==="

echo "=== $(date) STEP V2-3: Build retriv index (v2 model) ==="
python3 scripts/build_retriv_claim_index.py --collection patent_claims_v2 --model /lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2
echo "=== $(date) STEP V2-3 done ==="

echo "=== $(date) V2 PIPELINE COMPLETE ==="
