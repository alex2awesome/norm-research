#!/bin/bash
# Wait for Google Patents scraper to finish, then build v3 FAISS index +
# re-bake top-K retrievals using the extended corpus.
set -e
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
mkdir -p logs

echo "=== $(date) Waiting for Google Patents scraper to finish ==="
while pgrep -f fetch_missing_from_google_patents > /dev/null; do
    sleep 300  # 5-min poll
done
echo "=== $(date) GP scraper done ==="
sleep 60  # let final flush settle

echo "=== $(date) STEP 1: v3 FAISS index (extended corpus) ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/build_faiss_index_v3.py

echo "=== $(date) STEP 2: Re-bake top-K against v3 index ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/bake_top_k_retrievals.py \
    --index_dir /lfs/skampere3/0/alexspan/norm-research/indexes/patent_claims_v3 \
    --output /lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_final_outcome_cpc_balanced_with_rejections_with_retrievals_v3.csv.gz

echo "=== $(date) v3 PIPELINE DONE ==="
