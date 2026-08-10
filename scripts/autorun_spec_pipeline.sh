#!/usr/bin/env bash
# Autorun chain: waits for PV long-text download, then chunks → embeds → indexes
# spec passages. Also extracts clean §102 pairs in parallel.
set -e
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
mkdir -p logs

PV_GRANT_DIR=/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant
PV_PG_DIR=/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_pg

echo "=== $(date) Waiting for PV long-text download (g_detail_desc + pg_detail_desc) ==="
# Wait until both granted and pgpub detail-desc files exist for the latest year (2025)
# AND the downloader process has exited.
while pgrep -f download_patentsview_long_text > /dev/null; do
    sleep 300  # 5-min poll
done
echo "=== $(date) PV downloader has exited ==="

# Sanity-check: at least one g_detail_desc_text_*.tsv.zip and pg_detail_desc_text_*.tsv.zip exists
if ! ls $PV_GRANT_DIR/g_detail_desc_text_*.tsv.zip > /dev/null 2>&1; then
    echo "ERROR: no g_detail_desc_text files in $PV_GRANT_DIR"
    exit 1
fi
if ! ls $PV_PG_DIR/pg_detail_desc_text_*.tsv.zip > /dev/null 2>&1; then
    echo "ERROR: no pg_detail_desc_text files in $PV_PG_DIR"
    exit 1
fi

echo "=== $(date) STEP 1: extract clean §102 pairs (CPU, fast) ==="
python3 scripts/extract_clean_102_pairs.py 2>&1 | tee logs/clean_102_pairs.log

echo "=== $(date) STEP 2: chunk specs into paragraphs (CPU, multi-hour) ==="
python3 scripts/paragraph_chunk_specs.py 2>&1 | tee logs/chunk_specs.log

echo "=== $(date) STEP 3: embed all chunks with v2 BGE-M3 (GPU) ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/embed_spec_chunks.py 2>&1 | tee logs/embed_specs.log

echo "=== $(date) STEP 4: build FAISS IVF index ==="
python3 scripts/build_spec_faiss_index.py 2>&1 | tee logs/faiss_specs.log

echo "=== $(date) CHAIN DONE ==="
