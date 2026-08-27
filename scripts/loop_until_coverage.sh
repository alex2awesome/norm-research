#!/bin/bash
# Master loop: build local parquets → measure coverage → BQ fetch residual → re-measure → ...
# Loops until coverage ≥ 90% or no progress made.
set -e
export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
mkdir -p logs

GRANTED=/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/granted_patents_claim1.parquet
PGPUB=/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/pgpub_claims1.parquet
BQ_SUPPL=/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/bigquery_supplement.parquet

# Step 1: wait for local parquets if still building.
echo "=== $(date) waiting for granted parquet ==="
until [ -f "$GRANTED" ]; do sleep 60; done
echo "  granted OK"

echo "=== $(date) waiting for pgpub parquet ==="
until [ -f "$PGPUB" ]; do sleep 60; done
echo "  pgpub OK"

sleep 30  # let writers flush

# Step 2: measure initial coverage with both parquets.
echo "=== $(date) measuring initial coverage (granted + pgpub) ==="
set +e
python3 scripts/measure_citation_coverage.py 2>&1 | tee logs/coverage_iter0.log
rc=$?
set -e

if [ $rc -eq 0 ]; then
  echo "=== coverage threshold reached on iter 0 ==="
else
  # Step 3: iterate BQ fetches until >=90%.
  for iter in 1 2 3; do
    echo "=== $(date) iter $iter: BigQuery fetch ==="
    set +e
    python3 scripts/fetch_missing_from_bigquery.py 2>&1 | tee logs/bq_fetch_iter${iter}.log
    bqrc=$?
    set -e
    if [ $bqrc -ne 0 ]; then
      echo "  BQ fetch failed (rc=$bqrc), stopping iter $iter"
      break
    fi
    echo "=== $(date) iter $iter: re-measuring ==="
    set +e
    python3 scripts/measure_citation_coverage.py 2>&1 | tee logs/coverage_iter${iter}.log
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
      echo "  threshold reached on iter $iter"
      break
    fi
  done
fi

# Step 4: regardless of coverage, rerun v2 pair extraction with whatever we have.
echo "=== $(date) v2 pair extraction with final corpora ==="
python3 scripts/extract_anticipation_training_pairs_v2.py 2>&1 | tee logs/v2_pairs.log

# Step 5: re-fine-tune BGE-M3 on v2 pairs.
echo "=== $(date) waiting for v1 fine-tune to release GPU ==="
until ! pgrep -f finetune_bge_m3_anticipation.py > /dev/null; do sleep 60; done
echo "  GPU free"
echo "=== $(date) v2 fine-tune ==="
CUDA_VISIBLE_DEVICES=7 python3 scripts/finetune_bge_m3_anticipation.py \
    --base_model BAAI/bge-m3 \
    --pairs /lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/anticipation_training_pairs_v2.jsonl.gz \
    --batch_size 32 \
    --epochs 2 \
    --max_seq_len 512 \
    --lr 2e-5 \
    --out /lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2
echo "=== $(date) v2 fine-tune done ==="

# Step 6: build retriv index over full claim DB with v2 model.
echo "=== $(date) building retriv index (v2) ==="
python3 scripts/build_retriv_claim_index.py --collection patent_claims_v2 --model /lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2

echo "=== $(date) ALL DONE ==="
