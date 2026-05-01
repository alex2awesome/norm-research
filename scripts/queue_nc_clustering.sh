#!/usr/bin/env bash
# Run full clustering pipeline (Steps 1-6) for all NC tasks with cached extractions.
# Uses gpt-5-mini for labeling + BGE-large (~3 GB GPU). No Llama needed.
# Order: smallest first.

set -u

export HOME=/lfs/skampere3/0/alexspan
export OPENAI_API_KEY=$(cat /lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt)
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HF_MODULES_CACHE=/lfs/skampere3/0/alexspan/hf_modules
export CUDA_VISIBLE_DEVICES=7

ROOT=/lfs/skampere3/0/alexspan/norm-research
cd $ROOT
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python

run_task() {
  local tag=$1
  local agency_arg=$2
  local out=outputs/local_explanations/${tag}_star_local_llama33
  local log=${out}.clustering.log

  if [ ! -f "$out/explanation_cache/star_local_v1.jsonl" ]; then
    echo "SKIP $tag: no extraction cache" | tee -a $log
    return
  fi

  echo "=== Clustering $tag @ $(date) ===" >> $log

  local agency_flag=""
  if [ -n "$agency_arg" ]; then
    agency_flag="--agency $agency_arg"
  fi

  $PY -u scripts/run_full_clustering_pipeline.py \
    --dataset notice-and-comment $agency_flag \
    --output-dir $out \
    --proposer-model openai/gpt-5-mini \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --similarity-model-base BAAI/bge-large-en-v1.5 \
    --max-similarity-pairs 10000 \
    --n-canonical-features 30 \
    >> $log 2>&1

  echo "=== Clustering $tag DONE @ $(date) ===" >> $log
}

# Smallest first
run_task nc_cdc   cdc
run_task nc_uscis uscis
# NOAA skipped (no extraction yet)
run_task nc_fws   fws
run_task nc_epa   epa
run_task nc_overall ""

echo "ALL CLUSTERING COMPLETE @ $(date)"
