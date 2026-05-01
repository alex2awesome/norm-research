#!/usr/bin/env bash
# Sequentially extract STaR-Local features for notice-and-comment overall +
# 5 target agencies (CDC → USCIS → NOAA → FWS → EPA → NC overall).
# Order: smallest first so results accrue fast; biggest (EPA, NC-overall) last.

set -u

export HOME=/lfs/skampere3/0/alexspan
export OPENAI_API_KEY=$(cat /lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt)
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HF_MODULES_CACHE=/lfs/skampere3/0/alexspan/hf_modules
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=7

ROOT=/lfs/skampere3/0/alexspan/norm-research
cd $ROOT
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python

run() {
  local tag=$1
  local agency_arg=$2
  local out=outputs/local_explanations/${tag}_star_local_llama33
  local log=${out}.extract.log
  mkdir -p $out
  echo "=== Extract $tag → $out @ $(date) ===" >> $log
  if [ -n "$agency_arg" ]; then
    $PY -u scripts/extract_only.py \
      --dataset notice-and-comment --agency $agency_arg \
      --output-dir $out >> $log 2>&1
  else
    $PY -u scripts/extract_only.py \
      --dataset notice-and-comment \
      --output-dir $out >> $log 2>&1
  fi
  echo "=== Extract $tag DONE @ $(date) ===" >> $log
}

# Smallest first
run nc_cdc   cdc
run nc_uscis uscis
run nc_noaa  noaa
run nc_fws   fws
run nc_epa   epa
run nc_overall ""    # overall goes last (141K train)

echo "ALL EXTRACTIONS COMPLETE @ $(date)"
