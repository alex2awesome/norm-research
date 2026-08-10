#!/usr/bin/env bash
# Coordinate-then-fetch-then-build chain for acad-C full S2 standardization.
#
# 1. WAIT for the best_papers S2 fetch (PID 2737667) to finish so we don't
#    split the per-IP S2 rate.
# 2. Run the full acad-C S2 fetch (resumable into s2_cache_full.jsonl).
# 3. On completion, rebuild openalex_citations_v3.csv.gz and probe v2-vs-v3.
#
# Run under nohup. HOME pinned to /lfs so AFS-token loss can't bite.
set -u
export HOME=/lfs/skampere3/0/alexspan
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
DIR=/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/openalex_citations
cd "$DIR" || exit 1

BP_PID=2737667
echo "[$(date)] waiting for best_papers fetch PID $BP_PID to finish..."
while ps -p "$BP_PID" >/dev/null 2>&1; do
  sleep 300
done
echo "[$(date)] PID $BP_PID gone; starting acad-C full S2 fetch."

$PY s2_fetch_full.py --input s2_todo.csv --cache s2_cache_full.jsonl \
    --report-every 200 2>&1
echo "[$(date)] fetch finished. Building v3."

$PY build_v3_s2.py --fetch s2_cache_full.jsonl --out openalex_citations_v3.csv.gz 2>&1
echo "[$(date)] v3 build done. Probing v2-vs-v3 TF-IDF floor."

$PY probe_v3_tfidf.py 2>&1
echo "[$(date)] ALL DONE."
