#!/usr/bin/env bash
# Download PatentsView spec text bulks (granted + pre-grant pub), 2001-2025.
# Sizes are estimates; pgpub spec files are typically 4-12 GB each per year.

set -euo pipefail

PG_DIR="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_pg"
G_DIR="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant"
PG_BASE="https://s3.amazonaws.com/data.patentsview.org/pregrant_publications"
G_BASE="https://s3.amazonaws.com/data.patentsview.org/download"

download() {
    local url="$1"; local outdir="$2"; local fname=$(basename "$url")
    cd "$outdir"
    if [[ -f "$fname" ]]; then
        echo "  have $fname"; return 0
    fi
    echo "  fetching $fname"
    wget -c --no-verbose --tries=3 "$url" || echo "  WARN: failed $fname"
}

mkdir -p "$PG_DIR" "$G_DIR"

# ---- Granted: brief summary + detailed description (year-split) ----
for year in $(seq 2001 2025); do
    download "$G_BASE/g_brf_sum_text_${year}.tsv.zip" "$G_DIR" || true
    download "$G_BASE/g_detail_desc_text_${year}.tsv.zip" "$G_DIR" || true
done

# ---- Pre-grant pub: brief summary + detailed description (year-split) ----
for year in $(seq 2001 2025); do
    download "$PG_BASE/pg_brf_sum_text_${year}.tsv.zip" "$PG_DIR" || true
    download "$PG_BASE/pg_detail_desc_text_${year}.tsv.zip" "$PG_DIR" || true
done

echo "done. sizes:"
du -sh "$G_DIR"/g_*detail_desc* "$G_DIR"/g_*brf_sum* 2>/dev/null | head
du -sh "$PG_DIR"/pg_*detail_desc* "$PG_DIR"/pg_*brf_sum* 2>/dev/null | head
