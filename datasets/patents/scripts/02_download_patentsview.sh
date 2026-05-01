#!/usr/bin/env bash
# Download PatentsView pre-grant publication and granted patent tables from S3.
# Pre-grant = "draft" (as-filed text); granted = "final" (post-amendment text).
#
# We download claims (key text), abstracts, basic metadata, and the
# pg_granted_pgpubs_crosswalk that maps pre-grant pgpub_id -> patent_id.

set -euo pipefail

PG_DIR="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_pg"
G_DIR="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant"
mkdir -p "$PG_DIR" "$G_DIR"

PG_BASE="https://s3.amazonaws.com/data.patentsview.org/pregrant_publications"
G_BASE="https://s3.amazonaws.com/data.patentsview.org/download"
G_CLAIMS_BASE="https://s3.amazonaws.com/data.patentsview.org/claims"

download() {
    local url="$1"
    local outdir="$2"
    local fname=$(basename "$url")
    cd "$outdir"
    if [[ -f "$fname" ]]; then
        echo "Already have $fname, skipping"
        return 0
    fi
    echo "Downloading $fname..."
    wget -c --no-verbose --tries=3 "$url" || {
        echo "  WARN: failed to download $fname"
        return 1
    }
}

# ---- Pre-grant: metadata + crosswalk ----
PG_FLAT=(
    "pg_published_application.tsv.zip"            # 284 MiB - app metadata
    "pg_published_application_abstract.tsv.zip"   # 1.47 GiB - abstracts
    "pg_granted_pgpubs_crosswalk.tsv.zip"         # 106 MiB - pgpub <-> patent map
    "pg_cpc_current.tsv.zip"                      # 946 MiB - classification
)
for f in "${PG_FLAT[@]}"; do
    download "$PG_BASE/$f" "$PG_DIR" || true
done

# ---- Pre-grant: claims, year-split (2001-2025) ----
for year in $(seq 2001 2025); do
    download "$PG_BASE/pg_claims_${year}.tsv.zip" "$PG_DIR" || true
done

# ---- Granted: metadata + citations ----
G_FLAT=(
    "g_application.tsv.zip"           # 67 MiB
    "g_patent.tsv.zip"                # 219 MiB
    "g_patent_abstract.tsv.zip"       # abstracts
    "g_us_patent_citation.tsv.zip"    # 2.08 GiB - includes IDS + examiner citations
    "g_foreign_citation.tsv.zip"      # 695 MiB
    "g_cpc_current.tsv.zip"           # 473 MiB
)
for f in "${G_FLAT[@]}"; do
    download "$G_BASE/$f" "$G_DIR" || true
done

# ---- Granted: claims, year-split (2001-2025 to match PG coverage) ----
for year in $(seq 2001 2025); do
    download "$G_CLAIMS_BASE/g_claims_${year}.tsv.zip" "$G_DIR" || true
done

echo "PatentsView download complete. Sizes:"
du -sh "$PG_DIR" "$G_DIR"
