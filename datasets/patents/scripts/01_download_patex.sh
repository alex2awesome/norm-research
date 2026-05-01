#!/usr/bin/env bash
# Download USPTO Patent Examination Research Dataset (PatEx) 2022 release.
# Provides event log (transactions) and per-application metadata.
# Used to derive "first-draft approved" labels from examination history.

set -euo pipefail

OUT_DIR="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patex"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

BASE_URL="https://data.uspto.gov/ui/datasets/products/files/ECOPAIR/2022"

FILES=(
    "transactions.csv.zip"       # event log (2.02 GiB) — primary source for labels
    "application_data.csv.zip"   # per-app metadata (887 MiB)
    "event_codes.csv.zip"        # code lookup (25 KB)
)

for f in "${FILES[@]}"; do
    if [[ -f "${f%.zip}" ]]; then
        echo "Already extracted: ${f%.zip}, skipping"
        continue
    fi
    if [[ ! -f "$f" ]]; then
        echo "Downloading $f..."
        # data.uspto.gov returns the SPA HTML shell unless we send a browser UA
        UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        curl -L -C - --fail --retry 3 --retry-delay 10 \
            -A "$UA" \
            -H "Accept: application/octet-stream" \
            -o "$f" "$BASE_URL/$f"
    fi
    echo "Extracting $f..."
    unzip -o "$f"
done

echo "PatEx download complete. Files:"
ls -lh *.csv 2>/dev/null || true
