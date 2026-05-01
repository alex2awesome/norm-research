#!/usr/bin/env bash
# Download USPTO Office Action Research Dataset (OARD) — 2017 release.
# Provides per-application citation table including BOTH granted and abandoned apps.
# Source: https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/

set -euo pipefail

OUT_DIR="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/oard"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

BASE="https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

FILES=(
    "office_actions.csv.zip"   # 79 MB — keyed by ifw_number
    "rejections.csv.zip"        # 90 MB — per-rejection breakdown
    "citations.csv.zip"         # 450 MB — per-app cited references (PRIMARY for our use)
)

for f in "${FILES[@]}"; do
    if [[ -f "${f%.zip}" ]]; then
        echo "Already extracted: ${f%.zip}, skipping"
        continue
    fi
    if [[ ! -f "$f" ]]; then
        echo "Downloading $f..."
        curl -L -C - --fail --retry 3 --retry-delay 10 -A "$UA" -o "$f" "$BASE/$f" || {
            echo "  Primary failed; trying ODP fallback URL..."
            ALT="https://data.uspto.gov/bulkdata/datasets/ptoffact/files/$f"
            curl -L -C - --fail --retry 3 --retry-delay 10 -A "$UA" -o "$f" "$ALT" || true
        }
    fi
    if [[ -f "$f" ]]; then
        size=$(stat -c%s "$f")
        if [[ "$size" -lt 1000000 ]]; then
            echo "  WARN: $f is suspiciously small ($size bytes)"
            head -c 200 "$f"
        else
            echo "Extracting $f..."
            unzip -o "$f"
        fi
    fi
done

echo "OARD download complete. Files:"
ls -lh *.csv 2>/dev/null || true
