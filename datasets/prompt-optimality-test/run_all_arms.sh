#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python run_official_gepa.py hover    --max-metric-calls 600 --val-n 100
python run_official_gepa.py hotpotqa --max-metric-calls 600 --val-n 100
python run_official_gepa.py aime2025 --max-metric-calls 300 --val-n 17
echo ALL_ARMS_DONE
