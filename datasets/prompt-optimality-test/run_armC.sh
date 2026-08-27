#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export ZAI_KEY_FILE=$HOME/.z-ai-api-key-spangher.txt
python run_unit_recombination.py hover    --max-metric-calls 600 --val-n 100
python run_unit_recombination.py hotpotqa --max-metric-calls 600 --val-n 100
python run_unit_recombination.py aime2025 --max-metric-calls 300 --screen-n 5 --panel-n 8 --val-n 17
deactivate 2>/dev/null || true
# final pass: rescore picks up the new arm (resumable), analyze rebuilds the summary
source .venv/bin/activate && python phase4_rescore.py; deactivate
python3 analyze.py
echo ARMC_AND_FINAL_ANALYSIS_DONE
