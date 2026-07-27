#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
need() { [ -f "runs/$1/official/result.json" ] && [ -f "runs/$1/inhouse/result.json" ]; }
until need hover && need hotpotqa && need aime2025; do sleep 300; done
echo "ARMS COMPLETE — starting rescore"
source .venv/bin/activate
python phase4_rescore.py
deactivate
python3 analyze.py
echo PHASE34_DONE
