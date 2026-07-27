#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p runs/hover/momega_v2 runs/hotpotqa/momega_v2
ZAI_KEY_FILE=$HOME/.z-ai-api-key-alexander-spangher.txt python run_momega_v2.py hover \
  > runs/hover/momega_v2/run.log 2>&1 &
P1=$!
ZAI_KEY_FILE=$HOME/.z-ai-api-key-spangher.txt python run_momega_v2.py hotpotqa \
  > runs/hotpotqa/momega_v2/run.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo MOMEGA_V2_BOTH_DONE
