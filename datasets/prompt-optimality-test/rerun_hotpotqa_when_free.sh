#!/usr/bin/env bash
# Wait for the four in-flight GLM streams to end, then rerun hotpotqa on both arms in parallel,
# split across the two subscription keys (rate-limit fix).
set -uo pipefail
cd "$(dirname "$0")"
for pid in 67038 69520 87673 87674; do
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
done
source .venv/bin/activate
ZAI_KEY_FILE=$HOME/.z-ai-api-key-alexander-spangher.txt \
  python run_official_gepa.py hotpotqa --max-metric-calls 600 --val-n 100 &
PA=$!
ZAI_KEY_FILE=$HOME/.z-ai-api-key-spangher.txt \
  python run_inhouse_gepa.py hotpotqa --max-metric-calls 600 --val-n 100 &
PB=$!
wait $PA; wait $PB
echo HOTPOTQA_RERUNS_DONE
