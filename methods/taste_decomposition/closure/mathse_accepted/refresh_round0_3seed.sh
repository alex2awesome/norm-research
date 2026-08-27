#!/bin/bash
# Re-run every round-0 readout that reads the DENSE side, once seeds 1 and 2 land, so
# the curve's r = 0 anchor and the whole position audit are 3-seed rather than seed-42.
#
# The bank side (census, ablations' VA rows, splits, alignment gate) is seed-independent
# and is NOT recomputed.
#
# Usage (from the campaign dir):  bash refresh_round0_3seed.sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}

python3 fetch_dense.py
python3 oof_alignment_gate.py > /dev/null      # must still PASS (bank side unchanged)
python3 round0.py            > round0_3seed.log 2>&1
python3 position_line.py     > position_line_3seed.log 2>&1
python3 position_matched.py  > position_matched_3seed.log 2>&1
python3 length_stratification.py > length_stratification_3seed.log 2>&1
echo "ROUND0_3SEED_REFRESH_DONE"
python3 - <<'PY'
import json
d = json.load(open("mathse_accepted_r0_context.json"))
print(json.dumps({"T": {k: v for k, v in d["T"].items() if isinstance(v, dict)},
                  "delta_tier1": d["round0_delta_TIER1_pooled"],
                  "delta_tier2": d["round0_delta_TIER2_within_question"]},
                 indent=1, default=float))
PY
