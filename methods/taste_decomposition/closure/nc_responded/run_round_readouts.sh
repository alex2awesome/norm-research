#!/bin/bash
# Every per-round readout the freeze asks for, in one command.
#   bash run_round_readouts.sh 1
set -e
R=${1:?round}
cd "$(dirname "$0")"
echo "=== curve (states 0..$R) ==="
python3 stage4_curve.py --upto "$R"
echo "=== swap pair ==="
python3 swap_readout.py --upto "$R"
echo "=== Track-B discount ==="
python3 track_b_discount.py --round "$R"
echo "=== missing mass (both tracks) ==="
python3 missing_mass.py --round "$R"
echo "=== round $R readouts complete ==="
