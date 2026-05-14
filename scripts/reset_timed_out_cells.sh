#!/bin/bash
# Move R2/R3 outputs for cells with timeouts to .timeout_v1.json so the
# watchers re-fire them.
set -u
ROOT=/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy
cd "$ROOT"

# Cells with TO > 0 from scan_r2_timeouts.py
CELLS=(
  code-review_general
  code-review_hyper_specific
  code-review_specific
  creative-writing_general
  grant-funding_general
  grant-funding_specific
  humor_general
  humor_specific
  legal-outcome-prediction_general
  legal-outcome-prediction_specific
  math-stackexchange_general
  math-stackexchange_hyper_specific
  notice-and-comment_general
  notice-and-comment_specific
  patents_specific
  peer-review_general
  peer-review_specific
  press-releases_general
  press-releases_hyper_specific
  press-releases_specific
)

for cell in "${CELLS[@]}"; do
  for suffix in _r2.json _r2_expanded.json _r2_as_r3_input.json _r3.json _r3_expanded.json; do
    f="${cell}${suffix}"
    if [[ -f "$f" ]]; then
      mv "$f" "${f%.json}.timeout_v1.json"
      echo "moved $f"
    fi
  done
done

# Also clear the R2 and R3 lock dirs for these cells
LOCKS=/lfs/skampere3/0/alexspan/norm-research/outputs/r2_locks
rm -f $LOCKS/*.lock
LOCKS=/lfs/skampere3/0/alexspan/norm-research/outputs/r3_locks
rm -f $LOCKS/*.lock
echo "cleared locks"
