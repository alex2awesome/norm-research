#!/usr/bin/env bash
set -euo pipefail

# Keep both resume-safe N&C verifier orders moving through a single GPU slot.
# The per-order runner owns stable-availability checks and the live quota
# guard.  This supervisor retries only quota-preempted runs, never launches a
# second copy of an active order, and invokes the existing hash-pinned
# downstream chain only after both completion metadata files exist.

REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
VERIFY_DIR="$DATA/production_v1/verifier"
SLOT="$REPO/scripts/tools/silver_match_v3/run_notice_quota_slot.sh"

cd "$REPO"

for order in hashed original; do
  meta="$VERIFY_DIR/notice-and-comment.primary.verify.$order.jsonl.meta.json"
  while [[ ! -f $meta ]]; do
    if pgrep -f "[r]un_notice_quota_slot.sh $order" >/dev/null; then
      echo "$(date -Is) active_order=$order waiting"
      sleep 10
      continue
    fi

    echo "$(date -Is) starting_order=$order"
    set +e
    "$SLOT" "$order"
    status=$?
    set -e
    if (( status != 0 && status != 42 )); then
      echo "$(date -Is) order=$order unexpected_status=$status; retrying after cooldown" >&2
      sleep 30
    fi
  done
  echo "$(date -Is) completed_order=$order"
done

# Both verifier orders are sealed, so the existing pipeline skips GPU work and
# performs only its pinned combine/finalize steps.
NOTICE_VERIFY_ORIGINAL_GPU=0 NOTICE_VERIFY_HASHED_GPU=0 \
  bash scripts/tools/silver_match_v3/run_notice_production_chain_parallel.sh
