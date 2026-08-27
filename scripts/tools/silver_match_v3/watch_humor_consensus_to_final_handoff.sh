#!/usr/bin/env bash
set -euo pipefail

# Local CPU-only watcher: wait for canonical exact consensus, relocate only
# artifact paths, publish exact bytes to sk3, and invoke the fail-closed remote
# dev gate/final freezer.  Training is intentionally outside this script.

LOCAL_ROOT=outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/truth_consensus_v1
TRAINING_TRUTH=$LOCAL_ROOT/training_truth_v1
CE_TRUTH=$LOCAL_ROOT/ce_truth_v1
RELOCATED=$LOCAL_ROOT/final_handoff_relocation_v1
REMOTE=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1
REMOTE_CONSENSUS=$REMOTE/consensus_relocated_v1
LOG=$LOCAL_ROOT/FINAL_HANDOFF_WATCH_EVENTS.log

while [[ ! -f "$TRAINING_TRUTH/MANIFEST.json" || ! -f "$CE_TRUTH/REPORT.json" ]]; do
  sleep 30
done

test ! -e "$RELOCATED"
python -u -m scripts.tools.silver_match_v3.relocate_consensus_truth_handoff \
  --manifest "$TRAINING_TRUTH/MANIFEST.json" \
  --ce-report "$CE_TRUTH/REPORT.json" \
  --source-validation outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/truth_collection_v1/validation.json \
  --output-root "$RELOCATED" \
  --published-output-root "$REMOTE_CONSENSUS" >>"$LOG" 2>&1

ssh sk3 "test ! -e '$REMOTE_CONSENSUS' && mkdir -p '$REMOTE_CONSENSUS'"
rsync -a "$RELOCATED/" sk3:"$REMOTE_CONSENSUS/"
ssh sk3 "cd '$REMOTE/repo_snapshot' && bash scripts/tools/silver_match_v3/run_humor_final_handoff_after_consensus.sh" >>"$LOG" 2>&1

