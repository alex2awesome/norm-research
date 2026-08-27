#!/usr/bin/env bash
# batch_value_census_sk3.sh — run the §12.3 value census on every behavior-census checkpoint present.
# CPU-ONLY (no GPU): safe to run concurrently with the GPU behavior sweep. Skips tasks whose
# …_alpha_probe_sigs.npz checkpoint does not exist yet, so it can be re-run as the sweep produces more.
#
# ENV:  TASKS="..."  GEPARESERVE=60  NPROBES=300  OUT=...
set -uo pipefail
REPO=/lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HUGGINGFACE_HUB_CACHE=/lfs/skampere3/0/shared_hf_cache/hub
export HF_HUB_OFFLINE=1
PY="${PY:-/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python}"
TASKS="${TASKS:-math news-homepages patents law notice-and-comment creative-writing humor peer-review}"
RESERVE="${GEPARESERVE:-60}"; NPROBES="${NPROBES:-300}"
AOUT="$HOME/outputs/alpha_probe"          # behavior checkpoints live here
VOUT="${OUT:-$HOME/outputs/value_census}"
BLOG="$HOME/logs/value_census_batch.log"
mkdir -p "$VOUT" "$(dirname "$BLOG")"
cd "$REPO"
echo "$(date) VALUE-CENSUS batch start" | tee -a "$BLOG"

for TASK in $TASKS; do
  CKPT="$AOUT/${TASK}_alpha_probe_sigs.npz"
  if [ ! -f "$CKPT" ]; then
    echo "$(date) skip ${TASK}: no checkpoint yet" | tee -a "$BLOG"; continue
  fi
  echo "$(date) >>> ${TASK} value census" | tee -a "$BLOG"
  "$PY" -m methods.metric_implementer.experiments.run_value_census \
        --task "$TASK" --checkpoint "$CKPT" \
        --gepa-reserve "$RESERVE" --n-probes "$NPROBES" --out-dir "$VOUT" \
        > "$HOME/logs/value_census_${TASK}.log" 2>&1
  SUMMARY=$("$PY" -c "
import json
r=json.load(open('$VOUT/${TASK}_value_census.json'))
print('alpha=%.3f alpha_V=%.3f gap=%.3f MV0=%.4f recovered=%.0f%% ver=%s' % (
    r['alpha_behavior'], r['alpha_V_terminal'], r['breadth_gap'], r['MV0'],
    r['frac_label_recovered']*100, r['decision']['verdict']))" 2>/dev/null || echo "PARSE-FAIL")
  echo "$(date) <<< ${TASK}  ${SUMMARY}" | tee -a "$BLOG"
done
echo "$(date) VALUE-CENSUS batch DONE" | tee -a "$BLOG"
