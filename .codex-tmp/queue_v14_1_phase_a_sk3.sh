#!/usr/bin/env bash
set -euo pipefail

root=/lfs/skampere3/0/alexspan/cr3-v14.1-roadmap
code=$root/code
outputs=$root/outputs
metrics=/lfs/skampere3/0/alexspan/cr3-v13.1/manifests/tier_b.json
mkdir -p "$root/logs" "$outputs/probe_extensions" "$outputs/ladder"
export HOME=/lfs/skampere3/0/alexspan
export V13_PATH_REWRITE_JSON='{"
/lfs/skampere2/0/alexspan/cr3-v13.1/assets/tier_b":"/lfs/skampere3/0/alexspan/cr3-v13.1/assets/tier_b","
/lfs/skampere2/0/alexspan/cr3-v12/inputs":"/lfs/skampere3/0/alexspan/cr3-v13.1/assets/tier_b_inputs"}'
# Remove formatting whitespace introduced for readability before JSON parsing.
export V13_PATH_REWRITE_JSON=${V13_PATH_REWRITE_JSON//$'\n'/}
export V14_MODEL_PATH_OVERRIDES_JSON='{
"meta-llama/Llama-3.1-8B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
"meta-llama/Llama-3.3-70B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"}'

free_checks=0
while (( free_checks < 3 )); do
  mapfile -t rows < <(nvidia-smi -i 5,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  if [[ ${#rows[@]} -ne 2 ]]; then
    echo "unable to read exactly GPUs 5 and 7" >&2
    exit 1
  fi
  free=1
  for row in "${rows[@]}"; do
    memory=${row%%,*}; utilization=${row##*,}
    memory=${memory// /}; utilization=${utilization// /}
    if (( memory >= 1000 || utilization >= 5 )); then free=0; fi
  done
  if (( free )); then free_checks=$((free_checks + 1)); else free_checks=0; fi
  if (( free_checks < 3 )); then sleep 60; fi
done

"$code/scripts/run_v14_campaign_sk3.sh" 7 \
  --phase extend-probes --out-root "$outputs/v14" \
  --metrics-manifest "$metrics" --probe-extension-root "$outputs/probe_extensions" \
  --probe-corpus-manifest "$root/probe_sources/manifest.json" --run-sha 3ee59ef

"$code/scripts/run_ceiling_ladder_sk3.sh" freeze "$outputs/ladder" \
  --metrics-manifest "$metrics" --probe-extension-root "$outputs/probe_extensions"
touch "$root/READY_FOR_LOCAL_REFERENCE"
