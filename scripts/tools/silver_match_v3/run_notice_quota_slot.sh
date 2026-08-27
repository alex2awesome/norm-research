#!/usr/bin/env bash
set -euo pipefail

# Resume one N&C verifier order only after the server-wide occupied-GPU count
# has remained below the declared cap for several consecutive polls.  The
# quota guard terminates this job if another workload subsequently raises the
# occupied count above the cap.

order=${1:-hashed}
max_occupied=${MAX_OCCUPIED_GPUS:-4}
stable_polls=${STABLE_POLLS:-6}
poll_seconds=${POLL_SECONDS:-5}

if [[ $order != original && $order != hashed ]]; then
  echo "order must be original or hashed" >&2
  exit 2
fi

REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
MODEL=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
PROD="$DATA/production_v1"
CANDIDATES="$PROD/candidates/notice-and-comment.all-corpora.primary.nemotron_adapter.jsonl"
PRIMARY="$PROD/adjudicator/notice-and-comment.primary.original.jsonl"
OUTPUT="$PROD/verifier/notice-and-comment.primary.verify.$order.jsonl"
LOG="$PROD/verifier/notice-and-comment.primary.verify.$order.log"
GPY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
PINNED_VERIFY_SHA=797e6ade28dba5c3493e28c6fb5c0123d9877e6354649e595f9690860b3afc7e

export HOME=/lfs/skampere3/0/alexspan
export XDG_CACHE_HOME=/lfs/skampere3/0/alexspan/.cache
export TORCHINDUCTOR_CACHE_DIR=/lfs/skampere3/0/alexspan/.cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/lfs/skampere3/0/alexspan
export VLLM_NO_USAGE_STATS=1
export PYTHONPATH=.
cd "$REPO"

if [[ -f "$OUTPUT.meta.json" ]]; then
  echo "$(date -Is) already_complete=$OUTPUT"
  exit 0
fi

actual_sha=$(sha256sum scripts/tools/silver_match_v3/verify_gemma.py | cut -d' ' -f1)
if [[ $actual_sha != "$PINNED_VERIFY_SHA" ]]; then
  echo "verify_gemma.py hash mismatch: $actual_sha" >&2
  exit 3
fi

candidate_gpu=
stable=0
while (( stable < stable_polls )); do
  mapfile -t memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  occupied=0
  for used in "${memory[@]}"; do
    (( used > 100 )) && (( occupied += 1 ))
  done

  current_candidate=
  if (( occupied < max_occupied )); then
    for gpu in "${!memory[@]}"; do
      if (( memory[gpu] <= 100 )); then
        current_candidate=$gpu
        break
      fi
    done
  fi

  if [[ -n $current_candidate && $current_candidate == "$candidate_gpu" ]]; then
    (( stable += 1 ))
  elif [[ -n $current_candidate ]]; then
    candidate_gpu=$current_candidate
    stable=1
  else
    candidate_gpu=
    stable=0
  fi
  echo "$(date -Is) occupied=$occupied cap=$max_occupied candidate=${candidate_gpu:-none} stable=$stable/$stable_polls"
  (( stable >= stable_polls )) || sleep "$poll_seconds"
done

# Recheck immediately before launch.  The guard below owns enforcement after
# launch and fail-closes if another process races us for a GPU.
occupied=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 100 { n++ } END { print n + 0 }')
selected_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((candidate_gpu + 1))p")
if (( occupied >= max_occupied || selected_used > 100 )); then
  echo "$(date -Is) launch_recheck_failed occupied=$occupied gpu=$candidate_gpu used=$selected_used"
  exec "$0" "$order"
fi

echo "$(date -Is) launching_order=$order gpu=$candidate_gpu occupied_before=$occupied"
CUDA_VISIBLE_DEVICES="$candidate_gpu" "$GPY" -u -m scripts.tools.silver_match_v3.verify_gemma \
  --manifest "$DATA/manifest.json" \
  --candidates "$CANDIDATES" \
  --primary "$PRIMARY" \
  --output "$OUTPUT" \
  --prompt scripts/tools/silver_match_v3/prompts/verify_match_v1.txt \
  --prompt-addon scripts/tools/silver_match_v3/prompts/verify_notice_shepherded_v2.txt \
  --order-mode "$order" \
  --max-alternatives 49 \
  --context-chars 1200 \
  --description-chars 260 \
  --example-chars 180 \
  --max-examples 0 \
  --batch-size 128 \
  --gpu-memory-utilization .88 \
  --max-model-len 8192 \
  --max-tokens 180 \
  --seed 29 \
  --resume >>"$LOG" 2>&1 &
target_pid=$!

set +e
scripts/tools/silver_match_v3/guard_gpu_quota.sh "$target_pid" "$max_occupied" "$poll_seconds"
guard_status=$?
wait "$target_pid"
job_status=$?
set -e

echo "$(date -Is) order=$order gpu=$candidate_gpu guard_status=$guard_status job_status=$job_status"
if (( guard_status == 42 )); then
  exit 42
fi
if (( job_status != 0 )); then
  exit "$job_status"
fi
test -f "$OUTPUT.meta.json"
