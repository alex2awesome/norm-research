#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context/adjudicator_k50/press-releases/gepa_clean_v2
BASE="$ROOT/optimize_gepa_v1/verifier_third_design_v1"
REPO=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
MODEL=/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb
OUT="$BASE/gemma_author_candidates_v1"
LOCK=/tmp/alexspan_sk3_four_gpu_launch.lock

if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite $OUT" >&2
  exit 2
fi

while true; do
  if mkdir "$LOCK" 2>/dev/null; then
    trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
    mapfile -t USED < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    declare -A PROJECT_UUIDS=()
    while IFS=',' read -r UUID PID MEMORY; do
      UUID="${UUID// /}"
      PID="${PID// /}"
      MEMORY="${MEMORY// /}"
      OWNER="$(ps -o user= -p "$PID" 2>/dev/null | tr -d ' ' || true)"
      if [[ "$OWNER" == "alexspan" ]] && (( MEMORY >= 1024 )); then
        PROJECT_UUIDS["$UUID"]=1
      fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
    ACTIVE="${#PROJECT_UUIDS[@]}"
    GPU=""
    for INDEX in "${!USED[@]}"; do
      if (( USED[INDEX] >= 1024 )); then
        ACTIVE=$((ACTIVE + 1))
      elif [[ -z "$GPU" ]] && (( USED[INDEX] < 100 )); then
        GPU="$INDEX"
      fi
    done
    if (( ACTIVE < 4 )) && [[ -n "$GPU" ]]; then
      echo "$(date -Is) launching PR Gemma author on physical GPU $GPU; active lanes before launch=$ACTIVE"
      cd "$REPO"
      export HOME=/lfs/skampere3/0/alexspan
      export PYTHONPATH=.
      export CUDA_VISIBLE_DEVICES="$GPU"
      "$PY" -u -m scripts.tools.silver_match_v3.author_pr_verifier_gepa_candidates_gemma \
        --training-report "$BASE/author_packet_v1/REPORT.json" \
        --training-examples "$BASE/author_packet_v1/examples.jsonl" \
        --aggregate-taxonomy "$BASE/input_design_v2/aggregate_error_taxonomy.json" \
        --base-prompt "$ROOT/verifier_dev_v2/verifier_author/accepted_v4/prompt.frozen.txt" \
        --model "$MODEL" \
        --output-root "$OUT" \
        --seed 2026071333 \
        --examples-per-class 12 \
        --max-model-len 32768 \
        --max-tokens 3000 \
        --gpu-memory-utilization 0.86
      exit $?
    fi
    rmdir "$LOCK"
    trap - EXIT
  fi
  echo "$(date -Is) waiting: four sk3 GPU lanes remain active or no untouched GPU is free"
  sleep 15
done
