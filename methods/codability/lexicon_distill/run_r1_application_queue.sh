#!/usr/bin/env bash
# Score and build immutable R1 candidates after the certified pooled adapter lands on sk2.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU TASK [TASK ...]" >&2
  exit 2
fi

gpu=$1
shift
repo=/lfs/skampere2/0/alexspan/norm-research
python=/lfs/skampere2/0/alexspan/envs/gemma4-similarity-lora-v1/bin/python
model=/lfs/skampere2/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/518276fb130dc81caf9a4f772e65e63ef2526493
run=$repo/outputs/lexicon/similarity_lora_v1
data=$repo/outputs/lexicon/similarity_distill_v1
adapter=$run/adapters/pooled_R1_primary
calibration=$run/reports/calibrate_R1_primary.json

cd "$repo"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/lfs/skampere2/0/alexspan/.cache/huggingface
export XDG_CACHE_HOME=/lfs/skampere2/0/alexspan/.cache
export TMPDIR=/lfs/skampere2/0/alexspan/tmp

while [[ ! -f "$adapter/adapter_model.safetensors" ]]; do sleep 60; done

for task in "$@"; do
  cell=$run/hierarchy/$task/R1
  inputs=$cell/pairs.jsonl
  outputs=$cell/scores.jsonl
  scoring=$cell/scoring.json
  partition=$cell/partition.gemma-r1-primary.candidate.json
  graph=$cell/graph.gemma-r1-primary.json
  [[ -f "$cell/nodes.json" && -f "$inputs" ]] || {
    echo "missing frozen inputs for $task" >&2
    exit 1
  }

  if [[ ! -f "$outputs" || ! -f "$scoring" ]]; then
    [[ ! -e "$outputs" && ! -e "$scoring" ]] || {
      echo "partial score artifact for $task; refusing overwrite" >&2
      exit 1
    }
    while true; do
      read -r memory utilization < <(
        nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits | tr -d ','
      )
      [[ "$memory" -le 2048 && "$utilization" -eq 0 ]] && break
      sleep 60
    done
    echo "[$(date -Is)] scoring $task/R1 on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -m \
      methods.codability.lexicon_distill.score_hierarchy_pairs \
      --inputs "$inputs" --outputs "$outputs" --report "$scoring" \
      --protocols "$data/protocols.json" --model "$model" --adapter "$adapter" \
      --batch-size 16 --max-length 1024
  fi

  while [[ ! -f "$calibration" ]]; do sleep 60; done
  "$python" - "$calibration" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("certified") is not True:
    raise SystemExit("R1 threshold calibration is not certified")
PY
  if [[ ! -f "$partition" || ! -f "$graph" ]]; then
    [[ ! -e "$partition" && ! -e "$graph" ]] || {
      echo "partial graph artifact for $task; refusing overwrite" >&2
      exit 1
    }
    "$python" -m methods.codability.lexicon_distill.build_hierarchy_candidate \
      build-partition --inventory "$cell/nodes.json" --pair-inputs "$inputs" \
      --pair-outputs "$outputs" --calibration "$calibration" \
      --partition "$partition" --report "$graph"
  fi
  echo "[$(date -Is)] completed $task/R1"
done
