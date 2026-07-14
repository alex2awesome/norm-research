#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "$script_dir/.." && pwd)

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHASE OUT_ROOT [PHYSICAL_GPU] [ceiling_ladder arguments...]" >&2
  exit 2
fi

phase="$1"
out_root="$2"
shift 2

case "$phase" in
  freeze|native-audit|aggregate)
    physical_gpu=""
    ;;
  constructor|executor)
    if [[ $# -lt 1 ]]; then
      echo "$phase requires exactly one physical sk3 GPU" >&2
      exit 2
    fi
    physical_gpu="$1"
    shift
    case "$physical_gpu" in
      0|5|6|7) ;;
      1|2|3|4)
        echo "HARD STOP: sk3 physical GPU $physical_gpu is permanently forbidden" >&2
        exit 64
        ;;
      *)
        echo "invalid sk3 physical GPU $physical_gpu; permitted values are 0,5,6,7" >&2
        exit 64
        ;;
    esac
    exec 211>"/tmp/cr3-ceiling-ladder-sk3-gpu-${physical_gpu}.lock"
    if ! flock -n 211; then
      echo "physical GPU $physical_gpu already has a ceiling-ladder lock" >&2
      exit 75
    fi
    ;;
  reference)
    echo "reference is intentionally local/API-only; do not run it through the sk3 launcher" >&2
    exit 64
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export METRIC_IMPLEMENTER_LFS_HOME=/lfs/skampere3/0/alexspan
export V13_PATH_REWRITE_JSON=${V13_PATH_REWRITE_JSON:-'{"/lfs/skampere2/0/alexspan/cr3-v13.1/assets/tier_b":"/lfs/skampere3/0/alexspan/cr3-v13.1/assets/tier_b","/lfs/skampere2/0/alexspan/cr3-v12/inputs":"/lfs/skampere3/0/alexspan/cr3-v13.1/assets/tier_b_inputs"}'}
export CEILING_MODEL_PATH_OVERRIDES_JSON=${CEILING_MODEL_PATH_OVERRIDES_JSON:-'{"meta-llama/Llama-3.3-70B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b","meta-llama/Llama-3.1-8B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659","mistralai/Mistral-Small-24B-Instruct-2501":"/lfs/skampere3/0/shared_hf_cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724","Qwen/Qwen2.5-32B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"}'}
CEILING_PYTHON=${CEILING_PYTHON:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "$physical_gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$physical_gpu"
  export CEILING_PHYSICAL_GPU="$physical_gpu"
else
  export CUDA_VISIBLE_DEVICES=""
fi

cd "$code_root"
exec "$CEILING_PYTHON" -m methods.metric_implementer.experiments.ceiling_ladder \
  --phase "$phase" --out-root "$out_root" "$@"
