#!/usr/bin/env bash
set -euo pipefail

root=${V14_SK2_ROOT:-/lfs/skampere2/0/alexspan/cr3-v14.1-two-lane}
code=$root/code
out=$root/outputs/fast
metrics=$root/manifests/tier_b.json
sources=$root/probe_sources
extension_gpu=${V14_EXTENSION_GPU:-6}
qwen_gpu=${V14_QWEN_GPU:-1}

mkdir -p "$out" "$root/logs"
export HOME=/lfs/skampere2/0/alexspan
export HF_HOME=$HOME/.cache/huggingface
export V14_PYTHON=$HOME/miniconda3/bin/python
export PYTHONPATH=$code
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

wait_free() {
  local device=$1 checks=0 memory util
  while (( checks < 3 )); do
    IFS=, read -r memory util < <(
      nvidia-smi -i "$device" \
        --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    )
    memory=${memory// /}
    util=${util// /}
    if (( memory < 1000 && util < 5 )); then
      checks=$((checks + 1))
    else
      checks=0
    fi
    (( checks == 3 )) || sleep 60
  done
}

llama8=/lfs/skampere2/0/shared_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
qwen14=$HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
[[ -d "$llama8" && -d "$qwen14" ]]
export V14_MODEL_PATH_OVERRIDES_JSON
V14_MODEL_PATH_OVERRIDES_JSON=$(printf \
  '{"meta-llama/Llama-3.1-8B-Instruct":"%s","Qwen/Qwen2.5-14B-Instruct":"%s"}' \
  "$llama8" "$qwen14")

if [[ ! -f "$out/probe_extensions/manifest.json" ]]; then
  wait_free "$extension_gpu"
  "$code/scripts/run_v14_campaign_sk3.sh" "$extension_gpu" \
    --phase extend-probes --out-root "$out" \
    --metrics-manifest "$metrics" \
    --probe-extension-root "$out/probe_extensions" \
    --probe-corpus-manifest "$sources/manifest.json" \
    --run-sha v14.1-two-lane-20260714
fi

CUDA_VISIBLE_DEVICES= "$V14_PYTHON" -m \
  methods.metric_implementer.experiments.run_v14_value_campaign \
  --phase design --scoring-lane fast --out-root "$out" \
  --metrics-manifest "$metrics" \
  --probe-extension-root "$out/probe_extensions" \
  --run-sha v14.1-two-lane-20260714

if [[ ! -f "$out/template_freeze.json" ]]; then
  CUDA_VISIBLE_DEVICES= "$V14_PYTHON" -m \
    methods.metric_implementer.experiments.run_v14_value_campaign \
    --phase seed-freeze --out-root "$out" \
    --template-freeze "$out/template_freeze.json"
fi

wait_free "$qwen_gpu"
"$code/scripts/run_v14_campaign_sk3.sh" "$qwen_gpu" \
  --phase constructor --out-root "$out" \
  --template-freeze "$out/template_freeze.json" \
  --decoder-family qwen --channels mcq behavioral
touch "$root/WAITING_FOR_MODEL_STAGING"
