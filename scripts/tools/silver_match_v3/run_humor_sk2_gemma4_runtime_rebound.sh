#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/lfs/skampere2/0/alexspan/norm-research-silver-v3
readonly PYTHON=/lfs/skampere2/0/alexspan/envs/gemma4-sk3-mirror-20260713/bin/python
readonly QUEUE="$REPO/outputs/silver_match_v3/humor/verifier_select_expansion_v3/HUMOR_FRESH_SELECT_GPU7_QUEUE_SK3_GEMMA4_RUNTIME.json"
readonly CACHE=/lfs/skampere2/0/alexspan/cache/gemma4-runtime-20260713

test "$(hostname -f)" = skampere2.stanford.edu
test "$(sha256sum "$PYTHON" | awk '{print $1}')" = 683f3274c99ec7b25746d0aeef2abce1724ec672d83518f9d359ca315e0fb27d
test "$(sha256sum "$QUEUE" | awk '{print $1}')" = 00291efdaab6d7e809f654376c84ff0f86ba1ab90472c59ad1150e8e5c4e5b12

mkdir -p \
  "$CACHE/xdg" \
  "$CACHE/torch_extensions" \
  "$CACHE/triton" \
  "$CACHE/flashinfer_workspace_base" \
  "$CACHE/flashinfer_jit" \
  "$CACHE/cuda" \
  "$CACHE/vllm" \
  "$CACHE/torchinductor" \
  /lfs/skampere2/0/alexspan/tmp

export HOME=/lfs/skampere2/0/alexspan
export XDG_CACHE_HOME="$CACHE/xdg"
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions"
export TRITON_CACHE_DIR="$CACHE/triton"
export FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer_workspace_base"
export FLASHINFER_JIT_DIR="$CACHE/flashinfer_jit"
export CUDA_CACHE_PATH="$CACHE/cuda"
export VLLM_CACHE_ROOT="$CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR=/lfs/skampere2/0/alexspan/tmp

cd "$REPO"
exec "$PYTHON" -u -m scripts.tools.silver_match_v3.run_humor_fresh_select_gpu_queue \
  --queue "$QUEUE" \
  --run \
  --poll-seconds 10
