#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/lfs/skampere2/0/alexspan/norm-research-silver-v3
readonly PYTHON=/lfs/skampere2/0/alexspan/miniconda3/bin/python3.11
readonly QUEUE="$REPO/outputs/silver_match_v3/humor/verifier_select_expansion_v3/HUMOR_FRESH_SELECT_GPU7_QUEUE.json"
readonly OVERLAY=/lfs/skampere2/0/alexspan/env-overlays/transformers5-for-vllm016
readonly CACHE=/lfs/skampere2/0/alexspan/cache/gemma3-vllm016-tf5-20260713

test "$(hostname -f)" = skampere2.stanford.edu
test "$(sha256sum "$PYTHON" | awk '{print $1}')" = 72f5d6f0b451f1d8891607f51de620b024a026adb0caaded916a4f43eab7f6e3
test "$(sha256sum "$QUEUE" | awk '{print $1}')" = 6063149a982abcab61d249a414975ef829d7373e487c3b9681ba77990bd0487f
test "$(sha256sum "$REPO/scripts/tools/silver_match_v3/run_humor_fresh_select_gpu_queue.py" | awk '{print $1}')" = b06f7b5f42085e0d6a1418cf500c6019ebb3a026b82bae3d48f51d3029f84f8b
test "$(cd "$OVERLAY" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')" = e61123315dd7925e4f7c7dd50687b155d70bb0fae881472231d5b25154917626

mkdir -p \
  "$CACHE/xdg" \
  "$CACHE/torch_extensions" \
  "$CACHE/triton" \
  "$CACHE/flashinfer_workspace_base" \
  "$CACHE/cuda" \
  "$CACHE/vllm" \
  "$CACHE/torchinductor" \
  /lfs/skampere2/0/alexspan/tmp

export PYTHONPATH="$OVERLAY"
export HOME=/lfs/skampere2/0/alexspan
export XDG_CACHE_HOME="$CACHE/xdg"
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions"
export TRITON_CACHE_DIR="$CACHE/triton"
export FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer_workspace_base"
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
