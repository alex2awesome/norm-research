#!/bin/bash
# Scaled E7 launch on sk3 (vLLM offline batch, 1 GPU). No OpenRouter.
# Usage (on sk3, inside the repo):
#   bash methods/metric_implementer/run_scale_sk3.sh search   # mint prompt iterations
#   bash methods/metric_implementer/run_scale_sk3.sh measure  # mega-batch score -> long table
#   bash methods/metric_implementer/run_scale_sk3.sh all
#
# Run under nohup so it survives disconnect; HOME is pinned to /lfs so AFS-token loss in
# nohup jobs does not break ~/.cache ([[feedback_sk3_afs_tokens]]). One GPU only
# ([[feedback_gpu_usage]]) — the orchestrator keeps ONE model resident at a time by design.
set -u
MODE="${1:-all}"
cd "$(dirname "$0")/../.."                       # repo root

# --- sk3 env (matches the repo's 137 canonical vLLM recipes) -----------------------------
export HOME=/lfs/skampere3/0/alexspan            # BEFORE any torch/vllm import
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/lfs/skampere3/0/shared_hf_cache  # shared model snapshots
# Pick ONE free GPU (verify with nvidia-smi first; set explicitly, do not auto-grab all):
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP=$(date +%Y%m%d_%H%M)
OUT=/lfs/skampere3/0/alexspan/norm-research/outputs/metric_implementer_scale
LOG=$OUT/scale_${MODE}_${STAMP}.log
mkdir -p "$OUT"

echo "[scale] mode=$MODE gpu=$CUDA_VISIBLE_DEVICES out=$OUT  (log: $LOG)"
# --out-root keeps registries + long table on /lfs (not the AFS-backed laptop checkout).
nohup python -m methods.metric_implementer.scale "$MODE" \
    --manifest pilot \
    --out-root "$OUT" \
    >>"$LOG" 2>&1 &
echo "launched pid $! ; tail -f $LOG"

# NOTES for the real run (edit before launching):
#  - Swap --manifest pilot -> a full-manifest entry once metric/dataset caps are tuned
#    (manifest.full_manifest(metrics_per_task=40, ...)); wire a 'full' name into scale._manifest.
#  - Default tiers (manifest.DEFAULT_TIERS): Llama-3.1-8B, Llama-3.3-70B, Qwen2.5-72B — all
#    must be in $HF_HOME. gemma-3-4b and llama-1b are intentionally EXCLUDED (degenerate in the
#    06-12 pilot). Strong/reviser model = Qwen2.5-72B by default.
#  - Models load ONE at a time; a 70B BF16 needs ~140GB (fits a B200). For a 122B MoE use the
#    FP8 path (dtype='auto', kv_cache_dtype='auto'; do NOT set kv_cache_dtype=fp8 -> '!!!!').
#  - passes=3, n_items=60 per the pilot's protocol fix; raise for tighter CIs.
