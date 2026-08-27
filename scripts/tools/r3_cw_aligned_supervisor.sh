#!/usr/bin/env bash
# Aligned CW Q1 launch (2026-07-01). Same-family primary Llama ladder (3B/8B/70B-FP8) + Qwen
# replication, scored with the §12.6 m_bar_omega orbit-averaged target (--orbit-target 4).
# 4 GPUs in parallel (1/2/3/7), explicit user direction. NO GLM: rescore_executor only re-scores
# the frozen GLM-proposed Omega (from llama8b_glm); 8B/3B reuse existing sigs (--retarget-mi-only,
# ~200x cheaper), 70B/Qwen full-rescore to complete the ladder. Covers the 39 metrics that have Omega.
set -o pipefail
export HOME=/lfs/skampere3/0/alexspan
export PYTHONPATH=/lfs/skampere3/0/alexspan/norm-research
REPO=/lfs/skampere3/0/alexspan/norm-research
OUT=/lfs/skampere3/0/alexspan/outputs/r3_cw
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
cd "$REPO" || exit 1

M8B=meta-llama/Llama-3.1-8B-Instruct
M3B=meta-llama/Llama-3.2-3B-Instruct
M70B=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
MQWEN=Qwen/Qwen3.5-122B-A10B-FP8

echo "[$(date '+%m-%d %H:%M')] aligned Q1 launch: 8b/3b retarget + 70b/qwen full-rescore (orbit-4)"

# GPU1: 8B orbit-retarget (reuse 8B sigs from llama8b_glm source)
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.93 nohup "$PY" -u -m methods.metric_implementer.experiments.rescore_executor \
  --src-dir "$OUT/llama8b_glm" --target-model "$M8B" --out-dir "$OUT/aligned_8b_orbit" \
  --task creative-writing --n-metrics 0 --n-probes 300 --orbit-target 4 --retarget-mi-only --skip-existing \
  > "$OUT/aligned_8b.log" 2>&1 &
P8=$!

# GPU2: 3B orbit-retarget (reuse 3B sigs from llama8b_to_llama3b)
CUDA_VISIBLE_DEVICES=2 VLLM_GPU_MEM_UTIL=0.93 nohup "$PY" -u -m methods.metric_implementer.experiments.rescore_executor \
  --src-dir "$OUT/llama8b_to_llama3b" --target-model "$M3B" --out-dir "$OUT/aligned_3b_orbit" \
  --task creative-writing --n-metrics 0 --n-probes 300 --orbit-target 4 --retarget-mi-only --skip-existing \
  > "$OUT/aligned_3b.log" 2>&1 &
P3=$!

# GPU3: 70B-FP8 full-rescore + orbit (complete the primary ladder; src = GLM Omega)
CUDA_VISIBLE_DEVICES=3 VLLM_GPU_MEM_UTIL=0.93 nohup "$PY" -u -m methods.metric_implementer.experiments.rescore_executor \
  --src-dir "$OUT/llama8b_glm" --target-model "$M70B" --out-dir "$OUT/aligned_70b_orbit" \
  --task creative-writing --n-metrics 0 --n-probes 300 --orbit-target 4 --skip-existing \
  > "$OUT/aligned_70b.log" 2>&1 &
P70=$!

# GPU7: Qwen-122B full-rescore + orbit (replication panel; FLASHINFER_MOE_FP8=0)
CUDA_VISIBLE_DEVICES=7 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_GPU_MEM_UTIL=0.93 nohup "$PY" -u -m methods.metric_implementer.experiments.rescore_executor \
  --src-dir "$OUT/llama8b_glm" --target-model "$MQWEN" --out-dir "$OUT/aligned_qwen_orbit" \
  --task creative-writing --n-metrics 0 --n-probes 300 --orbit-target 4 --skip-existing \
  > "$OUT/aligned_qwen.log" 2>&1 &
PQ=$!

echo "[$(date '+%m-%d %H:%M')] launched 8b=$P8(GPU1) 3b=$P3(GPU2) 70b=$P70(GPU3) qwen=$PQ(GPU7)"
echo "$P8 $P3 $P70 $PQ" > "$OUT/aligned_pids.txt"
