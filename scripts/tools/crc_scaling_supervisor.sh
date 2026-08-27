#!/usr/bin/env bash
# Overnight capture-recapture SCALING supervisor: runs the open-weight executor ladder on one task's
# R2 metrics at high N (M_freegen=600, n_probes=300), parallel across free GPUs with per-metric
# --skip-existing resume + timeout. Conditional CRC (upper/lower bounds) + order permutation are then
# run CPU-side by crc_analyze. Crash-resilient: relaunch resumes from checkpoints.
# Honors: HOME=/lfs (AFS tokens), HF_HOME shared cache, 1 job/GPU (small models stacked), no GLM,
# kill-by-PID only (this script kills only its own PIDs via timeout).
set -u
REPO=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
GEMMA_PY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
OUTROOT=/lfs/skampere3/0/alexspan/outputs/crc_scaling
TASK=${TASK:-creative-writing}; LEVEL=${LEVEL:-R2}; BUCKET=${BUCKET:-general}
NPROBES=${NPROBES:-300}; MFREE=${MFREE:-600}
LOGDIR=$OUTROOT/_supervisor; mkdir -p "$LOGDIR"
SUPLOG=$LOGDIR/supervisor.log

run_job () {   # args: gpu pybind env_prefix model n_metrics suffix
  local gpu=$1 pybin=$2 envpre=$3 model=$4 nmet=$5 suffix=$6
  local out="$OUTROOT/$suffix"; mkdir -p "$out"
  local log="$LOGDIR/$suffix.log"
  echo "[$(date '+%H:%M:%S')] START $suffix gpu=$gpu model=$model n=$nmet" >> "$SUPLOG"
  ( cd "$REPO" && \
    env $envpre HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache \
        CUDA_VISIBLE_DEVICES=$gpu VLLM_GPU_MEM_UTIL=0.93 VLLM_MAX_MODEL_LEN=12288 \
        TOKENIZERS_PARALLELISM=false \
        timeout --kill-after=120 32000 "$pybin" -m methods.metric_implementer.experiments.run_alpha_probe \
        --task "$TASK" --level "$LEVEL" --r2-bucket "$BUCKET" --largest-first --n-metrics "$nmet" \
        --target-model "$model" --M-freegen "$MFREE" --n-probes "$NPROBES" --no-glm \
        --skip-existing --out-dir "$out" > "$log" 2>&1 )
  echo "[$(date '+%H:%M:%S')] DONE  $suffix  ckpts=$(ls "$out"/*_sigs.npz 2>/dev/null | wc -l)" >> "$SUPLOG"
}

echo "[$(date '+%H:%M:%S')] ===== supervisor start TASK=$TASK LEVEL=$LEVEL M=$MFREE probes=$NPROBES =====" >> "$SUPLOG"

# ---- Wave 1: 5 models on 4 free GPUs (llama-3b+8b stacked on G1) ----
run_job 1 "$PY" ""                                        meta-llama/Llama-3.1-8B-Instruct   12 llama8b  &
sleep 20
run_job 1 "$PY" ""                                        meta-llama/Llama-3.2-3B-Instruct   12 llama3b  &
sleep 20
run_job 2 "$PY" ""                                        nvidia/Llama-3.3-70B-Instruct-FP8   4 llama70b &
sleep 20
run_job 3 "$PY" ""                                        google/gemma-2-27b-it               6 gemma27b &
sleep 20
run_job 7 "$PY" "VLLM_USE_FLASHINFER_MOE_FP8=0"          Qwen/Qwen3.5-122B-A10B-FP8         12 qwen122b &
wait
echo "[$(date '+%H:%M:%S')] ===== WAVE-1 complete =====" >> "$SUPLOG"

# ---- Wave 2: Gemma-4 stretch (gemma4 env) on a freed GPU ----
run_job 1 "$GEMMA_PY" ""                                  google/gemma-4-31b-it               4 gemma31b &
wait
echo "[$(date '+%H:%M:%S')] ===== ALL complete =====" >> "$SUPLOG"
# auto-run the CPU scaling analysis so morning has the table
( cd "$REPO" && "$PY" -m methods.metric_implementer.experiments.crc_analyze \
    --scaling llama8b:$OUTROOT/llama8b,llama3b:$OUTROOT/llama3b,gemma27b:$OUTROOT/gemma27b,\
llama70b:$OUTROOT/llama70b,qwen122b:$OUTROOT/qwen122b,gemma31b:$OUTROOT/gemma31b \
    --json $OUTROOT/crc_scaling_summary.json > $LOGDIR/crc_scaling_table.txt 2>&1 )
echo "[$(date '+%H:%M:%S')] scaling table -> $LOGDIR/crc_scaling_table.txt" >> "$SUPLOG"
