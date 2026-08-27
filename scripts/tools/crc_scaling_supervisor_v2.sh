#!/usr/bin/env bash
# v2 capture-recapture SCALING supervisor — RUBRIC-FIRST (canonical) measurement, with the safe
# efficiency levers: #2 scan-probe subset (NSCAN env; 0 = full 300), #6 form_invariance sig-reuse
# (automatic in the new code), and #5 rep-refine OFF (gated on --text-first, which is NOT used —
# prompt_ordering_check showed text-first inflates B_E ~2x, a real form-sensitivity, not a free win).
# Canonical FP8 snapshot PATH (NOT the HF id -> resolves to the broken /hub/ mirror -> NaN readouts).
# Crash-resilient (--skip-existing resumes from checkpoints). 1 job/GPU, llama-3b+8b stacked on G1.
# Honors: HOME=/lfs (AFS tokens), HF_HOME shared cache, kill-by-PID only (timeout owns its PIDs).
set -u
REPO=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
GEMMA_PY=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
OUTROOT=/lfs/skampere3/0/alexspan/outputs/crc_scaling
TASK=${TASK:-creative-writing}; LEVEL=${LEVEL:-R2}; BUCKET=${BUCKET:-general}
NPROBES=${NPROBES:-300}; MFREE=${MFREE:-600}; NSCAN=${NSCAN:-0}; NMET=${NMET:-30}
LLAMA70B_FP8=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
GEMMA4=/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb
LOGDIR=$OUTROOT/_supervisor; mkdir -p "$LOGDIR"
SUPLOG=$LOGDIR/supervisor_v2.log
SCANFLAG=""; [ "$NSCAN" -gt 0 ] 2>/dev/null && SCANFLAG="--n-scan-probes $NSCAN"

run_job () {   # args: gpu pybind env_prefix model n_metrics suffix
  local gpu=$1 pybin=$2 envpre=$3 model=$4 nmet=$5 suffix=$6
  local out="$OUTROOT/$suffix"; mkdir -p "$out"
  local log="$LOGDIR/$suffix.log"
  echo "[$(date '+%H:%M:%S')] START $suffix gpu=$gpu model=$model n=$nmet scan=$NSCAN" >> "$SUPLOG"
  ( cd "$REPO" && \
    env $envpre HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache \
        CUDA_VISIBLE_DEVICES=$gpu VLLM_GPU_MEM_UTIL=0.93 VLLM_MAX_MODEL_LEN=12288 \
        TOKENIZERS_PARALLELISM=false FLASHINFER_DISABLE_VERSION_CHECK=1 \
        timeout --kill-after=120 32000 "$pybin" -m methods.metric_implementer.experiments.run_alpha_probe \
        --task "$TASK" --level "$LEVEL" --r2-bucket "$BUCKET" --largest-first --n-metrics "$nmet" \
        --target-model "$model" --M-freegen "$MFREE" --n-probes "$NPROBES" $SCANFLAG --cmi-thresh 0.15 \
        --no-glm --skip-existing --out-dir "$out" > "$log" 2>&1 )
  echo "[$(date '+%H:%M:%S')] DONE  $suffix  ckpts=$(ls "$out"/*_sigs.npz 2>/dev/null | wc -l)" >> "$SUPLOG"
}

echo "[$(date '+%H:%M:%S')] ===== v2 start TASK=$TASK M=$MFREE probes=$NPROBES scan=$NSCAN n=$NMET (rubric-first) =====" >> "$SUPLOG"
# Wave 1: 5 executors on G1,G1(stack),G2,G3,G7. ahmedah/animjha on G0/G4-G6 — never touch.
run_job 1 "$PY"    ""                                meta-llama/Llama-3.1-8B-Instruct   $NMET llama8b   &
sleep 25
run_job 1 "$PY"    ""                                meta-llama/Llama-3.2-3B-Instruct   $NMET llama3b   &
sleep 25
run_job 2 "$PY"    "FLASHINFER_DISABLE_VERSION_CHECK=1" "$LLAMA70B_FP8"                     4     llama70b_fp8 &
sleep 25
run_job 3 "$GEMMA_PY" ""                             "$GEMMA4"                           4     gemma31b    &
sleep 25
run_job 7 "$PY"    "VLLM_USE_FLASHINFER_MOE_FP8=0"   Qwen/Qwen3.5-122B-A10B-FP8         $NMET qwen122b   &
wait
echo "[$(date '+%H:%M:%S')] ===== WAVE-1 complete =====" >> "$SUPLOG"
# auto-run the CPU scaling analysis so morning has the table
( cd "$REPO" && PYTHONPATH="$REPO" "$PY" -m methods.metric_implementer.experiments.crc_analyze \
    --scaling llama3b:$OUTROOT/llama3b,llama8b:$OUTROOT/llama8b,llama70b_fp8:$OUTROOT/llama70b_fp8,\
gemma31b:$OUTROOT/gemma31b,qwen122b:$OUTROOT/qwen122b \
    --json $OUTROOT/crc_scaling_summary.json > $LOGDIR/crc_scaling_table.txt 2>&1 )
echo "[$(date '+%H:%M:%S')] scaling table -> $LOGDIR/crc_scaling_table.txt" >> "$SUPLOG"
