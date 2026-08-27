#!/usr/bin/env bash
# sweep_alpha_probe_sk3.sh — sequential ALPHA-PROBE behavior census across tasks on ONE GPU.
# Purpose: hunt for any task where α < 0.5 (a coverable / low-dim reachable behavior space).
# Runs each task's full breadth sample back-to-back (1 GPU rule), logs α + decision per task so
# progress is monitorable. Structured tasks first (more likely low-dim); rich tasks last.
#
# ENV OVERRIDES:  TASKS="..."  M=450  NPROBES=300  RESERVE=60  MODEL=...  PY=...
set -uo pipefail
REPO=/lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache            # weights live here (see launch_alpha_probe_sk3.sh)
export HUGGINGFACE_HUB_CACHE=/lfs/skampere3/0/shared_hf_cache/hub
export HF_HUB_OFFLINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export TOKENIZERS_PARALLELISM=false
export VLLM_GPU_MEM_UTIL=0.93
PY="${PY:-/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
M="${M:-450}"; NPROBES="${NPROBES:-300}"; RESERVE="${RESERVE:-60}"
# structured (likely low-α) first, then rich (likely high-α)
TASKS="${TASKS:-math news-homepages patents law notice-and-comment creative-writing humor}"
OUT="$HOME/outputs/alpha_probe"
SWEEP_LOG="$HOME/logs/alpha_probe_sweep.log"
mkdir -p "$OUT" "$(dirname "$SWEEP_LOG")"
cd "$REPO"

# detect ONE free GPU and PIN it for the whole sequential sweep
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2 < 2000 {print $1; exit}')
if [ -z "${FREE:-}" ]; then
  echo "$(date) no idle GPU (<2000 MiB) — aborting sweep" | tee -a "$SWEEP_LOG"; exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE"
echo "$(date) SWEEP start GPU=${FREE} M=${M} tasks: ${TASKS}" | tee -a "$SWEEP_LOG"

for TASK in $TASKS; do
  LOG="$HOME/logs/alpha_probe_${TASK}_gpu${FREE}.log"
  echo "$(date) >>> TASK=${TASK} start (log ${LOG})" | tee -a "$SWEEP_LOG"
  "$PY" -m methods.metric_implementer.experiments.run_alpha_probe \
        --task "$TASK" --target-model "$MODEL" --M "$M" --n-probes "$NPROBES" \
        --gepa-reserve "$RESERVE" --no-glm --out-dir "$OUT" > "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "$(date) <<< TASK=${TASK} FAILED rc=${RC} (tail: $(tail -1 "$LOG"))" | tee -a "$SWEEP_LOG"
    continue
  fi
  SUMMARY=$("$PY" -c "
import json
r=json.load(open('$OUT/${TASK}_alpha_probe.json'))
print('alpha=%.3f dec=%s D=%d/N=%d disc_L1=%.3f' % (
    r['alpha_terminal'], r['decision'], r['D'], r['N'],
    r['signature_diagnostics']['mean_between_sig_l1']))" 2>/dev/null || echo "PARSE-FAIL")
  echo "$(date) <<< TASK=${TASK}  ${SUMMARY}" | tee -a "$SWEEP_LOG"
done
echo "$(date) SWEEP DONE" | tee -a "$SWEEP_LOG"
