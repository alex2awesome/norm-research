#!/usr/bin/env bash
# launch_alpha_probe_sk3.sh — ONE-COMMAND metric-level ALPHA-PROBE on sk3 (α_i per R2 cluster).
# Loads the vLLM executor ONCE and sweeps R2 clusters for one task. See run_alpha_probe.py.
#
# ENV OVERRIDES:  TASK=creative-writing  R2BUCKET=general  NMETRICS=30  MFREEGEN=60  NPROBES=300
#                 RESERVE=60  NO_GLM=1  LARGEST=1  METRICSTART=0  GEPAREGISTRY=...  MODEL=...  PY=...
set -uo pipefail
REPO=/lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache            # weights live here (not ~/.cache)
export HUGGINGFACE_HUB_CACHE=/lfs/skampere3/0/shared_hf_cache/hub
export HF_HUB_OFFLINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export TOKENIZERS_PARALLELISM=false
export VLLM_GPU_MEM_UTIL=0.93
PY="${PY:-/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
cd "$REPO"

FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2 < 2000 {print $1; exit}')
if [ -z "${FREE:-}" ]; then
  echo "No idle GPU (<2000 MiB). Wrap: ./launch_when_gpus_free.sh 1 \"$0\""; exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE"

TASK="${TASK:-creative-writing}"; R2BUCKET="${R2BUCKET:-general}"
NMETRICS="${NMETRICS:-30}"; MFREEGEN="${MFREEGEN:-60}"
NPROBES="${NPROBES:-300}"; RESERVE="${RESERVE:-60}"
OUT="$HOME/outputs/alpha_probe_metric"
LOG="$HOME/logs/alpha_probe_metric_${TASK}_gpu${FREE}.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

ARGS=(--task "$TASK" --r2-bucket "$R2BUCKET" --target-model "$MODEL"
      --n-metrics "$NMETRICS" --M-freegen "$MFREEGEN" --n-probes "$NPROBES"
      --gepa-reserve "$RESERVE" --out-dir "$OUT")
[ "${NO_GLM:-0}" = "1" ] && ARGS+=(--no-glm)
[ "${LARGEST:-0}" = "1" ] && ARGS+=(--largest-first)
[ -n "${METRICSTART:-}" ] && ARGS+=(--metric-start "$METRICSTART")
[ -n "${GEPAREGISTRY:-}" ] && ARGS+=(--gepa-registry "$GEPAREGISTRY")

echo "$(date): idle GPU ${FREE} → metric α-probe: task=$TASK bucket=$R2BUCKET n=$NMETRICS M_fg=$MFREEGEN"
echo "  args: ${ARGS[*]}"
nohup "$PY" -m methods.metric_implementer.experiments.run_alpha_probe "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$(date): LAUNCHED pid=$PID"
echo "  log: $LOG   |   tail -f $LOG"
echo "  done?: grep -E 'METRIC-LEVEL α summary|summary →' $LOG"
echo "  KILL (PID only, reap EngineCore child):"
echo "    CHILD=\$(ps --ppid $PID -o pid= | head -1); [ -n \"\$CHILD\" ] && kill \$CHILD; kill $PID"
