#!/bin/bash
# Side-by-side generation: base Llama-3.1-8B vs trained LoRA on the same
# held-out artifacts. Lets us eyeball whether SFT actually moved the model.
#
# Inputs:
#   RUN_NAME (default: explore_contrastive_cw)  -- the iter dir whose LoRA we evaluate
#   ITER (default: 0)                           -- which iter's LoRA
#   EVAL_RUN_NAME (default: explore_contrastive_cw_eval)
#   N_EVAL (default: 50)                        -- artifacts to generate from
#   GPU (default: 1)
set -euo pipefail

export HOME=/lfs/skampere3/0/alexspan
export PATH=/lfs/skampere3/0/alexspan/miniconda3/bin:$PATH
cd /lfs/skampere3/0/alexspan/norm-research
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_MOE_FP8=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

TASK="${TASK:-creative_writing}"
RUN_NAME="${RUN_NAME:-explore_contrastive_cw}"
ITER="${ITER:-0}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-explore_contrastive_cw_eval}"
N_EVAL="${N_EVAL:-50}"
GPU="${GPU:-1}"

LOG_DIR="logs/articulation_star/${EVAL_RUN_NAME}"
mkdir -p "$LOG_DIR"

LORA_ABS="$(pwd)/outputs/articulation_star/${TASK}/${RUN_NAME}/iter_$(printf %02d "$ITER")/lora"

if [[ ! -f "${LORA_ABS}/adapter_config.json" ]]; then
  echo "ERROR: LoRA not found at $LORA_ABS"; exit 1
fi

# ── Base 8B generation ────────────────────────────────────────
echo "[$(date '+%F %T')] === base 8B generation (GPU $GPU, eval iter=0) ==="
CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.generate_rationales \
  --task "$TASK" --run_name "${EVAL_RUN_NAME}_base" --iter 0 \
  --n_train_subsample "$N_EVAL" --n_rationales_per_input 1 \
  > "$LOG_DIR/gen_base.log" 2>&1

# ── Trained LoRA generation ───────────────────────────────────
echo "[$(date '+%F %T')] === trained-LoRA 8B generation (GPU $GPU, eval iter=1) ==="
CUDA_VISIBLE_DEVICES=$GPU python -m methods.articulation_star.generate_rationales \
  --task "$TASK" --run_name "${EVAL_RUN_NAME}_lora" --iter 0 \
  --n_train_subsample "$N_EVAL" --n_rationales_per_input 1 \
  --lora_path "$LORA_ABS" \
  > "$LOG_DIR/gen_lora.log" 2>&1

OUT_BASE="outputs/articulation_star/${TASK}/${EVAL_RUN_NAME}_base/iter_00/rationales.jsonl"
OUT_LORA="outputs/articulation_star/${TASK}/${EVAL_RUN_NAME}_lora/iter_00/rationales.jsonl"
echo "[$(date '+%F %T')] === done ==="
echo "  base: $OUT_BASE"
echo "  lora: $OUT_LORA"
