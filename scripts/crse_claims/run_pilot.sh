#!/usr/bin/env bash
# run_pilot.sh — end-to-end CR.SE claim-extraction + verification pilot.
# Intended to be executed on sk3 (the working dir below is sk3-specific).
#
# Steps:
#   0) sanity-check the chosen GPU is one of OUR own and has enough free mem.
#   1) Pre-flight: extract on 20 answers; if yield is broken, abort.
#   2) Full pilot: extract on 300 answers.
#   3) Flatten + validate.
#   4) Run check_claims.py to verify claims and emit per-answer features.
#
# Usage:
#   bash scripts/crse_claims/run_pilot.sh        # uses default GPU 6
#   GPU=4 bash scripts/crse_claims/run_pilot.sh  # override
set -euo pipefail

ROOT=/lfs/skampere3/0/alexspan/norm-research
cd "$ROOT"

export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/.cache/triton
export VLLM_CACHE_ROOT=/lfs/skampere3/0/alexspan/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/lfs/skampere3/0/alexspan/.cache/torchinductor
export XDG_CACHE_HOME=/lfs/skampere3/0/alexspan/.cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "$TMPDIR"

PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
GPU=${GPU:-6}
INPUT=datasets/code-review/crse_balanced_v1/crse_v1_propensity_balanced.csv.gz
OUTDIR=outputs/crse_claims_pilot
mkdir -p "$OUTDIR"

echo "=== GPU state check ==="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
echo "GPU $GPU free: $free_mem MiB"
if [ "$free_mem" -lt 25000 ]; then
  echo "FATAL: GPU $GPU has only ${free_mem} MiB free; aborting."
  exit 1
fi

echo
echo "=== Step 1: Pre-flight extraction (20 answers) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/crse_claims/extract_claims.py \
    --input "$INPUT" \
    --output "$OUTDIR/preflight_claims.jsonl" \
    --pilot-n 20 --stratify-by judgement --seed 42 \
    --validate-only-n 20 \
    --batch-size 20 \
    --gpu-memory-utilization 0.55

# Quick yield gate — abort the pilot if pre-flight is broken.
yield=$($PY -c "
import json
ok = sum(1 for L in open('$OUTDIR/preflight_claims.jsonl') if json.loads(L).get('extracted'))
print(ok)
")
echo "Pre-flight validated: $yield / 20"
if [ "$yield" -lt 6 ]; then
  echo "FATAL: pre-flight yield $yield/20 < 6. Aborting pilot."
  exit 1
fi

echo
echo "=== Step 2: Full pilot extraction (300 answers) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/crse_claims/extract_claims.py \
    --input "$INPUT" \
    --output "$OUTDIR/claims.jsonl" \
    --pilot-n 300 --stratify-by judgement --seed 42 \
    --batch-size 300 \
    --gpu-memory-utilization 0.55

echo
echo "=== Step 3: Flatten + validate ==="
$PY scripts/crse_claims/validate_claims.py \
    --claims "$OUTDIR/claims.jsonl" \
    --out-flat "$OUTDIR/claims_flat.jsonl" \
    --out-summary "$OUTDIR/validate_summary.json"

echo
echo "=== Step 4: Check claims ==="
$PY scripts/crse_claims/check_claims.py \
    --claims-flat "$OUTDIR/claims_flat.jsonl" \
    --pool "$INPUT" \
    --out-verdicts "$OUTDIR/verdicts.jsonl" \
    --out-features "$OUTDIR/answer_features.jsonl"

echo
echo "=== Pilot complete ==="
ls -lh "$OUTDIR"
