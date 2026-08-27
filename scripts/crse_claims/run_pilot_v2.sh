#!/usr/bin/env bash
# run_pilot_v2.sh — re-run of the CR.SE claim pilot AFTER the three blocking
# fixes (anti-pattern few-shots; raw-HTML code detection; JS/Java/C# checkers).
#
# Reuses the EXACT 300-answer sample from the first pilot (answer_ids pulled
# from outputs/crse_claims_pilot/claims.jsonl).
#
# Usage (sk3):
#   bash scripts/crse_claims/run_pilot_v2.sh          # default GPU 7
#   GPU=6 bash scripts/crse_claims/run_pilot_v2.sh
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
export MCS_BIN=/lfs/skampere3/0/alexspan/envs/mono/bin/mcs
mkdir -p "$TMPDIR"

PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
GPU=${GPU:-7}
GPU_UTIL=${GPU_UTIL:-0.13}   # stacked on our OWN busy GPU; 0.13*183GB ~= 24GB
INPUT=datasets/code-review/crse_balanced_v1/crse_v1_propensity_balanced.csv.gz
POSTS=datasets/codereview_se/posts.parquet
OLDDIR=outputs/crse_claims_pilot
OUTDIR=outputs/crse_claims_pilot_v2
mkdir -p "$OUTDIR"

echo "=== GPU state check ==="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
echo "GPU $GPU free: $free_mem MiB (need >= 26000 for util=$GPU_UTIL)"
if [ "$free_mem" -lt 26000 ]; then
  echo "FATAL: GPU $GPU has only ${free_mem} MiB free; aborting."
  exit 1
fi

echo
echo "=== Step 0: reuse the pilot sample (300 answer_ids) ==="
$PY -c "
import json
ids = [json.loads(L)['answer_id'] for L in open('$OLDDIR/claims.jsonl')]
open('$OUTDIR/sample_ids.txt','w').write('\n'.join(map(str, ids)))
print('wrote', len(ids), 'ids')
"

echo
echo "=== Step 1: extraction (same 300 answers, fixed prompt) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/crse_claims/extract_claims.py \
    --input "$INPUT" \
    --output "$OUTDIR/claims.jsonl" \
    --pilot-n 300 --seed 42 \
    --restrict-ids "$OUTDIR/sample_ids.txt" \
    --batch-size 300 \
    --gpu-memory-utilization "$GPU_UTIL"

echo
echo "=== Step 2: Flatten + validate ==="
$PY scripts/crse_claims/validate_claims.py \
    --claims "$OUTDIR/claims.jsonl" \
    --out-flat "$OUTDIR/claims_flat.jsonl" \
    --out-summary "$OUTDIR/validate_summary.json"

echo
echo "=== Step 3: Check claims (multi-language) ==="
$PY scripts/crse_claims/check_claims.py \
    --claims-flat "$OUTDIR/claims_flat.jsonl" \
    --pool "$INPUT" \
    --posts-parquet "$POSTS" \
    --out-verdicts "$OUTDIR/verdicts.jsonl" \
    --out-features "$OUTDIR/answer_features.jsonl"

echo
echo "=== Pilot v2 complete ==="
ls -lh "$OUTDIR"
