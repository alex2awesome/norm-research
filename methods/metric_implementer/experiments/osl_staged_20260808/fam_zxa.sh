#!/bin/bash
# Same-family decompression lane (z×a fam arms; spec 2026-07-09 addendum).
# Phase 1: llama70b + qwen25-72b author explanation+dossier for all 72 slate metrics.
# Phase 2: build freeze_zxa_<task>_fam_v1.json (gates identical to v1).
# Phase 3: run the llama + qwen ladders on the fam arms, humor first. Resumable.
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
export HF_HUB_OFFLINE=1
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=$1
mkdir -p $O/logs

for AUTHOR in llama70b qwen25-72b; do
  # skip authoring pass if this author already has 72 valid rows
  N=$($PY - <<EOF 2>/dev/null
import json,os
fp="$O/zxa_authoring_fam/$AUTHOR.json"
print(sum(1 for r in json.load(open(fp)) if r.get("valid")) if os.path.exists(fp) else 0)
EOF
)
  [ "$N" = "72" ] && { echo "GPU$GPU FAM author $AUTHOR already 72/72 $(date)" >> $O/FLEET_STATUS; continue; }
  echo "GPU$GPU FAM AUTHOR START $AUTHOR $(date)" >> $O/FLEET_STATUS
  CUDA_VISIBLE_DEVICES=$GPU $PY $O/author_fam_arms.py --author $AUTHOR \
    >> $O/logs/fam_author_${AUTHOR}.log 2>&1
  echo "GPU$GPU FAM AUTHOR END $AUTHOR rc=$? $(date)" >> $O/FLEET_STATUS
done

# second authoring pass to mop up invalid rows (resume-safe: only redoes invalid)
for AUTHOR in llama70b qwen25-72b; do
  CUDA_VISIBLE_DEVICES=$GPU $PY $O/author_fam_arms.py --author $AUTHOR \
    >> $O/logs/fam_author_${AUTHOR}.log 2>&1
done

$PY $O/build_zxa_freeze_fam.py >> $O/logs/fam_freeze_build.log 2>&1 || {
  echo "GPU$GPU FAM FREEZE BUILD FAILED $(date)" >> $O/FLEET_STATUS; exit 1; }
echo "GPU$GPU FAM freezes built $(date)" >> $O/FLEET_STATUS

for TD in humor creative_writing peer_review math; do
  for EX in llama1b llama3b llama8b qwen25-3b qwen25-7b qwen25-14b qwen25-32b llama70b qwen25-72b; do
    OUT=$O/mbar_zxafam_${TD}_${EX}.npz
    [ -s $OUT ] && continue
    echo "GPU$GPU FAM START $TD $EX $(date)" >> $O/FLEET_STATUS
    CUDA_VISIBLE_DEVICES=$GPU $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --n-forms 1 --executor $EX --freeze $O/freeze_zxa_${TD}_fam_v1.json --out $OUT \
      >> $O/logs/zxafam_${TD}_${EX}.log 2>&1
    echo "GPU$GPU FAM END $TD $EX rc=$? $(date)" >> $O/FLEET_STATUS
  done
done
echo "GPU$GPU FAM-LANE-DONE $(date)" >> $O/FLEET_STATUS
