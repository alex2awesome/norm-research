#!/bin/bash
# GPU7 morning chain (2026-07-09): gemma2-9b z×a re-pass (block-size fix now actually
# reaches the engine) -> FLASH_ATTN fallback if the FlashInfer assert survives ->
# mechbat small ladder -> relaunch newmodels chain (mistral-24b/qwen25-32b; downloads
# already restarted in background after the 12h hang was killed).
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TMPDIR=/lfs/skampere3/0/alexspan/tmp TRITON_CACHE_DIR=/lfs/skampere3/0/alexspan/tmp/triton
B=/lfs/skampere3/0/alexspan; R=$B/norm-research; PY=$B/envs/ai_usage/bin/python
export PYTHONPATH=$R; cd $R
O=$B/outputs/osl_multi; GPU=7

bash $O/run_zxa_small.sh $GPU

# fallback: if gemma2-9b still failed on the FlashInfer assert, retry once with FLASH_ATTN
if [ ! -s $O/mbar_zxa_humor_gemma2-9b.npz ] && grep -q "FlashInfer block_size" $O/logs/zxa_humor_gemma2-9b.log 2>/dev/null; then
  echo "GPU$GPU GEMMA2-9B FLASH_ATTN FALLBACK $(date)" >> $O/FLEET_STATUS
  for TD in humor creative_writing peer_review math; do
    OUT=$O/mbar_zxa_${TD}_gemma2-9b.npz
    [ -s $OUT ] && continue
    env VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_BLOCK_SIZE=32 CUDA_VISIBLE_DEVICES=$GPU \
      $PY -m methods.metric_implementer.experiments.osl_sweep --mbar-only \
      --n-forms 1 --executor gemma2-9b --freeze $O/freeze_zxa_${TD}_v1.json --out $OUT \
      >> $O/logs/zxa_${TD}_gemma2-9b.log 2>&1
    echo "GPU$GPU ZXA FALLBACK END $TD gemma2-9b rc=$? $(date)" >> $O/FLEET_STATUS
  done
fi

bash $O/run_mechbat.sh $GPU
bash $O/run_newmodels.sh $GPU
echo "GPU$GPU MORNING-CHAIN-DONE $(date)" >> $O/FLEET_STATUS
