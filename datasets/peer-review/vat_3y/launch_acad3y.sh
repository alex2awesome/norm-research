cd /lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/vat_3y || { echo CD_FAIL; exit 1; }
export HOME=/lfs/skampere3/0/alexspan
GEMMA=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
D=/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/vat_3y
setsid env HF_HUB_OFFLINE=1 HF_HOME=/lfs/skampere3/0/shared_hf_cache \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 \
  $GEMMA "$D/score_va_gemma_3y.py" --util 0.85 \
    --input "$D/union_toscore.jsonl" --out "$D/union_scores.npz" \
  < /dev/null > "$D/acad3y_score.log" 2>&1 &
echo "LAUNCHED pid=$!"
