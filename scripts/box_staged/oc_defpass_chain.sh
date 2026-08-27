#!/bin/bash
export HOME=/lfs/skampere3/0/alexspan
cd $HOME/mention_auc
G4PY=$HOME/envs/gemma4/bin/python
MG4=/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb
for T in peer cw pr humor; do
  TX=${T}_probe_texts.jsonl; KEY=probe_id
  CUDA_VISIBLE_DEVICES=0 VLLM_GPU_MEM_UTIL=0.85 $G4PY score_within_metric.py \
    --texts $TX --manifest ocdef_${T}_manifest.json --key $KEY \
    --model $MG4 --chunk 14 --out ocdef_${T}_probes_g4.json >> loc_defpass.log 2>&1
  echo "$(date -u +%FT%TZ) defpass $T done" >> loc_defpass.log
done
echo "$(date -u +%FT%TZ) OC DEFPASS CHAIN DONE" >> loc_defpass.log
