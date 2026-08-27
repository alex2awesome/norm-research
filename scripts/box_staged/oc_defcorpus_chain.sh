#!/bin/bash
export HOME=/lfs/skampere3/0/alexspan
cd $HOME/mention_auc
G4PY=$HOME/envs/gemma4/bin/python
MG4=/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb
run() {
  CUDA_VISIBLE_DEVICES=0 VLLM_GPU_MEM_UTIL=0.85 $G4PY score_within_metric.py \
    --texts $1 --manifest ocdef_$2_manifest.json --key $3 \
    --model $MG4 --chunk 14 --out ocdef_$2_corpus_g4.json >> loc_defcorpus.log 2>&1
  echo "$(date -u +%FT%TZ) defcorpus $2 done" >> loc_defcorpus.log
}
run peer_paper_texts.jsonl peer paper_id
run cw_story_texts.jsonl cw post_id
run pressrel_score_texts.jsonl pr source_id
run humor_score_texts.jsonl humor source_id
echo "$(date -u +%FT%TZ) OC DEFCORPUS CHAIN DONE" >> loc_defcorpus.log
