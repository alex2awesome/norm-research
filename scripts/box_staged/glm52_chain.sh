#!/bin/bash
# GLM-5.2 matched prompt-judge trial chain (2026-08-14). Phases A -> select -> C -> analyze.
export HOME=/lfs/skampere3/0/alexspan
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
cd $HOME/mention_auc || exit 9
L=glm52_chain.log
echo "$(date -u +%FT%TZ) CHAIN START" >> $L
$PY api_field_runner_patient.py --backend zai_anthropic --model glm-5.2 \
    --prompts glm52_trial_a_prompts.jsonl --out glm52_trial_a_results.jsonl \
    --concurrency 8 --max-tokens 1024 >> $L 2>&1
echo "$(date -u +%FT%TZ) PHASE A DONE" >> $L
$PY glm52_matched_trial.py select >> $L 2>&1
$PY glm52_matched_trial.py build_c >> $L 2>&1
$PY api_field_runner_patient.py --backend zai_anthropic --model glm-5.2 \
    --prompts glm52_trial_c_prompts.jsonl --out glm52_trial_c_results.jsonl \
    --concurrency 8 --max-tokens 1024 >> $L 2>&1
echo "$(date -u +%FT%TZ) PHASE C DONE" >> $L
$PY glm52_matched_trial.py analyze >> $L 2>&1
echo "$(date -u +%FT%TZ) GLM52 TRIAL COMPLETE" >> $L
