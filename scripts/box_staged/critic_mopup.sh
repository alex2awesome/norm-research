#!/bin/bash
export HOME=/lfs/skampere3/0/alexspan
while kill -0 2916624 2>/dev/null; do sleep 60; done
cd $HOME/mention_auc
for i in 1 2 3; do
  /lfs/skampere3/0/alexspan/miniconda3/bin/python api_field_runner_patient.py --backend openrouter --model qwen/qwen-2.5-72b-instruct --prompts $HOME/outputs/objective_comparison_v1/critic_all_prompts.jsonl --out $HOME/outputs/objective_comparison_v1/critic_all_results.jsonl --concurrency 8 --max-tokens 40 >> loc_critic_mopup.log 2>&1
  echo "$(date -u +%FT%TZ) mopup pass $i done" >> loc_critic_mopup.log
done
echo "$(date -u +%FT%TZ) CRITIC COMPLETE" >> loc_critic_mopup.log
