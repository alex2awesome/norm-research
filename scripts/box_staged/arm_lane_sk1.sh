#!/bin/bash
# generic arm lane, sk1 port: $1=gpu $2=port $3=bench $4=arm $5=budget $6=tag $7=gepa_seed(default 0)
export HOME=/lfs/skampere1/0/alexspan
export HF_HOME=$HOME/.cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536 2>/dev/null
GPU=$1; PORT=$2; BENCH=$3; ARM=$4; BUD=$5; TAG=$6; SEED=${7:-0}
D=$HOME/norm-research/datasets/prompt-optimality-test
VLLM=$D/.venv/bin/vllm; CACHE=$HOME/.cache/huggingface/hub
cd $D || exit 9
L=logs/arm_${TAG}.log
[ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | awk '$1>2000')" ] && { echo "GPU$GPU BUSY - abort" | tee -a $L; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
M=Qwen3-8B; SNAP=$(ls -d $CACHE/models--Qwen--$M/snapshots/*/ | head -1)
echo "$(date -u +%FT%TZ) $TAG: $BENCH $ARM budget=$BUD seed=$SEED gpu$GPU port$PORT (sk1)" | tee -a $L
nohup $VLLM serve $SNAP --served-model-name $M --port $PORT --host 127.0.0.1 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
  > logs/vllm_${TAG}.log 2>&1 &
VP=$!
UP=0; for i in $(seq 1 90); do sleep 10; curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q "$M" && { UP=1; break; }; done
[ $UP != 1 ] && { echo "$(date -u +%FT%TZ) SERVER FAILED" | tee -a $L; exit 1; }
export DSPY_CACHEDIR=$HOME/dspy_cache_${TAG}
./.venv/bin/python paperexact_arms.py $BENCH --arm $ARM \
   --task-lm openai/$M --api-base http://127.0.0.1:$PORT/v1 --lm-cache-off \
   --budget-calls $BUD --gepa-seed $SEED --reflection-model "local:$M@http://127.0.0.1:$PORT/v1" \
   --run-tag $TAG --eval-threads 32 --test-passes 5 --max-tokens 24000 2>&1 | tail -20 | tee -a $L
f=runs_paperexact/$BENCH/Qwen3-8B/${ARM}_${TAG}/result.json
[ -f "$f" ] && echo "ARTIFACT ok: $(python3 -c "import json;j=json.load(open('$f'));print(j.get('seed_test'),j.get('best_test'),j.get('budget_calls'))")" | tee -a $L || echo "ARTIFACT MISSING" | tee -a $L
CORE=$(for p in $(pgrep -f "VLLM::EngineCore"); do pp=$(ps -o ppid= -p $p 2>/dev/null|tr -d " "); [ "$pp" = "$VP" ] && echo $p; done)
kill $VP $CORE 2>/dev/null
echo "$(date -u +%FT%TZ) $TAG COMPLETE" | tee -a $L
