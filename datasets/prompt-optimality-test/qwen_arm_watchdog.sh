#!/usr/bin/env bash
# Crash-proof auto-restart watchdog for wedge-prone arms (litellm dead-socket bug, Qwen+GLM).
# NO set -e/-u. Every 5 min, per watched arm: if EXACTLY one proc exists, it is 0.0% CPU, and
# proposals.jsonl is stale >15 min, kill-by-PID + relaunch. Both conditions => never kills a
# slow eval. GLM arms need a live z.ai; Qwen arms need a live tunnel (checked before relaunch).
cd "$(dirname "$0")" || exit 0
LOG="runs_paperexact/qwen_watchdog.log"
QLM="openai/Qwen3-8B"; QBASE="http://127.0.0.1:8077/v1"
GBASE="https://api.z.ai/api/anthropic"

qwen_live() { timeout 8 curl -s -m 6 "$QBASE/models" >/dev/null 2>&1; }
zai_live() {
  timeout 40 python3 -c "
import json,urllib.request
k=open('$HOME/.z-ai-api-key-spangher.txt').read().strip()
r=urllib.request.Request('$GBASE/v1/messages',data=json.dumps({'model':'glm-4.7','max_tokens':8,'messages':[{'role':'user','content':'ok'}]}).encode(),headers={'x-api-key':k,'anthropic-version':'2023-06-01','content-type':'application/json'})
urllib.request.urlopen(r,timeout=35).read()" 2>/dev/null
}

# relaunch <bench> <lm-tag-dir> <full python arg string...>
relaunch_qwen() {  # bench
  local B="$1"
  nohup .venv/bin/python paperexact_arms.py "$B" --arm unitrecomb --task-lm "$QLM" \
    --api-base "$QBASE" --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 8000 \
    --max-units 96 --confirm-add-val --budget-calls 24000 \
    > "runs_paperexact/$B/Qwen3-8B/unitrecomb_v4_run.log" 2>&1 &
  echo "$(date -u +%FT%TZ) RELAUNCHED $B(qwen) v4 pid $!" >> "$LOG"
}
relaunch_glm() {  # bench keyfile
  local B="$1"; local KEY="$2"
  HF_DATASETS_TRUST_REMOTE_CODE=1 nohup .venv/bin/python paperexact_arms.py "$B" --arm unitrecomb \
    --task-lm anthropic/glm-5.2 --api-base "$GBASE" --api-key-file "$KEY" \
    --temperature 0.6 --top-p 0.95 --max-tokens 32000 --max-units 96 \
    > "runs_paperexact/$B/glm-5.2/unitrecomb_run.log" 2>&1 &
  echo "$(date -u +%FT%TZ) RELAUNCHED $B(glm) pid $!" >> "$LOG"
}

# check <bench> <lmdir> <provider> <extra-for-glm-keyfile>
check() {
  local B="$1"; local LMDIR="$2"; local PROV="$3"; local KEY="${4:-}"
  if [ "$PROV" = qwen ]; then qwen_live || return 0; else zai_live || return 0; fi
  local pids; pids=$(pgrep -f "paperexact_arms.py $B --arm unitrecomb" 2>/dev/null)
  local n; n=$(printf '%s\n' "$pids" | grep -c .)
  [ "$n" = "1" ] || return 0
  local cpu; cpu=$(ps -o %cpu= -p "$pids" 2>/dev/null | tr -d ' ')
  local f="runs_paperexact/$B/$LMDIR/unitrecomb/proposals.jsonl"
  local age=9999
  [ -f "$f" ] && age=$(( ( $(date +%s) - $(stat -f %m "$f" 2>/dev/null || echo 0) ) / 60 ))
  if [ "$cpu" = "0.0" ] && [ "$age" -gt 15 ]; then
    echo "$(date -u +%FT%TZ) $B/$LMDIR WEDGED pid $pids cpu $cpu age ${age}m -> restart" >> "$LOG"
    kill "$pids" 2>/dev/null; sleep 2
    if [ "$PROV" = qwen ]; then relaunch_qwen "$B"; else relaunch_glm "$B" "$KEY"; fi
  fi
}

echo "$(date -u +%FT%TZ) watchdog START (qwen+glm)" >> "$LOG"
while true; do
  check livebench Qwen3-8B qwen 2>>"$LOG"
  # ifbench moved to sk2 (v5, localhost vLLM on 8078 — no tunnel, no dead-socket treadmill);
  # local ifbench retired 2026-07-22 (user directive: push harder).
  check pupa      Qwen3-8B qwen 2>>"$LOG"  # pupa-GLM done; only pupa-Qwen (v4) remains
  check hotpot    glm-5.2  glm  "$HOME/.z-ai-api-key-spangher.txt" 2>>"$LOG"
  sleep 300
done
