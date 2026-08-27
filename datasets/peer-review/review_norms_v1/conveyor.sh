#!/bin/bash
# Review-norm extraction conveyor (Leg B full corpus, user 2026-07-09 "all reviews").
# Headless Max-plan Sonnet workers; atomic claims; snap + anchor validation; resume-safe.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
WORKER=$1
TPL=$(cat "$D/extract_prompt_template.txt")
log() { echo "[w$WORKER $(date +%H:%M:%S)] $*" >> "$D/logs/conveyor.log"; }
validate() {  # $1=input $2=output ; rc 0 = pass
python3 - "$1" "$2" <<'PY'
import json, sys
inp = {r["review_id"]: r["review_text"] for r in json.load(open(sys.argv[1]))}
try:
    lines = [json.loads(l) for l in open(sys.argv[2])]
except Exception:
    sys.exit(1)
if len(lines) < max(3, int(len(inp) * .8)):
    sys.exit(1)
q = [(l["review_id"], p["quote"]) for l in lines for p in l.get("passages", [])]
if not q:
    sys.exit(1)
snap = sum(1 for rid, quote in q if quote in inp.get(rid, ""))
if snap / len(q) < .85:
    sys.exit(1)
anch = [l for l in lines if str(l["review_id"]).startswith("ANCH")]
if len(anch) < 2 or any(len(a.get("passages", [])) < 2 for a in anch):
    sys.exit(1)
PY
}
fails=0
for dir in "$D/inputs" "$D/inputs_full"; do
  for f in "$dir"/batch_*.json; do
    name=$(basename "$f" .json)
    [ "$dir" = "$D/inputs_full" ] && name="full_$name"
    out="$D/outputs/$name.jsonl"
    [ -s "$out" ] && continue
    mkdir "$D/claims/$name" 2>/dev/null || continue
    log "claim $name"
    prompt="${TPL//__INPUT__/$f}"; prompt="${prompt//__OUTPUT__/$out}"
    claude -p "$prompt" --model sonnet --permission-mode acceptEdits --max-turns 30 \
      > "$D/logs/$name.out" 2>&1
    if [ -s "$out" ] && validate "$f" "$out"; then
      log "done $name ($(wc -l < "$out" | tr -d ' ') reviews)"
      fails=0
    else
      log "FAIL $name (rc=$? size=$(wc -c < "$out" 2>/dev/null || echo 0))"
      mv "$out" "$out.bad" 2>/dev/null
      rmdir "$D/claims/$name" 2>/dev/null
      fails=$((fails+1))
      [ $fails -ge 3 ] && { log "3 consecutive fails — backing off 15 min"; sleep 900; fails=0; }
      sleep 30
    fi
  done
done
log "queue drained"
