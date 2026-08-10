#!/usr/bin/env bash
# Usage: label_lc_python_2k_shard.sh <shard_idx>
# Reads shard_NN.json, runs claude --print --model sonnet, writes labels_NN.json
set -u
SHARD_IDX=$1
WORK=/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/lc_python_2k_work
SHARD=$(printf "%02d" "$SHARD_IDX")
IN="$WORK/shard_${SHARD}.json"
OUT="$WORK/labels_${SHARD}.json"
LOG="$WORK/log_${SHARD}.txt"

if [[ -f "$OUT" ]]; then
  echo "[$SHARD] already done"
  exit 0
fi

PROMPT='You are labeling pairs of (candidate Python code, editorial Python code) for the same LeetCode problem. The editorial is the canonical reference solution.

For each pair, label:
  1 = candidate is SIMILAR to editorial — same approach AND similar stylistic choices (variable naming, structure, idioms)
  0 = candidate is NOT SIMILAR — different approach OR very different style choices, even if approach matches

Be holistic: consider both algorithm choice AND surface style (naming, structure, comments, idioms).

Output a single JSON list (no preamble, no markdown fence). One object per pair, in the same order as input:
[{"pair_id": "<id>", "label": 0 or 1, "brief_reason": "<5-12 words>"}]

Pairs:
'

# Concatenate prompt + shard JSON, send to claude
{
  printf '%s' "$PROMPT"
  cat "$IN"
} | claude --print --model sonnet > "$OUT.raw" 2> "$LOG"

# Extract JSON list (strip any code fences just in case)
python3 - "$OUT.raw" "$OUT" <<'PY'
import sys, json, re
raw = open(sys.argv[1]).read()
# Strip ```json ... ``` fences
m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw, re.DOTALL)
if m:
    txt = m.group(1)
else:
    # find first [ ... last matching ]
    s = raw.find('[')
    e = raw.rfind(']')
    if s == -1 or e == -1:
        print("NO_JSON_FOUND", file=sys.stderr); sys.exit(2)
    txt = raw[s:e+1]
data = json.loads(txt)
with open(sys.argv[2], 'w') as f:
    json.dump(data, f)
print(f"shard {sys.argv[2]}: {len(data)} labels")
PY
ec=$?
if [[ $ec -ne 0 ]]; then
  echo "[$SHARD] parse failed (ec=$ec)"
  exit $ec
fi
echo "[$SHARD] done"
