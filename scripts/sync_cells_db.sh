#!/bin/bash
# Bidirectional sync of the cells DB between local and sk3.
#
# Each judge partition has ONE owner machine (no merge conflicts):
#   judge=claude/                          → owned by local
#   judge=llama_bf16/                      → owned by sk3
#   judge=qwen_thinking_fp8/               → owned by sk3
#   judge=qwen_thinking_fp8_20x1/          → owned by sk3
#
# Strategy: rsync each partition FROM its owner TO the other side.
# That way the owner's version always wins; no conflicts possible.

set -e
LOCAL=/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_db/cells_v1
SK3=sk3:/lfs/skampere3/0/alexspan/norm-research/outputs/v2_db/cells_v1

echo "=== local → sk3 (claude partitions) ==="
rsync -av --include='*/' --include='*/judge=claude/**' --exclude='*' \
  "$LOCAL/" "$SK3/" 2>&1 | tail -5

echo ""
echo "=== sk3 → local (everything except claude) ==="
rsync -av \
  --include='*/' \
  --include='*/judge=llama_bf16/**' \
  --include='*/judge=llama_fp8/**' \
  --include='*/judge=llama_fp8_smoketest/**' \
  --include='*/judge=qwen_thinking_fp8/**' \
  --include='*/judge=qwen_thinking_fp8_20x1/**' \
  --include='*/judge=qwen_thinking_fp8_20x1_r2post/**' \
  --include='*/judge=qwen_thinking_v1/**' \
  --include='*/judge=qwen_thinking_v2/**' \
  --include='*/judge=qwen_thinking_v3/**' \
  --include='*/judge=qwen_nothink/**' \
  --include='*/judge=qwen_fp8_early/**' \
  --exclude='*' \
  "$SK3/" "$LOCAL/" 2>&1 | tail -5

echo ""
echo "=== local DB state ==="
find "$LOCAL" -name 'data.parquet' | sort | while read f; do
  rel=${f#$LOCAL/}
  size=$(du -h "$f" | cut -f1)
  echo "  $size  $rel"
done
