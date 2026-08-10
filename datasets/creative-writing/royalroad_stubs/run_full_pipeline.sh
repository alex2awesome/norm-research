#!/bin/bash
# Full RoyalRoad-stubs pipeline. Designed for nohup on sk3:
#   HOME=/lfs/skampere3/0/alexspan nohup bash run_full_pipeline.sh > logs/full_run.log 2>&1 &
# Every step is checkpoint-resumable; safe to re-run after a crash.
set -u
export HOME=/lfs/skampere3/0/alexspan
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
DIR=/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/royalroad_stubs
SCRIPTS="$DIR/scripts"
cd "$DIR"

echo "=== $(date) step 1: enumerate stubs ==="
$PY "$SCRIPTS/scrape_search_listings.py" --mode stubs --data-dir "$DIR"

echo "=== $(date) step 2: enumerate control pool (2000 pages = 40K) ==="
$PY "$SCRIPTS/scrape_search_listings.py" --mode controls --data-dir "$DIR" --max-pages 2000

echo "=== $(date) step 3: prioritize controls ==="
$PY "$SCRIPTS/prioritize_controls.py" --data-dir "$DIR" --k 8

echo "=== $(date) step 4: FIRST BATCH stub recovery (220) ==="
$PY "$SCRIPTS/recover_wayback.py" --side stub --data-dir "$DIR" --limit 220

echo "=== $(date) step 5: FIRST BATCH control recovery (220, priority order) ==="
$PY "$SCRIPTS/recover_wayback.py" --side control --data-dir "$DIR" \
    --order-file "$DIR/raw/control_priority.txt" --limit 220

echo "=== $(date) step 6: build first-batch dataset ==="
$PY "$SCRIPTS/build_dataset.py" --data-dir "$DIR"

echo "=== $(date) step 7: FULL stub recovery ==="
$PY "$SCRIPTS/recover_wayback.py" --side stub --data-dir "$DIR"

echo "=== $(date) step 8: FULL control recovery (priority order, cap 6000) ==="
$PY "$SCRIPTS/recover_wayback.py" --side control --data-dir "$DIR" \
    --order-file "$DIR/raw/control_priority.txt" --limit 6000

echo "=== $(date) step 9: build final dataset ==="
$PY "$SCRIPTS/build_dataset.py" --data-dir "$DIR"

echo "=== $(date) DONE ==="
