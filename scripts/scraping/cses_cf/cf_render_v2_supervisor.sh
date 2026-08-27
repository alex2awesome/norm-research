#!/usr/bin/env bash
# cf_render_v2_supervisor.sh
# ==========================
#
# Keep running cf_blog_render_scraper_v2.py over the growing
# contest_blog_map_v2.parquet until both:
#   (a) the discovery process (cf_discover_tutorials_v2.py) has exited, AND
#   (b) the last render pass made no progress (todo == 0 after resume filter).
#
# Each pass:
#   - calls scrape_all() which reads contest_blog_map_v2.parquet, filters
#     blog ids that already have rendered_html/blog_{id}.html (>5KB), then
#     processes the remainder.
#   - exits cleanly when nothing to do.
#
# We sleep between passes so discovery can accumulate more rows. The v2
# render instance uses 6-8 s jitter (set in cf_blog_render_scraper_v2.py)
# to stay polite alongside the v1 instance (PID 3634835, 3-5 s jitter).

set -uo pipefail

export HOME=/lfs/skampere3/0/alexspan
ROOT=/lfs/skampere3/0/alexspan/norm-research
cd "$ROOT"

PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python3
RUN_LOG="$ROOT/datasets/codeforces_delta/render_v2_supervisor.out"
DISCOVERY_PROC=cf_discover_tutorials_v2

echo "[supervisor] starting at $(date -u +%FT%TZ)" >> "$RUN_LOG"

PASS=0
while true; do
    PASS=$((PASS+1))
    echo "[supervisor] pass=$PASS starting at $(date -u +%FT%TZ)" >> "$RUN_LOG"
    "$PY" scripts/scraping/cses_cf/cf_blog_render_scraper_v2.py \
        >> "$ROOT/datasets/codeforces_delta/render_scrape_v2.out" 2>&1
    echo "[supervisor] pass=$PASS finished at $(date -u +%FT%TZ)" >> "$RUN_LOG"

    DISCOVERY_RUNNING=0
    if pgrep -f "$DISCOVERY_PROC" >/dev/null 2>&1; then
        DISCOVERY_RUNNING=1
    fi

    # If discovery is done AND last pass had nothing to do, we're done.
    NOTHING_NEW=0
    if [ -f "$ROOT/datasets/codeforces_delta/render_scrape_v2_progress.json" ]; then
        FINAL=$("$PY" -c "import json; d=json.load(open('$ROOT/datasets/codeforces_delta/render_scrape_v2_progress.json')); print(d.get('total', 0))" 2>/dev/null || echo 0)
        if [ "$FINAL" = "0" ]; then
            NOTHING_NEW=1
        fi
    fi

    if [ "$DISCOVERY_RUNNING" = "0" ] && [ "$NOTHING_NEW" = "1" ]; then
        echo "[supervisor] discovery done and nothing new — exiting." >> "$RUN_LOG"
        break
    fi

    echo "[supervisor] sleeping 600s (discovery_running=$DISCOVERY_RUNNING nothing_new=$NOTHING_NEW)" >> "$RUN_LOG"
    sleep 600
done

echo "[supervisor] DONE at $(date -u +%FT%TZ)" >> "$RUN_LOG"
