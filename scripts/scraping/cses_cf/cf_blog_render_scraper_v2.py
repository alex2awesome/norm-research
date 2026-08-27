"""
cf_blog_render_scraper_v2.py
============================

Phase-2 wrapper around the existing `cf_blog_render_scraper.py` that:
  - reads its blog_entry_id queue from contest_blog_map_v2.parquet (the
    output of the discovery pass) instead of contest_blog_map.parquet.
  - writes its progress / JSONL log to v2-suffixed paths so it cannot
    collide with the v1 instance running as PID 3634835.
  - runs with longer per-request delays (6-8 s jitter) so the combined
    rate against codeforces.com from two concurrent Camoufox instances
    stays polite.

Rendered HTML files share the same `rendered_html/blog_{id}.html` namespace
as the v1 instance — this is intentional (and safe) because:
  - the v2 queue is built by deduping out every blog_id already in the v1
    map (see cf_discover_tutorials_v2.py), so concurrent writes to the
    same file are impossible.
  - the v1 `already_done()` resume check picks up anything either instance
    has finished, so each blog is rendered at most once.

CLI: same as the upstream script (--limit, --validate). No --ids needed
since the queue source is fixed.

APPEND-ONLY: never modifies any existing data file.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Import the upstream module and patch its globals before calling main().
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts/scraping/cses_cf")
import cf_blog_render_scraper as base  # noqa: E402

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/codeforces_delta")

# v2 queue + log paths (disjoint from v1).
base.CONTEST_BLOG_MAP = ROOT / "contest_blog_map_v2.parquet"
base.LOG_JSONL = ROOT / "render_scrape_v2_log.jsonl"
base.RUN_LOG = ROOT / "render_scrape_v2.out"
base.PROGRESS = ROOT / "render_scrape_v2_progress.json"

# Be politer than the v1 instance (which runs at 3-5 s) so the combined
# rate stays roughly in the same envelope.
base.DEFAULT_JITTER = (6.0, 8.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated blog_entry_ids to render (overrides map).")
    args = ap.parse_args()

    only_ids = None
    if args.ids:
        only_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

    asyncio.run(
        base.scrape_all(
            limit=args.limit,
            only_ids=only_ids,
            dry_validate=args.validate,
        )
    )


if __name__ == "__main__":
    main()
