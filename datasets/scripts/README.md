# `datasets/scripts/` — shared rubric-corpus fetchers

This directory is **not a dataset**. It holds the shared, task-agnostic
fetcher scripts that feed every per-task `online-rubrics/` collection
pipeline (see e.g. `datasets/peer-review/online-rubrics/`,
`datasets/creative-writing/online-rubrics/`, etc.).

## Purpose

Each per-task `online-rubrics/` directory holds rubric-bearing source
material (submission guidelines, editorial criteria, prize/award rules,
review rubrics, historical canon, rejection corpora). These scripts are
the shared crawlers that populate `online-rubrics/raw/` and the shared
`urls-visited.csv` log per task.

## Conventions

- **Filename prefix encodes the wave** the file came from:
  - `wavee_` — Wayback Machine snapshots of canonical rubric URLs
  - `waveh{2,4,6,7}_` — live-web waves (h2 = canonical editions, h4/h6 = pre-1900 foundational texts, h7 = ML paper-quality rubrics)
  - `rej_`   — rejection-corpus wave (negative examples; query labels start with `rejection:`)
- **On-disk layout per task** (always under `datasets/<task>/online-rubrics/`):
  - `raw/<prefix><sha1_12>.<ext>` — fetched bytes
  - `urls-visited.csv` — shared cross-wave log (per-task seen-set; new fetches dedup against this)
  - `<prefix>_log.csv`, `<prefix>_seen.txt` — per-wave attempt log + dedup set
- **URL hash:** first 12 hex chars of `sha1(url)`. Same URL → same filename across reruns.
- **Politeness:** Wayback waves are throttled (≤4 workers, sleeps, retries); live waves use 8–10 threads, 6–16 MB caps, 25–90 s timeouts.

## Scripts inventory

| Script | Purpose | Output |
|---|---|---|
| `wavee_fetch.py` | Generic threaded URL-list fetcher (live web), `wavee_` prefix | `raw/wavee_<hash>.<ext>` + `urls-visited.csv` |
| `wavee_wayback_fetch.py` | CDX-driven Wayback snapshots (yearly strides, ~6 per URL) for hard-coded `TARGETS` per task | `raw/wavee_<hash>.html`, query=`wave_e` |
| `wavee_wayback_fetch_pass2.py` | Sequential polite retry for URLs missing after pass 1 (HTTP fallback) | same as above |
| `wavee_wayback_logging_from_files.py` | Offline: scan `raw/wavee_*.html` for Wayback markers and backfill `urls-visited.csv` (no network) | CSV rows only |
| `wavee_wayback_logging_recover.py` | Re-run CDX to map orphaned `raw/wavee_*.html` files back to their canonical URLs | CSV rows only |
| `waveh2_fetch.py` | Live-web fetcher for versioned editions of canonical rubrics | `raw/waveh2_<hash>.<ext>` |
| `waveh4_fetch.py` | Pre-1900 foundational texts (Gutenberg, IA, Wikisource, Perseus, BNF, HathiTrust) | `raw/waveh4_<hash>.<ext>` + `waveh4_log.csv` + `waveh4_seen.txt` |
| `waveh6_fetch.py` | Same shape as h4, separate wave bucket | `raw/waveh6_*` + `waveh6_log.csv` |
| `waveh7_fetch.py` | ML paper-quality rubrics (TMLR criteria, Bengio essay, Ng CS230, NeurIPS award announcements) | `raw/waveh7_*` + `waveh7_log.csv` |
| `rej_fetch.py` | Rejection-corpus fetcher; query label SHOULD start with `rejection:` | `raw/rej_<hash>.<ext>` |
| `scrape_reddit_arctic.py` | Generic Arctic Shift Reddit scraper (preserves full comment tree, monthly-sharded gzipped JSONL, resumable) | `<output_dir>/raw/{sub}_{kind}_{YYYYMM}.jsonl.gz` |

## How to use

All wave fetchers share the same CLI:

```bash
python datasets/scripts/wavee_fetch.py <task> <urls_file> <query_label>
python datasets/scripts/waveh7_fetch.py peer-review urls_waveh7_ml_paper_quality.txt waveh7:ml_paper_quality
python datasets/scripts/rej_fetch.py    creative-writing rejection_urls.txt        rejection:slush_pile
```

`<task>` resolves to `datasets/<task>/online-rubrics/`. URLs already in
that task's `urls-visited.csv` are skipped.

Reddit scraper (different shape):

```bash
python datasets/scripts/scrape_reddit_arctic.py \
  --subreddit WritingPrompts --output-dir /lfs/.../comments \
  --start-ts 2010-01-01 --end-ts now --kind comments --delay 1.0
```

## Related

- Per-task corpora live at `datasets/<task>/online-rubrics/` for
  `code-review`, `creative-writing`, `grant-funding`, `humor`,
  `legal-outcome-prediction`, `math-stackexchange`, `news-homepages`,
  `notice-and-comment`, `patents`, `peer-review`, `press-releases`.
- The fetchers here are the only writers to those `raw/` dirs and the
  shared `urls-visited.csv` logs.
