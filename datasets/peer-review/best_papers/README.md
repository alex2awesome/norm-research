# Best Papers (acad-B2)

Best-paper awards in computer science (Jeff Huang's curated list) joined to
OpenAlex, with same-venue same-year y=0 candidate pools.

- **y=1**: won a best-paper award at the venue that year.
- **y=0 candidates**: published at the same venue, same year, no award.

## Pipeline

| step | script | output |
|---|---|---|
| 1. scrape | `scrape_best_papers.py` | `best_papers_awards.csv` |
| 2. join to OpenAlex | `join_openalex.py` | `best_papers_joined.csv` (+ `join_cache.jsonl`) |
| 3. finalize labels | `finalize_labels.py` | `best_papers_labels.csv` |
| 4. y=0 pools | `pull_venue_pools.py` | `pools/{venue}.parquet`, `sources_resolved.json`, `pool_coverage.csv` |

Source: https://jeffhuang.com/best_paper_awards/ — one table per year
(1996–2025), venue blocks via `th.category-name` rowspans.

Join: OpenAlex `title.search` filtered to publication_year ±1, with a full
`search` fallback (OpenAlex sometimes stores only the pre-colon part of a
title, e.g. CIKM'17 "Hike: A Hybrid ..." is stored as "Hike"). Candidates
scored by normalized-title `SequenceMatcher` ratio with pre-colon-equality
credit; accept ≥ 0.85. Most of the join actually ran through
`../openalex_citations/batch_title_join.py` (pipe-OR'd quoted phrases, ~15
titles per 10-credit search request) on skampere hosts, because OpenAlex's
free tier is now $1/day = 10,000 credits per IP and plain per-title searches
blew the budget; results were merged into `join_cache.jsonl`, which
`join_openalex.py` resumes from.

Text normalization everywhere (labels + pools): NFKC, curly→straight quotes,
en/em-dash→hyphen. `text` = `TITLE\n\nABSTRACT` (abstract decoded from
`abstract_inverted_index`).

## Venue → OpenAlex source resolution (the messy part)

OpenAlex conference mapping is fragmented: a MAG-era generic source per venue
(coverage dies ≈2021), per-year IEEE/ACM proceedings sources, and sometimes a
journal that carries the proceedings (VLDB → "Proceedings of the VLDB
Endowment", SIGMETRICS → POMACS, SIGMOD 2023+ → "Proceedings of the ACM on
Management of Data"). `pull_venue_pools.py` resolves each venue by the union
of (a) sources where the matched award papers actually live and (b) a
`display_name.search` sweep filtered by per-venue accept/reject regexes.
Every accepted source is logged in `sources_resolved.json`; per-award-year
pool sizes in `pool_coverage.csv` (`resolved` = pool ≥ 20 works). Treat
venue-years with `resolved=False` as missing, not as small venues.

## Key numbers (2026-06-12 build)

- **1,819 awards** scraped (verified == sum of venue-block rowspans), 32
  venues, 1996–2025. The page has grown well past the old ~700–900 estimate
  (recent years have 3–8 awards/venue).
- **OpenAlex join: 1,505/1,819 = 82.7%** (threshold 0.85 normalized-title
  similarity). 173 queried-but-unmatched; **141 never queried** — daily
  credit budgets ran out; run `../openalex_citations/finish_leftovers.sh`
  from a host with budget to top off (resumable). Abstract coverage among
  matched: **94.6%**. Manual inspection of 10 joins (DOI-verified against
  proceedings prefixes): 10/10 correct.
- **Pools**: 29 venue parquets, ~66 MB, on sk3. Per-award-year resolution in
  `pool_coverage.csv` (resolved = pool ≥ 20 same-year works):
  - good: AAAI 100%, VLDB 96%, IJCAI 85%, NSDI 82%, NeurIPS 77%, ICSE 70%,
    ICML 59%, SODA 59%, OSDI 50%
  - broken (OpenAlex simply lacks venue records): PLDI, PODS, SOSP, STOC,
    UIST, ISCA at 0%; ACL/CHI/CIKM/CVPR/FOCS/FSE/INFOCOM/MOBICOM/S&P/WWW
    < 15%; SIGCOMM and SIGGRAPH resolve to no usable source at all.
  - Famous sanity cases: GoogLeNet/ResNet (CVPR), XGBoost (KDD'16) and
    Neural Collaborative Filtering (WWW'17) have NO venue source in OpenAlex
    (sourceless or repository-only locations) — the brokenness is OpenAlex's,
    not the resolver's. Upgrade path: DBLP TOC membership + batched title
    join, as done for Task B (acad-C).
- **Pool usage**: pools contain ALL same-venue works incl. the winners —
  exclude `openalex_id`s present in `best_papers_labels.csv` when sampling
  y=0.
- One real trap documented in `pull_venue_pools.py`: award-derived sources
  must be regex-filtered or journal twins (CVPR→TPAMI, NeurIPS→JACM) and
  outright mismatches inject tens of thousands of foreign works. PACMPL is
  never accepted for PLDI (mixes POPL/OOPSLA/ICFP); POMACS (SIGMETRICS) and
  PACMMOD (SIGMOD) are venue-pure and accepted.

## sk3

`/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/best_papers/`
(labels + joined + awards + join cache + `pools/*.parquet` +
`pool_coverage.csv` + `sources_resolved.json`)
