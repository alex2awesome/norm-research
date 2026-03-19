# Running Research Notes

Key decisions and context for dataset processing and modeling.

## Creative Writing (LitBench)
- **Source**: `LitBench-Train.csv` — preference pairs (chosen/rejected stories with upvotes) from r/WritingPrompts
- **Binarization**: Combined chosen + rejected stories into single rows. `judgement = 1` if `upvotes > 100`, else `0`.
- **Output**: `datasets/creative-writing/litbench-to-train.csv.gz` — 87,654 rows (27,393 positive, 60,261 negative)
- **Processing notebook**: `notebooks/2026-02-17__data-processing.ipynb`

## Humor (New Yorker Caption Contest)
- **Source**: `newyorker_caption_ratings.csv.gz` — 2.2M Turker ratings across 362 contests
- **Scale**: Mean rating 1–3 (1=not funny, 2=somewhat funny, 3=funny). Extremely skewed: 90% of captions rated 1.0–1.4.
- **Binarization decision**: `judgement = 1` if `mean >= 1.3`, else `0`. This splits roughly at the median.
- **Note**: Even contest winners average only ~1.94. The `rank` column (0=winner, editor-selected) is a separate signal from crowd ratings.
- **TODO**: Preprocessing script not yet written.

## Press Releases
- **Source**: Press releases + news article coverage mappings. Filtered out XBRL data dumps, duplicates, and certain companies (go_factset).
- **Binarization**: `judgement = True` if the press release was picked up by a top-domain news outlet, else `False`.
- **Output**: `datasets/press-releases/press_release_modeling_dataset.csv.gz` — 128,131 rows (53,780 positive, 74,351 negative)
- **Processing notebook**: `notebooks/2026-02-17__data-processing.ipynb`

## Grant Funding
- **NIH RePORTER** (`datasets/grant-funding/nih_exporter/`): 2.8 GB of compressed ZIPs (1985–2024). Metadata + abstracts only — no proposal text, no accept/reject labels. Not useful for modeling as-is.
- **Open Grants** (`datasets/grant-funding/open-source-grants/processed/`): ~12 MB. Small set of voluntarily shared proposals with labeled outcomes.

## Peer Review

Massive multi-venue peer review corpus assembled from OpenReview API, eLife API, and F1000Research XML corpus. All data in `datasets/peer-review/`.

### OpenReview Conferences & Journals

Crawler: `PeerRead/code/data_prepare/crawler/openreview_crawl.py`. Requires `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` env vars (account: spangher@usc.edu). Supports year-based conferences and rolling journals (TMLR).

**ICLR 2018–2025** — `PeerRead/data/openreview/iclr/{year}/reviews/*.json`
The best single source: all submissions (accepted + rejected) with full review text, making it the only major AI venue with complete reject coverage.

| Year | Papers | Reviews | Accepted | Rejected | Notes |
|------|--------|---------|----------|----------|-------|
| 2018 | 935 | 2,784 | 337 | 496 | v1 API (Blind_Submission) |
| 2019 | 1,419 | 4,332 | 502 | 917 | |
| 2020 | 2,213 | 6,721 | 687 | 1,526 | |
| 2021 | 2,594 | 10,022 | 859 | 1,735 | |
| 2022 | 2,617 | 10,202 | 1,094 | 1,523 | Re-crawled 2026-03-18 (v1 fallback needed) |
| 2023 | 3,792 | 14,335 | 1,573 | 2,219 | Re-crawled 2026-03-18 (v1 fallback needed) |
| 2024 | 7,404 | 28,028 | 2,261 | 3,519 | v2 API native |
| 2025 | 11,672 | 46,748 | 3,708 | 5,019 | v2 API native |
| **Total** | **32,646** | **123,172** | **11,021** | **16,954** | **33.8% acceptance rate** |

**NeurIPS 2021–2025** — `PeerRead/data/openreview/neurips/{year}/reviews/*.json`
Heavily skewed toward accepted papers (~95% of visible submissions are accepted). NeurIPS has ~25% real acceptance rate, but rejected authors must opt-in to publish.

| Year | Papers | Reviews | Accepted | Rejected |
|------|--------|---------|----------|----------|
| 2021 | 2,768 | 10,736 | 2,632 | 136 |
| 2022 | 2,824 | 10,330 | 2,671 | 153 |
| 2023 | 3,395 | 15,175 | 3,218 | 177 |
| 2024 | 4,237 | 16,648 | 4,035 | 202 |
| 2025 | 5,540 | 22,369 | 5,286 | 254 |
| **Total** | **18,764** | **75,258** | **17,842** | **922** |

**ICML 2023–2025** — `PeerRead/data/openreview/icml/{year}/reviews/*.json`
Only 2025 has review text. 2023–2024 are accepted papers only with metadata (OpenReview doesn't expose reviews for those years).

| Year | Papers | Reviews | Accepted | Rejected | Notes |
|------|--------|---------|----------|----------|-------|
| 2023 | 1,828 | 0 | 1,828 | 0 | Metadata only, no review text |
| 2024 | 2,610 | 3 | 2,610 | 0 | Metadata only (1 paper has reviews) |
| 2025 | 3,422 | 13,102 | 3,260 | 162 | Full data |
| **Total** | **7,860** | **13,105** | **7,698** | **162** | |

**TMLR (rolling journal)** — `PeerRead/data/openreview/tmlr/reviews/*.json`
Excellent source: rolling journal with both accepted and rejected papers, full review text. Uses different OpenReview invitation patterns (`/-/Review` instead of `/-/Official_Review`, `directReplies` instead of `replies`).

| Papers | Reviews | With Reviews | Accepted | Rejected | Under Review |
|--------|---------|-------------|----------|----------|-------------|
| 6,333 | 17,858 | 5,751 (91%) | 3,716 | 1,688 | 929 |

**COLM 2024–2025** — `PeerRead/data/openreview/colm/{year}/reviews/*.json`
New conference (Conference on Language Modeling). Only accepted papers are public.

| Year | Papers | Reviews | Accepted | Rejected/Withdrawn |
|------|--------|---------|----------|--------------------|
| 2024 | 303 | 1,149 | 299 | 4 |
| 2025 | 418 | 1,586 | 418 | 0 |
| **Total** | **721** | **2,735** | **717** | **4** |

### OpenReview Totals

| | Papers | Reviews | Accept | Reject |
|---|--------|---------|--------|--------|
| **All OpenReview** | **66,324** | **232,128** | **40,994** | **19,730** |

### Non-OpenReview Sources

**eLife** — `elife/data/*.json`
Post-2023 "reviewed preprint" model: no binary accept/reject. Instead, each preprint gets an eLife Assessment with significance (e.g., "important", "valuable") and strength (e.g., "compelling", "solid", "incomplete") labels, plus 2–3 public reviews. Many also have author responses. Fetched via public REST API (`api.elifesciences.org`), no API key needed.

Fetcher: `elife/fetch_elife_reviews.py`

| Preprints | Reviews | With Assessment | With Author Response | Domain |
|-----------|---------|-----------------|---------------------|--------|
| ~4,286 (fetching) | ~10K est. | ~100% | ~80% | Life sciences (neuroscience, cell biology, genetics, etc.) |

As of snapshot: 1,252 fetched, 3,183 reviews, 1,246 with reviews, 1,067 with author response. Fetch still running.

**F1000Research** — `f1000research/data/*.json`
Open-access multidisciplinary journal with post-publication peer review. All reviews are public with named reviewers (open review), reviewer affiliations, and ORCIDs. Uses a 3-level decision system: Approved / Approved with Reservations / Not Approved. Multiple article versions supported (latest version fetched). Reviews from all versions included in each article's JSON.

Fetcher: `f1000research/fetch_f1000_reviews.py`. Source XML URLs in `f1000research/xml_urls_unique.txt` (17,288 article-version URLs → 11,369 unique articles).

| Articles | Reviews | Approved | Approved w/ Reservations | Not Approved | Domain |
|----------|---------|----------|--------------------------|-------------|--------|
| ~11,369 (fetching) | ~30K est. | ~51% | ~39% | ~10% | Multidisciplinary (biology-heavy) |

As of snapshot: 987 fetched, 2,946 reviews. Recommendation breakdown: 1,489 Approved, 1,155 Approved w/ Reservations, 302 Not Approved. Fetch still running.

**ORB Dataset (SciPost + workshops)** — `orb-dataset/extracted/*.json`
Extracted non-ICLR submissions from the ORB Zenodo dataset. **Metadata and decisions only — no review text** (review text is in a separate 883MB pickle file we didn't download, and would overlap with our existing OpenReview data for AI venues).

| Source | Papers | Notes |
|--------|--------|-------|
| SciPost Physics | 1,737 | Physics journals; decisions include "In refereeing", "Accepted", etc. |
| SciPost Physics Proceedings | 708 | |
| SciPost Physics Core | 190 | |
| SciPost Lecture Notes + Codebases | 110 | |
| ML Reproducibility Challenge | 157 | 2020 + 2021 |
| MIDL (Medical Imaging) | 331 | 2018–2023 |
| Various NeurIPS/ICML/ACL workshops | ~2,218 | 110 total venues |
| **Total non-ICLR** | **5,451** | **Metadata only** |

### PeerRead Legacy Data

Pre-existing data in `PeerRead/data/` (not from our crawler):

| Source | Papers | Notes |
|--------|--------|-------|
| ACL 2017 | 137 | Train/dev/test splits, with reviews + parsed PDFs |
| CoNLL 2016 | 22 | Very small |
| ICLR 2017 | 427 | Legacy format, train/dev/test splits |
| arXiv CS (2007–2017) | 10,609 | ai/cl/lg categories, no reviews (citation-based labels) |
| NeurIPS 2013–2017 | ~2,420 | Accepted only (proceedings) |

### Known Limitations & Caveats

1. **NeurIPS/ICML reject bias**: Only ~5% of visible NeurIPS papers are rejected (opt-in). ICML 2023–2024 show accepted papers only. This makes accept/reject prediction trivially easy for these venues without careful handling.
2. **ICLR is the gold standard** for reject coverage: all submissions are published on OpenReview regardless of outcome.
3. **eLife has no accept/reject**: The post-2023 model publishes assessments for all reviewed preprints. Could use significance/strength as graded labels instead.
4. **F1000Research** publishes all submissions then reviews post-publication: "Not Approved" is the closest analog to rejection (~10% of reviews).
5. **ICLR 2022–2023 v1 fallback**: The v2 OpenReview API returns submissions but empty `details="replies"` for these years. The crawler falls back to v1 per-paper forum fetching (slower but gets reviews). Fixed 2026-03-18.
6. **TMLR invitation patterns** differ from conferences: uses `/-/Review` (not `/-/Official_Review`), `directReplies` (not `replies`), and `recommendation` (not `decision`).
7. **COLM** only publishes accepted papers' reviews.

### Grand Total (with review text)

| Source | Papers | Reviews | Domain |
|--------|--------|---------|--------|
| ICLR | 32,646 | 123,172 | AI/ML |
| NeurIPS | 18,764 | 75,258 | AI/ML |
| ICML 2025 | 3,422 | 13,102 | AI/ML |
| TMLR | 6,333 | 17,858 | AI/ML |
| COLM | 721 | 2,735 | AI/ML |
| eLife | ~4,286 | ~10K | Life sciences |
| F1000Research | ~11,369 | ~30K | Multidisciplinary |
| **Grand Total** | **~77,500** | **~272,000** | |

- **TODO**: Build unified modeling CSVs from raw JSONs.
- **TODO**: Run remaining eLife + F1000 fetches to completion (background processes still running as of 2026-03-18).
