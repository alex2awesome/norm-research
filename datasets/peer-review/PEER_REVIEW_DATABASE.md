# Peer Review PDF Database

**File:** `peer_review_pdfs.db` (SQLite3, ~13 GB uncompressed, ~1.9 GB gzipped)

## Overview

A structured database of academic peer review data with full paper text extracted from PDFs, review text, and accept/reject decisions. Covers 7 venues across ML, NLP, and life sciences from 2020-2026.

## Coverage

| Venue | Years | Papers | Reviews | PDFs | Accept/Reject |
|-------|-------|--------|---------|------|---------------|
| ICLR | 2020-2025 | ~27,800 | ~100K | ~28K (10K+ revisions) | Yes |
| NeurIPS | 2021-2025 | ~18,800 | ~70K | ~13K | Yes |
| ICML | 2023-2025 | ~7,900 | ~28K | ~6K | Yes |
| COLM | 2024-2025 | ~720 | ~2.7K | ~700 | Yes |
| TMLR | Rolling | ~6,400 | ~23K | ~12K (6K+ revisions) | Yes |
| EMNLP | 2023-2025 | ~6,000 | ~21K | ~6K | Yes |
| eLife | 2022-2026 | ~4,300 | ~15K | ~8K (multi-version) | Assessment only |

**Totals:**
- **~70,000 papers**
- **259,326 reviews**
- **75,477 PDF versions** (49,052 original submissions + 26,425 revisions)
- **22,625 papers with multiple PDF versions** (submission + camera-ready)
- **42,168 accepted / 22,899 rejected** (papers with known decisions)

## Tables

### `papers`
One row per paper.

| Column | Type | Description |
|--------|------|-------------|
| `paper_id` | TEXT PK | Unique ID (OpenReview forum ID or `elife_XXXXX`) |
| `source` | TEXT | `openreview` or `elife` |
| `venue` | TEXT | `ICLR`, `NEURIPS`, `ICML`, `COLM`, `TMLR`, `EMNLP`, `eLife` |
| `venue_id` | TEXT | Full venue ID (e.g., `ICLR.cc/2024/Conference`) |
| `year` | INT | Submission/publication year |
| `title` | TEXT | Paper title |
| `abstract` | TEXT | Paper abstract |
| `authors` | TEXT | JSON array of author names |
| `keywords` | TEXT | JSON array of keywords |
| `decision` | TEXT | Raw decision text (e.g., "Accept (Poster)", "Reject") |
| `accepted` | INT | `1` = accepted, `0` = rejected, `NULL` = unknown |
| `num_reviews` | INT | Number of reviews for this paper |

### `reviews`
One row per review (259K rows).

| Column | Type | Description |
|--------|------|-------------|
| `review_id` | INT PK | Auto-increment |
| `paper_id` | TEXT FK | Links to `papers.paper_id` |
| `review_text` | TEXT | Full review text (may include structured sections) |
| `is_meta_review` | INT | `1` if area chair / meta-review / eLife assessment |
| `score` | TEXT | Reviewer rating (venue-specific scale) |
| `confidence` | TEXT | Reviewer confidence score |
| `recommendation` | TEXT | Reviewer recommendation |
| `reviewer_name` | TEXT | Reviewer name (eLife only; OpenReview is anonymous) |

### `pdf_versions`
One row per extracted PDF (75K rows).

| Column | Type | Description |
|--------|------|-------------|
| `id` | INT PK | Auto-increment |
| `paper_id` | TEXT FK | Links to `papers.paper_id` |
| `version` | INT | `0` = original submission, `1`+ = revisions/camera-ready |
| `pdf_filename` | TEXT | Original PDF filename |
| `full_text` | TEXT | Full extracted text from PDF (pymupdf) |
| `sections` | TEXT | JSON dict mapping section headers to text (e.g., `{"introduction": "...", "methods": "..."}`) |
| `page_count` | INT | Number of pages in PDF |
| `char_count` | INT | Character count of extracted text |

## Indexes

- `idx_reviews_paper` — `reviews(paper_id)`
- `idx_pdf_paper` — `pdf_versions(paper_id)`
- `idx_pdf_paper_ver` — `pdf_versions(paper_id, version)`
- `idx_papers_venue` — `papers(venue, year)`
- `idx_papers_accepted` — `papers(accepted)`

## Example Queries

```sql
-- Count papers by venue and year
SELECT venue, year, COUNT(*) as n,
       SUM(accepted) as accepted, COUNT(*) - SUM(accepted) as rejected
FROM papers WHERE accepted IS NOT NULL
GROUP BY venue, year ORDER BY venue, year;

-- Get paper text + reviews for ICLR 2025
SELECT p.title, p.accepted, v.full_text, r.review_text, r.score
FROM papers p
JOIN pdf_versions v ON p.paper_id = v.paper_id AND v.version = 0
JOIN reviews r ON p.paper_id = r.paper_id AND r.is_meta_review = 0
WHERE p.venue = 'ICLR' AND p.year = 2025
LIMIT 10;

-- Find papers with both submission and revision PDFs
SELECT p.paper_id, p.title, p.venue, p.accepted,
       COUNT(*) as num_versions,
       MIN(v.char_count) as orig_chars, MAX(v.char_count) as final_chars
FROM papers p
JOIN pdf_versions v ON p.paper_id = v.paper_id
GROUP BY p.paper_id
HAVING num_versions > 1
ORDER BY num_versions DESC;

-- Compare text length between accepted and rejected
SELECT p.accepted,
       AVG(v.char_count) as avg_chars,
       AVG(v.page_count) as avg_pages,
       COUNT(*) as n
FROM papers p
JOIN pdf_versions v ON p.paper_id = v.paper_id AND v.version = 0
WHERE p.accepted IS NOT NULL
GROUP BY p.accepted;

-- Extract specific sections (e.g., all introductions)
SELECT p.paper_id, p.title,
       json_extract(v.sections, '$.introduction') as introduction
FROM papers p
JOIN pdf_versions v ON p.paper_id = v.paper_id AND v.version = 0
WHERE p.venue = 'NEURIPS' AND p.year = 2024
AND json_extract(v.sections, '$.introduction') IS NOT NULL
LIMIT 5;
```

## Data Sources

- **OpenReview** — crawled via `openreview-py` API (v1 + v2). Reviews, decisions, and PDFs fetched directly. Revision PDFs downloaded via Edit API attachment references.
- **eLife** — crawled via eLife public REST API. PDFs downloaded from `elifesciences/enhanced-preprints-data` GitHub repo. Text parsed with pymupdf.

## Notes

- PDF text extraction uses **pymupdf (fitz)**. Some PDFs (scanned, image-heavy) may have incomplete text.
- Section parsing is heuristic — looks for standard headers (Introduction, Methods, Results, etc.). The `sections` JSON may have `"preamble"` for text before the first recognized header.
- eLife does not use binary accept/reject. The `decision` field contains the eLife assessment text. `accepted` is NULL for eLife papers.
- TMLR is a rolling journal — `year` may be NULL for some entries.
- The `OPENREVIEW` venue with NULL year corresponds to TMLR papers.

## Created

March 2026. Data collected from OpenReview and eLife APIs.
