# Human-norm extraction inventory — CORRECTED 2026-06-14/15

**Goal (user, 2026-06-14)**: "For every dataset, have a full superset of all human norms that were explicitly given during review, comment or feedback processes that we haven't directly incorporated into the label y. These will be our anchors for metrics m."

**Correction note (2026-06-15)**: The original snapshot under-counted by treating each "task" as monolithic. Several tasks have multiple commentary-bearing component corpora; the llama_norm_extraction pipeline has only been pointed at a subset of them. Re-audit below.

## Verified facts re: corruption

The `extracted_qwen.jsonl.gz` files were written by `run_sk3_batch.py` using `gzip.open(..., "at")` (append-text). For long-running writers this produces concatenated deflate frames whose dictionaries don't share state — readers stop at the first inconsistent boundary, so the **readable line count is a lower bound** on what was actually written. The raw source corpora (parquet/csv.gz/jsonl) are untouched and intact. Recovery is straightforward: the runner is resumable (dedups by `unit_id`) and a fresh chunked writer (one file per N batches) avoids future occurrences. The bug is documented in `feedback_no_long_running_gzip_append`.

## Part 1 — What we already have, BY COMPONENT (not by task)

### peer_review — single component

| Component | Source | Source rows | Extracted | Coverage | State |
|---|---|---:|---:|---:|---|
| OpenReview/eLife/F1000/COLM/ACL reviews | `data/peer_review/peer_review_unified.parquet` (248 MB) | ~170,773 (per prior monitor) | 88,541 readable | **52%** | corrupted; runner not active |

### code_review — **3 components** (PRs / CR.SE / competition editorials)

| Component | Source (sk3) | Source rows | Extracted | Coverage | State |
|---|---|---:|---:|---:|---|
| GitHub PR comments | `data/code_review/code_review_pr_aggregated.csv.gz` (237 MB) | 141,644 | 92,997 readable | **66%** | corrupted; recoverable |
| **CodeReview.SE** | `datasets/codereview_se/posts.parquet` (307 MB); balanced pool `crse_v2_propensity_balanced.csv.gz` (71,510 rows); raw_dump has Posts.xml only (897 MB) — **no Comments.xml in dump** | ~80K balanced pool / 71,510 propensity-balanced | **0** | **NO llama_norm_extraction config** (only a separate claims/V/A pilot at `outputs/crse_claims_pilot_v2/`) |
| **Competition editorials** (LC + CF + CC) | `datasets/competition_unified/editorials.parquet` (58 MB) + `editorials_code_extracted_sectioned.parquet`; also `leetcode_editorials/`, `codeforces_delta/editorials_rendered_extracted.parquet` | ~ tens of thousands | **0** | **NO config** — editorials are pedagogical ("how to solve"), softer norm density |

### humor — multi-component, single config (`humor_multi`)

| Component | Bundled into source | State |
|---|---|---|
| reddit r/jokes 1M | `data/humor/standup_multi/filtered_threads.jsonl` rows; threads aggregate StandUpWorkshop / standupshots / StandUpComedy / AST forum / McSweeney's rejections / r/StandUpWorkshop comment trees | merged source = **49,762 threads**, extracted **27,366** (corrupted) → **55%** |

### press_releases — single component

| Component | Source | Source rows | Extracted | Coverage | State |
|---|---|---:|---:|---:|---|
| PR ↔ article editorial-summary pairs | `data/press_releases/pr_article_pairs_full.jsonl` | 100,000 | 82,600 readable | **83%** | corrupted; near-complete |

### notice_and_comment — 2 components (v1 + v2 backfill)

| Component | Source | Source rows | Extracted | Coverage |
|---|---|---:|---:|---:|
| v1 RTC sections | `rtc_sections.parquet` | 3,642 | 3,644 | **100%** ✓ |
| v2 backfill RTC | `rtc_sections_backfill.parquet` | 7,902 | 7,902 | **100%** ✓ |

### math — **2 components, only one extracted**

| Component | Source | State |
|---|---|---|
| mathlib PR review threads | `datasets/math/mathlib/thread_norms.jsonl` (86,001 lines, separate `extract_thread_norms_sk3.py` pipeline) | **DONE** ✓ |
| **Math.StackExchange comments-on-answers** | `datasets/math/stackexchange/` — has structural lint features, preference pairs, V/A verification eval. **No normative norm extraction** | **0** — only lint features & a V/A pilot exist |

### patents — single ad-hoc pipeline (separate from llama_norm_extraction)

| Component | Source | State |
|---|---|---|
| USPTO Office Actions (OARD) | `datasets/patents/processed/office_actions_v3/` + `office_actions_bulk_rest/` (many JSONL files) | task #112 in progress |

### Tasks with NO norm extraction (and no config)

| Task | Why |
|---|---|
| creative_writing | WP comment trees are SCRAPED on sk3 at `data/creative_writing/comments/raw/` (per-month JSONLs) but **no config** + no aggregated source built yet |
| legal_outcome_prediction | no config; this is the bucket the 7 integration tasks belong to |
| news_homepages | label is mechanical (spatial layout) — no human commentary corpus |

## Part 2 — Gaps from your new-datasets table (integration backlog)

These remain the 7 integration tasks (#132–#138) from yesterday's plan, still valid:

| # | Source | What's there now | Action |
|---|---|---|---|
| 132 | r/supremecourt | balanced built/train_v2_authgroup.csv.gz (230K rows; 60K balanced) | new config + bulk |
| 133 | Law.SE Comments.xml | full raw_dump on sk3 (8 XML files) | parse + config + bulk |
| 134 | CourtListener opinions | `.part` (download incomplete); 45 samples | finish download + config |
| 135 | OpenReview oral_spotlight + best_papers | labels exist; reviews already in unified_reviews | **post-hoc subset of peer_review extraction** — needs PR repair first |
| 136 | LegalAdviceUK + offtopic | 490M + 122M comments on sk3 | new config + subsample 50K |
| 137 | RoyalRoad reviews | only chapters scraped | new scraper + config |
| 138 | humor contest blurbs | sparse, currently filtered | audit; likely skip |

**Plus newly-surfaced gaps from this re-audit (5 more):**

| # | Source | Why it matters |
|---|---|---|
| **NEW** | **CR.SE Comments.xml** (request re-download of full SE dump — Comments.xml + PostHistory.xml + Votes.xml) | CR.SE answers + comments are direct critiques; covers the SE leg of the code_review trifecta |
| **NEW** | **CR.SE answers** (existing balanced pool) | even without Comments.xml, the answer prose itself is norm-bearing ("use Optional", "extract method", "this is O(n²)") |
| **NEW** | **Competition editorials** (LC + CF + CC) | pedagogical norm-bearing ("greedy fails because…", "use sliding window when…") |
| **NEW** | **Math.SE comments-on-answers** (from Math.SE dump's Comments.xml — task #113 still pending) | parallel to Law.SE: critiques of math-answer quality |
| **NEW** | **WritingPrompts comments** (scraped on sk3, not yet aggregated) | reader feedback on writing prompts; build aggregated source + config |

## Part 3 — Corrected overall scorecard

| Task | Components | Norm-extracted components | Lines (readable) | Coverage vs verified-source |
|---|---:|---:|---:|---|
| peer_review | 1 | 1 (corrupt) | 88,541 | 52% |
| code_review | **3** | **1 of 3** (corrupt) | 92,997 | **22%** of total commentary corpus (PRs only, partial) |
| humor | 1 (merged) | 1 (corrupt) | 27,366 | 55% |
| press_releases | 1 | 1 (corrupt) | 82,600 | 83% |
| notice_and_comment | 2 | 2 | 3,644 + 7,902 = 11,546 | 100% |
| math | 2 | 1 (mathlib only) | 86,001 | ~50% (Math.SE not done) |
| patents | 1 | 1 (in progress) | many files | partial |
| creative_writing | scrape only | 0 | 0 | 0% |
| legal_outcome | 0 (all new) | 0 | 0 | 0% |
| news_homepages | — | n/a | — | not a norm corpus |

**Net total**: ~390K rows of human norms extracted; **~5 components** still uncovered within tasks I had previously called "done" + 7 new corpora; if all gaps filled, total norm-extraction corpus would be roughly **1.0–1.2M extractions**.

## Part 4 — Revised priority order (after this re-audit + your "code_review = 3 components" correction)

1. **Repair peer_review + code_review_PR** gzip — peel readable lines, switch to chunked writer, resume. ~82K PR rows + ~50K CR rows still owed.
2. **CR.SE** — new config; bulk on existing 71,510 balanced pool. Re-download SE dump in parallel if we want Comments.xml.
3. **reddit_supremecourt** (#132).
4. **Law.SE Comments.xml** (#133).
5. **OpenReview oral/spotlight + best_papers subset** (#135) — no new bulk; runs after PR repair.
6. **Competition editorials** — moderate priority; pedagogical not adversarial.
7. **Math.SE Comments.xml** (#113) + **WritingPrompts comments aggregation + extraction**.
8. **LegalAdviceUK + offtopic** (#136), **CourtListener opinions** (#134), **RoyalRoad reviews** (#137), **humor contest blurbs** (#138).
