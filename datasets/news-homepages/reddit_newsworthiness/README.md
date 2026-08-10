# Reddit Newsworthiness (news-C cell: lay crowd, revealed)

Lay-crowd revealed newsworthiness judgments from r/news + r/worldnews post scores
(Arctic Shift dumps, 2023-11-01 → 2026-06-09 effective). One row = one link-post
headline; the crowd's upvotes are the judgment.

**Label semantics — LAY crowd, revealed preference.** This is the crowd cell of the
taste taxonomy: anonymous Reddit voters deciding what news deserves attention.
Contrast with the `news_homepages` cell (EXPERT, revealed): editors deciding homepage
spatial layout. Same artifact type (news headline), different judge population.

## Design

- **X** = headline title only (presentation-normalized: HTML-unescaped ×2, NFKC,
  curly→straight quotes, dashes, ellipsis, collapsed whitespace; outlet boilerplate
  trailing segments like "... | CNN" stripped via a frequency≥30 + top-3-domain-share≥0.6
  rule, 15 segments, 1,088 titles touched). `raw_title` kept. The model never sees
  domain/url (kept as metadata columns only).
- **y** (`judgement`) = score percentile within (subreddit × UTC posting-day) cell,
  average-rank based: top quartile (pct ≥ 0.75) → 1, bottom quartile (pct ≤ 0.25) → 0,
  middle dropped. The huge score==1 tie block (never-voted and/or mod-removed posts,
  ~89% of r/news) mostly lands in the middle and is dropped; negatives are therefore
  predominantly actively-downvoted or clearly-ignored-relative-to-cell posts.
- Cells with <20 posts dropped (none in practice; both subs run ~300+ posts/day).
- Filters: over_18, removed/deleted/empty titles, self posts (nonempty selftext or
  no/reddit-internal url), posts <3 days old at dump time (immature scores).
- Dedup: earliest post kept per normalized-title or normalized-URL cluster within a
  rolling 3-day window (chains extend); 132,603 dup rows dropped (same story posted
  many times — also a train/test leakage channel).
- **Splits**: stable hash, `int(md5(post_id),16) % 10` → 0-7 train / 8 eval / 9 test.

## Deconfounding (mandatory, mirrors Math.SE v3.3 propensity-decile balancing)

Text-free propensity model (logistic regression) for y=1 on confounds only:
hour-of-day (24), weekday (7), log post-age-at-dump, top-100 domain one-hots + other,
log author post-count (karma-farming proxy), subreddit. Predicted propensity binned
into deciles; pos/neg exactly balanced within each decile by downsampling.

- **Propensity 5-fold CV AUC = 0.844** — strong confounding, dominated by domain
  (youtube/spam-blog domains → y=0 with coefs −4..−5; ukrinform/euromaidanpress → y=1)
  and subreddit. Domain is IN the propensity features (v3.3 philosophy: balance on it,
  don't show it to the model).
- After balancing, propensity AUC on the kept set = **0.508** (≈ chance, as intended).
- Labels before balancing: news 25,229 pos / 2,777 neg; worldnews 52,755 pos / 38,085 neg.
- After: news 2,868/2,621; worldnews 21,688/21,935 → 49,112 rows, exactly 50/50 within
  propensity deciles.

## Dataset-first protocol results

- TF-IDF/LR floor (train→test): **AUC 0.738** test / 0.735 eval
  (before boilerplate strip: 0.738 — stripping changed nothing measurable, kept anyway).
- Per-subreddit test AUC: news 0.658, worldnews 0.750.
- Top pos features: russian, hamas, bans, russia, finland, canada, hezbollah, iran,
  zelenskyy, climate, ukraine — geopolitics topics the crowd upvotes.
- Top neg features: why, live, exclusive, how, what, epstein, stabbing, crypto, https,
  eurovision, "war latest", petition — question-headline clickbait, live-feed/megathread
  formats (rule-violating in r/worldnews), crypto spam, URLs in titles.
- Title length vs label corr: **+0.033** (negligible; pos 78.8 vs neg 76.8 chars).

## Known residual biases

- **Topic confound**: topic is genuinely part of lay newsworthiness here, but it means
  the floor partly reflects topic preference (Russia/Ukraine ↑) not headline craft.
- **Score retrieval recency**: dumps carry NO `_meta` key (0/621,352 rows have
  `retrieved_2nd_on`), so the promised ~36h second retrieval cannot be verified;
  scores are as-of-dump (2026-06-12). Posts <3 days old were dropped as a guard.
- **Moderator-removal channel**: removed link posts are frozen near score 1; most fall
  in the dropped middle, but some negatives may be removed-then-downvoted posts
  (flair kept in `link_flair_text` for auditing, e.g. "Not Appropriate Subreddit").
- 144 duplicate texts (~0.3%) survive dedup — same headline reposted >3 days apart
  (treated as new events by design); a tiny cross-split leakage channel.
- worldnews dominates after balancing (89% of rows) because r/news negatives are scarce.

## Files

- sk3: `/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/reddit_newsworthiness/`
  - `built/{train,eval,test}.csv.gz` — 39,209 / 4,979 / 4,924 rows
  - `manifest.json`, `build_log.txt`, `build_reddit_newsworthiness.py`
- Raw dumps (never delete): `/lfs/skampere3/0/alexspan/norm-research/datasets/taxonomy_dumps/{news,worldnews}_posts.jsonl`
- Columns: `text, judgement, raw_title, score, percentile, num_comments, subreddit,
  domain, post_id, author, created_utc, link_flair_text, propensity`
