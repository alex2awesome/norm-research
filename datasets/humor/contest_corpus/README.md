# Humor-Contest Label Corpus (humor A-cell, ordinal-within-pool)

Labeled artifacts from three long-running humor writing contests. Each row is
one entry that earned a place in a contest cycle's published results, with an
ordinal rank tier within that cycle's pool. Built 2026-06-12.

Final counts: **929 rows total, 513 with full text** — wergle_flomp 411
(378 with text, 2002-2025), to_hull_and_back 383 (labels only, 2014-2025),
erma_bombeck 135 (135 with text, 2010-2026).

## Output

`/.staging/norm-datasets/humor/contest_corpus/contest_corpus.jsonl` (staging
dir; synced to sk3 by the parent pipeline). One JSON object per row:

```
{contest, cycle, rank_tier, category, title, author, url, text, raw_text, text_available}
```

- `contest`: `wergle_flomp` | `to_hull_and_back` | `erma_bombeck`
- `cycle`: contest year (string)
- `rank_tier`: ordinal tier within the cycle (semantics per contest below)
- `category`: only for Erma Bombeck (e.g. `Humor - Global`); null otherwise
- `text`: normalized full text of the work (null if not freely posted)
- `raw_text`: unnormalized extraction (line breaks preserved for poems)
- `text_available`: bool

Per-source intermediates in the same staging dir: `wergle.jsonl`,
`hull.jsonl`, `erma.jsonl`. All fetched HTML is cached under
`html_cache/` (sha256-prefixed filenames); re-runs never re-fetch.

## Scripts (this dir)

| script | role |
|---|---|
| `fetch_util.py` | cached, rate-limited fetcher (1 req / 2 s / domain, academic UA) |
| `normalize.py` | shared text normalization (see below) |
| `scrape_wergle.py` | Wergle Flomp cycles + full poem texts |
| `scrape_hull.py` | To Hull And Back result lists (labels only) |
| `scrape_erma.py` | Erma Bombeck index + full essay texts (live + Wayback index for 2010-16) |
| `build_corpus.py` | merge + uniform re-normalization -> `contest_corpus.jsonl` |

Run order: the three scrapers (any order), then `build_corpus.py`.

## Sources & URL structures

### 1. Wergle Flomp Humor Poetry Contest (winningwriters.com), 2002-2025

- Cycle index: `https://winningwriters.com/our-contests/contest-archives`
- Cycle page: `.../contest-archives/wergle-flomp-humor-poetry-contest-{YYYY}`
  (tier `<h4>` headings followed by `Author, <a>Title</a>` rows; parsing
  stops at the "Contest Judge(s)" heading)
- Entry page: `https://winningwriters.com/past-winning-entries/{slug}` —
  full poem text after the `div.meta` block. Site emits unclosed `<br>` tags;
  parse with lxml and unwrap (not replace) `<br>` to keep mis-nested content.
- Tiers: `first_prize` > `second_prize` > `third_prize` >
  `honorable_mention` > `finalist` (tier set varies by year: 2002 had only
  first + HM; finalists listed 2006-2012; third prize absent some years).
- Full poem text available for essentially every listed entry.
- Editorial paragraphs on entry pages (illustration credits, "please enjoy
  this video", reprint notices) are filtered by pattern (`NOTE_PATTERNS`).
- Tombstone pages ("We regret this content is no longer available online",
  "Poem not displayed? Download the PDF") are detected
  (`TOMBSTONE_PATTERNS`) and yield `text_available=False` — EXCEPT pages
  with a `graphics/wergle/*.pdf` link, where the poem is recovered from the
  PDF via pypdf (2 poems: 2018 "The Swipe Sonnets", 2019 "Ancient Barbie").
  Title-echo pages (body = title only, poem missing) also yield no text.
- 33 entries without text: mostly the 2012 cycle (site removed ~21 of its
  poem pages) plus scattered tombstones in 2010-2025.
- Tier naming changed in 2024-25 to "Most Highly Commended"
  (`most_highly_commended`), replacing honorable mention.

Tom Howard contests on the same site (Tom Howard/Margaret Reid Poetry,
Tom Howard/John H. Reid Fiction & Essay) were checked and are NOT
humor-specific (categories are by form — verse/story/essay — not genre), so
they are excluded.

### 2. To Hull And Back humorous short story competition (christopherfielden.com), 2014-2025

- Landing page: `/to-hull-and-back-humorous-short-story-competition/`
  (note: the URL slug includes "humorous"; `to-hull-and-back-short-story-competition` 404s)
- Results: `.../results-{YYYY}/` for 2014-2019 (annual) and 2021, 2023, 2025
  (biennial). Consistent WordPress structure across years.
- Tiers: `first_prize` > `second_prize` > `third_prize` >
  `highly_commended` (3/yr) > `shortlist` (20/yr; includes the 6 winners,
  deduped to their higher tier, so 14 rows/yr) > `longlist` (20/yr, 2016+;
  disjoint from shortlist) > `special_mention` (~6-12/yr, most years;
  stories that didn't fit the contest but the judge wanted to flag).
- 2014-15 winner lines are `Title by Author`; 2016+ `Title, by Author`;
  list lines are `Author – Title`. Parsing is anchored on the author bio
  anchor links (`href="#AuthorName"`).
- **No full texts**: winning stories are published only in the paid To Hull
  And Back anthologies; `text_available=False` for all rows.

### 3. Erma Bombeck Writing Competition (Washington-Centerville Public Library), 2010-2026

- **WARNING**: `humorwriters.org` (the old Erma Bombeck Writers' Workshop
  domain referenced in older notes) has lapsed and is now casino spam. The
  competition is hosted by WCPL: `https://www.wclibrary.info/erma/winners/`.
- Index table: `td.erma_header` (year) / `td.erma_subheader` (category) /
  `td.erma_essay` (tier + author link).
- Detail page (full essay + title): `.../winners/detail/?id={N}&year={YYYY}&winner={W|H|F}`.
- The live index lists only 2018-2026 (biennial, same years as the Dayton
  workshop). Older detail pages are STILL SERVED live; we recover the
  2010-2016 index rows from a 2019 Wayback snapshot
  (`web.archive.org/web/20191112204602/...`) and fetch the essays live.
- Tiers per category-pool: `first_place` > `honorable_mention` (2/pool;
  more in some years) > `finalist` (2026 + 2018 only).
- Categories (= separate judging pools!): `Humor` and `Human Interest`, each
  split into `Global` and `Local Area` (Dayton-region residents). For
  humor-A-cell use, filter `category` startswith `Humor`; both kept per spec.
- Full essay text available for every row (~450-word limit per the rules).

## Ordinal-within-pool semantics

The comparison pool for ordinal modeling is `(contest, cycle)` — and for
Erma Bombeck `(contest, cycle, category)`, since Humor/Human-Interest ×
Global/Local are judged separately. Tiers are ordinal within a pool only;
prize money / tier counts changed across cycles, so do not compare tiers
across cycles or contests.

## Text normalization (CRITICAL)

Identical pipeline (`normalize.normalize_text`) applied to title/author/text
of ALL rows from ALL sites (standing rule: a 0.996 fake AUC was once caused
by typographic-quote leakage between sources):

1. `unicodedata.normalize("NFKC", s)` (also folds nbsp, ellipsis, ligatures)
2. curly quotes/apostrophes/guillemets/primes -> straight `'` and `"`
3. CRLF -> LF, per-line trailing-space strip (leading indent kept for poems)
4. collapse 3+ newlines -> 2 (stanza breaks preserved)
5. strip wrapper quotes when the whole string is one quoted span

`raw_text` keeps the unnormalized extraction. `build_corpus.py` re-derives
`text` from `raw_text` at merge time so per-scraper drift cannot occur.

## Known gaps

- **Hull**: no full texts (paid anthologies). Labels only.
- **Erma pre-2010**: cycles ~2001-2008 exist only as per-essay `.asp` pages
  in the Wayback Machine (e.g. `wclibrary.info/erma/2007globalhumor.asp`)
  with a different page structure; not collected.
- **Erma finalists**: only published for some cycles (2018, 2026).
- **Wergle**: a handful of very early entries are "Untitled (...)"; titles
  disambiguated by first-line excerpt as published by the site.
- **Wergle 2025 onward**: contest is ongoing; re-run scrapers to extend
  (cache makes re-runs cheap).
