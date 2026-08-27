# Verdict/curation source verification — humor + creative writing (2026-08-12)

Web-verified audit of proposed new "verdict" (binary accept/reject per item) and
"curation" (top-k from a pool) sources for the humor and creative-writing lanes.
Four parallel research agents; every claim below was checked against a live fetch
or flagged as unverifiable. Cross-referenced against what the repo already holds
(`datasets/humor/README.md`, `datasets/creative-writing/README.md`).

Verification questions per source: (1) scrapable/gettable? (2) volume?
(3) full universe — are BOTH the selected and non-selected items observable with text?

## Headline findings

- **Reedsy Prompts = best new source overall (CW curation, full universe).**
  367+ weekly contests at reedsy.com/creative-writing-prompts/contests/; each
  contest page lists all published entries full-text (1k–3k words), winner
  explicitly marked, shortlist labeled. ~20k–35k stories estimated. Plain HTML.
  Caveat: only ~40–70% of paid entries are approved+published, so the observable
  pool is the *approved* pool (moderation criteria unknown).
- **McSweeney's archive vs. "Best of" anthologies = best new humor curation.**
  Full archive scrapable with browser UA via /articles/archives?page=N
  (~1,458 pages ≈ 29,000 pieces, free full text) = the pool; three anthologies
  (Created in Darkness 2004; Best of Internet Tendency 2014 ~50 pieces; Keep
  Scrolling 2019) = curated top-k (~150–250 pieces, ~0.5–0.8% positive rate).
  TOCs not online — need manual transcription from the borrowable Internet
  Archive scan. robots.txt has Cloudflare `ai-train=no` rights signal.
- **Reddit dumps lane confirmed feasible.** Academic Torrents hosts full
  Pushshift+RaiderBDev dumps through 2025-12, incl. per-subreddit dumps (top
  40k subs) with `gilded`/`total_awards_received` fields (era-dependent:
  awards removed mid-2023, later relaunched). r/bestofWritingPrompts (~12.6k
  subs) and annual Best-of-WP award threads are joinable back to the full
  r/WritingPrompts pool via link IDs — full universe by construction.
- **Hugo short fiction = graded universe.** Nomination stats PDFs publish a
  top-~16 longlist WITH exact nomination counts per category/year (2025 short
  story: 610 ballots, 673 distinct works nominated, top-16 enumerated 110→12
  noms). Finalists overwhelmingly from free-full-text venues (Uncanny,
  Clarkesworld, Lightspeed); pool text partial + venue-biased. ISFDB dump =
  eligible universe (Cloudflare-blocked to bots; historically weekly MySQL
  backups). Design: finalist vs. longlist vs. free-venue pool, with explicit
  venue-coverage confound.

## Dead / hallucinated / disqualified

- **"JVR (Joke Variety & Rating)" — DOES NOT EXIST** (hallucinated dataset name).
- **McSweeney's "open pitch logs" — DO NOT EXIST**; submission process is private
  email. (Repo already knew: `mcsweeneys_rejections/` = 30 rejection letters,
  zero submission texts.)
- **`lars76/story-evaluation-llm`** — real repo (HF dataset is
  `lars1234/story_writing_benchmark`) but 100% synthetic: LLM-generated stories
  scored by LLM judges. Disqualified by construction (project studies HUMAN
  preference).
- **Submission Grinder / Duotrope** — metadata-only ecosystems; story text is
  never present, by design. Market-level acceptance-rate priors only
  (Grinder market pages scrapable with browser UA; Duotrope paywalled).
- **Pushcart / O. Henry / BASS / Leacock Medal** — winners-only, pool
  unobservable, text in print books. Not usable.
- **Wergle Flomp** — winners-only (~12 winners + names-only finalists from
  ~5,000 entries/yr). Repo already scraped it (`humor/contest_corpus/`, 411 rows).
- **Slush-pile accept/reject corpora with manuscript text** — confirmed none
  public anywhere (McGrath textCrunch is metadata-only, unreleased).

## NYC Midnight (user-flagged)

- Masterlist pages fetch WITHOUT login; content is an embedded Airtable shared
  view (JS-rendered — need headless browser or Airtable shared-view JSON).
- **Masterlists are voluntary self-registration**: entrants opt in by posting
  their story to the members-only forum and submitting the forum URL. Story
  texts live ONLY on the login-walled forum (not search-indexed).
- Volume: Short Story Challenge >4,000 R1 entrants; ~6–8 contests/yr, multiple
  rounds → plausibly 15–30k entries/yr. Forum community >31k writers.
- Representativeness: NOT determinable publicly; self-selected (engaged/
  confident writers over-post). Round-advance verdicts exist only for the
  self-posting subset → selection-biased partial universe. Plus scraping a
  members-only forum is an access/ethics question. **Shaky as a primary
  source; the rare upside is that eliminated entrants' texts do exist there.**

## Mid-tier / secondary

- **CoAuthor (Stanford)** — real, 1,445 sessions (830 creative), full universe of
  accepted+dismissed GPT-3 suggestion spans with text, free download at
  coauthor.stanford.edu. But judged items are model-suggestion spans
  mid-composition, not finished human stories — different construct; secondary
  at best.
- **SemEval-2021 HaHackathon** (10k texts, 20 crowd raters each, binary+0-5) and
  **SemEval-2020 Humicroedit/FunLines** (15,095+8,248 edited headlines, 0-3
  crowd funniness; HF: `SemEvalWorkshop/humicroedit`) — real, easy access,
  full-universe ratings, but CROWD annotations, not expert verdicts (cf. the
  NY-caption crowd-rating ruling in `feedback` memory). HaHackathon official
  host dead; GitHub mirrors, license murky.
- **Shouts & Murmurs** — chosen-only; ~700 pieces enumerable via monthly
  sitemaps (sitemap-YYYY-MM.xml, 60-month window), full text in raw HTML
  despite metered paywall; slush pile invisible.
- **The Onion** — ~600 pitched headlines/week → ~16 published (dream ratio) but
  pitch pool not public; datasets are published-headlines-only.
- **Vocal Media challenges** — winners + 1,000 runners-up + 1,025 finalists
  named for Fiction Awards; entries are public stories but no clean
  per-challenge entries directory found — needs a hands-on probe.
- **Rattle Readers' Choice** — clean micro-universe (10 published finalists,
  vote picks 1) but tiny (~10/yr × ~15 yrs). Main Rattle prize pool (5,169
  entries 2024) is private Submittable.
- **Wattpad Wattys** — winner/shortlist lists public; API dead since 2018;
  scraping gray-zone; pool (opted-in entrants) not reliably enumerable.
  Book-length works. Weak fit.
- **Nebula** — finalists/winners at nebulas.sfwa.org (60+ yrs); no public
  nomination tallies (unlike Hugo).

## Recommended additions (ranked)

1. Reedsy Prompts (CW curation, full universe, trivially scrapable)
2. McSweeney's archive vs. anthology TOCs (humor curation, full universe,
   sparse positives)
3. r/bestofWritingPrompts + Best-of-WP awards joined via Academic Torrents
   dumps (CW curation on the existing WritingPrompts pool — same-population
   curation vs. vote comparison)
4. Hugo finalist/longlist-with-counts/pool gradient (CW curation, graded)
5. Vocal (pending scraping probe); Rattle Readers' Choice (anchor-scale)

NYC Midnight: park unless forum-access ethics + posting-rate question resolved.

---

## Addendum (same day, hands-on verification + downloads)

**Reddit (Arctic Shift API, not torrents — richer than expected; UA header required, 500 ids/request batch endpoint):**
- r/bestofWritingPrompts FULL scrape done: **822 posts + 739 comments** (2014-05 → 2026),
  NOT the estimated 5–15K. 410 posts link directly to WP threads (391 comment-level
  = clean join keys); 226 self-posts with pasted text (parse selftext for links).
  Artifacts: `datasets/creative-writing/bestof_writingprompts/raw/`.
- Annual awards harvest done: **449 meta threads + 9,865 comments**; winners threads
  2016–2022 (7 years) with **247 winner/nominee WP links + usernames** in selftexts;
  nominations in comment trees. `bestof_writingprompts/award_threads/`.
- r/Jokes award re-pull (1M ids): interim @930K rows — gilded 0.19%, any-award 0.26%
  (2,435 posts, 5,150 award units; Silver/Gold/Platinum dominate), stickied 2,
  distinguished 15 → "structurally votes-only" CONFIRMED. Median score of gilded
  = 14,438 → awards concentrate on already-viral posts (NOT an independent signal;
  use as cost-weighted second crowd tier conditional on score). CAVEAT: award fields
  are as-of-archive-retrieval; pre-2023 award removal + retrieval timing undercount.
  `datasets/humor/reddit_jokes/jokes_awards_metadata.jsonl.gz`.
- r/nosleep monthly contest winner threads confirmed (2019–2022 sampled).

**RoyalRoad (all live-verified today):**
- ~135.6K fictions site-wide (6,779 search pages × 20); ~1,600 STUB.
- STUB = the commercial-pickup verdict (KU exclusivity chapter-pull, status filter).
- **Community Magazine Contest = full-universe judged curation**: each edition is a
  "magazine" fiction whose chapters = every entry's first chapter. 10 editions
  Jan 2022–Jun 2026: 41/56/202/146/171/181/265/333/357/411 ch ≈ **2,153 entries**;
  5 ranked winners/edition (~50 positives, ~1.4%); prompt+month+fresh-work design
  controls topic/tenure/popularity; 55 community judges (June 2026). Pool→platform-
  metric join is PARTIAL (source-fiction links voluntary; title-match fallback).
  Winners join fully via blog announcements.
- Writathon = word-count achievement (55,555 words/5 wks), NOT a quality signal.
- Rising Stars / Best Rated / Trending = vote-derived algorithms, not curation.
- Expansion scrape launched: `datasets/creative-writing/royalroad_expansion/scrape_royalroad.py`
  (magazine chapters → blogs → STUB listing → full listing enumeration; 1 req/s).

**Style Invitational layering:**
- Losers' archive site is **nrars.net** (nrars.org outdated).
- Substack era: contest runs inside Gene Weingarten's Substack (The Gene Pool),
  Week 165+ as of 2026-02; per-post reaction_count/comment_count via public
  Substack JSON API (reactions 7–37, comments 0–110). Comment→specific-entry
  mapping feasibility unprobed. Style Conversational (Empress's pick rationale,
  WaPo era) = articulated-norm source; check nrars.net mirrors.
- Twitter: tweetapi account exists but Substack is the better bet (user decision).

## Overnight collection wave (2026-08-13, launched ~00:30)

**Removal-verdict recovery — resolved after full raw-source audit:**
- All dump lineages (rebuilt per-subreddit, original monthly RC_2016-06, Arctic
  Shift live) = 0% removed-text survival. Every published removals corpus was
  captured realtime pre-2023; that firehose is gone.
- **PullPush** (realtime ES lineage): API limits ~10 ids/req + hard 429s.
  WP story comments: 4/4 test ids [removed] — harvester over all 333,834
  removed WP ids running on sk3 (`harvest_pullpush.py`, ~2 days) to measure
  the true intact rate at scale.
- **Wayback/ArchiveTeam = the jokes route.** Thread pages captured for ~85-95%
  of removed jokes 2021-2023; body extraction (rendered `data-click-id="text"`
  div — NOT selftext JSON, which SSR omits) recovers **6/22 = 27% of random
  removed 2021 jokes** (e.g. "opposite of Christopher Reeve?" -> "Christopher
  Walken"). NSFW-flagged (~4-7%) unrecoverable (SSR omits body). Projected
  15-25K removed-jokes-with-text 2021-2023.
  Pipeline: sk3 `jokes_wayback_pipeline.py` (stable-hash order → any prefix =
  uniform sample), log `jokes_wayback.log`.

**Running overnight** (all resumable, 1 req/s-class politeness):
1. Reedsy Prompts full scrape — laptop `datasets/creative-writing/reedsy_prompts/`
   (contest index → per-contest story lists → all story pages, ~25-35K pages)
2. RoyalRoad deep metrics — laptop `royalroad_expansion/scrape_deep_metrics.py`
   (all 1,584 stubs + top-5K followers + 5K stable-hash sample = full fiction
   pages w/ 5-dim scores, favorites, avg views, reviews)
3. McSweeney's archive — laptop `datasets/humor/mcsweeneys_archive/`
   (~1,458 index pages → ~29K piece pages; pool for anthology-curation cell)
4. PullPush WP removed-text harvester — sk3
5. Jokes Wayback pipeline — sk3
6. Wayback coverage probe (2022/2023 verification) — laptop

## 2026-08-13 — WP removed-comment Wayback retry: CLOSED (negative, definitive)

Retried recovering WritingPrompts removed-story text from Wayback by walking all
snapshots per thread ("find the to-be-deleted ones"). Findings:

1. **Capture inventory** (3 threads, 484/424/20 captures): 100% www.reddit.com
   HTML. Zero old.reddit captures, zero direct .json endpoint captures of
   thread pages.
2. **React-era (2018–mid-2023) permalink HTML is structurally empty**: the
   hydration blob contains only routing metadata (`models:{}`,
   `headCommentId:null`) — comment bodies were loaded via XHR the crawler
   never captured. Confirmed on 2 captures + earlier gxaplkn test.
3. **NEW CHANNEL FOUND — archived `/api/info.json?id=t1_<cid>`**: Wayback crawl
   sessions for comment permalinks also archived Reddit's raw JSON endpoint
   (visible via `sessionReferrer`). Raw bytes retrievable with the `id_`
   timestamp modifier. Coverage is real: 6/6 of the original sample and 5/6 of
   2021–2023 probe ids have such captures (near-zero pre-2021).
4. **But 0/12 intact.** Every archived info.json is already
   `body:"[removed]", author:"[deleted]"`. Timing is damning: one capture 73
   SECONDS after creation, already blanked (AutoMod instant removal); exactly
   one capture per comment. The captures are *triggered by the removal itself*
   (mod-log / Reveddit-style bot hitting Save Page Now) → post-removal by
   construction. There is no live-window snapshot to find.

Uniform 40-id probe (stable-hash over the 333,834 removed ids, 2015–2025):
cdx coverage 6/40 (all 2018+, dense only 2021–2023), intact 0/40.

**Conclusion**: WP removed-story TEXT is unrecoverable retrospectively by any
known channel (dumps 0%, PullPush gated, Wayback post-removal-triggered).
r/Jokes remains the recoverable removal-verdict cell (submissions pipeline,
~17% and climbing on sk3). Clean CW verdict alternative: Kindle Scout.

Removed-id universe by year (wp_removed_ids.jsonl.gz, n=333,834):
2015:37k 2016:77k 2017:39.5k 2018:43k 2019:34k 2020:27k 2021:22k 2022:20.5k
2023:16k 2024:10k 2025:7k — metadata (id, thread, timing, score) intact even
though text is not; removal RATE analyses remain possible.

### 2026-08-14 addendum — SSR-era probe (answers "is there a non-React era that works?")
Both flanking eras DO render comment bodies server-side (pre-2018 old-reddit
HTML, n=153,528 removed ids; shreddit Aug-2023+, n=22,581). Probed 12+12
removed comments: for each, enumerated ALL thread captures AFTER comment
creation and fetched up to 3 earliest. Result **0/24 recovered**:
- pre-2018: threads typically have exactly ONE post-creation capture, taken
  much later (broad crawls), comment absent (removed/paged out).
- shreddit: 6/12 threads had post-creation captures (up to 153) but comment id
  absent from all fetched HTML except one preload-URL reference at +550h, no
  body. Removal beats capture.
Failure mode in SSR eras = TIMING, not structure. WP text recovery stays CLOSED.

## 2026-08-14 — overnight wave 2

**WP removals, targeted-window campaign (user-directed)**: bulk CDX dump of the
whole r/WritingPrompts/comments/ prefix (677 pages, 3.68M rows) -> local join vs
removed created_utc. Results: own-permalink SSR captures = only 5 old-era (2/5
recovered = full pre-removal stories); 3,620 shreddit "permalink" captures are
removal-triggered empty shells (svc partials OMIT removed comments, no
placeholder) -> dropped unfetched. Old-era thread pages 48h window: 94 unique
texts from ~3,100 pages. **Universe is 100% top-level** (story-slot comments);
recovered texts median 147 chars -> WP removal verdict is mostly FORMAT-NORM
enforcement (non-story top-level comments), plus occasional real stories (6-9K
chars). Round 2 (7-day window, 3,133 more old-era pages) running. Realistic
final: 150-400 texts = audit-scale, not classifier-scale.

**Jokes speedup**: slow harvester ~10-day pace (2 req/joke) at 3,900/106,937
(649 texts, 16.6%). Bulk CDX dump of r/Jokes/comments/ prefix (1,702 pages)
running; join script deployed -> fetch-only harvester sorted by capture delay
(~2.5-3 days, skips the ~38% with no captures).

**Kindle Scout COMPLETE**: 1,037 campaigns, 709 terminal on-page verdicts
(471 not-selected / 222 published / 16 selected), 613 with >500-char excerpts;
raw HTML in datasets/creative-writing/kindle_scout/raw/. Winners list capped at
180 (pages 10+ never archived) — superseded by on-page action-bar verdicts.

**RoyalRoad removal question (user)**: NOT usable as verdict. Deleted fictions
= bare 404, no mod-vs-author attribution; author-deletion correlates with
SUCCESS (KU pull); ~50K missing ids in 3-186,093 range but 4/15 sampled
"missing" ids were live-but-unlisted (listing coverage holes). RR verdict
signals remain STUB + Community Magazine Contest.

**Reedsy**: 15,779 story pages. **SO Votes count**: relaunched (first attempt
died silently).

### WP recovery FINAL (2026-08-14)
Round 2 (7-day window) complete. **414 unique pre-removal texts** total
(sk3: reddit_dumps/wp_recovered_final.jsonl.gz). Years: 2015:65 2016:110
2017:196 2018:43. Length: median 171 ch; >300ch: 124; >800ch (story-like): 47;
max 9,733. Character: predominantly format-norm removals (non-story top-level
comments) + a tail of real removed stories. Audit/validation-scale corpus, not
training-scale. Wayback cost: ~7,400 page fetches + 677 CDX pages total.

Reedsy approval-gate quantified (user Q): 262 contests report both counts:
69,987 entries -> 31,283 approved (45%; median 43%/contest, p10 35%, p90 63%).
Rejected text never public — verdict-count observable, text not.

## 2026-08-15 — consolidation wave (analysis-ready datasets)

- **Kindle Scout PARSED**: campaigns_parsed.jsonl — 1,006 campaigns, 995 w/
  excerpt text (median 31,285 chars = full first-chapters from expand-excerpt
  captures). 709 terminal verdicts: 238 accept / 471 reject, ALL with excerpts.
  Strongest rejected-text verdict cell in the collection.
- **Reedsy PARSED (incremental)**: stories_parsed.jsonl.gz — 18,811 stories
  (67 contests complete so far), full text (median ~9.9K chars), likes,
  winner badge (validated 64/64 vs contest-index "Won by") + 130 shortlist
  badges. Parser re-runs incrementally as scrape proceeds.
- **Jokes fast harvester**: 7,600+/63,959, 3,031 texts at 39.9% (delay-sorted).
- **SO Votes census DONE**: accepted 12,444,260 / bounty-closed 290,786 /
  up 183.2M / down 24.7M — verdict+curation+voting on one population, 14x Math.SE.
- **Kept-side builds launched (sk3)**: jokes_kept_universe (non-removed,
  text-intact, same window/filters) + wp_kept_sample (top-level kept stories:
  same-thread-as-removal contrast pairs + 2% stable-hash background).

### 2026-08-15 — WP 414 is now cross-archive FINAL
User challenged "final". Checked every remaining archive:
- **Common Crawl**: index dumped for all 31 crawls 2015-2017 (old-reddit era).
  Only 1,271 WP thread captures TOTAL (reddit throttles CC); join vs removed
  timestamps = **0 captures within 7d of any removed comment**. CLOSED.
- **Memento aggregator** (federates archive.today, national libraries, etc.):
  zero non-IA mementos for sampled WP thread URLs. CLOSED.
- **archive.today** directly: hard 429 to bots, but covered via Memento above.
- Kindle Scout universe: NO centralized tracker ever existed (author blogs +
  Goodreads threads only) → 1,037 archived campaigns = practical ceiling.
414 recovered texts = final across Wayback + CC + Memento + dumps + PullPush.

## 2026-08-16 — Reveddit-channel resolution + PROSPECTIVE collector launched

**User challenged "no author metadata" — resolved with a clean asymmetry:**
- Removed COMMENTS: text survives in author's public old.reddit listing, but
  author blanked in every retrospective source (verified: 1 known / 333,745 in
  dumps; live api/info blanks too). Channel needs names we mostly lack.
  Reply-mention mining: 120 candidates → live listing harvest → PROVEN
  (recovered full 2,257-char removed story mnflqz0 via u/Zeznex).
- Removed SUBMISSIONS (Jokes): author SURVIVES (97-100% 2020-25!) but selftext
  blanked in ALL public views (listing expando, RSS, /api/expando all
  "[removed]"). Channel dead for posts.
The two blanking policies interlock: comments keep text/lose name; posts keep
name/lose text. Explains Reveddit's posts-show-title-only limitation.

**PROSPECTIVE COLLECTOR RUNNING (sk3)**: wp_prospective_collector.py polls
old.reddit r/WritingPrompts live comment stream (SSR bodies) every 180s,
captures id/link_id/author/body/created pre-removal; labels removed/kept at
+5d via Arctic Shift batch API every 6h. Data:
sk3:reddit_dumps/wp_prospective/{captures,labels}.jsonl.gz. First poll: 25/25
rows complete. Expected: ~20 removals/day labeled (2025 rate) with full text
+ author — builds the WP verdict cell prospectively, no wall.

## 2026-08-16 — GitHub PR cells collected

**datasets/github-prs/** (new):
- issue_reactions__<repo>.jsonl.gz — 97,005 closed PRs across 13 gold repos
  (merge flag, size, PR reactions/comments, linked-issue reactions). Findings:
  PR-level reactions ~0 (dead channel); linked-issue coverage 4-33%/repo;
  linked issue WITH reactions only 2,708 PRs (2.8%) → issue-reactions usable
  as demand covariate on the linked subset, NOT a primary community y.
- Rust release cell: release_highlights/rust/ — pool_entries.jsonl.gz (4,031
  changelog entries, 147 versions, PR ids) + 133 announcement HTMLs (curated
  highlights; label by matching entries into announcement text at model time).
- Mathlib cell: release_highlights/mathlib/posts/ — 16 "This month in mathlib"
  posts (Aug 2021-May 2024), 623 curated PR refs; pool =
  issue_reactions__leanprover-community__mathlib{,4}.jsonl.gz (57,628 closed
  PRs). ★ CAVEAT: mathlib merges via BORS -> GitHub merged flag FALSE for
  bors-landed PRs (mathlib 1,910/18,590, mathlib4 368/39,038 "merged") — true
  merge verdict needs bors/label pass (ready-to-merge label or commit-in-master
  check) before using as verdict y.
- Changelog census: traefik/nomad changelogs are EXHAUSTIVE categorization
  (every PR listed) — no top-k selection; curation only exists in editorial
  release ANNOUNCEMENTS (Rust/K8s-class projects).

## 2026-08-16 — ★ CORRECTION: Reedsy "45% approval gate" was a MISPARSE
Cross-check of 262 contests: public contest-listing count ≈ card "entries"
(median ratio 0.96), NOT the card "stories" number. Card shows two separate
populations: "N contest entries" (paid $5 submissions, ~all public) vs
"M stories" (likely free publish-to-profile prompt responses). The earlier
claim "69,987 entries → 31,283 approved (45%)" is WRONG — real editorial
rejection of paid entries ≈ ~4% (rule/content violations only).
**NEVER quote a 45% Reedsy approval gate.** Reedsy verdict leg is WEAK
(~4% reject, text unobservable, only counts); Reedsy's real value = curated
(winner+shortlist of ~200/contest) + community (likes) + full text.
Rejected-text recovery: no public pre-approval state ever exists → no archive
race; only route = direct research collaboration with Reedsy.

## 2026-08-19 — r/Jokes removal corpus CLOSED: 16,755 pre-removal texts
Consolidation (sk3 `jokes_removed_final.jsonl.gz`, readback-verified): three
harvest shards merged + dedup by id — sk3 slow shard 699, sk3 fast shard
11,531 (readable prefix) + 4,521 via gzip multi-member RESYNC past a corrupt
block (438 new beyond prefix overlap), laptop shard 4,087 (mostly overlapping).
**Final: 16,755 unique removed jokes with pre-removal text = 15.7% of the
106,937 removed-joke universe.** Kept universe 184,774. Laptop harvester
stopped by PID after sk3 finished the full 63,959-page queue (laptop was
re-walking covered pages). Recovery-rate note: the queue was delay-sorted, so
the 15.7% overall rate masks the early high-yield band (~39% at the front).

## 2026-08-21 — WP prospective collector: first labeling pass + shard repair

First Leg-B labeling pass fired 04:53: **36 labeled → 35 live, 1 absent**
(removed-or-author-deleted; ~2.8% catch rate so far, n tiny). Collection
healthy: 1,110 comments captured over 5.1 days (~220/day).

**Defect found and fixed**: `captures.jsonl.gz`'s first gzip member was
truncated by the Aug-17 restart (kill mid-write). The labeler's plain
GzipFile read died at ~row 282, which would have PERMANENTLY hidden every
later capture from Leg B. Repaired by magic-byte resync (282+828 → 1,110
unique rows, zero loss; originals kept as `.bak_20260821` /
`.corrupt_20260821`), and the collector now reads captures via
`read_captures_resilient()` (resyncs past truncated members — same fix
pattern as the jokes shard, 2026-08-20). Collector restarted pid 2878627,
verified: `start: 1114 seen, 35 labeled`, immediate pass labeled 1
previously-hidden row. Collector lives ONLY on sk3:
`/lfs/.../data/reddit_dumps/wp_prospective_collector.py` (no repo copy —
divergence recorded here per code-sync rule).
