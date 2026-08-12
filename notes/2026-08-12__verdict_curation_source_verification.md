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
