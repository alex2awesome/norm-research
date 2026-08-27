# BBC Most Read (news-C cell: lay crowd, REVEALED ATTENTION)

Lay-crowd revealed-attention judgments from BBC News **"Most Read"** lists, harvested
from the Wayback Machine. One row = one BBC headline; the judgment is whether the
reading public made it a most-read story that day.

**Label semantics — LAY crowd, REVEALED preference.** This is a crowd cell of the
taste taxonomy: the anonymous BBC reading public, via what they actually clicked /
read, deciding which stories get attention. It complements the `reddit_newsworthiness`
cell (lay crowd, revealed votes) and contrasts with `news_homepages` (EXPERT editors,
revealed homepage *placement*). Same artifact type (a news headline), different judge
population (BBC readers) and a cleaner, non-algorithmic source: the Most Read module
is **server-rendered, ranked by aggregate reads**, not by a personalized feed.

- **y = 1**: headline was on the BBC Most Read list for that capture (rank kept, 1-10).
- **y = 0**: a same-day BBC headline NOT on that day's Most Read list (control pool).
- **X** = headline text, presentation-normalized (HTML-unescape ×2, NFKC, curly→straight
  quotes, dashes, ellipsis, collapse whitespace; `raw` headline kept). Section/format
  prefixes the React era inlines ("Video", "Watch", "Live", "In pictures", …) are
  stripped — presentation, not headline craft. Model never sees the URL/section (kept
  as metadata).

Both classes come from the **same Wayback capture pipeline** (same HTML, same parser
family), so there is no cross-source typography leak. In the React era (2024-25) the
control headlines are pulled from the homepage's *other* `card-headline` `<h2>` nodes
(headline only) — NOT raw anchor text, which would append "… 8 hrs ago · Africa"
metadata and leak the class.

## Source: Wayback Machine, two server-rendered channels

The BBC Most Read markup changed several times. Validated per-era selectors:

| Era | Channel | Wayback URL | Container selector | Status |
|---|---|---|---|---|
| ~2014–2017 | A: dedicated page | `bbc.{com,co.uk}/news/popular/read` (+ `…/read.fragment`) | `ul.most-popular-page__list > li.most-popular-page-list-item > a.…__link`, rank in `span.…__rank` | **clean ranked top-10** |
| ~2017–2023 | B: homepage module (Morph) | `bbc.{com,co.uk}/news` | `ol.nw-c-most-read__items > li > a.nw-c-most-read__link`, rank in `span.nw-c-most-read__rank` | **inline, server-rendered** |
| ~2024–2025 | B: homepage module (React) | `bbc.{com,co.uk}/news` | `section[data-analytics_group_name="Most read"]`, items `[data-testid="cambridge-card"]`; headline `[data-testid="card-headline"]` (`<h2>`), rank `[data-testid="card-order"]` | **inline (DOM + embedded JSON)** |

**Parses that look right but are WRONG (rejected):** the homepage's *"Most watched"*
video carousel and the rotating *Top Stories* index carousel both render headings
that say "Most …" and lists prefixed "1:"/"Video"; an early heading-based parser
caught these. The production parsers target the specific Most-Read containers above,
so carousel/nav text is excluded. (See `scripts/feasibility.py` for the failure mode.)

**The 2016–2017 homepage Most Read is AJAX** (`data-fetch-url="/news/popular/read.fragment"`)
— empty in the captured HTML. For that era we harvest the **dedicated page** (Channel A)
for the list, and pull **controls from the same-day homepage capture** (the homepage
still lists dozens of other articles even when its most-read module is AJAX-loaded).

## Confound (this cell is flagged YELLOW): PLACEMENT / promotion

Most-read reflects placement and promotion as much as intrinsic newsworthiness: a story
splashed at the top of the homepage gets clicked more, so it becomes most-read partly
*because* editors promoted it. We mitigate and document, not eliminate:

- **rank** and **capture `timestamp`** are kept on every positive row.
- The **control pool is same-day** (and largely same homepage capture), which partially
  matches day, news cycle, and topic-mix between the classes.
- The separate **`news_homepages`** dataset (homepage spatial position, expert layout)
  can later **instrument** placement: condition most-read odds on measured homepage
  position to separate "read because promoted" from "read despite/independent of it."
- Residual: stories never placed on the homepage can't enter the most-read list at all,
  so y=1 is conditioned on having had some homepage exposure; the control pool shares
  that conditioning (it's also homepage headlines), which is the point.

## Dataset-first protocol results

See `manifest.json` / `build_log.txt` for the exact run. The numbers below are the
**first-pass build** (2015-2017 Channel-A slice only; the full 2014-2025 crawl is
left running — see Coverage).

**First-pass (2015-2017): 1,303 most-read + 2,019 controls, 87 capture-days.**

| metric | value |
|---|---|
| TF-IDF/LR floor (raw) | eval **0.660** / test **0.662** |
| TF-IDF/LR floor (length-decile-matched) | eval **0.639** / test **0.603** |
| headline length vs label corr | **+0.65** (pos mean 50.5 chars, neg 35.8) |
| top POS features | says, after, election 2015, crisis, newspaper headlines, train driver, British, Top Gear |
| top NEG features | is, what, EU, talks, behind, Iran, vote, Greek, US, threat, political, Russia |

**The +0.65 length correlation is a real SOURCE-asymmetry confound in this slice**
(not headline craft): the 2015-2017 homepage index lists non-lead stories as
~35-char truncated teasers, while the dedicated most-read page shows full ~50-char
headlines. Stripping format markers does not remove it (the teasers ARE shorter).
The **length-decile-matched floor (0.60-0.64)** is the trustworthy signal for this
slice — net of length there is still a real topical/register signal: the crowd
reads human-interest / disaster / celebrity / "newspaper headlines" digests more,
and EU/Iran/Greek-debt/process-politics stories less. The 2018-2025 Morph/React
slices (added by the full crawl) draw positives AND controls from the SAME homepage
capture, so they do NOT have this source-length asymmetry and are the cleaner slices.

Cleaning applied symmetrically to both classes (see `strip_prefix`): leading rank
"N:" and clip-duration "M:SS" prefixes, "Full article"/"Video"/"Watch" markers, and
trailing card metadata. Controls that are VIDEO CLIPS (clip duration anywhere, or a
leading/trailing Watch/Video/Listen) are DROPPED — a video caption is a different
register that never appears in the most-read article list, so keeping them would be
a register/typography leak (it injected fake number-features "46/00/08" before the
fix). Section is `news` for ~99.7% of rows (BBC most-read is overwhelmingly /news/),
so the "Sport:"/"Business:" prefix confound is minimal here.

### Spurious-feature reasoning (read alongside the floor)

- **Section prefix / topic.** BBC most-read skews to human-interest, oddity, and
  world-news ("rat on a plane", "employees shut inside coffins", royal/celebrity deaths)
  over UK-politics process stories. Topic *is* genuinely part of lay attention, but it
  means part of any floor is topic preference, not headline craft — same caveat as the
  Reddit cell. Section is kept as metadata for auditing, never shown to the model.
- **Length / format markers.** "Video"/"Watch"/"Live" prefixes are stripped so the model
  can't key on them; check `len_label_corr` in the manifest for residual length signal.
- **Breaking-news / "Live".** Live-feed headlines are a format, not craft — stripped.

## Coverage & biases

**Channel coverage is complementary in TIME** (measured via `scripts/coverage_probe.py`):

| Channel | Era with good Wayback coverage | Distinct daily captures |
|---|---|---|
| A: dedicated `/news/popular/read` page | **2015–2017** (BBC retired the page ~2018) | ~615 days (2015 ≈162, 2016 ≈240, 2017 ≈213; ~0 after) |
| B: homepage Morph module | **2018–2023** | from homepage captures (thousands/yr) |
| B: homepage React module | **2024–2025** | from homepage captures |

So Channel A supplies clean ranked most-read for 2015-2017 and Channel B supplies it
for 2018-2025 — together spanning the full window. The full deduped crawl worklist is
**7,300 unique captures** (`bbc.{com,co.uk}/news` homepage daily + the dedicated page,
collapsed `timestamp:8`, www/non-www and .com/.co.uk overlaps removed).

- The 2014 (and most pre-2015) homepage captures have NO inline most-read (AJAX era);
  they still contribute ~80 control headlines each, used to match 2015-era Channel-A
  positives by day.
- Lists are top-10; positives per day ≤ 10. Scale comes from **many capture-days**, not
  list depth. Dedup keeps the earliest occurrence of each normalized headline
  ("most-read wins": if a headline is ever most-read it's labeled y=1 at its first
  most-read capture; otherwise its earliest control occurrence is y=0).
- Edition skew: `bbc.com/news` serves a US/international edition, `bbc.co.uk/news` a UK
  edition; both are harvested and the `channel`/host is kept. Their most-read mixes
  differ slightly (US vs UK reader interest).

## Files

- sk3: `/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/bbc_mostread/`
  - `built/{train,eval,test}.csv.gz` — stable-hash split (md5(headline_id) % 10 → 0-7/8/9)
  - `raw/captures.jsonl` — one record per Wayback capture (most_read list + others); **append-only, never deleted**
  - `manifest.json`, `build_log.txt`
  - `scripts/` — `scrape_bbc_mostread.py` (Wayback harvest, resumable), `build_bbc_mostread.py` (label+dedup+floor+split), plus feasibility/validation diagnostics
- Columns: `text, judgement, rank, timestamp, day, section, channel, parser, href, headline_id`

## Politeness

Wayback ≤ 1 req/sec, exponential backoff on 429/5xx, real User-Agent with contact
`alex2awesome@gmail.com`. The scraper is resumable (rebuilds the done-set from the
existing JSONL) and append-only.
