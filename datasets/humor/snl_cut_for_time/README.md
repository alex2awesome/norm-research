# SNL cut-for-time vs aired (humor verdict dataset)

Collected 2026-07-28.

## Verdict definition

At SNL, every sketch that reaches **dress rehearsal** is performed before a live
audience; after dress, producers decide the final live lineup. A sketch labeled
**`cut_for_time`** survived the writers' table read and made it into the dress
rehearsal but was dropped before the live broadcast — a producer-level
rejection at the last gate. A sketch labeled **`aired`** was broadcast in the
live show. Both verdicts are therefore *producer accept/reject decisions over
professionally written, fully produced sketches* — a strong pairwise humor/
quality signal at matched production values.

Caveats:
- Cut-for-time reasons are not purely quality (timing, topicality, host
  fatigue, standards), so the label is "not chosen for air", not "unfunny".
- The Fandom index is community-maintained and NOT exhaustive; NBC only began
  releasing cut sketches online around S39 (2013). Coverage before 2013 is
  sparse (a few famous ones back to S26).

## Files

- `snl_catalog.jsonl` — one row per sketch:
  `title, host, season, episode (S##E## when known), year (season start year),
  verdict: aired | cut_for_time, summary, source, transcript_path, url`.
- `yt_pilot_results.jsonl` — pilot (max 40 cut sketches): YouTube URL found by
  strict title match on the official "Saturday Night Live" channel, plus
  caption transcript when retrievable
  (`status: ok | no_transcript:* | no_confident_match`).
- `raw/` — raw preservation, parse-second:
  - `cut_for_time_wikitext.json` — MediaWiki API parse of
    https://snl.fandom.com/wiki/Cut_For_Time (page HTML is Cloudflare-blocked
    to curl; the api.php endpoint is not).
  - `category_<season>_p<N>.html` — snltranscripts.jt.org season category
    listing pages (categories `13..18` = seasons starting 2013..2018,
    `2019..2023` = those start years).
  - `season_2013.html`, `season_2014.html` — static season index pages
    (`/13/2013.phtml`, `/14/2014.phtml`); later years 404 (site moved to WP).
  - `sitemap*.xml`, `snltranscripts_home.html` — discovery artifacts.
  - `transcript_samples/*.html` — full transcript pages: ALL 20 cut-for-time
    transcripts hosted by snltranscripts + 12 random aired ones.
  - `yt_auto_subs/*.txt` — YouTube caption transcripts from the pilot.

## Provenance & method

1. **Cut catalog**: `snl.fandom.com/wiki/Cut_For_Time` via MediaWiki
   `action=parse` API (raw JSON preserved). 67 `{{lineup3}}` entries,
   S26E9 (2001) .. S47E12 (2022). No per-sketch wiki pages exist (checked), so
   no YouTube URLs on the wiki; pilot URLs come from yt-dlp search restricted
   to the official SNL channel with strict normalized-title matching.
   `Category:Cut for Time` is empty (checked).
2. **Aired positives**: snltranscripts.jt.org season categories 2013+ →
   transcript title + URL per sketch. 20 listings titled "Cut for Time..." are
   relabeled `verdict=cut_for_time` (they are transcripts of cut sketches —
   bonus: full text for cut items).
3. **Pilot captions**: `yt-dlp --flat-playlist` search only (no video
   downloads), captions via `youtube-transcript-api`. Many official SNL
   uploads have captions disabled (`TranscriptsDisabled`); URLs recorded
   regardless.

Politeness: 2.5s (site) / 4.5s (YouTube) delays, browser UA, robots.txt
checked: snltranscripts.jt.org allows all relevant paths; fandom.com HTML is
bot-challenged but its public API is the sanctioned machine interface.

## Counts

See `COUNTS` section at bottom of this file (regenerate with
`python3 build_catalog.py`).

- catalog rows: 2,636
  - `cut_for_time`: 87 (67 Fandom index + 20 snltranscripts cut-for-time
    transcripts; overlap between the two subsets is possible for S44 items)
  - `aired`: 2,549 (snltranscripts listings, seasons 2013-2023; includes all
    segment types: sketches, monologues, Weekend Update segments, some music
    performances — filter by title if a sketch-only subset is needed)
- transcript samples saved: 32 (20 cut + 12 aired)
- YouTube pilot (40 cut sketches attempted): 36 confident official-channel URL
  matches; 6 caption transcripts retrieved (`ok`); 30 videos have captions
  disabled by the channel (`no_transcript:TranscriptsDisabled`);
  4 `no_confident_match`.
- cut_for_time rows with a URL: 56/87; with transcript text: 26/87
  (20 snltranscripts HTML + 6 YouTube captions)

## Resume / extend

- Re-run `python3 scrape_snl.py` — cached pages are skipped; add categories to
  `CATS` for more seasons (categories exist back to `75`).
- Re-run `python3 build_catalog.py` after new raw pages.
- Pilot: `python3 pilot_yt_transcripts.py` rewrites `yt_pilot_results.jsonl`
  for the first 40 wiki entries; raise the `n >= 40` cap to extend.
- To get full transcripts for all aired rows: iterate `url` fields of
  `snl_catalog.jsonl` (2.5s delay); only 32 fetched so far as samples.
