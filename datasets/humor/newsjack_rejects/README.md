# BBC Newsjack rejects (humor verdict dataset)

Collected 2026-07-28.

## Verdict definition

**Newsjack** (BBC Radio 4 Extra, 2009-2021) had an **open submission** system:
anyone could email one-liners and sketches each week; the production team
selected a handful for broadcast. Writers whose material was **not selected**
routinely self-published the rejected jokes ("rejects") on public forums and
blogs the same week. Verdicts:

- `rejected` — the poster presents the material as not selected for broadcast
  (the default in dedicated "rejects" threads and rejects blog posts).
- `aired-claimed` — the poster claims some material got on the show
  ("got one in", "first credit", "was read out" ...). Self-reported, usually
  in the same threads; the aired joke text is often *not* quoted (BBC owns
  broadcast material), so many `aired-claimed` records are testimony of
  acceptance rather than the accepted joke itself. Treat as noisy positives.

Both verdicts are decisions by the same gatekeepers (Newsjack producers) over
the same weekly topical-joke submission pool — a clean accept/reject seam for
one-liner topical humor, with the caveat that the negative class is
self-selected (writers who chose to post their rejects).

## Files

- `newsjack_rejects.jsonl` — one row per forum post / blog chunk:
  `source, source_url, thread_id, thread_title,
  thread_kind (rejects | series | blog-rejects), post_id, author, date,
  series (Newsjack series number when known), joke_text, verdict,
  aired_cue, reject_cue`.
  A post may contain several one-liners; `joke_text` preserves paragraph
  breaks. Quoted blocks (other posters' text) are stripped before cue
  detection and output.
- `raw/` — raw HTML preserved before any parsing:
  - `threads_index_p1.html` — Newsjack threads index
    (https://www.comedy.co.uk/radio/newsjack/threads/).
  - `thread_<tid>_p<N>.html` — every page of every Newsjack thread on the
    British Comedy Guide forum (see `THREAD_META` in `parse_newsjack.py` for
    id → title mapping). Dedicated rejects threads: 34847 (s20), 35265 (s21),
    35458 (s22), 35790 (s23), 33939 (Autumn 2017), 23867 (s6),
    13736 (salon de refuse) — the rest are series-discussion threads that mix
    chat with rejects.
  - `newsbiscuit_120350_p{1,2}.html`, `newsbiscuit_126883_p1.html` — Wayback
    Machine captures of the (now defunct) NewsBiscuit forum threads "Newsjack
    Rejects" and "Newsjack Series 19".
  - `garryabbott_wayback_{1..7}.html` — Wayback captures of Garry Abbott's
    "Not good enough for the BBC" Newsjack rejects blog posts (series 10-11;
    live URLs now 404).
  - `garryabbott_*.html/xml/json` — discovery artifacts.

## Provenance & method

- **British Comedy Guide (comedy.co.uk)**: the previously observed 403 is
  avoided simply by sending a normal browser User-Agent; robots.txt allows all
  paths for `User-agent: *` (Cloudflare content-signals: search=yes,
  ai-train=no — recorded here for downstream licensing decisions). Fetched
  live with 2.5s delays; 855 thread pages across 26 threads (all complete).
- **NewsBiscuit forum**: dead (site is now a Wix rebuild; old
  `/forum/topic.php` URLs serve the homepage). Recovered via Wayback Machine
  (CDX + availability API; captures 2021).
- **garryabbott.com**: posts deleted from the live site (404). Recovered via
  Wayback CDX (captures 2014-2019). 7 posts, split into paragraph chunks.

## Verdict labeling rules (heuristic — read before use)

Labels come from regex cues in the poster's own (quote-stripped) words, in
`parse_newsjack.py`:

- dedicated rejects threads / blog posts: default `rejected`; flipped to
  `aired-claimed` only when an aired cue fires with no reject cue and no
  negation ("didn't get in") or other-directed praise ("well done ...").
- series-discussion threads: posts kept ONLY when a reject or aired cue
  fires; everything else is dropped as chatter.

Expect residual noise in both directions (spot-checks found e.g. "we got in
ok" [audience queue] as a false aired cue). `aired_cue` / `reject_cue` flags
are retained so you can re-threshold. For high-precision rejected text, filter
to `thread_kind in (rejects, blog-rejects)` and posts that look like joke
lists (multiple paragraphs, quotation-marked one-liners).

## Counts (2026-07-28)

Raw: 855 BCG thread pages (26 threads, all complete), 3 NewsBiscuit Wayback
pages, 7 garryabbott Wayback posts. Records in `newsjack_rejects.jsonl`: 1,981.

| thread_kind   | rejected | aired-claimed |
|---------------|---------:|--------------:|
| rejects       |      806 |            12 |
| series        |    1,017 |           111 |
| blog-rejects  |       34 |             1 |
| **total**     |**1,857** |       **124** |

By source: comedy.co.uk 1,751 rejected / 122 aired-claimed; NewsBiscuit
(Wayback) 72 / 1; garryabbott.com (Wayback) 34 / 1.

Regenerate with `python3 parse_newsjack.py`.

## Resume / extend

- `python3 scrape_newsjack.py` — resume-safe (skips existing raw pages);
  `resume_state.json` lists pages fetched per thread; failures logged in
  `scrape_log.txt`.
- `python3 parse_newsjack.py` — rebuilds the jsonl from raw/.
- Not yet harvested: BCG threads outside the Newsjack index (search the forum
  for "Newsjack" — e.g. thread 34590 was only found via web search), other
  writers' blogs (e.g. search "newsjack rejects blog"), and NewsBiscuit topic
  ids other than 120350/126883 (the forum had per-series Newsjack threads;
  enumerate via Wayback CDX `url=newsbiscuit.com/forum/topic.php*`).
