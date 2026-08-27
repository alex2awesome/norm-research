# Washington Post Style Invitational — nrars.org "Book of Weeks" archive

Weekly humor contest run by The Washington Post (March 1993 – Dec 2022; later
"The Invitational" on Substack). Each week the Empress/Czar published a prompt;
results ran ~2-4 weeks later with editorially ranked verdicts: one winner,
~4-5 named runners-up, and a batch of honorable mentions.

## Provenance
- Source: the Losers' community archive at nrars.org (Not Ready for the
  Algonquin Roundtable Society). Entry page: https://nrars.org/nrarsarchives.html
  (does not link the Book of Weeks directly); the working index is the master
  contest list at
  `https://nrars.org/0 The Book of Weeks/mastercontestlist0all.html`
  (saved here as `raw/mastercontestlist0all.html`), which tables all 1,367
  contest weeks (weeks 1-1530, 1993-2023) with per-week links to plain-text
  files under `archive/new/01 text/NNNN.txt`.
- Collected 2026-07-28 with curl, UA "academic-research-collection", 1 s
  delay between requests, sequential. Note: nrars.org serves a certificate
  that fails strict verification; fetched with `curl -k`.
- Each `01 text/NNNN.txt` file contains that week's NEW contest prompt plus a
  "Report from Week N" section holding the RESULTS (verdicts) of an earlier
  week. FILE numbers diverge from WEEK numbers after the Post's late-1999
  hiatus (e.g. week 402's file is 0333.txt); `raw/master_index.json` holds the
  authoritative week -> file mapping parsed from the master list.

## License notes
- Underlying entries are Washington Post-published reader submissions; the
  nrars.org archive is a fan/community preservation effort with no stated
  license. Treat as research-use-only; do not redistribute publicly.

## What was collected this session
- `raw/mastercontestlist0all.html` — full archive index (all 1,367 weeks).
- `raw/master_index.json` — parsed index: week, date, title, synopsis
  (= contest prompt), contest_txt, results_txt.
- `raw/01_text/` — **332 text files (0001.txt-0333.txt; 0004.txt was probed
  first), all files referenced by weeks 1-402. Zero download failures.**
- `raw/archive_index.html`, `raw/nrars_home.html` — entry pages, for the record.

## Parsed output: style_invitational.jsonl
Built by `parse_results.py` (heuristic plain-text parsing; formats drift over
the years — raw files are preserved so parsing can always be redone). One JSON
object per entry:

| field | meaning |
|---|---|
| week_id | contest week whose results these are ("Report from Week N") |
| contest_prompt | that week's prompt synopsis from the master index |
| entry_text | the submitted humor entry, usually with author attribution in-line |
| tier | `winner` / `runnerup` / `honorable_mention` |

Counts (verified 2026-07-28):

| | value |
|---|---|
| files yielding a results section | 320 of 332 |
| weeks with parsed results | 316 (weeks 2-329) |
| rows total | 9,637 |
| tier = winner | 322 |
| tier = runnerup | 1,227 |
| tier = honorable_mention | 8,088 |

Typical per-week shape: 1 winner, 4-6 runners-up, 10-30 honorable mentions.
Winner count slightly exceeds week count because a few weeks published
two-part results.

Known parse limitations (all recoverable from raw/):
- 0001-0003.txt predate the first results report; 0084.txt and 0096.txt lack
  the "Report from Week" marker; ~7 more files use headers the heuristics
  miss. 19 rows are >600 chars (likely editorial banter merged into an entry).
- Author attributions are left inside entry_text.

## How to resume (remaining ~1,100 weeks)
1. `raw/master_index.json` already indexes ALL weeks (through week 1530,
   March 2023). The full archive references 1,653 unique `01 text` files.
2. Append the not-yet-downloaded file names to `raw/download_list.txt`
   (e.g. regenerate with the week<=402 filter in this README's history raised
   or removed), then rerun `raw/dl.sh` — it skips files already present and
   logs failures to `raw/dl_failures.log`. Keep the 1 s delay.
   Note dl.sh processes `download_list.txt` line-by-line; ensure the file ends
   with a trailing newline or the last entry is skipped.
3. Rerun `python3 parse_results.py` — it re-scans everything in `raw/01_text/`
   and rewrites `style_invitational.jsonl`.
4. Later eras also exist as `.doc`/`.htm`/PDF under `archive/new/02 docs`,
   `03 html from docs`, `04 html scrape or search`, `06 fishwrap PDFs`,
   `07 E-version PDFs` if a week's text file is missing; directory listing is
   403-forbidden, so go through the master list links.
