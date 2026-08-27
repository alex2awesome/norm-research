# McSweeney's Rejection Corpus — Harvest Log

Harvest date: 2026-06-01
Final pair count: **30** (verbatim editor responses)

## Output

`pairs.jsonl` — one pair per line, schema as specified.

## Per-source notes

### 1. Sean Hewlett WordPress (2007) — **11 pairs harvested**
(Yielded 11 unique editor responses after consolidation.)
URL: https://sahewlett.wordpress.com/2007/11/09/rejection-emails-i-have-received-from-mcsweeneys-with-the-messages-written-between-the-lines-spelled-out/

- Required a second fetch with explicit "find at least 10" prompt — first pass only surfaced 4.
- Yielded 13 quoted rejections in source; consolidated to 11 unique pairs (excluded 2 that were near-duplicates of #1, just truncated re-quotes for the "between the lines" gag).
- Submitted piece text NOT included by Sean in any case. We have rejection-only for all 11.
- Editors: Chris (Monks), Jess, Benjamin (Cohen), John Warner.
- One typo corrected: "on going to pass" -> "I'm going to pass" (hewlett_10), flagged in notes.

### 2. Victor Beigelman / Footbridge Substack — **8 pairs harvested**
URL: https://footbridge.substack.com/p/my-failed-mcsweeneys-submissions

- All 8 quoted rejections extracted cleanly with submission titles and dates.
- Beigelman explicitly does NOT reproduce submission text in his post; only titles + editor response.
- All editor responses are from Chris (Christopher Monks).
- Date range: 2013-07 to 2023-04 — useful longitudinal slice.

### 3. Rejection Wiki — Internet Tendency — **4 pairs harvested**
URL: https://www.rejectionwiki.com/index.php?title=McSweeney%27s_Internet_Tendency

- First fetch timed out (60s). Retried successfully on second attempt.
- Templates, not personal pairs. No author, no submission. Marked as `author: null`.
- Useful as baselines for what a "standard form pass" looks like.

### 4. Rejection Wiki — Quarterly — **5 pairs harvested**
URL: https://www.rejectionwiki.com/index.php?title=McSweeney%27s_Quarterly

- Fetched cleanly first try.
- These are Quarterly (print) rejections, not Internet Tendency. Different editors: Rita Bullwinkel, Chelsea Hogue, Jordan.
- Long-form polite-form-letter style.

### 5. arsenalofwords (2019) — **1 pair harvested**
URL: https://arsenalofwords.com/2019/03/03/winter-2019-rejections-one-story-mcsweeneys-electric-literature-and-more-literary-journals/

- Got title ("Terminus") and full verbatim editor response. No submission text. Quarterly, unsigned.

### 6. Himanshu Medium — **0 pairs**
URL: https://medium.com/@himanshuiswriting/my-rejected-submission-to-mcsweeneys-301a0f39f84a

- Article shows submission text but the actual editor email is only shown as an image and NOT transcribed in prose. Per task constraint (only verbatim quoted), skipped.

### 7. Charleston City Paper / Paul Bowers — **1 pair harvested**
URL: https://charlestoncitypaper.com/tips-for-getting-published-on-mcsweeneys-internet-tendency/

- The "ignorance of Boilen" rejection found and quoted in full. Submitted piece was about a fictional medieval hair-metal band "LIONS MAYNE"; full piece text not reproduced in the article.

### 8. Kimberly Harrington interview Part 1 & Part 2 (Medium) — **0 pairs**
URLs:
- https://honeystaysuper.medium.com/a-conversation-with-chris-monks-managing-editor-of-mcsweeney-s-internet-tendency-4c269133ffff
- https://honeystaysuper.medium.com/a-conversation-with-chris-monks-managing-editor-of-mcsweeney-s-internet-tendency-part-2-death-28ed2f36a218

- Discuss rejection process and "jerk folder" anecdotes but contain ZERO verbatim Monks rejection text. Skipped.

### 9. Carlos Greaves Substack — **0 pairs**
URL: https://shadesofgreaves.substack.com/p/my-first-mcsweeneys-submission

- Greaves reproduces his full submission ("I Know Sports Are Stupid…") but only describes the rejection as "a polite, form rejection email from Chris" — verbatim text not provided. Skipped.

### 10. Julie Vick Substack + julievick.com — **0 pairs**
URLs:
- https://julievick.substack.com/p/tips-for-writing-for-mcsweeneys
- https://julievick.com/2020/10/01/on-submitting-to-mcsweeneys/

- General advice articles, no quoted rejection text. Skipped.

### 11. J.P. Melkus "Most Unfairly Rejected" (Medium / The Clap) — **0 pairs**
URL: https://medium.com/the-clap/my-most-unfairly-rejected-mcsweeneys-humor-pieces-of-all-time-fdb66ab48719

- Satirical list of fictional submission titles, no actual editor correspondence. Skipped.

### 12. Tom Mitchell "Writing Despite Rejection" — **0 pairs**
URL: https://tommycm.medium.com/writing-despite-rejection-8dc49e6d80e9

- HTTP 410 Gone. Article no longer available.

## Additional searches performed

- `site:substack.com mcsweeneys rejected Chris Monks` — surfaced Greaves, Vick, Sweetman, Warner, Sacks (none with verbatim quotes beyond what was already harvested).
- `"chris monks" rejection email quoted "going to pass"` — re-surfaced same set.
- `"rejected by mcsweeney" 2024 OR 2025 blog` — found Mitchell (410) and unrelated McSweeney's-published satirical pieces.
- `"mcsweeney's internet tendency" rejection letter personalized blog post writer` — no new sources.
- `"kimberly harrington" mcsweeney rejection "ignorance"` — led to the Bowers/Boilen quote.

## What was NOT obtained

- Full submitted-piece text is missing for every pair (30/30). Authors typically reproduced their submission titles but withheld the body. This is intrinsic to the available web record.
- The one case where the submission text IS public (Greaves) lacks the verbatim rejection. Not pairable.

## Substantiveness assessment

Of the 30 verbatim rejections:

- **Substantive craft critique (~9 pairs):** identify a specific weakness (premise vs. execution, hook timing, target too easy, category saturation, similarity to prior pieces, voice good but premise weak, editor's own knowledge gap). Examples: beigelman_01, beigelman_04, beigelman_05, hewlett_09, hewlett_11, bowers_01.
- **Brief polite pass with light positive note (~10 pairs):** "Sharp satire, but..." / "Some fun moments here, but..." / "Certainly amuses, but..." — acknowledges merit, gives no diagnostic detail.
- **Pure form / template (~10 pairs):** Quarterly form letters and short "going to pass" notes with no content.

Personalized Internet Tendency rejections from Christopher Monks dominate the substantive tier. Quarterly rejections (Bullwinkel, Hogue, Jordan) are universally form-letter style. The Sean Hewlett (2007) batch is mostly short/curt, consistent with the era's editorial style before personalization became a stated policy.

Overall: there is signal — Monks's rejections especially carry diagnostic feedback in roughly a third of the corpus — but the small N (30) and lack of paired submission text bound this dataset to a supplementary/anchor role rather than a training corpus.
