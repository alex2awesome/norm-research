# RoyalRoad Stubs — cw-A cell (web serial → publisher/KU deal)

Taste-taxonomy cell **cw-A**: expert/market *revealed* selection over creative
writing. Did this web serial get picked up for commercial publication?

## Label semantics

- **y=1**: fiction has official **STUB** status on RoyalRoad
  (`/fictions/search?status=STUB`). RoyalRoad's own tooltip: "Parts of this
  fiction have been removed by the author, likely due to third party
  exclusivity contracts such as Kindle Unlimited." I.e., the author pulled
  chapters because the work was picked up (almost always Amazon KU).
- **y=0**: non-stub fictions from the followers-ordered search listing,
  greedy 1:1 matched to stubs on listing metadata
  (log followers, log pages, rating, tag Jaccard — see
  `prioritize_controls.metric_dist`).

**X** = earliest ~3 chapters (lowest chapter ids; chapter ids are
platform-globally monotonic), recovered from **Wayback Machine** captures for
BOTH sides — never live pages — so the archive-availability filter applies
symmetrically. `x_source='wayback'` on every row.

## Enumeration date

2026-06-12 (listings + Wayback recovery started; recovery is resumable and
append-only).

## Known biases / caveats

- **Survivorship**: deleted (non-stubbed) fictions never enter either pool.
- **Wayback-coverage filter**: archive coverage correlates with popularity;
  applied to both sides identically, but the matched sample skews popular.
- **Stub-date ambiguity**: RoyalRoad does not expose when stubbing happened;
  follower/pages metadata is as-of enumeration (post-deal for y=1). Matching
  on it is therefore approximate; followers also keep accruing after stubbing.
- **Label noise in y=0**: a control may have a publishing deal without stub
  status (author kept all chapters up), or may get stubbed later.
- **Earliest-chapters approximation**: we take the 3 lowest *archived*
  chapter ids; if chapter 1 was never captured, "rank 1" is the earliest
  archived chapter, not necessarily chapter 1.
- Old `royalroadl.com` (pre-2018 domain) captures are not queried.
- Anti-scrape watermarks: RoyalRoad injects hidden (`display:none` /
  `speak:never`) watermark elements with randomized CSS classes; the
  extractor parses inline `<style>` blocks and strips them.

## Presentation normalization

`text` = `display(raw_text)`: NFKC, curly→straight quotes, em/en dash →
`--`/`-`, ellipsis → `...`, strip wrapper quotes, collapse whitespace
(paragraph breaks kept). `raw_text` retained. This is mandatory after the
cross-source-formatting-leak incident (fake 0.996 AUC from typographic quotes).

## Splits

Stable hash on fiction id: `int(md5(str(fid)).hexdigest(),16) % 10` →
0–7 train, 8 eval, 9 test. Never seeded shuffle.

## Pipeline (scripts here; data on sk3)

Data root: `sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/royalroad_stubs/`
(`raw/` listings+CDX+chapter jsonl, `built/royalroad_stubs_v1.jsonl`, `logs/`).

1. `scrape_search_listings.py --mode stubs` — enumerate STUB fictions (~1.5K).
2. `scrape_search_listings.py --mode controls --max-pages 2000` — control pool
   (followers-desc listing, 40K), STUB-labeled cards excluded.
3. `prioritize_controls.py` — per-stub k-NN, round-robin priority order for
   control recovery.
4. `recover_wayback.py --side {stub,control}` — CDX per fiction → earliest
   substantial capture of the 3 lowest-id chapters → prose extraction
   (≥200 words). Checkpointed (`raw/*_done.txt`), append-only jsonl.
5. `build_dataset.py` — symmetric availability filter, greedy 1:1 match,
   per-chapter rows, normalization, stable splits.

Politeness: ≤1 req/s per host, UA carries contact email, exponential backoff
on 429/5xx. Wayback from sk3 is flaky (~50% timeouts) — retries handle it.
