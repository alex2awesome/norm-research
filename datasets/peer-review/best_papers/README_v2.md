# Best Papers v2 (acad-B2) — DBLP + Semantic Scholar rebuild

Rebuild of the best-paper-award taste cell that sidesteps OpenAlex's broken
CS-flagship venue pools. **v1 (OpenAlex)** could only place 311 awards in a
resolved venue-year (429 in any resolved pool) because OpenAlex has no usable
conference source for PLDI/SOSP/STOC/CHI/CVPR/ISCA/SIGCOMM/SIGGRAPH/UIST/PODS
and several others. **v2 enumerates proceedings membership from DBLP** (which
defines venue × year membership cleanly) and joins citations + abstracts from
**Semantic Scholar** (S2), the same machinery the acad-C citations cell used.

The original `best_papers_awards.csv` / `best_papers_labels.csv` / `pools/`
are **never overwritten**; v2 lives under `v2_build/` and writes
`built/best_papers_v2_{train,eval,test}.csv.gz`.

## Method

1. **DBLP venue-year pools** (`v2_build/`): the award list spans 32 venues ×
   688 venue-years (1996–2025). We pull the full proceedings membership of
   each award venue-year from the static DBLP XML dump
   (`parse_dblp_best_papers.py dblp.xml.gz`). The dump is used instead of the
   DBLP search API because the API rate-limits per-IP within minutes (the API
   fetcher `fetch_dblp_best_papers.py` stalled on 429/conn-error backoff after
   a single venue-year). Membership = `<inproceedings>` whose `<crossref>`
   matches an anchored main-track regex (`^conf/pldi/2019$` etc.; suffixed
   workshop/companion/datasets-and-benchmarks volumes excluded). Recent
   CS-flagship years moved to **issue-numbered journals** and are recovered by
   `recover_journal_venues.py`:
     - PLDI 2023+ → PACMPL (`number="PLDI"`, vol = year−2016)
     - FSE 2024+ → PACMSE (`number="FSE"`, vol = year−2023)
   And journal-published proceedings handled in the main parser:
     - VLDB → PVLDB (vol = year−2007), SIGGRAPH → ACM TOG (vol = year−1981),
       SIGMETRICS 2017+ → POMACS (vol = year−2016),
       SIGMOD 2023+ → PACMMOD (vol = year−2022).

   **Coverage: 664 / 688 venue-years enumerated, 198,027 pool papers,
   1,773 / 1,819 awards now fall in an enumerable venue-year** (vs old 311
   usable / 429 in a resolved pool — a 5.7× recovery). All previously-broken
   CS flagships recovered: PLDI, SOSP, STOC, FOCS, CHI, CVPR, ISCA, SIGCOMM,
   SIGGRAPH, UIST, PODS, ICCV, S&P, INFOCOM, MOBICOM. The 46 still-missing
   awards are mostly pre-2000 years (DBLP crossref naming changes:
   PODS/SIGMOD/SIGIR/WWW 1996–2000) and a couple of 2025 venue-years not yet
   in the dump (NeurIPS 2025).

2. **S2 citation + abstract join** (`s2_join_best_papers.py`, reused from
   `openalex_citations/s2_title_join.py`): for each pool paper,
   `GET /graph/v1/paper/search/match?query={TITLE}&fields=title,citationCount,year,venue,externalIds,abstract`.
   Accept iff exact / ≥0.92-difflib normalized-title match AND |year−target|≤1
   AND venue-token plausibility (DBLP membership is authoritative, so an empty
   or generic S2 venue string is not a reject). Resumable `s2_cache.jsonl`,
   cache key = `dblp_key`. **Hand-validated 30 awards-first matches: 29/29
   accepted matches correct (0% mismatch); the 1 rejection was a genuine S2
   wrong-paper return correctly caught by the title guard.** Accept rate ~95%.
   S2 unauthenticated throttles to ~5-request bursts then 429 (no Retry-After);
   a fixed 2.5 s backoff keeps a steady drip. The full 27,025-paper fetch
   (awards + ≤40 stable-sampled non-awards per venue-year, prioritized so the
   recovered CS flagships complete first) runs under nohup.

3. **Dataset** (`build_dataset.py`): y=1 = award papers, y=0 = same-venue-year
   non-award papers (DBLP pool minus awards). **Abstract required** (S2).
   X = `TITLE\n\nABSTRACT`, presentation-normalized (NFKC, curly→straight
   quotes, en/em/minus dashes→hyphen, ellipsis→`...`, whitespace collapsed);
   `title_raw`/`abstract_raw` kept. Metadata: `cited_by_count` (S2
   citationCount), `venue`, `year`. Stable split by `md5(dblp_key)%10`:
   0–7 train, 8 eval, 9 test.

4. **Citation deconfound** (`deconfound_bounds.py`). Awards are more cited —
   partly *why* they won, but committees pick **before** most citations accrue,
   so `cited_by_count` is **post-treatment**. We report three numbers honestly:
   - **(a) citation-only AUC** — the confound (a no-text model on log-citations).
   - **(b) raw text TF-IDF floor** — text signal, citation-entangled.
   - **(c) citation-decile-matched text floor** — match y=1/y=0 on
     `cited_by_count` deciles **within venue-year** (Math.SE v3.3 style), then
     re-measure the text floor; this is the de-confounded text signal.
   - **(d) within-venue-year stratified AUC** — awards are budget-bound B1
     (~1–3 / vy), so we also rank within each venue-year.

5. **Bounds + V** (`deconfound_bounds.py`, `v_feature_probe.py`): TF-IDF floor
   (raw + citation-matched), top features (venue/citation leakage watch),
   text-length confound, and a deterministic **V-feature probe** that reuses
   the per-aspect Python `score(text)` programs at
   `runs/validity_full/v2/peer_review/codegen_claude/` (654 programs) to confirm
   V > 0. Optional bge upper bound if GPU 1 is free.

## Files (`v2_build/`)
- `parse_dblp_best_papers.py` — DBLP XML-dump → `dblp_pool.csv` (32 venues)
- `recover_journal_venues.py` — PACMPL/PACMSE issue-number recovery (PLDI/FSE)
- `fetch_dblp_best_papers.py` — DBLP search-API fetcher (fallback; rate-limited)
- `build_s2_input.py` — awards + ≤40 stable non-awards/vy → `s2_input.csv`
- `s2_join_best_papers.py` — S2 match join → `s2_cache.jsonl`
- `build_dataset.py` — join + normalize + split → `best_papers_v2_full.csv.gz`
- `deconfound_bounds.py` — 3 AUCs + within-vy + features + length
- `v_feature_probe.py` — deterministic V-feature AUC

## sk3
`/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/best_papers/v2_build/`
(`dblp.xml.gz`, `dblp_pool.csv`, `award_dblp_keys.csv`, `s2_input*.csv`,
`s2_cache.jsonl`, built CSVs). Outputs in `best_papers/built/`.

## Results

### DBLP venue recovery (complete)
- **664 / 688 venue-years enumerated** (vs OpenAlex: only the ~half of venues
  with usable pools), **198,027 pool papers**, **1,773 / 1,819 awards now in an
  enumerable venue-year** (old: 311 usable / 429 in any resolved pool).
- **CS flagships recovered** (all 0% in OpenAlex): PLDI, SOSP, STOC, FOCS, CHI,
  CVPR, ISCA, SIGCOMM, SIGGRAPH, UIST, PODS, ICCV, S&P, INFOCOM, MOBICOM —
  plus journal-published PLDI 2023+ (PACMPL) / FSE 2024+ (PACMSE) via
  issue-number recovery, and VLDB/SIGGRAPH/SIGMETRICS/SIGMOD journal volumes.
- 1,743 awards matched to a DBLP key (exact 1,713 + fuzzy ≥0.92 30). Pool
  membership spot-check: SOSP'19 (38), STOC'19 (113), CHI'13 (392), SIGGRAPH'24
  (283), CVPR'16 (643) — 100% of awards found in their own venue-year pool.
- 46 awards still unrecovered: pre-2000 PODS/SIGMOD/SIGIR/WWW (DBLP crossref
  naming changes) + a few 2025 vy not yet in the dump.

### S2 join
- Match gate validated by hand on 30 awards-first matches: **0% mismatch among
  accepted (29/29 correct); the 1 reject was a real S2 wrong-paper, caught.**
  Accept rate ~95%.
- **Abstract coverage is the binding constraint.** S2 abstracts are patchy for
  older ACM/IEEE CS papers (CHI/UIST/PLDI/ICSE pre-2018 often lack them); recent
  + ML venues are well-covered. We **require S2 abstracts on BOTH sides** (no
  OpenAlex backfill) to avoid a cross-source abstract-presence leak — at the
  cost of size, skewed toward recent/abstract-rich venue-years.
- Throughput: S2 unauthenticated throttles to ~5-request bursts then 429
  (no Retry-After); fixed 2.5 s backoff → **~22 accepts/min sustained**. The
  full 27,025-paper fetch is **~20 h overnight under nohup** (prioritized so the
  recovered CS flagships and awards complete first).

### Preliminary subset (fetch in progress; PLDI-heavy)
A 443-paper preliminary slice (121 awards, 322 non-awards, 13 venues but
PLDI-dominant: 363/443) confirms the full pipeline produces every required
number:
| metric | AUC | note |
|---|---|---|
| (a) citation-only | **0.637** | the confound |
| (b) raw text TF-IDF | 0.838 | **venue-confounded** (PLDI non-awards vs CHI/CVPR awards) |
| (c) citation-decile-matched text | n/a on this slice | matched test set too small (1:1 per-decile match) |
| (d) **within-venue-year text** | **0.583 mean / 0.700 median** | venue-controlled, the honest number |
| V-feature (654 progs), pooled | 0.742 | V>0 but venue-confounded |
| length-only | 0.351 | awards slightly shorter |

Top pooled TF-IDF features are venue tokens (y=1: users/social/vision/sensing;
y=0: compiler/semantics/program) — i.e. the pooled floor is mostly
venue-detection on this imbalanced slice. **The within-vy stratified AUC is the
trustworthy metric** until cross-venue balance arrives.

### Auto-build on fetch completion
`auto_build_on_done.sh <fetch_pid>` (running under nohup) waits for the S2
fetch to finish, then rebuilds `best_papers_v2_full.csv.gz`, writes splits,
and runs `deconfound_bounds.py` + `v_feature_probe.py` + bge (GPU 1 if free).
The full cross-venue dataset will have a large enough citation-matched test set
for a stable (c), and balanced venues so the pooled floor stops being venue
detection. Watch `v2_build/auto_build.log`.

### Files written
- `built/best_papers_v2_{train,eval,test}.csv.gz` (preliminary; auto-build
  overwrites with full data). Columns: `dblp_key, venue, year, label, title,
  abstract, text, title_raw, abstract_raw, cited_by_count, s2_paper_id,
  s2_year, split`. Split = `md5(dblp_key)%10` (0–7 train / 8 eval / 9 test).
- v1 `best_papers_awards.csv` / `best_papers_labels.csv` / `pools/` untouched.
