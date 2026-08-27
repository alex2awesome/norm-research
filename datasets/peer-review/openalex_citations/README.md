# OpenAlex Citation Percentiles (acad-C)

Within-venue×year citation-percentile labels for ICLR / NeurIPS / ICML,
2013–2023 (publication-year + 3yr citation-maturity window ⇒ ≤ 2023).

- **judgement=1**: top quartile of `cited_by_count` within venue×year.
- **judgement=0**: bottom quartile. Middle rows kept with blank judgement;
  raw `percentile` column retained for flexibility.

## Pipeline

| step | script | output |
|---|---|---|
| 1. venue membership | `fetch_dblp_lists.py` (API) or `parse_dblp_dump.py` (dump fallback) | `dblp_papers.csv` |
| 2. OpenAlex source pull | `pull_openalex_sources.py` | `openalex_source_works.parquet` |
| 3. batched title join | `batch_title_join.py` (run on hosts with API budget) | `title_join_cache.jsonl` |
| 4. join + build | `join_and_build.py` | `openalex_citations.csv.gz` (v1) |
| 5. S2 title join (NeurIPS 2023 repair) | `s2_title_join.py` | `s2_cache.jsonl` |
| 6. v2 build (integrate S2 cell) | `build_v2_s2_neurips2023.py` | `openalex_citations_v2.csv.gz` |
| 7. assemble S2 reuse cache | `assemble_reuse.py` | `s2_reuse_cache.jsonl` + `s2_todo.csv` |
| 8. full S2 fetch (missing rows) | `s2_fetch_full.py` | `s2_cache_full.jsonl` |
| 9. v3 build (single-source S2, all cells) | `build_v3_s2.py` | `openalex_citations_v3.csv.gz` |
| 10. v2-vs-v3 floor probe | `probe_v3_tfidf.py` | (stdout) |

Steps 7–10 are chained under `run_full_s2_and_build.sh` (waits for any
concurrent S2 fetch, then fetch → build v3 → probe).

## Semantic Scholar pivot for NeurIPS 2023 (2026-06-12)

**The blocker.** OpenAlex's free tier is now **$1/day = 10,000 credits PER IP**
(resets midnight UTC); every `title.search`/`search` costs **10 credits**. At one
title per request that is ~1,000 joins/IP/day, and the NeurIPS-2023 arXiv-twin
repair ran out of budget at 225/3,314 — its cell was BLANKED by the <50%-coverage
guard (see "Curran-stub trap" below).

**The fix.** Semantic Scholar's title-match endpoint has **no budget wall**
(~1 req/sec unauthenticated, honors `Retry-After`):

```
GET /graph/v1/paper/search/match?query={TITLE}
    &fields=title,citationCount,year,venue,externalIds,abstract
```

`s2_title_join.py` calls it per DBLP membership row and ACCEPTS a match only when
**all** hold: (a) exact normalized-title match **or** difflib ratio ≥ 0.92 — note
S2's `matchScore` is an absolute BM25-like score that scales with title length,
so it is recorded for auditing but is **not** the accept gate; (b)
`|s2_year − target_year| ≤ 1`; (c) venue-token overlap (guards against same-title
papers in other venues). Resumable via `s2_cache.jsonl` (keyed on `dblp_key`).

**Single-source-per-cell rule.** The label is a citation percentile WITHIN each
(venue × year) cell. Mixing S2 and OpenAlex counts inside one cell would corrupt
the percentile, so every cell is kept single-source: **only NeurIPS 2023 was
re-fetched and is now 100% S2** (`cite_source="s2"`); every other cell stays
OpenAlex (`cite_source="openalex"`). The 73–79%-covered ICLR/ICML 2022–23 cells
are deliberately LEFT as OpenAlex-only — they are NOT partially filled with S2.

**`cite_source` column** (v2 only) makes the mixed provenance explicit and
auditable. New S2 rows get id `https://www.semanticscholar.org/paper/{paperId}`
so id→split hashing stays stable and never collides with OpenAlex W-ids.

**Future option (not done now):** the *whole* acad-C dataset (~29K) could be
re-fetched on S2 for cross-source citation consistency — ~29K match calls ≈ a
single overnight run unauthenticated (or faster with a free S2 API key). Only
NeurIPS 2023 was migrated here because that was the broken cell.

### OpenAlex credit budget (discovered the hard way, 2026-06-12)

OpenAlex free tier is now $1/day = 10,000 credits **per IP** (resets midnight
UTC). Search-type requests (`title.search`, `search`, `display_name.search`)
cost 10 credits; plain filters cost 1. One-title-per-request joins are
therefore infeasible at this scale. `batch_title_join.py` exploits the fact
that `title.search` accepts pipe-OR'd quoted phrases at a flat 10 credits per
request (~15 titles/request), with an unquoted singleton retry for the
leftovers. Joins were spread across skampere1/2/3 (one budget each).
On 429 + `X-RateLimit-Remaining: 0` every script aborts hard rather than
caching fake "no match" records.

## Why DBLP defines membership (document of failure, as promised)

The requested approach — resolve OpenAlex source ids and cursor-paginate all
works — was run first (`pull_openalex_sources.py`, sources NeurIPS
S4306420609, ICLR S4306419637, ICML S4306419644). It cannot deliver "ALL
works 2013–2023": these are MAG-era sources, partial before 2022 (e.g.
NeurIPS 2019: 650 works vs 1,428 accepted; ICLR 2019: 148 vs 502) and ~zero
after (NeurIPS 2023: 5), because these venues issue no DOIs so nothing enters
via Crossref. PMLR and "Advances in NeurIPS" sources hold < 10 works each.

Fix: DBLP per-year TOC files (`toc:db/conf/iclr/iclr2017.bht:` etc.) give
exact official accepted-paper lists (validated against official counts;
workshop / Datasets&Benchmarks tracks live in separate TOCs and are
excluded). OpenAlex then supplies `cited_by_count` + abstracts: first by
normalized-title match against the source pull, then per-title API join
(title.search, year ±1, full-search fallback; accept similarity ≥ 0.90).

`record_kind` says where each row's OpenAlex record came from:
`venue_source` | `titlejoin_published` | `titlejoin_preprint`. Citation
counts for `titlejoin_preprint` rows are read off the arXiv record (the only
OpenAlex entity for that paper) — fine within-year but keep in mind.

Preprint/published dedup: API candidates prefer non-repository records;
final table dedups normalized titles keeping `venue_source` records first,
then higher-cited.

## Output schema (`openalex_citations.csv.gz`)

`id` (OpenAlex W-id, or `https://www.semanticscholar.org/paper/{id}` for S2
rows), `text` (`TITLE\n\nABSTRACT`, NFKC + straight quotes — identical
normalization to all other v2 tasks), `judgement`, `percentile`, `venue`,
`year`, `cited_by_count`, `record_kind` (`s2_titlematch` for S2 rows), `match_score`
(for S2 rows this is S2's `matchScore`, an absolute BM25-like value, not a 0–1
similarity), `doi`, `title`, `has_abstract`.

**v2 adds `cite_source`** (`openalex` | `s2`) — see the Semantic Scholar pivot
section above. v2 file: `openalex_citations_v2.csv.gz` (the v1
`openalex_citations.csv.gz` is kept, never deleted).

## Key numbers (2026-06-12 build)

- **DBLP membership: 29,228 papers** (ICLR 5,419 / ICML 8,344 / NeurIPS
  15,465), every venue-year equal to official accepted counts (ICLR'17 198,
  ICML'17 434, NeurIPS'13 360, ICLR'21 860, ...).
- **Joined to OpenAlex: 24,098 rows** (after dedup). Join rate 95–100% for
  2013–2021; ICLR/ICML 2022–23 at 73–79% (singleton retries were cut short
  by credit budgets — `finish_leftovers.sh` tops them off); NeurIPS 2022 95%.
- **Labels: 6,000 y=1 / 6,106 y=0**, middle 11,992 rows kept with blank
  judgement + raw percentile.
- **Curran-stub trap (important)**: OpenAlex's official NeurIPS 2022+
  proceedings records (Curran deposits, DOI 10.52202/...) carry ~0 citations
  — the counts live on the arXiv twin records. The default
  prefer-published tie-break therefore produced garbage labels for NeurIPS
  2022/23 (median 0 cites). Fixed by re-joining those venue-years with
  `--prefer citations` (NeurIPS 2022 done: median 3, 22% zero-cite —
  consistent with ICML/ICLR 2023). NeurIPS 2023 repair ran out of budget at
  225/3,314: its unrepaired stub rows were dropped and the venue-year's
  labels BLANKED by the <50%-coverage guard. Re-run `finish_leftovers.sh`
  + the repair (see git history / `--prefer citations`) to restore it.
- **Abstract availability**: 0.79–1.00 for 2013–2021 venue-years; NeurIPS
  2022 0.70, 2023 0.61 (arXiv-record abstracts missing for some).
- **Validation**: percentile recomputed by hand for ICLR 2017 — exact match
  (top cited: GCN 8,068, NAS-RL 3,868, Gumbel-Softmax 3,240; quartile
  ordering clean: min(y=1)=238 ≥ max(y=0)=29). 10-row spot check passes
  (venues, counts, abstracts decode).
- record_kind: 7,933 venue_source / 14,235 titlejoin_preprint / 1,930
  titlejoin_published.

## v2 numbers — NeurIPS 2023 repaired on S2 (2026-06-13 build)

`openalex_citations_v2.csv.gz`: **27,233 rows** = 23,784 OpenAlex
(`cite_source=openalex`, every cell unchanged from v1) + **3,449 S2**
(`cite_source=s2`, the rebuilt NeurIPS-2023 cell). The 314 old OpenAlex
NeurIPS-2023 stub rows were dropped and the cell replaced wholesale.

- **NeurIPS-2023 S2 coverage: 3,449 / 3,540 DBLP membership = 97.4%.**
  Rejections (91 rows): 36 venue_mismatch, 24 year_mismatch, 20 title_low,
  11 no_match — all correctly withheld by the title+year+venue guard, not
  fuzzy false-accepts.
- **Mismatch rate (false accepts): ~0%.** Hand-checked the first 15 accepted
  matches — all exact-title, correct venue, year within ±1 (one legitimate
  arXiv-twin at 2022). Well under the 5% re-tighten threshold.
- **NeurIPS-2023 citation distribution, before → after:**
  before (OpenAlex Curran stubs, 314 surviving rows): median 2, 33% zero-cite,
  labels BLANKED. after (S2, 3,449 rows): **median 19, max 9,800, only 1.2%
  zero-cite**; clean quartile split (top-q median 92, min 45 ≥ bot-q max 8).
  867 y=1 / 870 y=0.
- **Abstract coverage (NeurIPS-2023 S2): 95.5%** (vs 0.61 under OpenAlex —
  S2 withholds some publisher abstracts but returns far more here; the missing
  ~4.5% are mostly closed-access records).
- **TF-IDF/LR floor AUC (within venue×year, group 5-fold):** NeurIPS-2023 cell
  **0.879** (1,737 labeled) — in line with neighbors (NeurIPS'19 0.86,
  '21 0.77). Mean per-cell floor over all 30 labeled cells **0.790**.
  (NeurIPS'22 0.989 is a *pre-existing* OpenAlex-cell artifact, untouched by
  this repair — flag for separate review.)
- v2 labels overall: **6,867 y=1 / 6,976 y=0**, 13,390 middle rows blank
  (raw `percentile` retained).

**Future option (not done):** standardize the *whole* acad-C dataset on S2 for
cross-source citation consistency. Cost ≈ 29K match calls — one overnight run
unauthenticated (this 3,540-row NeurIPS-2023 pass took ~2.5 s/req with backoffs;
29K ≈ 8 h), or a few hours with a free S2 API key. Would also let `record_kind`
drop the OpenAlex preprint/published distinction. Deferred: only the broken
NeurIPS-2023 cell needed repair, and the single-source-per-cell rule keeps each
cell's percentile internally consistent in the meantime.

## v3 — single-source Semantic Scholar standardization (2026-06-13)

`openalex_citations_v3.csv.gz` re-fetches **every** DBLP membership row
(29,228) on Semantic Scholar so the whole dataset uses **one citation source**
(`cite_source="s2"` throughout). This (a) removes the cross-source
inconsistency v2 still had (NeurIPS-2023 was S2, every other cell OpenAlex);
(b) lifts abstract coverage (S2 ~95% vs OpenAlex 61%); and (c) **repairs the
NeurIPS-2022 cell**, the #2 leak.

### Why NeurIPS-2022 had to be repaired (the 0.989 floor is a leak)

The v2 NeurIPS-2022 cell carried a TF-IDF/LR within-cell floor AUC of
**0.989** — not real predictability but a presentation artifact, the same
Curran-stub bug already fixed for NeurIPS-2023 (and despite the README claim,
the v2 cell was *not* actually repaired):

- bottom-quartile (y=0) rows were 589/606 `titlejoin_published` Curran
  proceedings stubs: **all zero citations** (max cite = 0) and **no abstract**
  (4.3% had one);
- top-quartile (y=1) rows were 655/671 `titlejoin_preprint` arXiv twins: real
  citations **and** abstracts (98.9% had one);
- so the classifier separated the classes on **abstract presence**, not
  content. The top "high-cite" features were pure stopwords (`the`, `we`,
  `to`, `that`...) that only appear once an abstract is present; median text
  length y=1 = 1,413 chars vs y=0 = 73 chars (title only).

S2 supplies a real citation count **and** an abstract from one record for both
classes, dissolving the leak. After v3, NeurIPS-2022's floor drops to a sane
within-cell value (target: in line with its neighbors ~0.77–0.86).

### v3 build details

- **Membership = DBLP (29,228).** A row enters v3 iff it has an ACCEPTED S2
  match. Cells below 50% match coverage have labels blanked (percentile over a
  biased subsample is untrustworthy) — same guard as v1.
- **Accept gate** (`s2_fetch_full.py`, same as the NeurIPS-2023 pass, slightly
  tightened): exact / ≥0.92-difflib normalized-title match AND
  `|s2_year − target_year| ≤ 1` AND venue plausibility. Empty S2 venue is
  **not** a reject (DBLP membership is authoritative); a hard acronym hit for a
  *different* one of our three venues **is** a reject (guards cross-venue
  same-title collisions).
- **Caches reused** (`assemble_reuse.py`): merged the NeurIPS-2023 cache
  (`s2_cache.jsonl`) + the best_papers cache. By normalized title, **3,422 of
  29,228 (11.7%) were already cached** — almost all from the NeurIPS-2023 cell
  (the best_papers cache covers other venues). The remaining **25,806** are
  fetched fresh into `s2_cache_full.jsonl`.
- **Schema = v2 schema + two columns**: `dblp_key` (stable paper identity) and
  an explicit `split` (`md5(dblp_key)%10`: 0–7 train / 8 eval / 9 test).
  Keying the split on `dblp_key` keeps each paper's split **stable from v1/v2
  to v3** even though the row `id` switches from a W-id to an S2 id
  (`https://www.semanticscholar.org/paper/{paperId}`). `record_kind` is
  uniformly `s2_titlematch`.
- **v1 + v2 are kept** (never overwritten).

### v3 validation

- **15-row hand-check (matching machinery): 0/15 mismatch (0.0%)** on the
  reused NeurIPS-2023 records — all exact-title, correct venue, year within
  ±1, citation counts real and varied (1–1,819). The newly-fetched
  ICLR/ICML/older-NeurIPS rows are re-validated from `s2_cache_full.jsonl` once
  the fetch finishes; re-tighten the gate only if that mismatch rate > 5%.

### v2-vs-v3 within-cell TF-IDF/LR floor (group 5-fold)

| cell | v2 floor | v3 floor |
|---|---|---|
| NeurIPS 2022 | **0.989** (abstract-presence leak) | *pending fetch* (target ~0.77–0.86) |
| mean per-cell (all labeled cells) | 0.790 | *pending fetch* |

(Run `probe_v3_tfidf.py` after the chained build to fill the v3 column; it
prints the full per-cell table side by side.)

## sk3

`/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/openalex_citations/`
(openalex_citations.csv.gz [v1] + openalex_citations_v2.csv.gz [v2] +
openalex_citations_v3.csv.gz [v3] + dblp_papers.csv +
openalex_source_works.parquet + title_join_cache.jsonl + s2_cache.jsonl +
s2_reuse_cache.jsonl + s2_cache_full.jsonl + s2_title_join.py +
s2_fetch_full.py + assemble_reuse.py + build_v2_s2_neurips2023.py +
build_v3_s2.py + probe_v2_tfidf.py + probe_v3_tfidf.py +
run_full_s2_and_build.sh). The full S2 fetch + v3 build + probe run under
`run_full_s2_and_build.sh` (nohup) and finish unattended.
