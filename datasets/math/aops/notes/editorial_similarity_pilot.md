# Editorial-similarity pilot (AoPS forum ↔ wiki join)

*2026-06-11. Math analog of the competitive-coding "similarity to editorial"
treatment: do forum solutions that resemble the wiki's canonical ("editorial")
solutions earn more community thanks?*

## Pipeline

| Step | Script | Output |
|---|---|---|
| Wiki-page link census | `scripts/build_wiki_topic_map.py` | `wiki/wiki_forum_links.csv` (164 direct links — too sparse to be the join) |
| Salvage + post extraction (sk3) | sk3 `scripts/salvage_gz_priority.py` | sk3 `priority_posts.jsonl.gz` → laptop `forum_priority_posts.jsonl.gz` |
| Join by statement matching | `scripts/build_forum_wiki_join.py` | `wiki/forum_wiki_join.parquet` |
| Solution-post filter | `scripts/build_forum_solutions.py` | `forum_solutions.parquet` (+ copy at sk3 `/lfs/skampere3/0/alexspan/aops/forum_solutions.parquet`) |
| Similarity + analyses | `scripts/editorial_similarity_analysis.py` | `editorial_similarity.parquet` |

## How the join was actually built (no id-level mapping existed)

- The wiki dump's wikitext contains only **164** forum links across 4,937
  problem pages — the priority topic ids (6,780) were instead harvested from
  community **Contest Collection** categories
  (`scripts/fetch_contest_collection_ids.py`), whose per-problem structure was
  never persisted. The crawled ajax payload has **no title/source field**
  (`data_to_set_direct_on_topic` = counts only; tags are subject areas).
- So the join is **text matching**: forum *first posts* (the problem-statement
  repost) vs wiki `problem_statement`, char 3–5-gram TF-IDF cosine after
  markup normalization (BBCode vs wikitext, legacy `\minus{}/\equal{}`
  escapes, HTML comments, `<onlyinclude>`, asy blocks).
- Acceptance: cosine ≥ 0.75 outright; 0.55–0.75 requires numeric-literal
  overlap ≥ 0.5 (true reposts share their numbers + answer choices).
  Near-duplicate secondaries (≥ 0.85, within 0.02 of best) keep the
  AMC10/12 shared-problem double-pages.
- **Validation:** all 23 wiki pages with a direct topic link that text-matching
  also covered agree with the text match (one apparent disagreement turned out
  to be a stray wiki link: topic 489748 is IMO 2023 P2, not 2018 AMC12A P23).
  Eyeballed band samples: ~7/8 correct in the gated 0.55–0.75 band; residual
  trap = different problems with identical answer-choice lists.

### Crawl-side data loss (gzip-append corruption)

Only 3,747/6,780 priority topics were in the plain `.jsonl` shards. The legacy
`.jsonl.gz` shards are corrupted (long-running gzip append), but scanning for
gzip-member magic offsets and decompressing members independently salvaged
2,004 more → **5,749/6,780 topics with posts (85%)**; 1,029 ids remain
unrecoverable until re-crawled.

## Coverage

3,484/5,681 usable first posts matched → **3,437/4,877 wiki problems (70.5%)
have ≥1 matched crawled thread with posts**; 3,186 problems have ≥1 retained
solution post.

| contest | problems covered | coverage |
|---|---|---|
| AMC8 | 614/650 | 94.5% |
| AIME | 739/1035 | 71.4% |
| USAJMO | 68/96 | 70.8% |
| AMC10 | 828/1200 | 69.0% |
| AMC12 | 827/1200 | 68.9% |
| USAMO | 184/300 | 61.3% |
| IMO | 177/396 | 44.7% |

Low IMO coverage: older IMO threads phrase statements differently
(translations / original wording vs wiki transcription), and the 1,029
unrecovered topics bite here too.

## Forum solutions table

From 85,964 posts in matched threads: 82,433 replies → **32,120 retained
solution posts** (post_number > 1, ≥200 chars of non-quote content, math
markup present, bump/thanks posts dropped). Per problem: median 4, mean 10.1,
max 182. No thread pagination truncation (all topics fully fetched).
Counts by contest: AIME 7,194 / IMO 7,904 / USAMO 5,278 / AMC10 3,537 /
AMC12 3,464 / USAJMO 3,258 / AMC8 1,485.

## Similarity (TF-IDF cosine, max over the problem's wiki solutions)

Two featurizations on normalized text, quotes stripped: `sim_char`
(char_wb 3–5) and `sim_word` (word 1–2); Spearman(sim_char, sim_word) = 0.80.

| | mean | p25 | median | p75 |
|---|---|---|---|---|
| sim_char | 0.261 | 0.159 | 0.234 | 0.325 |
| sim_word | 0.178 | 0.086 | 0.142 | 0.220 |

By contest (sim_word mean): AIME .228, AMC8 .227, AMC10 .205, AMC12 .205,
USAJMO .158, USAMO .137, IMO .128 — proof contests sit lower (longer, more
heterogeneous writeups); short computational solutions align more.
1.4% of forum posts have sim_word > 0.9 — these are essentially *the* wiki
solution (wiki transcribers lift forum posts, so causality runs forum → wiki).

### Eyeballed pairs (sanity check)

- **High** (2014 USAJMO P1, sim_word .65): forum and wiki both open with the
  `(a-1)^5 ≥ 0` expansion trick — same approach, near-identical algebra. ✔
- **High** (1998 USAMO P1, sim_word .73): both use the mod-5 ≡ 1 observation +
  parity + CRT. Same proof, different ordering. ✔
- **Low** (2006 IMO P5, sim_word .02): forum post is an informal sketch
  ("the only gun I used was x−y | P(x)−P(y)"), wiki is a structured
  lemma-based proof. Genuinely different register/approach. ✔

High-sim pairs really do share the approach; the metric measures what we want
(at the surface level).

## (b) Does editorial-likeness predict thanks? **No — if anything, slightly negative, and it's an artifact.**

| analysis | result |
|---|---|
| pooled Spearman sim_word vs thanks | **−0.093** (p≈1e-60, n=31,192) |
| within-problem (≥3 sols, n=2,122 problems) mean ρ | **−0.035** (median −0.040, 42% positive, Wilcoxon p=8e-5) |
| control: post_number vs thanks within problem | **−0.550** mean ρ (early-mover advantage dominates thanks) |
| sim vs thanks within problem, **post_number partialled out** | mean ρ −0.019, p=0.10 (char: −0.008, p=0.61) → **null** |

Thanks are overwhelmingly a recency/position phenomenon (first replies
accumulate thanks for years). Once post order is controlled, editorial
similarity carries no detectable thanks signal.

## (c) Correctness-conditioned (AIME/AMC, answer keys available)

Answer extraction (`\boxed{...}` brace-balanced, then trailing `(A)–(E)` /
last-integer fallbacks): extracted for 65% of 15,642 solutions on answerable
problems (AIME 97%, AMC 33–48% — AMC answers are letters and often omitted);
76% of extracted answers match the key.

- **Correct solutions are much more editorial-like**: sim_word mean 0.262 vs
  0.155 for incorrect (Mann-Whitney p≈1e-256). Similarity to canonical is a
  real *correctness* signal.
- **But thanks don't reward correctness**: within-problem correctness vs
  thanks mean ρ = −0.076 (p=1e-6) — incorrect-extraction posts get *more*
  thanks. Likely an age artifact: "incorrect" posts are older (median post
  year 2019 vs 2021) and older posts both accumulate thanks and predate
  `\boxed` conventions (extraction errors masquerade as incorrect).
- **Taste-within-correct: null.** Among correct solutions, within-problem
  sim vs thanks mean ρ = 0.004 (p=0.94); post-order-partialled −0.006
  (p=0.77). Editorial-likeness does not predict community appreciation once
  correctness is fixed.

## Limitations

1. **Thanks is a poor preference label as-is** — dominated by post position
   (ρ ≈ −0.55) and thread age. Any downstream use must control position/age
   or use within-(problem, era) contrasts.
2. **Wiki↔forum circularity**: wiki solutions are frequently transcribed from
   these very threads (1.4% near-verbatim, many more partial). High similarity
   sometimes means "this post became the editorial", not "this post imitates
   the editorial".
3. **Surface similarity only**: TF-IDF cosine conflates approach identity with
   notation/verbosity. Proof contests (IMO/USAMO) get depressed sims partly
   from writeup-length mismatch.
4. **Join noise**: ~1–2% residual mismatch (identical answer-choice lists trap
   the numeric gate); IMO coverage only 45%.
5. **Answer extraction** under-covers AMC (letters often not stated) and
   mislabels some older posts → the correctness analyses are AIME-weighted.
6. **BBCode vs wikitext normalization**: legacy `\minus{}/\equal{}` escapes,
   `[asy]`/`<asy>` figures (dropped), wiki `<imath>/<cmath>` wrappers, HTML
   comments — all handled, but tabular/eqnarray formatting differences still
   depress sims for older posts.

## Recommended next steps

1. **Embedding-based similarity** (e.g. math-tuned sentence embeddings or
   Qwen embedding on sk3) to capture approach identity past notation; compare
   against TF-IDF on the eyeball set.
2. **LLM approach-classification**: label (forum, wiki) pairs as
   same/different approach directly; gives a cleaner "editorial-likeness"
   treatment and a typology of non-canonical approaches.
3. **De-confounded thanks label**: model thanks ~ post position × thread age,
   use residuals; or restrict to posts within the same week of the thread.
4. **Re-crawl the 1,029 unrecovered topics** (and the IMO statement-mismatch
   tail) to push coverage toward 85–90%.
5. **Use correctness, not thanks, as the first-class label** for AIME — the
   sim→correctness link (0.262 vs 0.155) is the strongest signal found here.

## Update 2026-06-11: de-confounded thanks (next-step #3 done)

See `notes/thanks_deconfounded.md` + `scripts/deconfound_thanks_analysis.py`
(label in `thanks_deconfounded.parquet`, col `thanks_resid`). OOF
gradient-boosted residualization of log1p(thanks) on position/age/popularity
(R²=0.52) zeroes the confounds (post_number pooled ρ −0.323→−0.004;
within-problem −0.550→+0.061 slight overshoot). Re-tests: the raw negative
sim→thanks (−0.093) was pure artifact — de-confounded it's +0.029 pooled and
null within-problem; taste-within-correct stays **null** for similarity
(AUC 0.50), but **length** emerges as a genuine style signal among correct
solutions (ρ=+0.109, AUC 0.594). The perverse correctness→thanks −0.076 also
dies (→ +0.014, p=0.37), confirming the age-artifact hypothesis.
