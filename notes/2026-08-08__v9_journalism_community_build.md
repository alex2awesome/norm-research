# V9 — Journalism/press VOTE column (crowd): the tweet-engagement cell

Build note for the journalism/press × VOTE-REVEALED cell of the 3×N decomposition
grid (`notes/2026-08-08__vat-3xN-decomposition-grid.md`), where it was listed as
"tweets **UNLABELED (V9)**". Status of this note: build record + ground-truth
report. Headline numbers land in §7 once the arms complete.

Agent: `claude-v9-journalism-tweets`. Coordinator logs the registry and strict
list; this note is the primary artifact record.

---

## 1. Channel choice: TWEETS, with the reddit-news arm documented as the option

The charge named two candidate label channels and required both be inventoried
before any instrument work. Both were inventoried. **The tweet-engagement
channel was chosen.** The evidence and the reasoning, so the choice can be
re-litigated if it turns out wrong:

### What each channel actually is

| | tweet engagement | reddit news arm |
|---|---|---|
| where | `datasets/news-homepages/twitter_engagement/tweet_engagement.jsonl` (sk3) | `datasets/news-homepages/reddit_newsworthiness/built/` (sk3) |
| rows | 52,112 scraped → 47,199 OK | 43,866 balanced (v3 topicstrat); 621K raw |
| label | none built — constructed here | `judgement` already binarized 50/50, splits materialized |
| item text | headline (`anchor_text`) | headline (`title`) — **also headline-only** |
| groups | outlet × day, 731 touched | subreddit × day, 2 subreddits |
| prior work | none | propensity-decile deconfounding, bge k=400 topic-strat, TF-IDF floor .579, register floor .547, leak policy in `COLUMN_MANIFEST.json` |

On raw readiness the reddit arm is plainly further along, and a first pass says
"use reddit". Three things reverse that for **this cell**:

**(a) The grid's journalism row wants a cross-y contrast on ONE population.**
The other two cells in this row are editorial pickup (press releases → coverage)
and homepage placement — both about what these same news organisations produce
and select. The tweet population is *exactly* the homepage-captured article URLs
from the same 9 outlets, carrying the same `anchor_text` headlines the homepage
CURATION cell is built on. So the journalism row's three cells can share a
population, a V bank and an A bank, which is the property the grid explicitly
prizes (peer review is celebrated in that file for "56× cross-y residual spread
on identical text+instruments"). The reddit population overlaps the homepage URL
corpus by only **15,420 URLs (2.3%)** and is dominated by different outlets
(jpost, pravda, yahoo). A reddit vote cell would be a cross-population contrast.

**(b) The within-outlet-day design kills the confound that reddit had to spend
machinery on.** In the reddit arm, per-domain P(y=1) runs from .008 (youtu.be) to
.918 (abcnews.go.com) — submission success is dominated by *which outlet* and
*which topic*, which is why propensity-decile deconfounding and k=400 topic
stratification were required. In V9 the outlet and the day are held CONSTANT
inside the grouping unit, so that confound cannot operate at all. Measured, not
assumed: outlet-day identity alone lands at chance and pooled AUC ≈ within-group
AUC (§5).

**(c) A-bank reuse is population-exact.** `datasets/news-homepages/va/rubrics.jsonl`
— 14 GEPA-revised news-values criteria — was authored to be scored on a homepage
headline from these outlets, and its criterion texts are literally headline-scoped.
V9 reuses it verbatim: **zero new criteria, zero re-GEPA**, mirroring V8's
"A-bank 100% REUSED, zero new judging".

**The reddit arm stays a documented option** and is genuinely strong. It is best
understood as a *different* cell — lay-crowd news judgement across the open web,
rather than the crowd's response to one outlet's own front page — and it is the
natural replication site if V9's result needs an out-of-population check. What
it still lacks is identical to what V9 needed: no `va/`, no rubric bank, no
`v_features.py`, no dense standard.

**Correction to the scout record while we are here:** the taxonomy note's claim
that 2023-11+ reddit dumps carry 36h-refetched final scores does **not** hold —
`manifest.json` records `rows_with_retrieved_2nd_on: 0` (0 of 621,352). Scores
are as-of-dump. Also only r/news and r/worldnews were collected; r/politics and
r/UpliftingNews never were. Anyone reaching for that arm should read
`datasets/news-homepages/reddit_newsworthiness/README.md` first.

---

## 2. MANDATORY FIRST STEP — label-channel ground truth

The V8 lesson (strict list): verify the label channel against raw reality before
any instrument work. Findings, none of which is visible from the scraper code.

### 2.1 Row-level integrity

- 52,112 scraped rows = **47,199 OK + 4,913 errors**, of which 4,898 are HTTP 404.
- **Zero duplicate URLs.** The corpus was pre-deduped on `scheme://netloc/path`
  (query and fragment dropped) and the scraper's resume set is keyed on `url`.
- 404s are **not** informative zeros. A genuine "no tweets found" is recorded as
  a *successful* row with `n_tweets=0` — 1,393 of those exist. The 404s look
  missing-at-random w.r.t. article prominence: their `appearances` distribution
  (34% appear once) matches the OK rows (37%), and their median `anchor_text`
  length is identical (74 chars). **They are dropped, not imputed as zero.**
  Per-group 404 rate is median 3.4%; the one group at 96.7% is removed by the
  coverage gate.
- `anchor_text` is never empty, median 12 words, only 239 duplicate strings.

### 2.2 The cap — the known undercount, measured rather than hidden

62.2% of OK rows carry `capped=true`. Reading `tw_scrape.py` is what makes this
interpretable:

- The cap is a **retrieval** limit, not a value ceiling: `sum_likes` is summed
  over at most `MAX_PAGES × 20` retrieved tweets.
- It is **uniform**: 717 of 731 touched groups cap at 100, and exactly **one**
  group mixes two cap ceilings. So the censoring is a group-constant, which a
  within-group rank absorbs by construction.
- The search is **`type: "Latest"`, not "Top"**. For a capped article we hold the
  ~100 most *recent* tweets at scrape time (2026-06/07) about an article
  published 2025-12..2026-04. The honest name for the measured quantity is
  **sustained/trailing Twitter attention**, not launch-day virality. This is a
  real limitation on what the cell can claim and it is load-bearing.
- **The cap does not collapse the ordering.** Capped rate is 75.1% in the top
  within-group tercile vs 41.5% in the bottom — informative, far from
  determinative. Among capped rows alone `sum_likes` still spans IQR
  [1602, 8056] and correlates ρ=.926 with `max_likes`. The compression is at the
  top, which is exactly the region a top-vs-bottom binarization discards.
- Because the cap is a label-channel property and not a headline property, it is
  **never a feature**. It is carried only as a diagnostic: `capped` alone scores
  AUC **.611** on eval, which quantifies how much of y is raw tweet VOLUME as
  opposed to per-tweet intensity. Any future build must keep it out of V/A/dense.

### 2.3 The facets agree — a single latent attention dimension

Median within-group Spearman over the 615 groups with ≥30 rows:

| pair | ρ |
|---|---|
| sum_likes × sum_retweets | .931 |
| sum_likes × max_likes | .912 |
| sum_likes × sum_views | .800 |
| sum_likes × sum_bookmarks | .801 |
| sum_likes × n_tweets | **.408** |

`n_tweets` is the outlier precisely because it is the facet the cap truncates
directly. The rest cohere, so the label is measuring one thing.

### 2.4 Coverage — the 8% figure is the wrong statistic

The scrape reached only 8% of the 662,855-URL corpus before the monthly tweetapi
quota drained (2026-07-17; **the scraper was NOT restarted — it spends the
user's paid quota**). That number is irrelevant to a within-group design,
because `prep_urls.py` ordered the corpus by `(first_day, outlet)` descending, so
the scraper **completed groups instead of sampling them**:

- median within-group coverage among touched groups: **1.000**
- 602 groups at ≥95% coverage (42,033 rows); 631 at ≥80% (44,121 rows)

The go/no-go gate in the charge was ≥1,500 rows across ≥100 groups. Actual after
all gates: **31,129 rows across 508 groups** — passed by ~20×.

### 2.5 Article bodies exist but are deliberately unused

`datasets/news-homepages/fulltext/` covers 31,085 / 47,199 engagement URLs
(65.9%). Coverage is **not** engagement-biased (mean within-group percentile
.5034 covered vs .4935 missing) but is severely **outlet**-biased by paywall:
latimes 99.9%, cnnbrasil 99.9%, cnn 92.3%, guardian 91.5% vs nytimes 19.9%,
reuters 8.7%, washingtonpost 6.4%. Splicing bodies in would make the evidence
base differ systematically by outlet and would break byte-identity between the A
bank's input and the dense arm's input. **Headline-only, for all three arms.**
This also matches the reused bank, whose criteria are headline-scoped.

---

## 3. Population and splits

`datasets/journalism-tweets/build_tweets_population.py` →
`datasets/journalism-tweets/va/population.csv.gz` (+ `population_manifest.json`).

**y (primary).** `judgement` = 1 if `sum_likes` is strictly ABOVE the median
`sum_likes` of its own (outlet, first_day) group, 0 if strictly BELOW, ties
dropped (318 rows). This is a verbatim mirror of the two sibling VOTE cells —
math.SE and V6 SO-votes both use "strictly above the median on its own group,
ties dropped" — so the vote column stays commensurable across fields.

**Secondary y's, carried and never merged into the primary:**
- `y_maxlikes` — median split on `max_likes`. The **censoring** robustness arm:
  a single most-liked tweet is far less sensitive to a 100-tweet retrieval cap
  than a sum is. Agrees with the primary on 87.2% of rows, so it is a genuine
  independent check rather than a relabelling.
- `y_quartile` — percentile ≥.75 → 1, ≤.25 → 0, middle dropped (15,843 rows).
  The reddit-news sibling's rule. Note it is **nested** inside the primary
  (agreement 100% by construction) — a harder-margin subset, not an independent
  label.

**Gates**, each with its row cost in the manifest: within-group coverage ≥.80;
corpus group size ≥20; ≥10 OK rows in the group; English-language outlets only.

**cnnbrasil is excluded from the primary population** and carried as a held-out
replication arm. It is 43% of the raw rows and is Portuguese; excluding it (i)
keeps the reused English news-values bank scored on the language it was written
for and (ii) makes the outlet set match the homepage CURATION cell this row will
be contrasted against. bbc dropped: 15 rows total.

**Result:**

| | value |
|---|---|
| rows | 31,129 |
| groups (outlet × day) | 508 |
| pos rate | .49998 |
| outlets | nytimes 9,026 · washingtonpost 5,707 · latimes 5,336 · guardian 5,028 · cnn 4,474 · reuters 1,558 |
| days | 2025-12-14 → 2026-04-10 (115) |
| group size | median 56, min 20, max 144 |
| split rows | train 24,903 / eval 3,114 / test 3,112 |
| split groups | 358 / 75 / 75 |
| split pos rates | .49998 / .5000 / .5000 |
| train minority | 12,451 |

**Splits** are grouped and stable-hash: `stable_hash_bucket_map` imported
verbatim from `datasets/patents/build_dense_standard_claimfell.py` (deterministic
greedy + 20-pass hill-climb bin-packer over `sorted(sizes, key=(-size, sha1(g)))`,
objective = row-count target + 2.5 × pos-rate match). No seeded shuffle
anywhere; groups are disjoint across train/eval/test.

---

## 4. Instruments

**V — 100% reused, imported not forked.** `datasets/news-homepages/va/v_features.py`,
23 deterministic label-blind headline surface features, computed on the headline
half via that module's own `headline_of()`. The V9 item *is* a news headline, the
object that bank was written for.

**A — 100% reused, zero new judging.** `datasets/news-homepages/va/rubrics.jsonl`,
14 GEPA-revised news-values criteria, scored by Gemma-4-31B via
`datasets/journalism-tweets/score_tweets_bank.py`, which imports the scoring
loop, shard checkpointing, NA parsing and anchor machinery verbatim from
`datasets/va_gemma_banks/score_va_gemma_banks.py` + `score_scaleupC_banks.py`.
Offline batch vLLM 0.23 (`envs/gemma4`), temperature 0, max_tokens 6, prefix
caching, spawn + `CUDA_DEVICE_ORDER=PCI_BUS_ID`, 7 shards, one GPU.

*Inherited limitation, recorded not papered over:* the homepage bank carries no
Track A / Track B field, so V9 cannot split A_real from A_surface the way V6
SO-votes can. Adding tracks would mean re-authoring the bank and forfeiting the
zero-new-judging reuse. Noted as a regression relative to V6.

**Anchors come from an INDEPENDENT channel.** Anchor rows are drawn from
`datasets/news-homepages/va/population.csv.gz`, whose label is homepage
PLACEMENT ("link rendered in the TOP half of the capture's top-30% zone") — an
editorial prominence signal independent of Twitter engagement. Anchoring on the
engagement y itself would make the battery partly circular with the quantity
under test. (Same discipline as V6, which anchored on `y_accepted` while testing
`y_vote`.) Per-shard 3-row blinded anchors (pos / neg / scrambled), rescored up
to 4× until pos > neg > scram, **plus** the extended battery at K=50 per class.

**Smoke before scale** (`--smoke 24`, the V6 lesson): overall NA **.307**, mean
.524, no criterion at 100% NA, no guided-JSON all-min collapse (per-criterion
means .325–1.00, modal shares .12–.54). The high-NA criteria are behaving
semantically, not failing — an entertainment headline genuinely has no "elite
political actor" (na .58) and no "active or imminent crisis" (na .83). At scale
the NA rate is stable: .349 (shard 0), .345 (shard 1).

### 4.1 The per-shard 3-row anchor check fails on this cell — three causes, all diagnosed

Every shard exhausted its 4 attempts with `valid=False`. The scored X matrices
are unaffected (anchors are a separate pass), but the failure is worth the
record because **two of the three causes are properties of the shared helper on
a headline-length item, not of this cell's judge**, and they will recur on any
future headline-only cell.

Shard 0: pos .636/.773/.625/.300 · neg .577/.444/.357/.700 · scram .778/.750/.500/.333
Shard 1: pos .615/.333/.438/.278 · neg .773/.500/.250/.300 · scram .556/.333/nan/.000

**(i) Entity survival in the scrambled control.** `score_va_gemma_banks.scramble`
pools tokens from the pos+neg pair, shuffles, and reverses ALTERNATE words —
so half the words stay intact and the pool is only two headlines wide. On a
200-word answer that destroys meaning. On a 14-word headline it leaves the
proper nouns standing:

    "more tuoba pathetic dewener Gaza snoitidnoC than NU says ni Hegseth's ..."
    "in noitibma to xilfteN Panama gnimoC Canal ekater Best lirpA Shows dna The VT"

A judge asked whether an "elite political actor is a central subject" of a
string containing "Hegseth's" answers 1.0 — correctly. And with ~35% of criteria
at NA, the row's `nanmean` is taken over few survivors, so one or two 1.0s
dominate. The scram leg is measuring entity survival, not coherence.
**Repair** in `datasets/journalism-tweets/battery_repaired.py`: keep the frozen
transformation, widen the token pool to 40 unrelated headlines.

**(ii) The pos-vs-neg leg is underpowered by construction, and this is real
rather than a bug.** The anchor channel is homepage PLACEMENT, chosen for
independence from the engagement y. That bank separates placement at ≈.60 AUC
over a full population, so a single-row-vs-single-row draw is a coin flip
weighted .6 — it inverts roughly 40% of the time. Four attempts of a 1-vs-1
comparison therefore cannot certify anything. **K=50 per class is the fix, which
is precisely why the charge mandates it**; the per-shard 3-row result should be
read as a smoke alarm, not as the certification.

**(iii) A logic bug in the shared checker: all-NA on nonsense is scored as
FAILURE.** Shard 1 attempt 2 returned `scram = nan` — the judge refused to score
gibberish on every criterion, which is the *ideal* response. But the validity
test is `m[0] > m[1] > m[2]`, and any comparison against nan is False, so the
best possible scrambled-control outcome is recorded as invalid. `run_battery`
already handles this correctly (it drops all-NA rows and reports
`n_anchor_rows_all_NA_dropped`); `score_bank`'s per-shard check does not.
**This is a shared-helper defect affecting every cell, not just V9** — flagged
for the coordinator rather than patched here, since editing
`score_va_gemma_banks.py` mid-wave would change other cells' behaviour.

With correct nan handling, shard 1 attempt 2 (pos .438 > neg .250) and shard 4
attempt 2 (pos .786 > neg .727) would both have PASSED. Shard 3 attempt 1 passed
outright (pos .792 > neg .667 > scram .250). So the honest read is 3 of 7 shards
valid, not 1 — the checker's nan handling costs two of them.

### 4.2 The K=50 battery — the binding certification, and what it reveals

Bank scoring completed all 7 shards with a very stable NA rate: .349 / .345 /
.336 / .347 / .345 / .348 / .341.

| leg | K=50 result |
|---|---|
| anchor_pos mean | .5574 (sd .1349, n=50) |
| anchor_neg mean | .4775 (sd .1674, n=50) |
| anchor_scram mean | .6078 (sd .1915, **n=30**) |
| **pos vs neg AUC** | **.6466** |
| coherent vs scrambled AUC | .3642 |
| all-NA rows dropped | **20 of 150** |

**The certification that matters PASSES.** At proper power the judge separates the
independent homepage-placement channel at **AUC .647** — comfortably above chance
and, notably, above the .5979 that a *fitted* linear model on this bank achieved
for placement on the homepage cell's own population. The instrument is
responsive and correctly ordered. The per-shard 1-vs-1 failures were noise, as
(ii) predicted.

**The scrambled leg is inverted (.364), and the cause is a selection effect
worth reporting to the program.** Note `anchor_scram` has n=**30**, not 50: all
20 dropped all-NA rows were scrambled ones. `run_battery` drops all-NA rows
because they carry no rank information — correct in general. But here the rows
it drops are precisely the scrambles that *succeeded* in destroying meaning,
leaving the 30 where recognisable entities survived (§4.1(i)). The scram arm
therefore measures "scrambles that failed to scramble", and conditioning on
survival guarantees a high score. This is not specific to V9: **any cell with a
high NA rate and a short item will have its scrambled control silently selected
in the same direction.** Flagged for the coordinator.

### 4.3 The repaired battery, and the real conclusion about scrambled controls

`battery_repaired.py` reran the identical battery with the scramble pool widened
to 40 unrelated headlines. pos/neg are unchanged by construction (same draws),
which is a useful control on the rerun itself:

| | shipped | repaired |
|---|---|---|
| anchor_pos mean | .5574 | .5574 |
| anchor_neg mean | .4775 | .4775 |
| **pos vs neg AUC** | **.6466** | **.6466** |
| anchor_scram mean | .6078 (n=30) | .5507 (n=**24**, sd .308) |
| coherent vs scrambled AUC | .3642 | **.4640** |
| all-NA rows dropped | 20 | **26** |

The repair moved the scram leg .364 → .464 and pushed more scrambles into the
all-NA bin (26 vs 20) — both in the predicted direction. It did **not** reach
.5, and chasing it further would be a mistake, because the residual is not a
scramble-quality problem. It is a **statistic** problem:

> A row's battery score is `nanmean` over the 14 criteria. A coherent headline
> answers ~9 of them and averages toward the middle. A scramble answers 1–3 and
> its mean is whatever those 1–3 happen to be — frequently 1.0. **The two row
> scores are not commensurable**, and no amount of better scrambling makes them
> so, because the incommensurability comes from the denominator. The surviving
> scram rows also have sd .308, roughly double the coherent arms — the signature
> of an average over a tiny, variable denominator.

**And the judge is in fact discriminating coherence perfectly well — through the
NA channel rather than the mean channel.** Of 100 coherent anchor rows, **0**
were all-NA; of 50 scrambled rows, **26** were (shipped: 20). That is a complete
separation at the extreme (Fisher exact p ≪ 1e-15). The judge answers news-value
questions about real headlines and correctly refuses to answer them about word
salad. `coherent_vs_scrambled_auc` simply reads the wrong channel for short,
high-NA items.

**Recommendation for the program (not patched here — it would change every
cell's numbers mid-wave):** for any cell whose items are short enough to push
per-row NA above ~.3, `run_battery` should score the coherence leg on the
**non-NA response count** (or report it alongside), and `score_bank`'s per-shard
check should treat a nan scram mean as the lowest rank rather than as a failed
comparison. Both are small changes to
`datasets/va_gemma_banks/score_{va_gemma,scaleupC}_banks.py`.

**Net certification for V9:** responsiveness and correct ordering on an
independent channel — **PASS at .647**. Coherence — **PASS on the NA channel
(0/100 vs 26/50)**, unreadable on the mean channel. The A matrices themselves are
unaffected by any of this; the anchors are a separate scoring pass.

**Dense (T).** `datasets/journalism-tweets/build_dense_bundle.py` →
`va/dense_standard_journalism_tweets/{data.csv,split/}`. Same rows, same grouped
split, same `text` column as V and A — byte-identical input across all three
arms. Recipe unmodified: Llama-3.1-8B LoRA r16/α32, lr 5e-5, batch 16, len 1024,
2 epochs, gradient-checkpointing, select-on-eval, seed 42 (then 1, 2). No
`class_weight_auto` — the cell is balanced by construction (train minority
12,451).

---

## 5. Pre-kill baselines on the frozen split

Run before spending GPU time, per the pre-kill checklist
(`datasets/journalism-tweets/va/prekill_baselines.json`):

| baseline | eval AUC | test AUC |
|---|---|---|
| headline char length alone | .4898 | .5015 |
| headline word count alone | .4893 | .5121 |
| V-only, 23 surface features (logistic) | .5239 | .5300 |
| TF-IDF 1–2gram (logistic) | .6041 | .6184 |
| TF-IDF **within-group** | .6021 | — |
| `capped` flag alone (label-side, NOT a feature) | .6111 | — |

Three things this establishes:

1. **The cell is alive.** Headline text carries real signal about which article
   the crowd amplified (TF-IDF .60–.62).
2. **It is not a length model.** Length sits at chance (.490/.501) — the
   Style-Invitational failure mode is structurally absent. This matters because
   engagement cells are exactly where a length artifact would be expected.
3. **The pooled number is honest.** Pooled .6041 ≈ within-group .6021. Because y
   is a within-group median split, group identity cannot predict y, so the V8
   "pooled AUC is 92% group composition" trap cannot occur here. This is
   confirmed again in the ledger via outlet-day-identity-alone.

---

## 6. Deviations from the frozen program conventions

Every deviation, with its reason:

1. **No CONTEXT block**, unlike the homepage sibling which appends the other
   headlines in the same snapshot. Here the group IS the outlet-day, so a
   sibling-headline block would be group-CONSTANT — zero within-group rank
   information at ~500× the prompt cost.
2. **Article bodies not used** despite existing for 65.9% of rows — outlet-biased
   by paywall (§2.5), would break byte-identity across arms.
3. **No A_real / A_surface split** — the reused bank has no `track` field (§4).
4. **cnnbrasil excluded** from the primary population (§3), carried as a
   replication arm rather than dropped from the record.
5. **`y_quartile` is nested inside the primary y**, so it is a margin-hardening
   subset and not an independent binarization check. The independent check is
   `y_maxlikes`.
6. Robustness y's are scored with instruments **frozen on the primary y** (OOF
   predictions re-scored, not refit), so they are transfer checks rather than
   independent fits.

---

## 7. The cross-y contrast — the payoff of choosing this channel

This is the reason the tweet channel was picked over reddit (§1a), and it is
available **before** any instrument finishes, because it is a property of the
labels alone.

The homepage CURATION cell and V9 can be joined on exact headline text (the
homepage `va` population has no url column). Restricting to homepage headlines
carrying a single consistent placement label:

- **861 rows carry BOTH an engagement y and a placement y**, spanning 218
  outlet-day groups and all six outlets (nytimes 257, latimes 193, guardian 192,
  washingtonpost 122, reuters 59, cnn 38).

| | placement = bottom | placement = top |
|---|---|---|
| engagement = low | 153 | 287 |
| engagement = high | 205 | 216 |

- P(top-half homepage placement \| HIGH engagement) = **.513**
- P(top-half homepage placement \| LOW engagement) = **.652**
- **φ = −.141** (n=861; SE ≈ .034, so ≈ 4.1 SE from zero)

**The two y's point in OPPOSITE directions on identical text.** Articles the
editors promoted to the top half of the front page drew *less* Twitter
engagement than the ones they placed lower. This is the same structural
signature the V8 N&C co-signing build reported (co-signed comments got a 43.9%
agency response vs 79.4%, φ = −.160) — a second field where the expert-judgement
column and the crowd column anti-correlate rather than merely differ in residual
size. That makes journalism the natural companion to N&C in the grid's
"cross-y contrast is STRUCTURAL, not a residual-size difference" argument.

**Caveats, load-bearing:** the 861 joined rows are not a random sample of either
population — the homepage `va` population was capped at 1,700 rows per outlet by
hash-ordered snapshot prefix and covers its own day range, so the intersection is
opportunistic. And homepage placement here means "top vs bottom half of the
top-30% zone", i.e. a contrast *among already-prominent links*, not promoted vs
buried. The sign is robust to those caveats; the magnitude should not be
point-quoted as the field's verdict-vs-vote coupling.

---

## 8. Results

n = 31,129 · groups = 508 (outlet × day) · pos rate = .4999 · all three arms on
byte-identical headline text.

### 8.1 Layer-1 ledger

| quantity | value | group-bootstrap 95% CI |
|---|---|---|
| V_lin | .5271 | [.5203, .5339] |
| V_nl (mean of 3 GBM seeds) | .5399 | seeds .5418 / .5377 / .5401 |
| A_lin | .5661 | [.5588, .5730] |
| VA_lin | .5704 | [.5632, .5772] |
| **VA_nl (mean of 3 seeds)** | **.5947** | seed0 [.5883, .6016] |
| VA_nl seeds | .5951 / .5952 / .5939 | **spread .0014** |
| **T (dense, eval seed-mean)** | **.6300** | per-seed .6273 / .6303 / .6323 |
| T (dense, test seed-mean) | .6478 | per-seed .6493 / .6483 / .6457 |
| Δ_interact | **+.0247** | [+.0189, +.0309], P(>0) = 1.00 |
| Δ_total (T − VA_lin) | +.0596 | |
| Δ_beyond, pooled | +.0352 | *cross-population — do not quote* |

**Δ_beyond, SAME ROWS** (the honest figure — T's held-out rows only, so T is not
differenced against a VA pooled over a different row set):

| leg | n | VA_lin | VA_nl | T | **Δ_beyond** |
|---|---|---|---|---|---|
| eval | 3,114 | .5579 | .5952 | .6300 | **+.0348** |
| test | 3,112 | .5844 | .6265 | .6478 | **+.0212** |

**The cell is well-powered, unlike V8.** The dense T 3-seed spread is .0050
(eval), and Δ_beyond is +.0348 — the gap exceeds the seed noise by ~7×. The GBM
seed spread on VA_nl is .0014. This is the opposite of the V8 co-signing
situation, where the T spread (.0882) *exceeded* Δ_beyond (.0642) and forbade
point-quoting.

### 8.2 The pooled number is honest — measured, not asserted

- **outlet-day identity alone = .5000 exactly.** y is a within-group median
  split, so group identity carries literally zero signal. The V8 failure mode
  ("docket identity alone = 92% of pooled VA_nl") is structurally impossible here.
- pooled VA_nl .5991 vs **within-group** VA_nl .6003 (pair-weighted), and the
  program-convention min_n=20 figure is .6001 over all 508 groups / 31,129 rows —
  every row survives the min-20 filter. Pooled and within-group agree to .001.
- Headline length alone: .4972. Not a length model.
- `capped` flag alone: .5949 — reported because it quantifies how much of y is
  raw tweet volume. It is a label-channel property and is in none of V, A or the
  dense input.

### 8.3 Robustness

**Censoring (the cap).** Re-scoring the frozen instruments against `y_maxlikes`
— a median split on the single most-liked tweet, far less cap-sensitive than a
sum — gives VA_nl **.5844** (n=30,626, agreement with primary y .872). The
decomposition does not rest on the censored sum. The harder-margin `y_quartile`
arm gives VA_nl .6337 (n=15,843), higher as a wider margin should be.

**Per-outlet** — VA_nl is above chance in all six, so this is not one outlet's
artifact:

| outlet | n | VA_nl | A_lin |
|---|---|---|---|
| washingtonpost | 5,707 | .6232 | .5347 |
| cnn | 4,474 | .6222 | .6179 |
| guardian | 5,028 | .6033 | .5826 |
| nytimes | 9,026 | .5969 | .5699 |
| latimes | 5,336 | .5700 | .5340 |
| reuters | 1,558 | .5512 | .5524 |

**Invalid-shard sensitivity.** Two of seven shards passed the (underpowered,
§4.1) per-shard anchor check — shards 3 and 5. Dropping the other five leaves
n=8,905, on which V .5169 / A .5627 / VA_lin .5618 / **VA_nl .5809** — the same
ordering and nearly the same level. The result does not depend on the shards
whose 1-vs-1 anchor draw happened to invert.

*(Correcting §4.1: the ledger records `invalid_shards = [0,1,2,4,6]`, i.e. 2 of 7
valid as shipped. With the nan bug fixed, shards 1 and 4 would also pass, giving
4 of 7.)*

**Overfit gap** (GBM train − OOF) on VA: .139 / .122 / .140 across seeds —
normal for this stack.

### 8.4 Reading

The articulated instruments recover a real but minority share of the crowd's
attention signal. A bank of 14 explicitly-stated news-value criteria, judged
label-blind off the headline alone, reaches **A_lin .5661** where chance is .500
and the dense ceiling on the same rows is **.6300**. Adding 23 deterministic
surface features and letting a GBM find interactions lifts that to **.5947**, and
the interaction term itself is small but certain (+.0247, P(>0) = 1.00) — most of
the articulated signal is already linear in the criteria.

What remains — **+.0348 eval / +.0212 test** — is what the dense model reads off
a 12-word headline that fourteen articulated news-value criteria plus surface
form do not capture. For calibration, TF-IDF on the same split reaches .604
eval / .618 test, i.e. **roughly the articulated bank's level, well under dense**:
the residual is lexical-but-unnamed rather than exotic.

Two caveats bound the claim. The label is *sustained/trailing* Twitter attention
(§2.2, `type: "Latest"` months after publication), not launch-day virality. And
the item is a headline, so this cell measures the articulability of *headline*
appeal, not of the article.

---

## 9. Artifacts

| what | path |
|---|---|
| population + manifest | `datasets/journalism-tweets/va/population.csv.gz`, `population_manifest.json` |
| pre-kill baselines | `datasets/journalism-tweets/va/prekill_baselines.json` |
| population builder | `datasets/journalism-tweets/build_tweets_population.py` |
| bank scorer | `datasets/journalism-tweets/score_tweets_bank.py` |
| dense bundle builder | `datasets/journalism-tweets/build_dense_bundle.py` |
| layer-1 runner | `methods/taste_decomposition/tweets_community_layer1.py` |
| A/V matrices | `outputs/va_gemma_banks_journalism_tweets/journalism_tweets_shard{0..6}.npz`, `journalism_tweets_meta.json` |
| anchor battery | `outputs/va_gemma_banks_journalism_tweets/anchor_battery.json` |
| dense bundle | `datasets/journalism-tweets/va/dense_standard_journalism_tweets/` |
| layer-1 ledger | `methods/taste_decomposition/results/journalism_tweets_ledger.json` |
| reused V bank | `datasets/news-homepages/va/v_features.py` |
| reused A bank | `datasets/news-homepages/va/rubrics.jsonl` |
| raw label channel | `datasets/news-homepages/twitter_engagement/tweet_engagement.jsonl` (sk3) |

All sk3 paths are under `/lfs/skampere3/0/alexspan/norm-research/`.
