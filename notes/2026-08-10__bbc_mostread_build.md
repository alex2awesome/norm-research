# BBC most-read — journalism community cell #2 (same-outlet readership)

Build note. Companion to V9 (`notes/2026-08-08__v9_journalism_community_build.md`),
which is journalism community cell #1. User directive behind this build: **test
whether V9's community signal reflects Twitter platform dynamics or generalises
to a same-outlet readership list.**

The design is a controlled comparison. Same field, same item type (a news
headline), same V bank, same A bank, same judge, same dense recipe, same
grouping logic. The only things that change are *which crowd* and *how it acts*:

| | V9 tweets | BBC most-read |
|---|---|---|
| crowd | Twitter users, cross-platform | BBC readers, same-outlet |
| action | amplification (like/RT of a link) | a click, aggregated into a ranked list |
| y | within outlet-day median split of summed likes | in the home page's MOST READ top-10 vs elsewhere on the same capture |
| group | outlet × day | capture day (BBC is one outlet, so outlet×day collapses to day) |
| era | 2025-12 → 2026-04 | 2017-06 → 2024-02 |

Agent: `claude-v9-journalism-tweets`. Coordinator logs registry/strict list.

---

## 1. Inventory and reuse

The dataset existed (`notes/2026-06-12__taste-taxonomy.md`: "BBC most-read
(built)"). Found at `datasets/news-homepages/bbc_mostread/` on sk3:

| artifact | what |
|---|---|
| `raw/captures.jsonl` | 7,298 Wayback captures of the BBC News home page, each with a ranked `most_read` list and an `others` list of every other link on the page |
| `built/{train,eval,test}.csv.gz` | 82,891 rows, cols `text, judgement, rank, timestamp, day, section, channel, parser, href, headline_id` |
| `built/bbc_mostread_v2_morph_topicstrat.csv.gz` | a topic-stratified variant |
| `manifest.json`, `scripts/`, `logs/` | build/scrape/validation record |

**Reused wholesale, nothing re-authored:** the rows and labels (filtered, see §2),
the 23-feature headline V bank
(`datasets/news-homepages/va/v_features.py`), the 14 GEPA-revised news-values A
bank (`datasets/news-homepages/va/rubrics.jsonl`) — the *same* bank V9 reused, so
the two journalism community cells are instrument-matched — the split bucketer
from `datasets/patents/build_dense_standard_claimfell.py`, and the frozen dense
recipe. The scrape was **not** re-run.

The A bank's reuse here is population-exact in a way it was not even for V9: BBC
is one of the outlets that bank was authored on, and the item is a home page
headline, which is exactly what its criteria are written to score.

---

## 2. Ground-truth pass on the label channel — four defects

The charge required verifying list-capture timing and coverage against the raw
captures before instrument work. The shipped 82,891-row pool is **not usable as
shipped**. None of these is visible from the build code; all four come from
counting.

### DEFECT 1 — capture type largely determines the label

| parser | n | pos rate |
|---|---|---|
| `morph` | 51,790 | **.4400** |
| `popular_page` | 7,520 | **1.0000** |
| `react` | 6,488 | **.9553** |
| *(none)* | 17,093 | **.0000** |

24,613 of 82,891 rows (**29.7%**) sit in strata that supply essentially one
class. `channel` is the same story (A: 7,520 rows, all positive). Any era or
lexical signature of those captures is free AUC — which is what inflates the
shipped manifest's length-matched TF-IDF floor to .720 eval / .711 test.

### DEFECT 2 — the length confound is an artefact of Defect 1

The shipped manifest records `len_label_corr` .2163 with pos/neg mean length
48.8 / 44.3, which reads as a real length effect. Within `morph` alone the
lengths are **44.83 / 44.76 — identical**. The entire apparent length signal is
`popular_page` (52.1) and `react` (59.3) being all-positive strata. A cell built
on the shipped pool would have reported a length effect that does not exist.

### DEFECT 3 — the shipped splits are not day-grouped

**3,343 of 3,421 days appear in more than one of train/eval/test.** Rows
themselves are unique (82,891 distinct `headline_id`, zero duplicated text), so
this leaks the *day* — the same news cycle's story cluster spans fit and
evaluation. Fatal for a within-day design.

### DEFECT 4 — a small link-kind stratum where the href alone fixes y

Only visible by bucketing `href`. Of the 51,790 morph rows, 1,184 (2.3%) are not
ordinary news articles, and there the label is near-deterministic:

| link kind | n | pos rate |
|---|---|---|
| `/sport/` | 280 | 1.000 |
| `/live/` | 104 | 1.000 |
| other `/news/` non-article | 232 | .931 |
| `/in-pictures` | 568 | .079 |

A photo gallery or a live page is a different *kind* of object from an article,
so keeping them lets the instruments learn "is this an article at all" instead
of "did readers choose this article".

### The repair, and why it is a filter not a rebuild

Restrict to `parser == "morph"` **and** ordinary dated news articles, then
re-split day-grouped. The row content is sound — verified next — so nothing needs
re-scraping or re-parsing. This is reuse-before-rebuild with the defects
recorded rather than inherited.

### Verification against the raw captures — passes exactly

- All **51,790** morph rows re-derive their label from `raw/captures.jsonl`:
  **label-matches-raw 51,790, MISMATCH 0, not-found 0.**
- All **22,787** positives re-derive their `rank`: **22,787 match, 0 mismatched.**
- `most_read` and `others` are **disjoint** within a capture (0 hrefs in both),
  so positive/negative assignment is unambiguous.
- Retention from raw is deliberate downsampling, recorded not hidden: **77.3%**
  of raw morph positives and **33.7%** of raw morph negatives survive (negatives
  were subsampled toward a ~44% positive rate).

### Timing — a real constraint on what this label means

Wayback captured the BBC home page overwhelmingly **just after midnight UTC**
(hour 00: 39,645 rows; hour 01: 7,017). The most-read module is a rolling
window, so a row labelled day D reports reading that happened mostly on **day
D−1**. The honest name is "was among the 10 most-read as of the small hours of
day D". 693 of 2,256 days carry two captures, the rest one.

---

## 3. Population and splits

`datasets/bbc-mostread/build_mostread_population.py` →
`datasets/bbc-mostread/va/population.csv.gz`.

| | value |
|---|---|
| rows | **50,761** |
| groups (capture day) | **2,251** |
| pos rate | **.4405** |
| within-day pos rate (mean) | .4505 |
| day range | 2017-06-09 → 2024-02-20 |
| section | 100% `news` |
| group size | median 23, min 10, max 55 |
| split rows | train 40,614 / eval 5,075 / test 5,072 |
| split groups | 1,796 / 249 / 206 |
| split pos rates | .4405 / .4404 / .4403 |
| train minority | 17,889 |
| headline length by class | 44.82 (neg) / 44.88 (pos) |
| positives' rank distribution | flat, 2,180–2,306 per rank 1–10 |

Gates: `parser == morph`; ordinary dated news article href; day carries ≥10 rows
and both classes (cost: 2 days too small, 4 single-class). Splits are grouped
stable-hash over days via the imported bucketer — no seeded shuffle, and the
build asserts no day spans two buckets.

**Class weighting:** pos rate .4405 with train minority 17,889 is mild, so the
frozen recipe runs unmodified with no `class_weight_auto`, matching V9 and V6.

---

## 4. Cross-corpus overlap — a same-rows contrast is NOT possible

The charge asked this be documented so a same-rows contrast could be run where it
exists. It does not exist:

- **V9 tweets ∩ BBC most-read = 0 rows.** V9 carries **zero** BBC rows (BBC
  contributed only 15 URLs to the whole tweet scrape and was gated out), and the
  two corpora share **zero** headlines — they do not even overlap in time
  (2025-12→2026-04 vs 2017→2024).
- **Homepage curation ∩ BBC most-read = 0 headlines**, despite the homepage `va`
  population carrying 1,701 BBC rows, again because of the era gap.

So the V9-vs-BBC comparison is **cell-level, not paired**: two independent
estimates of how articulable a journalism crowd signal is, sharing instruments
but not rows. No paired test is licensed, and any difference between the cells is
confounded with era and with outlet mix. Recorded here so the contrast is never
over-read.

---

## 5. Pre-kill baselines, and the one number that had to be chased

On the frozen grouped split (`datasets/bbc-mostread/va/prekill_baselines.json`,
computed on the pre-Defect-4 population; the gate removed 2.3% of rows):

| baseline | eval | test | V9 for comparison |
|---|---|---|---|
| headline char length | .4913 | .4890 | .4898 |
| headline word count | .4669 | .4641 | .4893 |
| V-only, 23 surface features | **.6586** | .6422 | .5239 |
| TF-IDF 1–2gram | **.7446** | .7511 | .6041 |
| TF-IDF within-day | .7517 | — | .6021 |
| **day/group identity alone** | **.5807** | — | **.5000** |

Two things needed chasing before these could be believed.

**(a) V-only .659 is three times V9's surface signal.** That prompted the
Defect-4 audit. After gating out non-article links the surface signal is
substantively real, not an artefact, and it is interpretable: negatives are
**3× more likely to be question-framed** (12.2% end in "?" vs 3.8% of positives)
and **3× more likely to open How/Why/What** (12.0% vs 4.0%). BBC readers
under-click explainer- and question-framed headlines relative to their home page
prominence. The V bank has `v_question_mark` and `v_interrogative_opening`, so it
picks this up directly.

**(b) Day identity alone is .5807, NOT .5000 — the key structural difference
from V9.** V9's y is a within-group *median split*, which forces every group to
a .500 positive rate and makes group identity worthless by construction. This
cell's y is natural membership, so the per-day positive rate varies with how many
links a capture carried. **Consequently the within-day readouts are this cell's
honest primary numbers and the pooled ones are secondary — the exact reverse of
V9.** This is the single most important thing to hold straight when comparing the
two cells, and it is enforced in the ledger: every matrix is reported pooled *and*
within-day.

The cell is alive, and is not a length model (length at/below chance on both
legs).

---

## 6. Instruments

**V** — `datasets/news-homepages/va/v_features.py`, 23 deterministic headline
features, imported verbatim. Identical to V9.

**A** — `datasets/news-homepages/va/rubrics.jsonl`, 14 GEPA-revised news-values
criteria, scored by Gemma-4-31B offline-batch (vLLM 0.23, `envs/gemma4`,
temperature 0, max_tokens 6, prefix caching, spawn + `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
8 shards, one GPU). Identical bank to V9, so no `track` field and hence no
A_real/A_surface split — the same inherited limitation V9 records.

**Anchors** — the *same independent channel and the same pool as V9*: homepage
PLACEMENT from `datasets/news-homepages/va/population.csv.gz`. Anchoring both
journalism cells identically is what makes their two judge certifications
comparable. K=50 per class plus per-shard blinded triples.

**The V9 scramble repair ships from the start.** V9 diagnosed that the frozen
`scramble` helper reverses only alternate words drawn from the pos+neg pair, so
on a ~12-word headline intact proper nouns survive and the judge correctly scores
them 1.0. Here the nonsense control pools tokens across 40 unrelated headlines.
V9's deeper finding still stands and is *not* repaired by this: a row score is
`nanmean` over 14 criteria, so a scramble answering 1–3 of them is not
commensurable with a headline answering 9 — read coherence off the **all-NA
rate**, not the mean.

**Truncation is in tokens, not chars.** Asserted at build: over 4,000 sampled
headlines, max 22 tokens, p99 16, zero over the 128-token budget.

**Smoke before scale, with an ENFORCED collapse gate** (exits non-zero on
failure): NA .286, mean .508, **COLLAPSE_GATE_PASS**, no criterion at 100% NA.
One criterion to watch at scale — "Economic impact reaching the reader" sat at
na .96 on the n=24 smoke (it was .73 in V9 at scale). The gate is re-enforced on
the assembled matrix in the ledger.

**Dense (T)** — same rows, same day-grouped split, same `text` column as V and A.
Llama-3.1-8B LoRA r16/α32, lr 5e-5, batch 16, len 1024, 2 epochs,
gradient-checkpointing, select-on-eval, seeds 42/1/2.

**Job hygiene** — both the scorer and the dense chain are launched with `setsid`
and their detachment asserted (`ppid == 1` verified for both). The dense chain
waits on the scorer's **PID** via `kill -0`, not `pgrep -f`: in the V9 build a
`pgrep -f score_tweets_bank.py` chain deadlocked because the launching shell's
own command line contained the heredoc text of the script it was writing, so the
chain matched its own parent and waited forever. That cost ~10 minutes of idle
GPU and is not repeated.

---

## 7. Results

n = 50,761 · groups = 2,251 capture days · pos rate .4405 · all arms on
byte-identical headline text. Bank NA rate across the 8 shards: .295–.303.

### 7.1 Layer-1 ledger

**Read the within-day column as primary** (§5b: day identity alone is .5814 here,
not .5000 as in V9).

| quantity | pooled | within-day | group-bootstrap 95% CI |
|---|---|---|---|
| V_lin | .6478 | .6482 | [.6432, .6525] |
| V_nl (3 seeds) | .6707 | .6718 | seeds .6709/.6709/.6702 |
| A_lin | .6879 | .6908 | [.6833, .6921] |
| VA_lin | .7054 | .7079 | [.7010, .7098] |
| **VA_nl (3 seeds)** | **.7332** | **.7370** | seed0 [.7293, .7379] |
| VA_nl seed spread | **.0010** | | |
| **T (dense, eval)** | **.8230** | | per-seed .8218/.8234/.8239 |
| T (dense, test) | .8097 | | per-seed .8080/.8116/.8095 |
| Δ_interact | **+.0282** | | [+.0257, +.0308], P(>0)=1.00 |
| Δ_total | +.1176 | | |
| Δ_beyond (pooled) | +.0898 | | *cross-population — do not quote* |

**Δ_beyond, SAME ROWS:**

| leg | n | VA_lin | VA_nl | within-day VA_nl | T | **Δ_beyond** |
|---|---|---|---|---|---|---|
| eval | 5,075 | .7048 | .7366 | .7376 | .8230 | **+.0864** |
| test | 5,072 | .7176 | .7407 | .7428 | .8097 | **+.0690** |

Well-powered: the dense T 3-seed spread is .0021 (eval) against Δ_beyond +.0864
— a ~40× margin. GBM seed spread .0010. Overfit gap on VA is .048–.063, notably
tighter than V9's .12–.14.

Pooled and within-day agree to ~.004 everywhere, so although day identity alone
is .5814, day *composition* is not what the instruments are exploiting.

Gates: **collapse gate PASS** on the assembled matrix; **OOF alignment gate**
abs_diff 0.0 for both VA_nl and VA_lin, shuffled counterfactual .4983/.4982.
Invalid shards [0,1,2,5,6,7] (the underpowered per-shard check, §7.3); dropping
them leaves n=12,643 with V .6428 / A .6820 / VA .6975 / VA_nl .7128 — same
ordering, same level.

### 7.2 Rank, era

**The instruments order the winners, not just identify them.** Frozen on the
binary y and re-scored against most-read rank (22,357 ranked positives),
Spearman vs −rank: VA_nl **+.144**, VA_lin +.135, A_lin +.129, V_nl +.065;
top-3-vs-bottom-3 AUC .600 / .593 / .590 / .546. The articulated bank carries
almost all of this — V alone is less than half of it.

**Era stability** — VA_nl by year: .719 (2017), .735, **.770 (2019)**, .744,
.748, .727, .706 (2023), .629 (2024, n=1,025 only). Stable across seven years
with a mild decline; 2024 is a thin partial year and should not be read as a
trend.

### 7.3 Anchor battery — failed as shipped, diagnosed, and recovered

| battery | pos | neg | scram | **pos-vs-neg AUC** | coherent-vs-scram |
|---|---|---|---|---|---|
| shipped (mixed-outlet anchors) | .4656 | .4778 | .5417 | **.481** | .389 |
| diagnostic (BBC-only anchors) | .5771 | .5341 | .6684 | **.602** | .387 |
| *V9 for reference* | .5574 | .4775 | .6078 | *.647* | *.364* |

The shipped battery **failed** its responsiveness leg at chance (.481), which
cannot be waved through. But the same bank scores A_lin **.6879 [.6833, .6921]**
on this cell's own y — a bank separating most-read at .69 is not broken. The
diagnosis, tested rather than asserted:

> The two journalism cells draw anchors from the same all-outlet homepage-
> placement pool, but their system prompts differ — V9 says "a major outlet's
> home page", this cell says "**the BBC News** home page". The anchor rows are
> mostly *not* BBC. So on anchor rows only, this cell's prompt asserts a
> provenance the item contradicts, while on all 50,761 real rows the prompt is
> true. That damages the battery and leaves the A matrix untouched.

Redrawing both anchor classes from the 1,701 BBC rows of the placement
population, changing nothing else, moves pos-vs-neg **.481 → .602** — in line
with V9's .647 and with the homepage cell's own fitted A_lin (.5979). Hypothesis
confirmed.

**Third transferable discipline finding from this cell pair: anchor rows must
match the provenance the system prompt asserts.** A cell-specific prompt silently
invalidates a shared anchor pool. (The other two are in V9 §4.1/§4.3.) The
coherence leg stays inverted (.387) for exactly the reason V9 §4.3 documents —
`nanmean` over a tiny, variable denominator — and is again unreadable on the mean
channel.

The scored A matrices were produced in a pass that never saw an anchor row, so
none of this touches the ledger.

### 7.4 The contrast this cell was built to make

Row overlap with V9 is **ZERO** (§4), so this is a **cell-level comparison, not
a paired test**, and every difference below is confounded with era (2017-2024 vs
2025-26) and with outlet mix (BBC-only vs six US/UK outlets). With that said:

| | BBC most-read (same-outlet readership) | V9 tweets (cross-platform) |
|---|---|---|
| V_lin | .6478 | .5271 |
| V_nl | .6707 | .5399 |
| A_lin | **.6879** | .5661 |
| VA_lin | .7054 | .5704 |
| VA_nl | **.7332** | .5947 |
| T | **.8230** | .6300 |
| Δ_interact | +.0277 | +.0244 |
| Δ_beyond (same-rows eval) | +.0864 | +.0348 |
| group identity alone | .5814 | .5000 |

**The user's question — platform dynamics or general crowd signal? — answers
"both, and the difference is one of degree not kind."** Every instrument is
stronger on the readership list: the articulated news-values bank alone moves
from .566 to .688, and the dense ceiling from .630 to .823. A headline predicts
what BBC readers will *click* far better than it predicts what Twitter will
*amplify*. That is what one would expect if Twitter amplification adds a large
component that no property of the headline can carry — who happened to tweet it,
what else was competing, cascade effects — while a same-outlet most-read list is
much closer to a clean readout of headline appeal.

But the *shape* of the decomposition is remarkably similar. Δ_interact is nearly
identical (+.028 vs +.024), i.e. in both cells the articulated criteria are
mostly linear and interactions add the same small certain increment. And the
articulated share of the achievable signal is comparable: VA_nl reaches
**89.1%** of T on BBC (.7332/.8230) and **94.4%** on V9 (.5947/.6300) — so the
tweets cell is, if anything, *proportionally* slightly more articulable even
though it is far less predictable in absolute terms. The residual is larger on
BBC in absolute AUC (+.086 vs +.035) because there is simply more signal there to
divide.

The one qualitative difference is structural rather than about magnitude: V9's y
is a within-group median split, which forces group identity to .5000 and makes
pooled ≈ within-group by construction; BBC's y is natural membership, so day
identity alone carries .5814 — yet pooled and within-day still agree to .004.
Both cells are honest, for different reasons.

---

## 8. Artifacts

| what | path |
|---|---|
| population + manifest | `datasets/bbc-mostread/va/population.csv.gz`, `population_manifest.json` |
| pre-kill baselines | `datasets/bbc-mostread/va/prekill_baselines.json` |
| population builder | `datasets/bbc-mostread/build_mostread_population.py` |
| bank scorer | `datasets/bbc-mostread/score_mostread_bank.py` |
| dense bundle builder | `datasets/bbc-mostread/build_dense_bundle.py` |
| layer-1 runner | `methods/taste_decomposition/bbc_mostread_layer1.py` |
| A/V matrices | `outputs/va_gemma_banks_bbc_mostread/bbc_mostread_shard{0..7}.npz`, `bbc_mostread_meta.json` |
| anchor battery | `outputs/va_gemma_banks_bbc_mostread/anchor_battery.json` |
| dense bundle | `datasets/bbc-mostread/va/dense_standard_bbc_mostread/` |
| layer-1 ledger | `methods/taste_decomposition/results/bbc_mostread_ledger.json` (+ `_oof_ids.npy`) |
| source captures | `datasets/news-homepages/bbc_mostread/raw/captures.jsonl` |
| source built pool | `datasets/news-homepages/bbc_mostread/built/` |
| reused V bank | `datasets/news-homepages/va/v_features.py` |
| reused A bank | `datasets/news-homepages/va/rubrics.jsonl` |

All sk3 paths under `/lfs/skampere3/0/alexspan/norm-research/`.
