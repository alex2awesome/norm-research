# Press-release newsworthiness — deconfounding audit (2026-06-25)

Companion to `notes/2026-06-22__legal-vat-audit.md`. Same spirit: take a corpus we
were about to push through the V/A/Taste passage, and first measure how much of its
"signal" is confound. **Verdict: most of it is.**

All artifacts in scratchpad `pr_audit/` (master tables, scripts). Durable output:
`datasets/press-releases/press_release_deconfounded.parquet`.

---

## TL;DR

- The reported dense ceiling (**Llama-8B LoRA ≈ 0.71**) is a **random-split** number on text that
  carries publisher identity, topic, and wire-service boilerplate. It is substantially inflated.
- Cleaning the text + **grouping the split by publisher** drops a linear (TF-IDF+LR) text model to
  **0.584** (topic retained as legitimate signal) and **0.546** within-topic.
- Decomposition of the apparent signal: **publisher memorization +0.056**, **topic-selection +0.038**,
  **within-topic cross-publisher craft ≈ 0.046 over chance**.
- ⇒ The "large tacit residual" story I pitched (rubric 0.55 vs dense 0.71) is **mostly confound, not
  taste.** Rubric methods at 0.53–0.58 were already near the honest deconfounded ceiling. Press
  releases are **not** the clean taste-dominated showcase — they look more like `news_homepages`
  (the body text barely contains what the label tracks once you remove who-sent-it).

---

## 0. Data provenance + a corruption finding

- Canonical training file `press_release_modeling_dataset_clean.csv.gz` is **gzip-truncated** — only
  **41,607 / 128,131 rows** are readable, and that prefix is **81% positive** (non-representative,
  the file is label-sorted). The dense 0.71 runs may have trained on a biased/partial copy, or an
  intact copy lives on sk3 — **not yet verified**. File preserved, not deleted.
- Reconstructed the modeling table from intact, id-keyed sources (`build_master.py`):
  `press_release_clean.jsonl.gz` (clean bodies) ⋈ `…dataset.csv.gz` (metadata + label) ⋈
  `…__doc_to_topic.csv` (25-topic model). **128,131 rows, 53,780 pos (42.0%) / 74,351 neg.**
- **Label** = `news_article_domain` non-empty = the release was covered by ≥1 tracked news outlet.
- **Text** used = clean VLLM-extracted body, falling back to raw scraped text when extraction is empty.

## 1. Confound profile (`confound.py`)

| Confounder | Magnitude | Notes |
|---|---|---|
| **Publisher identity** | **AUC 0.673** | leave-one-out company coverage-rate. 11,074 companies; the dominant leak. |
| **Topic** | **AUC 0.610** | 25-topic model; pos-rate 0.048 (Beauty) → 0.592 (Gov & Public Policy). Partly *real* signal. |
| Clean-extraction empty | AUC 0.459 | 49,248 rows (38.4%) have empty clean body (→ raw fallback); empties skew negative (0.368 vs 0.452). |
| Clean-body length | AUC 0.547 | positives ~1.8× longer (median 2,414 vs 1,357 chars). `model_len` masked by raw fallback. |
| Language | AUC 0.505 | 98.4% English; non-EN (2,113 docs) skews negative (pt 0.053, id 0.104). Small but real. |
| Year | AUC 0.471 | weak; pos-rate drifts 0.374 (2016) → 0.339 (2023); 30% of rows have no date. |

Both dominant confounds (publisher 0.673, topic 0.610) sit **near the dense 0.71** — i.e. you can nearly
reproduce the "ceiling" from metadata identity alone.

## 2. TF-IDF + LR leakage (`tfidf.py`, `tfidf2.py`)

Top **+covered** tokens are channel/identity, not craft: `prnewswire`, `news provided`, `press room`,
`press center`, `gartner`, year tokens (`2018`, `2013`). Top **−notcovered**: specific firms/outlets
(`wwe`, `cisco`, `citi`, `boeing`, `zacks`, `otcpk`, `aon`) and HTML cruft (`amp`=&amp;, `gt`=&gt;).

Wire-service marker prevalence (present-rate / pos|present / pos|absent):

| marker | present | pos·present | pos·absent |
|---|---|---|---|
| prnewswire | 8.1% | 0.505 | 0.412 |
| news provided by | 2.3% | 0.576 | 0.416 |
| for immediate release | 1.5% | 0.552 | 0.418 |

Each marker is leaky but localized; **scrubbing all wire/HTML/year/ticker tokens moved grouped AUC only
0.546 → 0.545.** The residual signal isn't a few boilerplate strings — it's spread across thousands of
entity tokens, which the company-grouped split already absorbs.

## 3. The deconfounding ladder

| Setting | AUC | what it removes |
|---|---|---|
| Dense Llama-8B (reported, random split) | ~0.710 | — (fully confounded) |
| TF-IDF+LR, raw text, random split | 0.675 | — |
| TF-IDF+LR, raw text, **publisher-grouped** split | 0.605 | publisher memorization |
| Deconfounded¹, random split, natural topic | 0.640 | empties + non-EN + length tails |
| **Deconfounded¹, publisher-grouped, natural topic** | **0.584** | **+ publisher identity** ← honest ceiling |
| Deconfounded¹, publisher-grouped, topic-balanced | 0.546 | + topic-selection (within-topic craft) |
| …same, boilerplate-scrubbed | 0.545 | + wire/HTML/year/ticker tokens |

¹ deconfounded = clean body only (drop empty extractions), English only, length 200–12,000 chars.

**Signal decomposition** (linear text model):
`0.640 random → 0.584 grouped` ⇒ **publisher memorization = +0.056**;
`0.584 natural-topic → 0.546 balanced` ⇒ **topic-selection = +0.038**;
remaining **0.546** = within-topic, cross-publisher craft = **+0.046 over chance.**

## 4. Durable artifact

`datasets/press-releases/press_release_deconfounded.parquet` (`build_deconfounded.py`):
- **72,315 rows** (56.4% of original retained), pos 45.3%.
- Columns: `id, judgement, text, model_len, company, group, split, year, topic, topic_label`.
- **Stable company-hash split** (sha1·company mod 100 → 80/10/10), **0 companies straddle splits**
  (per `feedback_stable_hash_splits`). `group` = company (solo PRs keyed by id) for `GroupKFold`.
- Topic distribution intact → balance per-fold at modeling time (artifact keeps the signal optional).

## 5. Implications for V/A/Taste

- The press-release "articulability gap" I advertised is **mostly a confound gap, not a taste gap.**
  A fair (publisher-grouped, cleaned) ceiling is ~0.58, and rubric methods already hit 0.53–0.58.
  There is little headroom for a "tacit residual above articulated norms" once confounds are removed.
- The faint real residual is interpretable: **+covered** lean institutional/policy/government
  (`health authorities`, `reserve board`, `government`); **−covered** lean promotional/product/penny-stock
  (`beauty`, `otcpk`, `zacks`, crisis-hotline boilerplate). That's a genuine but weak *source-authority +
  public-significance* norm — consistent with Galtung–Ruge news values, but worth only ~0.05 AUC.
- **Recommendation:** press_releases is a **low-ceiling / identity-driven** task, parallel to
  `news_homepages`. Keep it as a *contrast* domain (where neither articulation nor dense learning
  recovers much from body text), **not** the taste-dominated showcase. For a clean taste-dominated
  domain, **creative_writing** (dense still climbing to 0.90, unsaturated) is the better bet.

## 5b. V-layer (auditable / thin / checkable) — `auditable_v.py`

First rung of the V/A ladder, **deterministic regex features, NO LLM**, on the deconfounded
company-grouped split (n=72,315). Tests the "$$ / quantitative specificity" hypothesis directly.

**Every auditable feature is at chance** (univariate AUC, |AUC−0.5| ≤ 0.02 for almost all):
`n_percent` 0.521, `n_words` 0.517, `n_dates` 0.509, `n_quotes` 0.508; and critically the money axis
`n_dollar` 0.498 / `has_dollar` 0.497 / `dollar_density` 0.497 / `n_bigmoney` 0.495 — **dollar amounts
do not predict pickup.** Same for stats, named-source count, contact info, puffery.

| V-layer LR | random | company-grouped |
|---|---|---|
| ALL 18 auditable feats, natural-topic | 0.533 | **0.508** |
| ALL, topic-balanced | 0.522 | 0.509 |
| money/quant only (6) | 0.524 | 0.514 |
| size only (2, control) | 0.518 | 0.514 |
| **ALL minus size** | 0.524 | **0.500** |

⇒ **V ≈ 0.51 (chance).** The little signal in money/quant is just length (size-only matches it; removing
size → 0.500). **The verifiable layer is empty** — opposite of law, where thin checkable facts (statutory
thresholds, filing dates, damages) were outcome-determinative. Newsworthiness is not in countable specifics.

Side benefit: V being at chance is independent confirmation the **deconfounding held** — auditable features
capture exactly the surface properties (length, boilerplate, channel) that would leak if cleaning had
failed; they show nothing.

## 5c. A-layer + "V-from-rubrics" (70B-scored, `build_pr_A2.py`)

FP8-Llama-70B scored each of the 309 curated rubrics (`rubrics.jsonl`) per doc on a 600-balanced
sample of the deconfounded pool (185,400 prompts; NA rate 0.00; scores not collapsed — per-rubric AUC
spread 0.47–0.56). Scores cached `pr_A_scores.npz`. LR over scores, StratifiedKFold(5). A is
identity-robust by construction (features = rubric scores, no publisher token), so random CV ≈ grouped.

| Layer | AUC |
|---|---|
| **A — all 309 articulated rubrics** | **0.552** |
| **V-from-rubrics — 16 quantitative rubrics** (money+metrics+data) | **0.547** |
| &nbsp;&nbsp;↳ MONEY/funding [1,91,227,274] | 0.512 |
| &nbsp;&nbsp;↳ METRICS/NUMBERS [67,72,98,116,149,150,257,285,292] | 0.503 |
| &nbsp;&nbsp;↳ DATA-AS-CONTENT [97,117,154] | **0.567** |
| V — auditable counts (regex, §5b) | 0.508 |

**Findings:**
- **A = 0.552** ≈ the inferred ~0.55–0.58 ceiling, and *below* the deconfounded linear text (0.584).
  The full articulated rubric bank barely beats chance and adds nothing over the linear model — the
  rubric↔dense gap was confound, now confirmed by direct measurement.
- **The 16 quantitative rubrics (0.547) carry essentially ALL of A's signal (0.552)**; the other 293
  rubrics are near-redundant for prediction.
- **The "$$ / quantification" hypothesis fails BOTH ways:** raw counts 0.508 AND *judged* meaningful
  quantification 0.503 (metrics/KPIs/effect-sizes) — money rubrics 0.512. Quantification does not
  predict pickup however it is scored.
- **The one weak quantitative signal is DATA-AS-CONTENT (0.567)** — not "mentions numbers" but "the
  release IS original/proprietary data or research" ([154]/[117]/[97]). Top single rubrics: [154]
  proprietary-data-as-news 0.564, [97] data-literacy/datasets 0.538, [117] original-research 0.538,
  [67] data+sourcing 0.536, [285] external-benchmark 0.533. A genuine but faint data-as-news-hook norm.

**Final press-release V/A ladder:** V(counts) 0.51 · V(judged-quant) 0.547 · A(all rubrics) 0.552 ·
deconf-linear 0.584 · confounded-dense 0.71. Everything honest caps at ~0.55–0.58.

## 5d. Per-outlet quantitative V — pooled-chance hides real heterogeneity

The label is "covered by ANY of 17 tracked outlets" (`news_article_domain`, a stringified list in the
v1 CSV; 53,780 covered, 6,394 multi-outlet). Re-ran the auditable/quant V per covering outlet:
positives = deconf PRs covered by outlet X, negatives = deconf PRs covered by no one. Scripts
`per_outlet_v.py`, `per_outlet_topiccontrol.py`.

**Finding: the globally-chance quant V (0.508) masks sign-opposed per-outlet structure.**
Financial outlets prefer **percentage-dense** releases; general/tech outlets do not (or anti-prefer):

| outlet | n_percent pooled AUC | within-topic AUC | verdict |
|---|---|---|---|
| forbes | 0.555 | 0.542 | survives topic-control |
| foxbusiness | 0.554 | 0.537 | survives |
| wsj | 0.550 | 0.538 | survives |
| cnbc | 0.540 | 0.529 | survives |
| fortune | 0.519 | 0.534 | survives |
| nytimes / businessinsider | ~0.51 | ~0.51 | chance |
| cnn / cbsnews / techcrunch | 0.46–0.48 | 0.48–0.49 | at/below chance |

Topic-control (within-topic stratified, sample-weighted) drops the financial-outlet % signal only
~0.01–0.02 ⇒ it's a genuine **outlet-style preference**, not "financial outlets cover earnings PRs."
The financial +0.04 and general −0.03 cancel under the pooled "covered-by-anyone" label → pooled 0.51.

**Caveats:** effects small (AUC ~0.54; n=2.6–8.5k, CI ±~0.012 — real but modest). **big-money
(million/billion) does NOT survive control** (≈chance everywhere). The eye-catching ProPublica
$ 0.566 / bigMoney 0.603 is **n=68 and un-topic-controllable (nan)** → suggestive only. prweb (a PR
syndication wire) and `independent` had the highest pooled V_quant but money features NEGATIVE — artifacts
(short promo / small-n), not quantification.

**Implication:** V is **outlet-heterogeneous, not globally empty.** A per-outlet (or financial-vs-general)
target would surface a real auditable signal the binary any-outlet label destroys. Worth considering an
outlet-conditional framing if press_releases is kept in the paper.

**Quant-ONLY ranking** (`per_outlet_quantonly.py`, features = dollar/bigmoney/percent/numbers/densities,
no length/quotes/dates): literal max = **prweb 0.667 but INVERSE** (PR wire re-hosts number-SPARSE promo;
best feat n_numbers AUC 0.319) — an artifact, not "quant pops." Same inverse for independent/techcrunch.
**Highest genuine positive-direction real outlet = FoxBusiness 0.567** (n_percent 0.554, within-topic 0.537),
then Forbes 0.558, WSJ 0.537 — all via percentages, all survive topic-control. Strongest money-specific =
ProPublica big-money 0.603 but n=68, un-controllable → suggestive only. Lesson: rank by signed/positive quant
direction, not |AUC| — LR exploits negative associations (number-sparse → wire syndication).

## 5e. Outlet automation audit (13 subagents, byline parse + web-confirm)

For each covering outlet we joined its covered articles to the scraped `article_text` (coref-resolved,
URL-normalized join; `build_outlet_bylines.py` → per-outlet `outlet_bylines/<outlet>.jsonl`), then a
subagent per outlet parsed real bylines, bucketed articles, and web-confirmed practices. **Goal: is the
"financial outlets prefer %" signal real editorial taste, or an artifact of automated/syndicated/PR-republished
coverage mechanically echoing number-dense releases?**

| outlet | editorial | dominant non-editorial | verdict |
|---|---|---|---|
| **nytimes** | 52% (91% of parsed) | 1.4% wire + 1.6% DealBook | **TRUST** |
| **wsj** | 58% | 6% in-house Dow Jones Newswires | **TRUST** |
| cbsnews | 60% | 18% AP/Reuters wire + 22% PR/auto | down-weight |
| cnn | 33% | CNNMoney `/prnewswire/` verbatim republish (2014-20) | down-weight (filter newsfeeds) |
| businessinsider | 30% | 12% BI-Intelligence auto + PR + Markets Insider (≤50%) | down-weight |
| marketwatch | 29% | republished PR Newswire/BusinessWire through 2023; 58% no-byline | down-weight |
| fortune | 38% | **49% automated Fortune-500/ranking DB pages** + wire | down-weight |
| foxbusiness | 35% | **30% Zacks automated equity research** + Motley Fool + sponsored = 65% | **EXCLUDE** |
| forbes | **5%** | **69% contributor-network + 19% BrandVoice = 88%** | **EXCLUDE** |
| **independent** | 5% | **92% tickets.independent.co.uk (e-commerce ticketing, NOT news)** | **EXCLUDE/DROP** |
| **pbs** | 0% | **99.9% pbs.org shop/RSS/tag pages — 3 real NewsHour articles in 3000** | **EXCLUDE/DROP** |
| **prweb** | ~0% | 91% paid Cision PR-distribution wire | **EXCLUDE/DROP** |
| cnbc | 5% parsed | parse-limited (600-char head too short) | INCONCLUSIVE |
| washingtonpost | 1.7% parsed | parse-limited; Heliograf auto was small | INCONCLUSIVE |
| techcrunch | — | 0 URL matches to coref corpus | not auditable here |

**Three "outlets" are data artifacts, not news** — `prweb` (PR wire), `pbs` (99.9% retail-shop/nav pages),
`independent` (92% event-ticketing platform). DROP from the dataset entirely.

**Terminology (important — do NOT equate "non-editorial" with "automated"):** "non-editorial" = "not a vetted
in-house staff-journalist newsworthiness decision," spanning SIX categories, only one machine-generated:
staff-editorial (the real signal) · WIRE (other newsroom's human journalist) · CONTRIBUTOR (outside human,
minimal oversight) · SPONSORED (paid ad) · PR-REPUBLISHED (the release itself) · AUTOMATED (machine/template).
Genuinely machine-AUTOMATED is only foxbusiness Zacks (~30%), fortune DB pages (~49%), businessinsider
BI-Intelligence/Markets-Insider, + trivial NYT DealBook/WaPo Heliograf. **forbes is NOT automated** — 69%
human CONTRIBUTOR + 19% paid SPONSORED. marketwatch/cnn = PR-REPUBLISHED (article = the press release).

**Critical impact on §5d (the % finding):** 3 of the top-5 percentage-preferrers are confirmed heavily
NON-editorial, for DIFFERENT reasons — **foxbusiness** (true automation: Zacks earnings bot, ~30%),
**forbes** (human contributor-network 69% + sponsored 19% — Trefis/"Great Speculations" stock-analysts
self-select %-dense topics; NOT a machine), **fortune** (49% auto-generated 500/ranking DB pages =
revenue/market-cap numbers). All three echo release numbers without an editorial newsworthiness judgment, so
the "financial outlets prefer %-dense releases" signal is **largely a non-editorial artifact (automation +
contributor + PR-republish), NOT editorial taste.** It survives topic-control but NOT editorial-integrity
control. The one clean financial outlet, **WSJ (TRUST, 58% editorial), still shows % = 0.538 within-topic**
— a *small* genuine editorial residual, but WSJ is n=460 vs forbes n=8456. NYT (TRUST) is the other clean benchmark.

**Recommended cleanup:** (1) DROP prweb/pbs/independent; (2) for the V/per-outlet analysis, restrict to
EDITORIAL-byline articles (or to TRUST outlets wsj+nytimes) and re-test whether % survives; (3) treat
forbes/foxbusiness/fortune/marketwatch/businessinsider coverage as contaminated by automated financial content.
Caveat: cnbc/washingtonpost audits were parse-limited (my 600-char `head` snippet was too short for their chrome)
— inconclusive, not clean. Byline data: `outlet_bylines/`; tally: `pr_audit/automation_tally.md`.

## 5f. Outlet × category COUNT GRID (re-audit, full text, all matched articles)

Re-audit per refined trust rules. 12 per-outlet subagents (full-text classify ALL matched articles +
contributor-rules web research) + 3 degenerate counted deterministically. Pipeline: `regen_fulltext.py`
(per-outlet full-text files `outlet_full/`), per-outlet subagent classification, `grid_rows.tsv`,
`build_grid.py` (deterministic cross-check + PR-level retention). Categories: STAFF_EDITORIAL & WIRE_REP
(reputable journalist wire: AP/Reuters/Bloomberg/AFP/Dow Jones Newswires — NOT PR Newswire) = TRUST;
CONTRIBUTOR = trust IFF the outlet's program is vetted; PR_REPUBLISHED/SPONSORED/AUTOMATED/NON_ARTICLE = EXCLUDE.

| outlet | staff | wire | contrib | PR | spon | auto | non‑art | unknown | no‑text | TRUST | EXCL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| wsj | 21028 | 86 | 6 | 5074 | 152 | 1 | 1071 | 10783 | 2244 | 21120 | 6298 |
| fortune | 9830 | 5425 | 70 | 695 | 695 | 1 | 17365 | 309 | 1102 | 15325 | 18756 |
| businessinsider ⚠ | 219 | 2739 | 245 | 2254 | 283 | 9296 | 41 | 6533 | 12776 | 3203 | 11874 |
| cnbc | 16860 | 36 | 127 | 1793 | 215 | 13 | 1420 | 2634 | 8703 | 17023 | 3441 |
| cbsnews | 14323 | 2264 | 220 | 614 | 56 | 0 | 895 | 2346 | 1123 | 16807 | 1565 |
| forbes | 511 | 83 | 4759 | 177 | 2288 | 2 | 29 | 4062 | 6948 | 594 | 7255 |
| pbs | 2 | 0 | 0 | 0 | 0 | 0 | 16696 | 91 | 621 | 2 | 16696 |
| foxbusiness | 1003 | 37 | 0 | 130 | 81 | 247 | 521 | 225 | 13543 | 1040 | 979 |
| propublica | 74 | 0 | 20 | 5 | 4 | 0 | 634 | 15 | 11888 | 94 | 643 |
| nytimes | 4221 | 257 | 112 | 0 | 15 | 112 | 67 | 1967 | 4689 | 4590 | 194 |
| independent | 9 | 0 | 0 | 0 | 0 | 0 | 5378 | 265 | 5564 | 9 | 5378 |
| marketwatch | 350 | 133 | 0 | 191 | 388 | 0 | 1317 | 429 | 3131 | 483 | 1896 |
| cnn | 464 | 6 | 43 | 68 | 66 | 0 | 381 | 2857 | 1831 | 513 | 515 |
| washingtonpost | 2503 | 10 | 195 | 110 | 33 | 0 | 15 | 984 | 1753 | 2708 | 158 |
| prweb | 0 | 0 | 0 | 3391 | 0 | 0 | 0 | 0 | 1520 | 0 | 3391 |
| **TOTAL** | **71397** | **11076** | **5797** | **14502** | **4276** | **9672** | **45830** | **33500** | **77436** | **83511** | **79039** |

(techcrunch excluded: 0 articles join the coref text corpus. progressive omitted: 109 matched.)

**Article-level totals (273,486 covered articles across 15 outlets):** TRUSTED **83,511 (30.5%)** ·
EXCLUDED 79,039 (28.9%) · UNKNOWN 33,500 (12.2%, parse-fail but real articles) · NO_TEXT 77,436 (28.3%).

**Contributor-vetting research (per outlet):** VETTED (→ counts as trust): fortune (invite-only, edited),
wapo/nytimes/cnn (op-ed, fact-checked + COI disclosure), cnbc (invite-only experts), wsj (editorial-board
reviewed), cbsnews, businessinsider (syndication-team reviewed, credentials verified), propublica (vetted
partnerships). **EXCLUDED — Forbes:** the "vetted/invite-only since 2024" model POSTDATES the 2014–2023 data;
during our period the Forbes Contributor network was the notorious OPEN/UNVETTED platform → its 4,759
contributors excluded (Forbes trust = staff 511 + wire 83 = 594). N/A: foxbusiness/marketwatch (no program).

**Caveats — 83.5k is a LOWER BOUND:** (1) UNKNOWN 33.5k is largely staff editorial whose byline the parser
missed (wsj 10.8k, bi 6.5k, forbes 4k, cnn 2.9k, cnbc 2.6k) → true trusted is higher. (2) **businessinsider
LOW-CONFIDENCE** — byline parse 1.1%; its 9,296 AUTOMATED is likely inflated by "Markets Insider/BI
Intelligence" in site-wide nav and its 219 staff badly undercounted. (3) wsj PR_REPUBLISHED 5,074 looks high
(possible over-detection of "Business Wire"/"PR Newswire" mentions) — spot-check before trusting that cell.

**Where the data goes:** big losses concentrate in outlets NOT doing editorial newsworthiness anyway —
prweb (paid PR wire), pbs (16.7k retail-shop/nav pages), fortune (17.4k auto Fortune-500 DB pages),
independent (5.4k ticketing), forbes (contributor network). The trusted core is real journalism: wsj 21k,
cnbc 17k, cbs 16.8k, fortune-staff/wire 15.3k, nytimes 4.6k, wapo 2.7k.

**PR-LEVEL retention** (`fast_retention.py`, deterministic classifier over the 170k matched articles; a
positive PR is RETAINED if ≥1 covering article is trusted). Of **53,780 positive PRs**:
RETAINED (≥1 confirmed-trusted) **9,019 (16.8%)** · LOST-all-confirmed-junk **10,828 (20.1%)** ·
indeterminate-NO_TEXT 23,574 (43.8%) · indeterminate-UNKNOWN-parse 10,359 (19.3%).
**Do NOT read this as "keep only 17%."** The strict floor is dominated by NON-signals: 43.8% have no
scraped article text to classify (not bad — just unscraped; no_text is 86% for foxbusiness, 94% propublica),
and 19.3% are deterministic-parser UNKNOWN (mostly staff editorial whose byline was missed; the deterministic
classifier is cruder than the per-outlet subagents). **The defensible loss = the ~20% (10,828) whose ENTIRE
coverage is confirmed junk** (PR-wire/sponsored/automated/non-article) — i.e. never genuinely editorially
covered → effectively **mislabeled positives that should flip to negative / be dropped**. Honest bracket:
confirmed-clean 17% → plausible-keep ~80%; ~20% confirmed mislabeled. **Binding bottleneck = missing article
text, not editorial quality** — resolving no_text (re-scrape article bodies) would sharply raise the
confirmed-retained share. Raw grid saved: `datasets/press-releases/outlet_category_grid.tsv`.

## 5g. RESCUE (no_text via URL + unknown via better parse) + clean-set numerical V

12 rescue subagents re-classified ALL covered articles per outlet (with-text: better byline parse; no-text:
URL taxonomy) → per-article labels `aid_labels/*.jsonl`. Scripts `regen_fulltext.py` (no-text URL files
`outlet_notext/`), `run_clean_V.py`. **The deterministic classifier had massively UNDERCOUNTED** (parse-fail
→ UNKNOWN, no scraped text → unclassifiable). Rescue results (trusted = staff + reputable-wire + vetted-contrib;
Forbes-contrib excluded):

| outlet | trusted (det) | trusted (rescued) | note |
|---|--:|--:|---|
| cnbc | 17,023 | **29,808** | no-text 99% rescued, 0 unknown |
| businessinsider | 3,203 | **21,984** | det parse was 1.1% — catastrophic undercount |
| cbsnews | 16,807 | **21,731** | |
| wsj | 21,120 | **38,923** | rescue judged the 5,074 "PR-republished" = over-detection → staff |
| nytimes | 4,590 | **9,480** | 619 markets.on.nytimes.com press_release feed = PR |
| foxbusiness | 1,040 | 15,591 ⚠ | OVER-rescued (Zacks under /markets/ invisible to URL; text-grid≈7k) |
| fortune | 15,325 | 33,663 ⚠ | OVER-rescued (DB pages have dated URLs; text-grid≈15k) |
| cnn | 513 | 5,461 | | marketwatch | 483 | 4,505 | | washingtonpost | 2,708 | 5,455 |
| forbes | 594 | 2,042 | propublica | 94 | 121 (99% Dollars-for-Docs DB confirmed junk) |
| **TOTAL trusted articles** | **83,511** | **~188,764** (≈160k after Fortune/Fox correction) | |

**PR-LEVEL (final, incl techcrunch dated-URL=editorial): RETAINED 38,651 (71.9%) · LOST-confirmed-junk
6,373 (11.9%, mislabeled positives → flip to negative) · LOST-unknown-only 7,793 (14.5%, unresolved) ·
LOST-no-label 963 (1.8%).** The deterministic "keep only 17%" was a parse/missing-text artifact — true
retention ≈ **72%**, confirmed loss only **~12%** (up to ~28% worst-case if all unresolved are dropped).
**progressive = search.progressive.com insurance-corporate search pages → DROP** (joins prweb/pbs/independent
as the 4 non-news "outlets"). Fortune/Fox over-rescue caveat means the clean count is mildly optimistic.

**NUMERICAL V on the clean set — CONCLUSIVELY CHANCE.** Built balanced set (clean positives + topic-matched
downsampled deconfounded negatives, company-grouped). Numerical-V (dollar/%/numbers/densities) company-grouped
AUC: full-pool **0.508** · deterministic-clean (5,962 pos) **0.505** · **rescued-clean (22,156 pos) 0.508**.
Per-feature n_percent 0.514 / n_numbers 0.511 / dollar 0.501 (sliver = length, length-only 0.515).
⇒ **Quantification does NOT predict PR newsworthiness even on a large CLEAN trusted-coverage dataset.** The
earlier financial-outlet % signal was entirely the automation/contributor artifact, now excluded. Negative
selection: downsample existing negatives, topic-matched + same deconfounding (negatives are clean by
construction — no coverage → no junk articles).

## 6. Open items

- [ ] **Re-run the dense ceiling on the deconfounded, publisher-grouped split** to confirm the 0.71→~0.58
  correction holds for a non-linear model (GPU; ~1 run). Until then the dense drop is inferred from the
  linear grouped/random gap, not measured.
- [ ] Locate the **intact** clean training CSV on sk3 (or rebuild it) — the local canonical file is corrupt.
- [ ] Improve clean extraction: 38% empty bodies is high; many are recoverable from raw (the fallback is
  doing a lot of work and carries HTML cruft).

---

## §5h V-RESCUE (2026-06-26): codex vs subagent vs reference — "can we get ANYTHING?"

Goal: after numeric-V came back chance (0.508), do a deep rescue pass for ANY auditable feature
(esp. number-focused, also factual) that separates covered (j=1) vs not-covered (j=0) PRs honestly.
Ran THREE independent passes on the deconfounded parquet (72,315 rows; 32,789 pos / 39,526 neg):
(a) codex:codex-rescue (88 hand features), (b) my general-purpose subagent (TF-IDF + ~20 features),
(c) my own reference (per-outlet + qualitative + extraction-failure audit).
Honest eval = company-GROUPED 5-fold AUC + within-topic mean AUC. Bar to "find something" = >0.53 on BOTH.
Reference points: prior numeric-V 0.508 grouped; length-only 0.515; dense linear ceiling 0.584 grouped / 0.546 within-topic.

### Headline reconciliation (the two agents only *appeared* to disagree)
| feature set | n_feat | grouped AUC | within-topic AUC | clears both? |
|---|---|---|---|---|
| **ALL hand features** (codex) | 88 | 0.561 | 0.540 | yes |
| **CLEAN** (drop scrape-artifact/wire/structure-junk) | 74 | **0.554** | **0.534** | **yes** |
| factual_entity_structure | 26 | 0.539 | 0.526 | no (within-topic) |
| readability_structure | 20 | 0.537 | 0.530 | borderline |
| **NUMERIC-only** | 27 | 0.529 | **0.517** | **NO** (fails within-topic) |
| event_keywords | 18 | 0.524 | 0.514 | no |
| subagent "all_combined" | 20 | 0.505 | 0.548 | no (grouped) |
| best single univariate (sentence_count) | 1 | 0.524 | 0.520 | no |
| n_percent / n_years / n_dollar single | 1 | ~0.505 | ~0.505 | no (chance) |

- The codex "win" (0.565) vs subagent "nothing works" (0.505) is **purely a feature-count effect**: 88 vs 20 features. With ~74 *clean* auditable features you reach grouped 0.554 / within-topic 0.534 — i.e. **a broad bag of cheap features recovers essentially the entire honest dense ceiling (0.584/0.546)**. There is NO tacit/dense residual hiding above cheap features. But the ceiling itself is just low.
- **Numbers specifically still fail.** 27 number features together = grouped 0.529 / within-topic **0.517** (fails the topic-controlled bar); any single number feature is chance (0.50–0.52); **$ amounts are chance everywhere (~0.49)**. The earlier numeric-V=0.508 verdict holds even with a much richer numeric set.
- The residual signal is **diffuse and non-numeric**: document structure/length (more sentences, longer, short-headline present), "hard-news" event keywords (regulatory/legal, earnings/guidance, M&A, funding), readability, and **"cites research/survey/study"** (kw_research_survey, the strongest clean keyword) — each worth a hair; only their union reaches the ceiling. Not a clean single "V metric."

### TF-IDF top features (subagent) — the only strong text signal is a distribution artifact
- TOP coverage tokens: `prnewswire`, `news provided`, `provided by`, `eastern` (PRNewswire timezone dateline), `table`, `gartner`, `professor`, `university of`, `ceo of` — i.e. the **newswire-distribution fingerprint** + a few entity/topic confounds.
- TOP no-coverage tokens: company names (`activision`, `entergy`, `aon`, `citi`), soft-news (`celebrate`, `on facebook`, `climate change`, `sustainability`, `winter`).
- The wire fingerprint is exactly the **PR-republication confound** we removed last session — and on its own it is 0.482 grouped (chance). So even TF-IDF's "signal" is not editorial taste.

### Per-outlet differences (reference; label = covered by outlet O, topic-matched negs)
| | n_pos | $ AUC | % AUC | # AUC | wire AUC | note |
|---|---|---|---|---|---|---|
| forbes | 8456 | 0.497 | **0.554** | 0.528 | 0.502 | % / metrics |
| foxbusiness | 2678 | 0.518 | **0.549** | 0.532 | 0.488 | % / metrics |
| wsj | 460 | 0.452 | 0.538 | 0.515 | 0.470 | % |
| cnbc | 2460 | 0.510 | 0.526 | 0.522 | 0.523 | mild numeric |
| fortune | 1563 | 0.502 | 0.526 | 0.506 | 0.480 | |
| marketwatch | 1738 | 0.473 | 0.489 | 0.495 | **0.572** | **wire-driven** (republisher) |
| techcrunch | 1818 | 0.524 | 0.495 | 0.486 | 0.499 | funding $ up, **quotes 0.374 (fewer)** |
| nytimes / wapo / cnn / cbsnews | — | ~0.49 | ~0.49–0.51 | ~0.49 | ~0.50 | nothing |
- Avg % AUC: **financial outlets 0.528 vs general 0.499**. Financial outlets lean (weakly) on percentages/metrics; general outlets don't lean on numbers at all. TechCrunch logic = startup funding magnitude + fewer quotes. MarketWatch coverage is wire-distribution-driven (consistent with the Zacks/automation finding).

### Data-quality note
~7.2% of deconfounded rows are extraction-failure boilerplate ("does not appear to contain a press release", "raw page content", "news_release_found"). Weakly label-skewed (AUC 0.491) → noise, not a confound; dilutes signal slightly. Not worth re-cleaning for this conclusion.

### VERDICT
We could NOT get a usable number-focused V metric. Numbers (incl. $, %, magnitudes) are chance-to-marginal and fail topic control. The honest deconfounded ceiling (~0.55–0.58) is fully reachable with a broad bag of cheap auditable features — meaning press_releases has **no tacit residual above cheap features**, but its ceiling is low and the signal is diffuse (structure + hard-news event type + research-citation), not quantitative. Confirms press_releases = **low-ceiling identity/taste task**, not a V/A showcase. Scripts: scratchpad/pr_vrescue/{codex_*,mine_*,ref_analysis.py,fast_outlets.py}.

### §5i DENSE upper bound landed (2026-06-27) — the triple is V≈A≈dense≈0.58
- **Dense neural (bge-m3 embeddings, full 72k, company-grouped): LR 0.584 / within-topic 0.556; MLP 0.552.** Identical to the cheap-feature linear ceiling (0.584/0.546) and the cheap-V bag (0.554/0.534); rubric-A (n=600) grouped 0.543 / within-topic 0.515.
- Conclusion: every **document-local** method — cheap counts (V), 70B-judged rubrics (A), neural embeddings (dense) — hits the SAME ~0.58 grouped / ~0.55 within-topic wall. No nonlinear/semantic headroom (MLP≤LR), no articulability gap (A≤V), no tacit residual.
- Interpretation for the V-agenda: the wall is because the true drivers of coverage (announcer PROMINENCE, NOVELTY vs prior news, TIMELINESS/news-peg) are **relational** — they live outside the PR text, so no document-local model can see them. The only V metrics that could exceed 0.58 are **retrieval/KB/cross-checking** metrics that inject external information. Ops note: 70B A-run kept OOMing under GPU2 co-tenant contention; dense (bge-m3, light) prioritized and completed; 2400-PR A-rescore deferred to a freer GPU.

### §5j Relational / complex-V metrics (2026-07-01) — novelty & density are CHANCE
Built relational V metrics from bge-m3 PR embeddings + PR dates (offline, CPU). Company-grouped + within-topic:
| feature | grouped | within-topic | note |
|---|---|---|---|
| NOVELTY (max-cos to prior same-company + same-topic PR) | 0.485 | 0.499 | chance (covered PRs NOT more novel) |
| + competitive density (±14d crowdedness) | 0.485 | 0.499 | chance |
| n_prior_company (prolific-announcer proxy) | 0.422 | 0.443 | inverted noise, not a clean prominence signal |
| **missingness flags only** | 0.560 | 0.533 | ARTIFACT: 29% of PRs lack a date; date-availability = scrape provenance, not newsworthiness |
| all + missingness (naive combined) | 0.581 | 0.549 | = the missingness leak, NOT relational signal |
- **Verdict:** genuine relational novelty/redundancy and competitive density do NOT predict PR coverage (chance). The one seemingly-strong combined number was a date-missingness provenance leak; excluded. My predicted "best honest bet" (retrieval novelty) fails.
- **Timeliness/news-peg: NOT cleanly implementable** — the only dated news corpus we have (`raw_data/all-coref-resolved`, and only 10/41 shards present) is the *coverage-side* corpus, so PR→news similarity is circular with the label. Needs an independent general-news index we don't have. Not attempted (would produce a leaky false-positive).
- **Prominence (external Wikipedia pageviews):** building on laptop (sk3 has no internet); result pending.
- **Claim-checkability (70B FactEval-lite):** queued; all GPUs saturated, pending a free GPU.
Scripts: build_relational_offline.py, relational_offline.parquet, decomp_rel.py.

### §5k COMPLETE V-battery (2026-07-01) — nothing beats the ~0.58 document-local ceiling
All metrics on the deconfounded set (72,315 PRs), company-grouped / within-topic AUC:
| metric | family | grouped | within-topic |
|---|---|---|---|
| Dense neural (bge-m3) | document-local | 0.584 | 0.556 |
| Dense linear (cheap feats) | document-local | 0.584 | 0.546 |
| V cheap-feature bag (74) | document-local | 0.554 | 0.534 |
| A rubrics (309, 70B) | document-local | 0.543 | 0.515 |
| V numbers-only | document-local | 0.529 | 0.517 |
| Novelty (retrieval) | relational | 0.485 | 0.499 |
| Competitive density | relational | 0.506 | 0.500 |
| Prominence (WP pageviews, gated 44%) | external | 0.488 | 0.516 |
| Claim-checkability (70B FactEval-lite, n=2400) | external | 0.512 | 0.502 |
| Relational combined (nov+dens+prom) | relational | 0.487 | 0.500 |
| Timeliness / news-peg | — | NOT IMPLEMENTABLE (circular coverage corpus) | — |
| **Dense only** | — | **0.584** | **0.556** |
| **Dense + ALL relational** | — | **0.583** | **0.555** |

- **Stacking relational/external onto the dense model changes nothing (0.584→0.583).** Every relational/external metric is chance; none reach, let alone exceed, the ~0.58 document-local ceiling.
- Covered vs not-covered PRs are INDISTINGUISHABLE on announcer prominence (log-pageviews 5.66 vs 5.65) and verifiable-claim count (13.42 vs 13.43).
- **Final verdict:** deconfounded press-release newsworthiness has a hard ~0.58 grouped / ~0.55 within-topic ceiling. No V metric — thin/local, articulated-rubric, dense-neural, or relational/external (novelty, prominence, competitive density, claim-checkability) — beats it. No articulability gap (A≤V), no tacit residual (dense=cheap), no relational lever. The coverage residual above 0.58 is genuinely EXTERNAL/IDIOSYNCRATIC (newsroom timing, editor relationships, slot availability) = irreducible noise, not communicable taste. press_releases is a low-ceiling identity/taste task — a NEGATIVE example for the V/A agenda, useful as a floor/contrast, not a showcase.
Scripts: build_relational_offline.py, build_pr_claims.py, prom_build_wikipedia_prominence.py, combined_eval2.py; features: relational_offline.parquet, pr_prominence_feat.parquet, pr_claims_scores.npz.

### §5l Per-outlet V-metric PRESENCE (2026-07-01) — outlets have distinct selection profiles
Mean metric level for PRs covered by each outlet, vs the all-covered (pos) baseline. (Coverage predictive AUCs were ~chance within each outlet — this is about selection *profile*, not predictive power.)
- **FINANCIAL outlets index HIGH on numbers** (the selection profile): cnbc n_dollar 1.42x, wsj 1.22x; foxbusiness n_percent 1.36x, cnbc 1.29x, wsj 1.27x; wsj n_numbers 1.29x, numeric_density 1.23x, marketwatch 1.15x. Financial desks select number-dense PRs.
- **GENERAL outlets index LOW on numbers**: cnn n_dollar 0.63x / n_percent 0.52x; cbsnews n_percent 0.77x / n_bigmoney 0.73x. General-news desks select softer/feature PRs.
- **WSJ** = hard-financial signature: HIGH on all numbers, LOWEST n_quotes (0.59x), LOWEST wire (0.33x — gets scoops directly, not wire-fed).
- **MarketWatch** = most wire-distributed (1.89x) + numeric-dense — confirms the republisher/Zacks finding.
- **TechCrunch** = startup/funding signature: LOWEST n_quotes (0.50x), low n_percent (0.64x)/n_numbers (0.75x), but HIGHEST comp_density (1.72x — most crowded announcement space).
- **Prominence (log-pageviews) ~flat across all outlets** (0.92–1.11x) → every outlet's covered PRs come from equally prominent companies. Confirms prominence doesn't differentiate (chance AUC).
- **Relational novelty ~flat** (red_company/red_topic ~1.00x everywhere) → no outlet selects more/less novel PRs (chance AUC).
**Takeaway:** V-metrics ARE differentially present by outlet (clear beat/selection profiles), but differential *presence* ≠ coverage *predictive power*: selection is by beat/topic, and within-beat the covered-vs-not residual stays external/idiosyncratic. Script: /tmp/outlet_metrics.py (sk3).

### §5m Infilling machinery (global + ctree/MOB) + MCC certificate (2026-07-01)
Ran BOTH engines (per AGENT_PLAYBOOK) on a company-grouped split of the deconfounded set.
- Setup: 28-metric deterministic V-bank (free materialization), 70B vLLM judge (sk3 GPU1) for proposed-metric materialization, GLM-5.2 proposer (z.ai). z.ai subscription rate-limited the all-GLM path, so split: bulk judge = 70B, proposer = GLM (few calls). discover=989 / guard=585 / test=626 (372/129/215 companies). curated z=[topic, text_length] (m_z=2), n_perm=999, alpha=0.05, acceptance=guard-AUC gain ≥0.03.
- Baseline bank: GUARD AUC **0.499** (≈chance), TEST AUC **0.549**.
- **Global infilling:** ran; scored ~3 GLM-proposed candidates on 1574 items via 70B (~4700 judge calls); kept **0** — no metric beat the +0.03 gate over the ~0.50 base. → no (iii) corpus-uniform residual certified.
- **ctree/MOB:** STUMPED (terminals=1; no split on topic/length survived Bonferroni at n_perm=999) → no (i) moderation-shaped residual. Root flagged a gap (0.549 < 0.55); GLM proposed "formal_business_register", 70B scored → auc_gain **−0.009** → dropped. Gap not closable.
- **MCC certificate (n=626 test):** V_bits = **+1.05** bits (bank over null) → **N_lower = ceil(1.05/log2 3) = 1** (≈1 articulable metric-equivalent, essentially minimal). Dominance gate PASS (bank 0.549 ≤ dense 0.584 − 0.03) but **N_upper RIGHT-CENSORED** (no dense scaling curve, only single dense ceiling point). Articulable ceiling right-censored at ~0.584.
- **Verdict (MCC trichotomy):** a STUMP — no (i) moderation or (iii) corpus-uniform residual surviving Bonferroni at (n=989, m_z=2, n_perm=999); NOT "saturated." Consistent with the full-72k wall (V≈A≈dense≈0.55–0.58; relational adds nothing). The coverage residual is external/idiosyncratic.
- Caveats: (a) bank = V-layer code-metrics, not the 309-rubric A-layer (which would need 70B rubric materialization); (b) dev set is playbook-scale (company-grouped limits n); (c) base guard AUC ≈ chance so the +0.03 gate is a low bar — and still nothing cleared it. Driver: datasets/press-releases/run_infill_pr_sk3.py; log logs_pr_infill_sk3.log.

### §5n A-layer + GEPA at k≥3 (2026-07-02) — SHOWCASE confirmed: V<A<dense ladder
**Label-threshold reframing (§5 key):** the ≥1-outlet label was 88% single-outlet noise. At k≥3 (broad/consensus coverage, 1478 pos), the ceiling lifts and a real structure appears:
| layer | k=1 (old) | k≥3 grouped | k≥3 within-topic |
|---|---|---|---|
| V (cheap counts) | 0.531 | 0.628 | 0.627 |
| **A (40 rubrics, 70B)** | 0.543 | **0.648** | **0.645** |
| dense (bge-m3) | 0.584 | 0.705 | 0.705 |
- **Clean V < A < dense ladder at k≥3.** Articulated rubrics add ~0.02 over cheap counts; dense adds ~0.06 over A (tacit residual). Within-topic tracks grouped (not topic confound). press_releases FLIPS from floor (k=1) to showcase (k≥3).
- Stage 1: 40 coverage-selected rubric medoids, 2956-item balanced set (1478 pos ≥3 outlets / 1478 0-outlet neg), 556 companies, company-grouped CV. NA rate 65% (judge marked many rubrics N/A; imputed to 0.5). Top rubrics (univariate AUC): Company boilerplate completeness 0.585, Uncertainty communication 0.573, Limitations/caveats 0.571, Calls-to-action 0.557, ESG disclosure 0.546, Lede/5Ws 0.544, Original research as newsworthy 0.534, Investor clarity 0.533 — i.e. professional PR-craft dimensions.
- **Stage 2 GEPA POC:** Gemma-4-32B judge (served, gemma4 env) + GLM-5.2 proposer/reconstructor (z.ai zai_anthropic) via make_roles_mixed; objective=fidelity_scalar (recon R + reliability + ...). 6 viable rubrics × 2 rounds. 1/6 accepted: "Lede uses concrete details" seed_fid 0.635→acc 0.746 (+0.11). Others ran but mutants didn't consistently beat the cross-family acceptance gate (some regressed). Machinery works end-to-end; POC scale (head-of-file 40 rubrics, 2 rounds, GLM-quota-bound).
- Caveats: (1) A-layer NA rate 65% — signal concentrated in applicable subset; applicability-weighted scoring could sharpen. (2) k≥3 imbalance 1:27 (handled: stratified+grouped CV, AUC threshold-free). (3) GEPA is POC-scale; more rounds + coverage-selected rubrics would find more gains.
- Scripts: run_A_layer_k3.py (scores pr_A_k3_scores.npz), run_gepa_pr.py. Servers killed, GPU1/GPU3 freed.
