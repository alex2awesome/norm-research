# V6 — StackOverflow answer-votes cell (the software-code field's VOTE column)

Date: 2026-08-08. Charge: the next build in the goal's order after the completed
V8 N&C co-signing cell — fill the software-code row's missing vote column
(notes/2026-08-08__vat-3xN-decomposition-grid.md, row "Software code",
cell "SO votes: **UNBUILT (V6)**"). Design contract:
notes/2026-08-05__taste-decomposition-design.md §12 (production V+A+T pipeline)
and §14 (pre-kill checklist, binding).

Status: **BUILT.** Population, y, splits, V extractor (46), A bank (40 mined →
39 after the enforced collapse gate), 7/7 Gemma-scored shards, K=50 anchor
battery certified, 3-seed dense (T), Layer-1 + Layer-2(a) ledger with
ids-carried OOF at exact reproduction. One item-view sensitivity arm outstanding
(§6e), which the gate verdict is explicitly conditioned on.

**Headline: the software-code vote column is HIGHLY ARTICULABLE.** Same-rows
Δ_beyond is **−.0297 eval / −.0143 test** — the 39 stated criteria plus 46
surface features match or *beat* an 8B dense reward model on identical rows —
and they combine **additively** (Δ_interact −.0014, P(>0) = .25). Unlike every
previous vote cell, within-question readouts *exceed* pooled ones, because the y
is a within-question median split. Cross-y: vote and verdict are strongly
**aligned** here (φ = +.552), the opposite sign to N&C co-signing (φ = −.160).

---

## 1. The y definition, and why

### 1.1 What "community endorsement" is on StackOverflow

StackOverflow carries two distinct community signals on an answer, and they are
*different columns of the grid*:

| signal | who produces it | grid column |
|---|---|---|
| the asker ticks one answer as **accepted** | ONE person, the asker | VERDICT |
| every reader may **upvote / downvote** | the crowd | **VOTE — this cell** |

The vote column is the crowd's revealed preference and is the exact structural
analogue of the other vote cells in the grid: citations for peer review, upvotes
for creative writing, crowd votes for humor, co-signing for N&C.

**y_vote = 1 if the answer's raw net vote Score is STRICTLY ABOVE the median
answer Score on its own question, 0 if STRICTLY BELOW; ties at the median are
dropped.**

### 1.2 Why the within-question relative framing, and not a raw threshold

Three candidate y's were on the table (per the charge). Rationale, decided
before any instrument was built:

| candidate | verdict | why |
|---|---|---|
| **raw score threshold** (e.g. Score ≥ 3) | REJECTED | Raw answer score is dominated by *question popularity* — how many people ever loaded the thread. A raw-threshold y would largely rank questions, not answers. This is precisely the failure the V8 N&C build documented ("the pooled number is docket composition", identity-alone .6951 ≈ 92% of pooled VA_nl). Conditioning on the question removes the popularity offset **by construction** rather than hoping a nuisance control catches it later. |
| **within-question rank / median split** | **CHOSEN** | Neutralises question-popularity spillover by construction; every comparison is between answers that had the same thread, the same audience and (approximately) the same exposure window. It also makes the vote column commensurable with the verdict column, which is inherently within-question — so the cross-y contrast runs on identical rows with identical instruments. |
| **accepted-vs-votes contrast** | KEPT, BUT SEPARATE | `y_accepted` is carried as its own column and is **never merged into y_vote**. |

The third row is the **math.SE lesson the charge names**. The legacy
`so_python_v2` pool already in this repo defines its label as
`accepted AND Score>=3` versus `NOT accepted AND Score<=0` — a target that fuses
the verdict and the vote channel into one variable. That label cannot answer
either column of the grid: a positive is simultaneously "the asker chose it" and
"the crowd upvoted it", so any residual is unattributable. This build keeps the
two channels orthogonal and measures their relationship (§7) instead of
assuming it.

The chosen rule is a **verbatim mirror of the math.SE vote cell** — "1 = raw
vote score strictly above the median answer score on its own question, 0 =
strictly below; ties at the median dropped"
(`datasets/math-stackexchange/v2_va/population_manifest.json`). Mirroring is
deliberate: math.SE and SO are the two StackExchange vote cells in the grid, and
an identical y rule makes them a matched pair rather than two unrelated builds.

### 1.3 The exposure confound — named, and why the design already absorbs most of it

The obvious threat to any vote y is **exposure**: an answer posted earlier
accumulates votes for longer and sits higher on the page. Two of the three legs
of that confound are handled by the y definition itself (same question ⇒ same
thread, same audience, same total traffic). The leg that survives is
**answer order within the question**, and it is large here — measured, not
assumed (§5, the position line): position 1 is 62.9% vote-positive and position
5 is 33.5%, and position alone scores **.5974 pooled / .6552 within-question**.
That is why the position line is a required readout in this cell rather than an
appendix curiosity, and why it is reported beside every instrument number.

### 1.4 Ground truth on the label channel, before any instrument work

The V8 transferable lesson is binding here: *check the label channel against
ground truth before building on it.* Two findings, neither visible from the
upstream code:

1. **`Body` on this mirror is Markdown, not HTML.** The ingest script's
   docstring asserts "KEEP Body raw (HTML)". Measured over 400K answers:
   `<p>`/`<pre>` appear on **0.21%** of rows, while ``` fences appear on
   **61.3%** and inline `` `code` `` on **74.4%**. Consequence: the legacy pool
   builder's `strip_html()` — `re.sub(r"<[^>]+>", " ")` — is not a harmless
   no-op on this corpus, it is a **code shredder**: it deletes `List<int>`,
   `<module>`, `<class 'x'>` and every `a < b … c > d` span. This build keeps
   Markdown and never calls it. The V extractor parses Markdown directly.
2. **`Score` is the raw net vote** (upvotes − downvotes), observed range
   [−44, 17568], 19.4% exactly 0, 1.6% negative. No transformation, no
   agency-populated field standing in for it (contrast the N&C
   "Duplicate Comments" column, which was dead on arrival).

A third, smaller landmine was caught in the V extractor itself: the Markdown
blockquote regex `^\s*>` matches the **Python REPL prompt `>>>`**, which is
pervasive in this corpus. `v_n_blockquotes` uses `^\s*>(?!>)`.

### 1.5 Unit of analysis and the tie rule

One row per **answer**. Ties at the question median are dropped, exactly as
math.SE does: a tied answer has no within-question direction, so a coin-flip
label would inject pure noise into a balanced cell. The tie rate is material and
is recorded — **23.7%** of sampled rows (3,799 of 16,001), higher than math.SE's
because SO scores concentrate hard on small integers (Score 1 and 2 are the two
modal values). The dropped rows stay in `population.csv.gz` with
`y_vote` null; nothing is deleted.

---

## 2. Reuse log (reuse-before-rebuild)

Inventory ran before any build, on both the repo and sk3. The headline: **a full
StackOverflow corpus already existed and was never used for a grid cell.**

| component | source | reuse |
|---|---|---|
| **Raw corpus (32 GB, 58 shards)** | `sk3:…/datasets/stackoverflow_python/_raw_parquet/stackoverflow-posts-*.parquet` — HuggingFace StackOverflow posts mirror, downloaded 2026-06-11 | **100% — not re-downloaded** |
| **Python Q/A extraction** | `so_python_questions.parquet` (1.3 GB, 2,297,505 questions) + `so_python_answers.parquet` (974 MB, 3,212,022 answers), built 2026-06-11 by `datasets/stackoverflow_python/so_python_v2/ingest_shards_to_python.py` | **100% — not re-ingested** |
| Verifiability stratum (`framework`/`lib_data`/`stdlib`) | `build_so_python_pool_v2.py` tag sets | 100% — copied verbatim so the two populations stay comparable |
| Split bucketer | `datasets/patents/build_dense_standard_claimfell.py::stable_hash_bucket_map` | 100% — imported, not reimplemented |
| y construction rule | math.SE vote cell | rule mirrored, re-implemented over SO columns |
| V extractor shape | `datasets/math/stackexchange/va/v_features.py` | structure + the whole generic length/structure/register tail reused; the LaTeX block replaced by a code block |
| Layer-1 estimators | `layer1_gemma_cells` + `scaleupC_layer1` (`outer_folds`, `linear_oof_family1`, `gbm_oof_family1`, both bootstraps, `dense_T`, `run_cell`) | **100% — imported, not reimplemented**, so this cell is numerically comparable to `mathse_vote_score` |
| Gemma scoring loop, shard checkpointing, NA parsing, per-shard anchors, K≥50 battery | `datasets/va_gemma_banks/score_va_gemma_banks.py` + `score_scaleupC_banks.py` | **100% — imported**; only the bank builder is new |
| Dense recipe | `methods/dense/run_dense_standard_scaleupC.sh` | 100% — frozen recipe, no flags added |
| **The legacy `so_python_v2` POOL / balanced / pairwise artifacts** | `pool/so_python_v2_pool.csv.gz` etc. | **DELIBERATELY NOT REUSED** — see §2b |
| **A bank (40 criteria)** | — | **BUILT** (§4) |
| **y itself** | — | **BUILT** (§1) |

**Reuse fraction: everything upstream of the label was reused.** The build cost
one CPU pass over two parquets, one Gemma scoring run, and one dense training
run. No data was downloaded and no corpus was re-ingested.

### 2b. Why the legacy SO pool was not reused

`datasets/stackoverflow_python/so_python_v2/` ships a complete, propensity-
balanced, position-matched pool (210,860 rows) with a pairwise companion. It was
inspected and rejected as this cell's population for three reasons, all
recorded rather than assumed:

1. **Its label is the fused verdict+vote target** (`accepted AND Score>=3` vs
   `NOT accepted AND Score<=0`) — §1.2. Unusable for either grid column.
2. **It is question-DISJOINT by construction** (`after_question_disjoint_pos`
   in its manifest; 39,235 questions carrying both classes were *dropped*).
   Question-disjointness makes the within-question readout — this cell's
   docket-analog and its single most important robustness check — impossible.
3. Its own manifest records `p_positive_posted_earlier = 0.749`, i.e. the
   time-order confound was measured and then removed by dropping exactly the
   questions where it could be studied.

Its tag sets, stratum definition and year window were reused; its rows were not.

---

## 3. Population and splits

`datasets/stackoverflow-votes/build_so_votes_population.py`. Year window
2016–2023 (mirrors the legacy pool), min 50 chars, parent question must exist,
question must carry ≥2 in-window answers, and — mirroring math.SE — the sample
is restricted to questions carrying **both** signals (a defined vote y and an
accepted answer) so the cross-y contrast runs on identical rows. Whole questions
are drawn in `sha256("so-votes-v1|" + question_id)` order until the row target
is reached. No seeded shuffle.

Funnel: 3,212,022 answers → 2,331,470 in the year window → 2,321,933 ≥50 chars
with a live parent → 1,337,785 in multi-answer questions (516,480 questions) →
752,091 vote-defined → 227,434 questions carry both signals → sampled.

| | rows | questions |
|---|---|---|
| population (all rows) | 16,001 | 5,972 |
| **vote-defined (the analysis population)** | **12,202** | 5,972 |
| tie-at-median, dropped from y | 3,799 | — |

y_vote pos rate **.5238** — near-balanced, which is what a within-question
median split should produce. y_accepted pos rate .3732.

Question-grouped stable-hash 80/10/10 via the patents bucketer, pos-rate
matched:

| split | rows | questions | pos rate |
|---|---|---|---|
| train | 9,762 | 4,468 | .52377 |
| eval | 1,220 | 759 | .52377 |
| test | 1,220 | 745 | .52377 |
| **all** | **12,202** | **5,972** | **.52377** |

Pos rates match to five decimals.

### 3b. §14 pre-kill checklist, recorded UP FRONT

The design note's §14 binds before any dead/terminal verdict may be read off
this cell. All five items are on the record now rather than reconstructed later:

| item | value |
|---|---|
| (1) absolute minority-class count in train | **4,649** (of 9,762) — this cell is not label-starved, unlike mathlib (~360) |
| (2) simple baseline on the same split | **TF-IDF + logistic, question-grouped 5-fold: .6079** — well above chance, so a big model at chance would be a training failure, not a cell failure |
| (3) registry search for historic working runs | done — no prior V+A stack on SO votes exists anywhere in the repo (the legacy pool was never scored); this is a FIRST-FIT cell |
| (4) which design is under test | question-grouped 80/10/10, 5,972 groups; NOT an outlet/repo-held-out transfer design, so the homepage k=8 failure mode does not apply |
| (5) seed spread vs claimed effect | reported in the ledger; the 3-seed dense spread is quoted beside every Δ |

Trivial-channel baselines on the same split, for reference beside every
instrument number:

| channel | pooled AUC | within-question AUC |
|---|---|---|
| TF-IDF on the answer body | .6079 | .6160 |
| **answer position (first-answer advantage)** | **.5974** | **.6552** |
| answer char length | .5933 | .6000 |
| **question identity alone** | **.6043** | — |

Two things to notice immediately. First, **question identity alone is only
.6043** — a far smaller group leak than the N&C co-signing cell (.6951) or the
N&C responded cell (.9156). Second, and unlike every previously built vote cell,
**the within-question readouts are HIGHER than the pooled ones**, not lower.
Both are direct consequences of the y definition: a within-question median split
removes the between-question offset by construction, so there is little
composition left for a pooled number to be inflated by. This is the y-choice in
§1.2 doing exactly the work it was chosen to do.

### 3c. Spot-check of the y (dataset-first protocol)

Read before trusting. The extremes are what the construct predicts, and they
also make the case for §1.2 concrete.

| margin (Score − question median) | Score | q median | pos | question |
|---|---|---|---|---|
| **+562** | 660 | 98 | 1/3 | "How do I add default parameters to functions when using type hinting?" |
| **+394** | 399 | 4.5 | 1/14 | "How to find which columns contain any NaN value in Pandas dataframe" |
| **+346** | 347 | 1 | 1/5 | "Select rows in pandas MultiIndex DataFrame" (the canonical mega-answer) |
| **−80** | 23 | 103 | 3/3 | "How to test single file under pytest" |
| **−64** | 34 | 98 | 2/3 | "How do I add default parameters…" (same thread as the +562 row) |
| **−48.5** | 0 | 48.5 | 2/2 | "how to reset index pandas dataframe after dropna()" |

**The negatives are the argument for the within-question y.** An answer with
**23 upvotes** is a *loser* — it sits on a thread whose median answer has 103. A
raw-score threshold (the legacy pool used Score ≥ 3) would have scored it a
strong positive, and would have scored the 0-upvote answer on the 48.5-median
thread identically to a 0-upvote answer on a thread nobody read. The
within-question split separates "this answer was good" from "this thread was
popular"; the raw threshold cannot.

The cell's own score distribution is tight — min −16, p25 0, median 1, p75 3,
max 660 — which is why the tie rate is 23.7%: most SO answers live at 0, 1, or 2
votes.

Median body length: vote-positive **572** chars vs vote-negative **435**. Longer
is better here, the ordinary direction, and the opposite of the N&C co-signing
cell's inverted length signature.

Pos rate is flat across the verifiability strata — stdlib .5251 / lib_data .5203
/ framework .5260 — as a within-question median split must be, since the split
is computed inside a question and a question sits in exactly one stratum.

---

## 4. Instruments

### 4a. V — 46 deterministic, label-blind surface features

`datasets/stackoverflow-votes/va/v_features.py`, adapted from the math.SE
module. Input is the ANSWER BODY ONLY — the question text is stripped by the
caller so question length cannot leak into an "answer style" feature — and no
date, score, position or accept flag is inspected.

The generic tail (length, sentence structure, list markers, type-token ratio,
register lexicons: deductive, hedging, first/second person, instruction verbs,
meta-edit) is carried over from math.SE. The math-specific block (display math,
LaTeX density, proof framing, heavy machinery) is **replaced** by the code block
this corpus actually carries: fenced/indented code and its character share,
inline code spans and density, imports, shell/pip invocations, REPL prompts,
tracebacks, version pins, doc/GitHub/SO links, headings, blockquotes, code
comments, and the prose-to-code ratio — plus two diagnosis-register lexicons
(`v_cause_language`, `v_error_mention`) aimed at the same construct the A bank's
real pole targets.

**Early V-only leg** (run with the frozen Layer-1 estimators before the judge
landed, to validate the plumbing; `va/v_only_early.json`):

| quantity | value |
|---|---|
| V_lin | **.6347** |
| V_nl (mean of GBM seeds 0/1/2) | **.6379** (spread .0023) |
| V_interact = V_nl − V_lin | **+.0032** |
| V_lin within-question | **.6425** |

Two things this already settles. (i) **V carries essentially no nonlinearity
here** (+.0032, barely above the .0023 seed spread). That is the opposite of the
N&C co-signing cell, whose V_interact of +.0353 was the SURFACE-nonlinearity
signature that routed it to Track B — this cell shows no such signature, so
whatever interaction the VA stack finds is not a V-internal surface effect.
(ii) The **length direction is the ordinary one** here: `v_log_len` alone reads
.5933, i.e. longer answers get more votes. N&C co-signing was INVERTED (.4182).
Top single features are `v_n_inline_code` .5956, `v_paragraph_count` .5938,
`v_log_len` .5933 — a length-and-structure cluster with no single dominant
feature (largest |AUC − .5| = .096), so this is a composite surface signal
rather than one length proxy. V also beats the TF-IDF baseline (.6347 vs .6079).

### 4b. A — 40 criteria, mined then merged

`datasets/stackoverflow-votes/va/rubrics.jsonl`, built by
`va/build_rubrics.py` (which carries every merge decision in code).

Four label-blind proposer agents each read a disjoint batch of **train-split**
exemplars — whole questions with all their answers, no y attached, drawn in
`sha256("so-mine|" + question_id)` order — and proposed candidates in the
math.SE house style, each self-labelled Track A (real) or Track B (surface).
**70 candidates came back (53 A / 17 B).** The audit collapsed 22 duplicate
clusters — the four proposers converged hard, which is itself the redundancy
signal the program looks for — leaving:

* **36 Track A "real" criteria**, spanning cause/diagnosis (5), correctness (7),
  scope and version conditions (4), risk and edge cases (4), engagement with the
  asker's question (6), and craft/alternatives/handover (10).
* **4 Track B "surface" probes, declared spurious up front**: `Contains a code
  block` (the charge's named surface pole), `Contains an external hyperlink`,
  `Shows an output block`, `Uses a numbered or bulleted list`. They are scored in
  the SAME matrix and carry `track: "B"`, so every readout can split A_real from
  A_surface without a re-score. Their real-pole counterpart is the
  cause/diagnosis family a01–a05 ("Says why the original approach failed", "Fix
  is tied to its diagnosis", …) — the charge's real probe.

Every dropped candidate is listed with its reason in `build_rubrics.py`. One
drop is worth naming here: **"Answer exceeds typical length" was cut** even
though it was an honest Track B proposal, because V already carries
`v_log_len` / `v_word_count` / `v_prose_char_count` and scoring it into A would
re-inject the length channel into the A matrix and contaminate the V-vs-A
contrast, which is one of the ledger's load-bearing comparisons.

### 4c. The judge context — a real bug the smoke test caught

Scoring is Gemma-4-31B, offline-batch vLLM, temperature 0, one token per
(item, criterion), NA allowed. The first smoke run copied the math.SE context
verbatim (question **title** + answer) and produced an **NA rate of 51.8%, with
three criteria at 100% NA**: "Engages the asker's actual code", "Edge cases in
the asker's data addressed", "Corrects a misconception in the question".

The judge was not malfunctioning — it was answering NA correctly. A large share
of this bank is **relational to the asker's posted code**, and a StackOverflow
question carries its code in the **body**, not the title. math.SE's criteria
mostly judge an answer's internal argument, so a title sufficed there; here it
does not. The context now carries question title + tags + a truncated question
body + the answer. NA fell to **42.6%** and the relational criteria recovered
("Engages the asker's actual code" 100% → 17% NA; "Matches the asker's data
shape" 96% → 8%; "Pitched to the asker's evident level" 96% → 0%).

The question body is **group-constant** — every answer to a question sees the
same question text — so it cannot manufacture within-question rank.

This is the *validate-before-scaling* rule paying for itself: at n=16,001 × 40
criteria the title-only run would have burned a full scoring pass and produced
an A matrix with three dead columns.

Anchor discipline: every shard carries 3 blinded anchor rows (known positive /
known negative / scrambled), and the run is followed by the extended battery at
**K=50 per class**. **Anchor labels come from `y_accepted`, not from this cell's
`y_vote`** — the accept verdict is an independent quality channel on the same
rows, so the battery certifies the judge against a signal it is not being asked
to reproduce; anchoring on y_vote would be partly circular with the quantity
under test.

Shard 0 (2,251 items) validated the instrument end to end: the blinded anchors
ordered correctly on the **first** attempt — pos **.740** > neg **.534** >
scrambled **.000** — and the full-scale NA rate came in at **.385**, close to
the smoke's .426. Scrambled text scoring exactly 0.000 across 40 criteria is the
strong form of the check: the judge is not pattern-matching surface plausibility.

---

## 5. The position line (this cell's required covariate)

`position` = answer order within the question by CreationDate.

| position | rows | vote-positive rate | accept rate |
|---|---|---|---|
| 1 | 5,030 | **62.9%** | 55.3% |
| 2 | 4,844 | 46.8% | 42.6% |
| 3 | 1,223 | 45.6% | 26.3% |
| 4 | 561 | 39.2% | 16.6% |
| 5+ | 544 | **33.5%** | 5.9% |

The first-answer advantage is large and monotone on both y's. Position alone
scores .5974 pooled and **.6552 within-question** — i.e. *stronger* than TF-IDF
on the answer text (.6160 within-question). Any claim about articulated
criteria in this cell has to be read against this number, and the ledger
reports it beside every instrument.

---

## 6. Ledger

All seven bank shards scored (16,001 items × 40 criteria), overall NA **.3709**,
K=50 anchor battery certified, 3-seed dense complete.

### 6a. Anchor battery (K = 50 per class) — CERTIFIED

| quantity | value |
|---|---|
| anchor_pos mean | **.6516** (sd .1388, n 50) |
| anchor_neg mean | **.5741** (sd .1743, n 50) |
| anchor_scram mean | **.0048** (sd .0089, n 50) |
| ordering holds on means | **True** |
| **coherent vs scrambled AUC** | **1.000** |
| pos vs neg AUC | .622 |
| anchor rows all-NA (dropped) | 0 |

Scrambled text separates perfectly (AUC 1.000) — the judge is reading content,
not surface plausibility. The pos-vs-neg AUC of .622 is modest because the
anchor labels are `y_accepted`, a deliberately *independent* channel, read off
an unweighted mean of 40 criteria; the ordering is what the gate requires and it
holds. Per-shard blinded anchors passed on the first draw for 6 of 7 shards;
shard 1 needed one redraw (attempt 0 gave pos .569 < neg .667, attempt 1 gave
pos .595 > neg .579), which is the standard retry protocol, not a failure.

### 6b. Enforced collapse gate (modal > .98)

Applied as a **drop**, not a flag, on the analysis rows before any fit:

| criterion | modal share | action |
|---|---|---|
| `a12 Complexity claim matches the code` | **.9966** | **DROPPED** |

Everything else passed. **A carries 39 columns (35 real + 4 surface)**, and the
post-gate near-constant list is empty. The dropped criterion was already the
weakest in the smoke (92% NA): scaling claims are simply rare on this corpus.

### 6c. The ledger

n = 12,202 · pos .5238 · 5,972 questions · V = 46c · A = 39c

| quantity | value |
|---|---|
| V_lin / V_nl | .6347 / **.6379** (seed spread .0023) |
| **A_lin** | **.6969** |
| VA_lin / VA_nl | **.7003** / .6989 (seed spread .0012) |
| **Δ_interact** | **−.0014**, question bootstrap [−.0055, +.0025], P(>0) = **.25** |
| **T** (3-seed dense, eval) | **.7074** — seeds .7120 / .7103 / .6998, spread **.0122** |
| T (test) | .7050 — seeds .7144 / .7075 / .6930, spread .0214 |
| Δ_total, pooled convention | +.0071 |
| Δ_beyond, pooled convention | +.0085 |

**OOF reproduction gate: PASSES EXACTLY.** Every linear leg recomputes to its
ledger value with `abs_diff = 0.00e+00` (tolerance 1e-9); the OOF vectors are
persisted **with their item ids, groups, split, position and y_accepted** in
scored row order at `results/so_votes_oof_with_ids.npz`. The two nonlinear
poolings are both recorded and never conflated: **mean of seed AUCs = .6989**
(the ledger row) versus AUC of the seed-mean OOF = .7013.

### 6d. Same-rows Δ_beyond — the headline, and it is NEGATIVE

Rather than differencing T against a VA_nl pooled over all 12,202 rows, the
grouped-OOF VA predictions are restricted to **exactly** the dense split's rows.

| leg | n / pos | VA_lin | VA_nl | T | **Δ_beyond** | within-q VA_nl (pair-w) |
|---|---|---|---|---|---|---|
| eval | 1,220 / 639 | .7458 | **.7371** | .7074 | **−.0297** | .7191 (407 mixed q) |
| test | 1,220 / 639 | .7079 | **.7192** | .7050 | **−.0143** | .7057 (415 mixed q) |

**On the same rows the articulated instruments BEAT the 8B dense model on both
legs.** The pooled convention's +.0085 and the same-rows −.014/−.030 differ in
*sign*; the same-rows figure is the honest one (V8 precedent) and is what the
gate is read against.

**This triggers the §11 standing rule** (a final ledger where the dense/fused
upper bound fails to beat the bank). Before treating "the bank beats dense" as a
finding, the build tests the most likely artifact — see §6e. The journalism
homepage cell's `bank ≥ dense` claim was already falsified once as a
population-mismatch artifact, so the prior on this being real is low.

### 6e. Item-view asymmetry — the artifact this cell had to rule out

The A judge sees **question title + tags + question body + answer**. The dense
arm's item text is **question title + answer** (the math.SE-consistent view). A
therefore has strictly more information than T, which biases Δ_beyond *downward*
— i.e. toward over-stating articulability, the dangerous direction.

A one-seed **item-view sensitivity arm** was therefore trained on the A judge's
exact context string (`--view abank`, byte-identical construction to the judge's
`ctx`). Note this is not automatically the fairer view: the dense recipe
truncates at **1024 tokens** while the A-matched item runs to ~1,944 tokens at
p99, so prepending the question body pushes the *answer* out of the dense window
for a minority of items.

**Result — the asymmetry is NOT the explanation.** Seed-matched (both seed 42):

| item view | dense sees | eval AUC | test AUC |
|---|---|---|---|
| `title` (headline) | title + answer | .7120 | .7144 |
| **`abank` (A-matched)** | title + tags + question body + answer | **.7189** | **.7137** |
| seed-matched difference | | **+.0069** | **−.0007** |

Giving the dense model the judge's exact context moves T by **+.007 eval /
−.001 test** — well inside the 3-seed spread (.0122 eval / .0214 test), i.e.
indistinguishable from noise. And Δ_beyond stays **below zero on both legs**
under the A-matched view:

| leg | VA_nl (same rows) | T (`abank`) | **Δ_beyond** |
|---|---|---|---|
| eval | .7371 | .7189 | **−.0182** |
| test | .7192 | .7137 | **−.0055** |

So the negative Δ_beyond survives the correction. The context gap between the A
judge and the dense arm was real and worth measuring, but it is **not** what
produces "bank ≥ dense" here. The headline verdict stands on either view; the
`title` view remains the headline because it is the math.SE-comparable one.

**What "bank beats dense" may and may not be read as.** T is a *lower* bound on
the ideal M\*, not the ceiling. The defensible statement is therefore: **at 8B
LoRA dense capacity there is no measurable residual beyond the articulated
criteria on this cell** — the 39 criteria plus 46 surface features are an
efficient encoding of what this dense model finds. It is NOT evidence that no
residual exists for a stronger reader. A larger-T probe is the natural follow-up.

### 6f. Pooled versus within-question

Unlike every previously built vote cell, **within-question exceeds pooled on
every row of this table** — the y definition removing the between-question
offset by construction:

| matrix | pooled | within-question (pair-weighted, 4,819 mixed questions) |
|---|---|---|
| V_lin | .6347 | .6425 |
| **A_lin** | **.6969** | **.7043** |
| VA_lin | .7003 | .7098 |
| A_real_lin (35 real) | .6954 | .7022 |
| **A_surface_lin (4 probes)** | **.5975** | **.6114** |
| V + A_real_lin | .7008 | .7080 |
| V_nl | .6404 | .6427 |
| VA_nl | .7013 | .7071 |
| A_real_nl | .6939 | .6967 |
| **question identity ALONE** | **.6043** | — |

**Do NOT quote the program-convention within-group figure (min_n = 20,
n-weighted) for this cell.** It qualifies on **2 questions / 43 rows** — SO
questions almost never carry 20+ answers — so it is uninterpretable here. The
pair-weighted figure over all 4,819 mixed questions is this cell's within-group
readout.

### 6g. What the numbers say

1. **A dominates V, decisively and linearly.** A_lin .6969 vs V_lin .6347. This
   is the reverse of the N&C co-signing cell (where V beat A) and it holds
   within questions too (.7043 vs .6425).
2. **There is no nonlinear interaction to find.** Δ_interact = **−.0014** with
   P(>0) = .25 — VA_nl is, if anything, slightly *worse* than VA_lin, and the
   seed spread (.0012) is comparable to the effect. The articulated criteria
   combine **additively**. No tacit-combination story is available here.
3. **The declared surface probes carry real but much smaller signal**
   (A_surface .5975 vs A_real .6954) and the real pole leads the univariate
   screen: "Solves the precise operation asked" .6065, "Mechanism behind the fix
   explained" .5992, "Generalises the fix beyond this instance" .5870, "Fix
   targets the named failure" .5857. The charge's real-vs-surface probe contrast
   comes back cleanly on the real side.
4. **The instruments predict the ACCEPT y about as well as the vote y**
   (VA_nl against y_accepted = .7063, against y_vote = .6989). Combined with
   φ = +.552, the bank is measuring answer quality, which drives both columns.

### 6h. GATE VERDICT — **ARTICULABLE (Δ_beyond ≤ 0 on every reading); Layer 3 NOT ELIGIBLE**

Gate applied: design note §4 — *Layer 3 only where Δ_beyond > .02 after Layer 1.*

| reading | eval | test |
|---|---|---|
| same-rows, `title` view (headline) | **−.0297** | **−.0143** |
| same-rows, `abank` view (item-view corrected) | **−.0182** | **−.0055** |
| pooled convention | +.0085 | — |

- **Every same-rows reading is below zero**, and the most favourable reading of
  any kind (+.0085, pooled) is still four times below the .02 threshold. The
  cell does **not** qualify for a Layer-3 articulation-closure campaign, because
  there is no residual to close.
- The reading is **SO answer votes are highly articulable**: 39 stated criteria
  plus 46 surface features match or exceed what an 8B dense reward model
  extracts from the same text, and they combine **additively** (Δ_interact
  −.0014, P(>0) = .25).
- This cell is **well-powered**, unlike V8: train minority **4,649** (vs 342
  total positives there), and the 3-seed T spread (.0122 eval / .0214 test) is
  *smaller* than the |Δ_beyond| it is read against — precisely the condition V8
  failed. The §14 checklist is satisfied in full (§3b).
- **The leading artifact was tested and ruled out** (§6e): the item-view gap
  between judge and dense arm moves T by +.007/−.001, inside seed noise, and
  Δ_beyond stays negative.
- **Scope limit, binding on how this is quoted:** T lower-bounds the ideal M\*,
  so the claim is *no measurable residual at 8B LoRA dense capacity*, **not**
  "no residual exists". Quote it as "articulable at this dense capacity".

**§11 STANDING-RULE TRIGGER — logged, not self-cleared.** This ledger has
`T ≤ VA_nl` on both same-rows legs, which is the auto-audit condition. This build
tested and eliminated the most likely artifact (item view) and confirmed the
population is same-rows by construction (the dense arm trains on the identical
12,202-row population and split, so the journalism homepage cell's
population-mismatch failure mode cannot apply here). It should still be routed
to the audit rather than cleared from inside the build. Note there is **no
fusion arm** in this cell (no VAT / V3), so the rule's literal
`max(VAT/V3) ≤ VA_nl` form is not yet evaluable; a V3 arm is feasible here
because the bank was scored on **all 16,001 rows including train**, unlike
`code_v3` and `aops_curation`, which are blocked for want of train-split bank
coverage.

---

## 7. Cross-y contrast — vote versus verdict on identical rows

Same 12,202 answers, same instruments, two different deciders:

| | n | accepted by the asker |
|---|---|---|
| vote-positive (y=1) | 6,391 | **69.5%** |
| vote-negative (y=0) | 5,811 | **14.7%** |

φ = **+.552**.

**This is the opposite sign to the N&C cell**, and it is the cell's first
finding. In regulatory notice-and-comment, the verdict and vote columns point in
*opposite* directions on identical text (co-signed comments draw an agency
response 43.9% vs 79.4%, φ = −.160). On StackOverflow they are strongly
*aligned*: the asker and the crowd mostly want the same thing. Two fields, two
vote columns, built with the same machinery, and the verdict↔vote relationship
inverts between them — which is exactly the kind of contrast the 3×N grid exists
to produce, and it means "does the crowd agree with the decision-maker?" is a
field-level property rather than a constant.

The alignment is strong but far from identity (φ = .552 leaves ~70% of the
variance unshared), so the two columns remain distinct targets rather than one
relabelled twice.

**The instruments do not distinguish the two columns either.** The same VA_nl
stack, fitted on the vote y, scores **.7063 against `y_accepted`** versus
**.6989 against `y_vote`** — it predicts the verdict column slightly *better*
than the column it was trained on. Read together with φ = +.552, the bank is
measuring "is this a good answer", and on StackOverflow both the asker and the
crowd are largely answering that same question. That is the substantive contrast
with N&C, where the identical instruments succeeded within-docket on the verdict
column and sat at chance on the vote column because the two crowds wanted
*different things*.

---

## 8. Artifacts

| what | where |
|---|---|
| y builder + full rationale in docstring | `datasets/stackoverflow-votes/build_so_votes_population.py` |
| population (one row per answer, both y's, split, covariates) | `datasets/stackoverflow-votes/va/population.csv.gz` |
| build funnel + split/cross-y/position audit | `datasets/stackoverflow-votes/va/population_manifest.json` |
| §14 pre-kill baselines (TF-IDF, position, length, identity) | `datasets/stackoverflow-votes/va/prekill_baselines.json` |
| V extractor (46 features) | `datasets/stackoverflow-votes/va/v_features.py` |
| A bank (40 criteria) + every merge/drop decision | `datasets/stackoverflow-votes/va/build_rubrics.py` → `va/rubrics.jsonl` |
| Gemma scoring driver | `datasets/stackoverflow-votes/score_so_votes_bank.py` |
| scored A/V matrix + anchor battery | `outputs/va_gemma_banks_so_votes/` (sk3) |
| dense-standard bundle (headline `title` view) | `datasets/stackoverflow-votes/va/dense_standard_so_votes/{data.csv,split/}` |
| dense bundle (item-view sensitivity, `abank` view) | `datasets/stackoverflow-votes/va/dense_standard_so_votes_abankview/` |
| Layer-1 + Layer-2(a) driver | `methods/taste_decomposition/so_votes_layer1.py` |
| **ledger** | `methods/taste_decomposition/results/so_votes_ledger.json` |
| **OOF with ids** (repro-gated, diff 0.00e+00) | `methods/taste_decomposition/results/so_votes_oof_with_ids.npz` |
| anchor battery (K=50) | `outputs/va_gemma_banks_so_votes/anchor_battery.json` (sk3) |
| raw corpus (reused, not rebuilt) | `sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/` |

Discipline check: no data deleted (tie rows retained with null y; the legacy
`so_python_v2` pool untouched and retained); `latex/` untouched; anchors on
every judging batch (3 blinded per shard + K=50 battery); stable-hash grouped
splits with no seeded shuffle; pos-rate matched to five decimals.

**GPU ledger note.** The charge assigned GPU 3. Between the ledger check (GPU 3
free) and the launch, another agent's `flip_ladder.py mistral7b` claimed GPU 3
with 163 GiB, and this cell's first dense launch OOM'd against it. That process
was **not** signalled — it is not ours. The dense arm was restacked onto GPU 2
and the Gemma judge onto GPU 0 (util 0.75 to fit alongside a 34 GiB tenant),
one process per GPU, per the stacking rule. The later item-view sensitivity arm
took GPU 5 (verified 0 MiB / 0% before claiming). All long jobs were launched
`setsid --fork` and verified at PPID 1.

## 9. What this changes for the 3×N grid

The software-code row now has two of three columns built — PR merge (verdict)
and SO votes (vote) — and the pair is an unusually clean contrast because both
are code and both were built with the same machinery:

| | PR merge (VERDICT) | **SO votes (VOTE)** |
|---|---|---|
| group | repo | question |
| group identity alone | large repo leak | **.6043** (small) |
| within-group vs pooled | within-repo *below* pooled | **within-question ABOVE pooled** |
| Δ_beyond | within-repo **+.058 eval / +.039 test** | **−.0297 eval / −.0143 test** |
| Δ_interact | positive | **−.0014 (none)** |
| verdict↔vote alignment | — | φ **+.552** |

The two code columns land on **opposite sides of the articulability gate**: PR
merge clears it (+.058/+.039, Layer-3 round 0 already run) while SO votes sits
below zero. Same field, same instrument family, and the decision-maker column
has a residual the crowd column does not.

Cross-field, the vote column now reads:

| field | vote y | Δ_beyond | verdict↔vote |
|---|---|---|---|
| Peer review | citation pct | **+.1125** (not articulable) | — |
| Creative writing | upvotes | +.103 plateau | — |
| Math | math.SE vote | +.037 (clears gate) | — |
| **Software code** | **SO votes** | **−.014 / −.030 (articulable)** | **φ +.552** |
| Regulatory | co-signing | +.064 but underpowered, ~92% docket composition | φ −.160 |

**Vote columns are not a natural kind.** The crowd's preference is nearly fully
articulable on StackOverflow, substantially unarticulable on peer review, and on
N&C is mostly a property of the venue rather than the text. Anything the program
says about "revealed/crowd preference" as a category has to survive that spread.

Its nearest sibling is the strongest comparison available: math.SE and SO share
a platform, a y rule, and an instrument design, and they still differ
(+.037 vs −.014/−.030). Verifiability is the obvious candidate mechanism — a
Python answer can be run, a real-analysis answer cannot — and this cell already
ships the `stratum` column (stdlib / lib_data / framework) to test it, though
that is a follow-up, not part of this build.
