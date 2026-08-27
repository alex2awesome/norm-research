# Scale-up wave C — sequential instrument builds (task D7)

Date opened: 2026-08-07. Charge: four instrument builds in priority order, each
delivering a full V / A / VA_lin / VA_nl / T ledger to
`methods/taste_decomposition/results/<cell>_ledger.json`.

Terminology, spelled out per the standing rule: **V** = deterministic surface
features (regex/count level, label-blind); **A** = the articulated-criteria bank
scored one criterion at a time by **Gemma-4-31B** (google/gemma-4-31b-it, local
snapshot, offline-batch vLLM, one token from {1.0, 0.5, 0.0, NA}); **VA_lin** =
grouped out-of-fold logistic aggregation of the V+A matrix; **VA_nl** =
HistGradientBoosting aggregation of the SAME matrix on the SAME grouped folds,
seed-mean over {0,1,2}; **T** = the dense standard (Llama-3.1-8B LoRA r16/a32,
lr 5e-5, batch 16, max_len 1024, 2 epochs, gradient checkpointing,
select-on-eval) clean-eval AUC; **Δ_total** = T − VA_lin; **Δ_interact** =
VA_nl − VA_lin; **Δ_beyond** = T − VA_nl; **GEPA** = the prompt-iteration
phrasing pass an A bank must pass before it is quotable; **AUC** = area under
the ROC curve; **LDA** = latent Dirichlet allocation (the topic model whose
50 topics are this cell's grouping unit).

Standards followed: A-bank standard (GEPA-iterated phrasing + Gemma-4-31B judge,
`datasets/va_gemma_banks/score_va_gemma_banks.py` machinery, blinded anchors in
EVERY judging batch + extended battery at K ≥ 50 per class); dense standard
(`methods/dense/run_dense_standard_v4.sh`); stable-hash grouped splits (never a
seeded shuffle); validate-before-scaling (smoke inspection before any full run);
never delete data; one GPU at a time on sk3.

---

## BUILD ORDER + STATE

| # | build | state |
|---|---|---|
| 1 | reddit-jokes community: mature Gemma A bank + dense standard | **COMPLETE (seed-42 T; seeds 1,2 queued)** |
| 2 | math.SE V2 rebuild (un-binarized votes) + Gemma bank + dense | **COMPLETE (both y's)** |
| 3 | AoPS curation Gemma bank | **COMPLETE** (T reused, no dense trained) |
| 4 | homepage curation V+A stack + grouped dense arm | **COMPLETE — but T is NOT QUOTABLE (see below)** |

---

## BUILD 1 — reddit-jokes community

Cell: humor × community/crowd. Registry (`notes/2026-07-27__vat-run-registry.md`)
carries only the May-2026 floor-harness numbers **V .574 / VA .564†** († = floor
harness, verified ~.10–.15 below mature banks on bridge tasks) and a provisional
ungrouped **T .824p**. This build replaces both with mature-bank + grouped-dense
numbers.

### Population (FROZEN 2026-08-07)

Source `datasets/humor/reddit_humor_with_topics.csv.gz` (383,786 rows; columns
text / judgement / topic), verified row-for-row identical to the canonical
`reddit_humor_modeling_dedup.csv.gz` plus the LDA topic column. Same file exists
on sk3 as `reddit_humor_modeling_with_topics.csv.gz`.

- `row_id = sha1(text)[:20]` — verified unique over all 383,786 rows.
- Sample = first **16,000** rows under `sha256("jokes-va-v1|" + row_id)`
  (stable hash, no seeded shuffle). Builder:
  `datasets/humor/reddit_jokes/build_population.py`; frozen population at
  `datasets/humor/reddit_jokes/va/population.csv.gz`.
- **n = 16,000, pos-rate .496, 50 topic groups**; text median 77 chars, p95 604,
  max 3,508.
- Grouping unit = LDA topic. Chosen because r/Jokes posts have no natural
  container; near-duplicate reposts were already removed upstream by MinHash LSH
  (Jaccard ≥ .8). Pos-rate is ≈.50 inside every topic by construction of the
  labeller (range .422–.566 over the 50 topics), so group identity carries almost
  no label information — the grouped split is a lexical-domain control, not a
  leakage fix. A uniform stable-hash prefix (rather than whole-group draws) is
  used so that all 50 topics stay represented and the grouped folds exist.

### Dense standard (same population — apples-to-apples)

`datasets/humor/reddit_jokes/dense_standard/{data.csv,split/,manifest.json}`,
topic-grouped 80/10/10 via the frozen `stable_hash_bucket_map` bin-packer ported
verbatim from `datasets/humor/hashtagwars/build_dense_standard.py` (row-count AND
pos-rate balanced):

| split | topics | rows | frac | pos-rate |
|---|---:|---:|---:|---:|
| train | 40 | 12,837 | .8023 | .4977 |
| eval | 5 | 1,663 | .1039 | .4865 |
| test | 5 | 1,500 | .0938 | .4920 |

Because the dense arm trains on exactly the A/V-scored population, the eval rows
are a subset of the scored rows by construction — FREEZE CHANGE 2 (same-rows T)
is satisfied without a separate rescore job.

### V channel

`datasets/humor/reddit_jokes/va/v_features.py` — 27 deterministic label-blind
surface features, adapted from `datasets/humor/hashtagwars/va/v_features.py`
(prompt-dependent features dropped: an r/Jokes post has no contest prompt;
joke-form surface proxies added: sentence count, final-beat character share,
dialogue-verb count, quote marks, riddle/narrative opening flags, Flesch).

### A bank

Authored by the proposer + self-audit + GEPA phrasing loop seeded from the
364-name r/StandUpWorkshop + humor-craft rubric hierarchy
(`datasets/humor/standup_reddit/rubrics.jsonl`) and the parsed joke-craft corpus
(`datasets/humor/online-rubrics/claude-parsed/`), against 90 LABEL-BLIND
train-split exemplars. Target 45–48 criteria at
`datasets/humor/reddit_jokes/va/rubrics.jsonl`.

Scorer: `datasets/va_gemma_banks/score_scaleupC_banks.py` (imports the frozen
scoring loop from `score_va_gemma_banks.py`; only the bank builder is new).

### A bank — LANDED (47 criteria)

`datasets/humor/reddit_jokes/va/rubrics.jsonl`, ids a01–a47, every description
carrying the four inline anchors, no banned token (vote/popular/viral/repost/
length/formatting/emoji) anywhere in the file. Proposer dropped multi-beat
stand-up devices (escalation, callback, rule-of-three, tags, meter) as NA for
>70% of this corpus, and merged ~14 near-duplicate pairs.

### Smoke score (validate-before-scaling) — PASSED

40 items × 47 criteria on GPU 1. Overall NA .270, mean .844, **no criterion
collapsed to a single value**: the discriminating tail runs from
`a28 Central move is not a stock template` (mean .289) and
`a22 Frame is not contorted to reach the wordplay` (.315) up to near-universal
form checks like `a30 One governing comic idea` (.963). Five criteria sit above
70% NA (a21 .78, a23 .78, a46 .75, a06 .72, a45 .72) — honest conditional
branches, carried with median imputation + missingness indicators.

### STATE

- [x] population frozen, dense CSVs built
- [x] V features written
- [x] A bank authored (47 criteria)
- [x] blind incidental-vs-quality audit of the bank — **47/47 quality, 0
  incidental**; 1 flagged untestable (`a29 Observation has recognizable truth`
  — grounded in the judge's world beliefs, not in the text) and 2 redundant
  pairs (a09↔a28 stock-template, a10↔a20 both-readings-hold). Flags are
  RECORDED, not acted on: dropping criteria after seeing an audit would be
  selection on the instrument. Verdicts in the session scratchpad
  `jokes_bank_audit.json`.
- [x] smoke score (validate-before-scaling) — passed
- [ ] full Gemma scoring + K=50 anchor battery (RUNNING, GPU 1)
- [ ] dense standard 3 seeds
- [ ] Layer-1 ledger

---

## DISCOVERY (2026-08-07) — inputs located for builds 2–4

Recorded here so later builds do not repeat the search. `sk3:` = remote box
`/lfs/skampere3/0/alexspan/norm-research/...`; unprefixed = repo-local.

### Build 2 — math.SE
- The deleted `datasets/math-stackexchange/build_*.py` files are **not lost**: an
  identical copy of that directory lives at `datasets/math/stackexchange/`.
  `build_binary_dataset.py` is the binarizer: positive = accepted AND score ≥ 3,
  negative = score ≤ 0; **scores 1–2 and high-but-unaccepted answers are dropped**
  (the "signal gap"). That gap is exactly what the V2 rebuild has to undo.
- Best un-binarized table already on disk:
  `sk3:datasets/math-stackexchange/math_se_v3_1_pool.csv.gz` — 334,878 rows,
  326,498 question ids, columns `text, judgement, split, question_id, answer_id,
  answer_position, n_answers_on_question, answer_age_gap_days, answer_year,
  score, accepted, primary_tag, question_tags`. `score` has 196 distinct values
  (incl. negatives), `accepted` True 219,907 / False 114,971. **Caveat: this pool
  inherits the score-gap filter, so scores 1 and 2 are absent.** A gap-free census
  requires re-parsing `sk3:datasets/math-stackexchange/raw_dump/Posts.xml` (5.8 GB;
  `Score`, `ParentId`, question-row `AcceptedAnswerId`).
- Published binarized cell (V .565 / VA .673 / T .794):
  `sk3:datasets/math-stackexchange/math_se_v3_3_propensity_balanced.csv.gz`
  (99,722 rows), dense run `sk3:runs/math_se_v3_3_dense_llama8b/`.
- Old **Qwen**-scored A matrix (never to be mixed with a Gemma rescore):
  `datasets/math/stackexchange/a_metric_verdicts_all.jsonl` (22,521 rows, a01–a14
  + answer_type). Criteria spec with the full anchored rubric:
  `datasets/math/stackexchange/scripts/a_judge_spec.json` — a01 motivation-before-
  machinery, a02 audience calibration, a03 right generality, a04 no unjustified
  gaps, a05 words–symbols balance, a06 proof idea visible, a07 elegance, a08
  precision/rigor hygiene, a09 pedagogical scaffolding, a10 epistemic honesty,
  a11 directness, a12 reusable technique, a13 notation quality, a14 profundity.
- V features: `datasets/math/stackexchange/mathse_lint.py` → `mathse_lint_features.csv`.
- No `dense_standard/` exists for math.SE.

### Build 3 — AoPS
- Canonical labelled population: `sk3:runs/aops_same_approach_dense_llama8b/data_full.csv.gz`
  — 28,415 rows, columns `text, judgement, problem`; `text` = problem statement +
  forum body only (never the editorial solution); y = same-approach judge verdict;
  **group column = `problem`**. Provenance JSON in the same dir.
- Splits already exist (the de-facto dense standard): `split_full/{train,eval,test}.csv`;
  predictions `preds_eval.csv` / `preds_test.csv`; dense .769 grouped.
- V stack already built: `datasets/math/aops/v_features_same_approach.parquet` —
  25,454 × 20, y mean .690, 3,007 problem groups (cols `log_len, n_display_math,
  latex_density, is_correct_f, has_answer_f, v_boxed, v_hide_block, v_answer_stmt,
  v_deductive, v_meta_doubt, v_first_person, v_heavy_machinery, v_standard_tech,
  v_proof_framing, v_numeral_density, v_question_marks, p_ed`). Builder
  `datasets/math/aops/scripts/v_features_from_tfidf.py`.
- **No A bank exists** — this build authors the first one.

### Build 4 — homepage curation
- Clean population `datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz`
  (133,130 rows, exact 50/50, 13,666 snapshots, columns `text, judgement,
  snapshot_id`); the T .824-provisional population is
  `homepage_newsworthiness_topic_balanced_groupsplit.csv.gz` (183,708 rows) with
  split dir already on sk3. `text` = `HEADLINE: … \n\nCONTEXT: …`.
- y = homepage spatial placement (top-half vs bottom-half of the top-30% zone) —
  an editorial-prominence decision, not clicks.
- **Outlet is stripped from every shipped CSV** (it leaked: BBC 34% top vs
  WashPost 73% top). Outlet-held-out needs re-derivation, either by re-running
  `datasets/news-homepages/build_homepage_dataset.py` (8 outlets, emits `outlet`)
  or from `datasets/news-homepages/analysis/format_audit_rows.jsonl`
  (`text, judgement, snapshot_id, outlet`).
- Census A criteria (the .593 bank) are defined inline as `RUBRICS` in
  `datasets/news-homepages/analysis/run_A_newsvalues.py` (14 news-values items);
  the .531 LLM-mined bank is `NEW_METRICS` in `analysis/news_newmetrics.py`.
  Existing A npz matrices (sk3 only) cover a **4,400-row** snapshot-grouped
  balanced sample, `levels/applicable/y/g/names`, judged by a 70B model — not
  Gemma, so they are a separate instrument from the A-bank standard.
- No `dense_standard/` dir; existing dense sweep
  `sk3:runs/homepage_newsworthiness_groupsplit_sweep_llama8b/`.


---

## BUILD 2 — math.SE V2 rebuild (prepared while build 1 occupies the GPU)

### Raw rebuild — DONE

`datasets/math/stackexchange/build_multiy_v2.py` streams the 5.8 GB `Posts.xml`
dump (3,792,435 post rows → 1,641,406 questions / 2,146,413 answers; 508,076
questions carry ≥2 answers) and emits two **un-binarized, question-grouped**
populations with NO score censoring, into
`sk3:datasets/math-stackexchange/v2_multiy/`:

| population | n | pos-rate | questions |
|---|---:|---:|---:|
| `mathse_v2_accepted_verdict.csv.gz` | 857,709 | .3842 | 329,558 |
| `mathse_v2_vote_score.csv.gz` | 859,421 | .5167 | 407,240 |

- **accepted-verdict y**: 1 = the answer the ASKER accepted, 0 = another answer
  on the same question. No vote information enters y.
- **vote-score y**: 1 = raw vote score strictly ABOVE the median score of the
  answers on its own question, 0 = strictly below; ties at the median dropped
  (the caption crowd-C construction). No accept information enters y.

Both are within-question contrasts, so question difficulty, age, topic and
traffic are differenced out by construction. This undoes the old
`build_binary_dataset.py` signal gap (it fused accept AND score ≥ 3, and threw
away every answer scoring 1 or 2).

### A/V population — DONE

`datasets/math/stackexchange/build_v2_va_population.py` takes the
question-level INTERSECTION (questions carrying both signals), so ONE Gemma
scoring pass serves both y's (the Style Invitational two-y precedent; the y's
are always reported separately, never merged). Whole questions drawn in
`sha256("mathse-v2-va|" + question_id)` order.
Output `sk3:datasets/math-stackexchange/v2_va/`:

- population **n = 13,001 answers / 4,960 questions**; text median 587 chars,
  p95 2,388, max 11,597.
- `y_accepted` pos-rate **.3815** (all 13,001 rows).
- `y_vote` pos-rate **.5015** on the **11,629** rows where it is defined.
- two dense-standard arms, question-grouped 80/10/10, pos-rate matched to 4
  decimals: accepted 10,401/1,300/1,300 (3,968/496/496 questions);
  vote 9,303/1,163/1,163 (3,820/571/569 questions).

A bank: fresh Gemma re-derivation seeded by the OLD Qwen 1–5 axes a01–a14 in
`datasets/math/stackexchange/scripts/a_judge_spec.json` — **never mixed with the
old matrix**; proposer running, target 28–34 criteria at
`datasets/math/stackexchange/va/rubrics.jsonl`.

---

## BUILD 4 — homepage curation: the outlet problem (resolved on paper)

`outlet` was stripped at the very first save — `homepage_newsworthiness.csv.gz`
and every descendant are `text, judgement[, snapshot_id|topic]` only, and
`analysis/format_audit_rows.jsonl` holds just 60 rows. But `snapshot_id` is
**deterministically recomputable from the text**:
`sk3:scripts/add_snapshot_id_homepages.py` defines it as
`md5("|".join(sorted({headline[:50]} ∪ {ctx[:50] …})))[:16]`, computed on the
PRE-cleaning `homepage_newsworthiness_topic_balanced.csv.gz` and then carried
through the v3→v9 cleaning chain as a column.

Therefore outlet is recoverable without touching the frozen population:
re-run the *processing* stage of `datasets/news-homepages/build_homepage_dataset.py`
over `sk3:datasets/news-homepages/raw_data/{bbc,cnn,guardian,latimes,nytimes,
reuters,washingtonpost,wsj}/`, which emits `{text, judgement, outlet}`, recompute
`snapshot_id` from that raw text with the same function, and join the resulting
`snapshot_id → outlet` map onto `homepage_newsworthiness_clean_v9.csv.gz`. One
snapshot is one outlet's capture, so the map is 1:1.

### Outlet recovery — DONE (2026-08-07)

`datasets/news-homepages/analysis/recover_outlet_map.py` implements the plan
above but WITHOUT re-running the labelling pipeline (which would risk perturbing
the frozen population — its context field is built with an unseeded shuffle, and
its article/furniture classifier is retrained on each run). Instead it reads the
raw `*.hyperlinks.json` captures directly, builds a per-outlet set of normalised
link texts, and assigns each `snapshot_id` the outlet covering the most of that
snapshot's headlines (a snapshot is one outlet's capture, so this is a plurality
vote over its own rows; shared wire headlines cannot flip it).

Result on `homepage_newsworthiness_clean_v9.csv.gz` (133,130 rows / 13,666
snapshots) → `sk3:datasets/news-homepages/outlet_map_v9.csv.gz`:

| outlet | snapshots | rows |
|---|---:|---:|
| cnn | 3,350 | 48,770 |
| washingtonpost | 2,539 | 22,144 |
| latimes | 1,280 | 21,732 |
| nytimes | 842 | 13,355 |
| bbc | 1,069 | 9,608 |
| reuters | 606 | 8,585 |
| guardian | 1,019 | 4,968 |
| wsj | 61 | 1,059 |
| *ambiguous* | 2,900 | 2,909 |

**Median coverage share among resolved snapshots = 1.000** (every headline of the
snapshot found in that outlet's own capture set), and only **2.19% of rows** are
ambiguous — those are dropped from the outlet-held-out arm. WSJ resolves on only
61 snapshots (paywalled/JS-rendered captures), so it is a thin held-out fold and
must be flagged as such.

### Homepage A/V population — DONE (2026-08-07)

`datasets/news-homepages/build_va_population.py` →
`sk3:datasets/news-homepages/va/{population.csv.gz,dense_standard/}`.
Per outlet, whole snapshots in `sha256("homepage-va-v1|" + snapshot_id)` order
until a 1,700-row cap; capping per outlet (not a uniform prefix) is what makes an
outlet-held-out fold meaningful, since the natural mix is 37% CNN.

**n = 12,998 rows / 8 outlets / 1,229 snapshots, pos-rate .5006.**
Per-outlet rows ≈1,700 each (WSJ only 1,059 — its captures are largely
paywalled/JS-rendered — a thin fold, flagged).

**Correction to a carried-forward claim.** The README's outlet leak (BBC ~34%
top vs Washington Post ~73% top) describes the RAW dataset. In the balanced
`clean_v9` population the per-outlet positive rates are .472–.524 (BBC .524,
NYT .475, WSJ .472, WaPo .491), i.e. **outlet identity carries almost no label
information here**. The weak-instrument flag still stands — but on the y itself
(spatial placement is co-determined by layout, ad slots, image availability and
template rules), not on an outlet base-rate leak.

Dense arm is genuinely outlet-held-out: train {bbc, cnn, guardian, reuters,
washingtonpost, wsj} 9,580 rows / eval {nytimes} 1,712 / test {latimes} 1,706.
Because there are only 8 groups the bin-packer lands at .737/.132/.131 rather
than 80/10/10 — recorded, not tuned. Checkpoint selection therefore happens on a
single held-out outlet, which is what outlet-held-out means and is flagged.

### Homepage A bank — the 14 census criteria, GEPA-rephrased

`datasets/news-homepages/va/rubrics.jsonl`. Same 14 news-values concepts as the
`RUBRICS` list in `analysis/run_A_newsvalues.py` (the .593 census bank), rewritten
into the house form: a testable property plus inline 1.0 / 0.5 / 0.0 / NA anchors,
each `gepa_revision` recording what the pass changed (most often: replacing flat
keyword membership with a role/temporal/scale test, and adding an honest NA
branch). Concepts are unchanged so the cell stays comparable to the published
census number; the JUDGE changes from a 70B model on 4,400 rows to Gemma-4-31B on
12,998 rows, so the two are separate instruments and must not be differenced.

`datasets/news-homepages/va/v_features.py` — 23 deterministic headline-only
surface features. The CONTEXT half of the item text is deliberately excluded from
V: it is shared by every row of a snapshot, so any feature computed on it would be
a snapshot-identity feature rather than a property of the item.

---

## BUILD 3 — AoPS curation (prepared)

- A bank: `datasets/math/aops/va/rubrics.jsonl`, **44 criteria** a01–a44, all four
  anchors inline, grep-clean for editorial/wiki/matching/thanks and for every
  V-channel surface quantity. The proposer dropped identity-style method markers
  (synthetic vs coordinate vs bash, elementary vs heavy machinery) because they
  duplicate the `v_heavy_machinery` / `v_standard_tech` lexicons and would turn a
  subject-typical route into a quality signal.
- V: `datasets/math/aops/va/v_features.py` — 24 deterministic features; the nine
  lexicon regexes are ported VERBATIM from
  `datasets/math/aops/scripts/v_features_from_tfidf.py`. Deliberately excluded:
  `is_correct_f` and `has_answer_f` (LLM-judge fields) and `p_ed` (a LEARNED
  out-of-fold editorial-register score). Those three sit inside the published
  V .706, so **this build's V and the published .706 are not the same instrument**
  and must not be differenced.
- Population: `datasets/math/aops/build_va_population.py` →
  `sk3:datasets/math/aops/va/{population.csv.gz,dense_standard/}`. Whole problems
  in `sha256("aops-va-v1|" + problem)` order; **n = 13,071 / 1,573 problems,
  pos-rate .6846**; body median 677 chars, p95 2,699. 2,249 exact-duplicate bodies
  inside a problem were deduplicated (first kept). Dense split problem-grouped
  10,457/1,307/1,307 with pos-rates matched to 4 decimals.

---

## BUILD 2 — math.SE A bank

`datasets/math/stackexchange/va/rubrics.jsonl`, **32 criteria** a01–a32, fresh
Gemma re-derivation of the old Qwen 1–5 axes. Splits recorded by the proposer:
old a01 → plan-announcement + per-move motivation; a04 → dismissal-on-the-crux +
recoverability-of-omissions; a08 → symbol hygiene + quantifier/domain scoping;
a10 → overclaiming + localisation-of-doubt; a11 → right-target + right-deliverable
+ conclusion-asserted. Replaced (virtue words with no observable test): a07
elegance → "no step that fails to feed the conclusion" + "one organizing
mechanism"; a14 profundity → "names the obstacle the key move exists to defeat".
Eleven axes are new (verified-vs-asserted, hypothesis applicability, edge cases,
defective question handling, engagement with the asker's attempt, hint
actionability, citation usability, compound-claim coverage, correctness of the
conclusion, correctness of intermediates, viability of a proposed route).
Instrument note from the proposer: the judge sees the question **title** only
(the question body is not in the item text), so the affected criteria were
phrased to be decidable from title + answer alone.

V: `datasets/math/stackexchange/va/v_features.py` — 28 deterministic features on
the answer body only (question title stripped, so question length cannot leak
into an answer-style feature).

---

## GPU ORCHESTRATION (sk3, GPU 1 only — one GPU, processes stacked)

GPU ledger at launch: GPUs 0,2,3,4,6,7 busy with other people's work; **1 and 5
free; this wave uses GPU 1 and nothing else.**

Scripts installed on sk3:
- `scripts/run_scaleupC_gemma.sh` — env wrapper (HOME pin, HF offline, spawn,
  `VA_OUT_C=outputs/va_gemma_banks_scaleupC`) around
  `datasets/va_gemma_banks/score_scaleupC_banks.py`.
- `scripts/run_scaleupC_dense_chain.sh` — chains the five dense arms
  (jokes → mathse accepted → mathse vote → aops → homepage), 3 seeds each,
  through `methods/dense/run_dense_standard_scaleupC.sh` (frozen recipe,
  RUN_DONE-sentinel resumable).
- `scripts/scaleupC_supervise.sh` — waits for the running jokes Gemma job to
  exit, runs the three remaining banks' **smoke gate** (40 items each), then
  starts the dense chain stacked on the same GPU.

Stage plan:
1. jokes Gemma (RUNNING) → 2. smoke gate for mathse/aops/homepage → 3. dense
chain (background, stacked) ∥ chained Gemma scoring for mathse → aops →
homepage at `--util 0.55` (the utilisation the original va_gemma_banks run used
precisely so a LoRA training job could share the device) → 4. Layer-1 ledgers.

Ledgers are produced by `methods/taste_decomposition/scaleupC_layer1.py`
(`--cell jokes_community | mathse_accepted_verdict | mathse_vote_score |
aops_curation | homepage_curation`), writing
`methods/taste_decomposition/results/<cell>_ledger.json`. All five are FIRST-FIT
cells: no prior V+A stack of the same construction exists, so the linear leg IS
the first fit and there is no reproduction gate (the press-verdict precedent).
Δ_interact confidence intervals are GROUP-level bootstraps (FREEZE CHANGE 3),
and T is same-rows by construction in every cell because each dense arm trains
on the identical frozen population.

### Population artifact noted, not fixed (homepage)

The v9 cleaning chain strips "N min read" but not other mashed trailing
furniture: focal headlines in the frozen population still occasionally carry a
dek and a timestamp run together, e.g.
`"Bachelorette's new season pulled after ... allegationsThe decision comes after
footage emerged ... in an altercation with her ex-partner.9 hrs ago"`. This is a
property of the canonical population, so it is **flagged and left alone** — it
inflates `v_char_count` and gives the judge extra text on some rows. Any later
attempt to clean it would fork the population away from the published
`clean_v9` and from the T .824-provisional dense sweep.

---

## INVENTORY-AND-REUSE PASS (coordinator directive 2026-08-07)

One line per build: **what existed / what was reused / what was built and why.**

### 1. reddit-jokes community
- **Existed:** the May-2026 validity harness at `sk3:runs/validity_full/v2/humor/`
  (366-aspect humor pool `aspects.json`, Llama-bf16 judge prompts/responses,
  `judge_manifest*.json`) — the source of the floor-harness V .574 / VA .564†;
  the same pool is mirrored in-repo as `datasets/humor/standup_reddit/rubrics.jsonl`.
- **Reused:** that pool WAS the seed pool for the new bank — verified by name
  comparison: **364 of the 366 aspects are byte-identical to the seed list the
  proposer was given**, and the only two absent are TV-runtime-compliance items
  ("Half-hour TV episode runtime compliance (≈22–24 minutes)", "Content unit
  duration and release cadence") that cannot apply to a written one-liner. So no
  criterion mining was duplicated. The canonical labelled corpus, its MinHash
  dedup and its LDA topics were all reused unchanged.
- **Built, and why:** a 47-criterion Gemma-scored bank + a 27-feature V module +
  a 16,000-row frozen population + a topic-grouped dense arm. The May-2026
  artifacts cannot serve as the mature stack: their judge is Llama-bf16 rather
  than the A-bank standard's Gemma-4-31B, their aspects are performance/production
  criteria for stand-up rather than tests applicable to written jokes, and the
  dagger flag records that this harness sits ~.10–.15 below mature banks on
  bridge tasks. The registry also still lists T as ungrouped/provisional (.824p).

### 2. math.SE V2
- **Existed:** the binarized v3.3 cell (V .565 / VA .673 / grouped clean-eval
  **T .794**) with row-level predictions in `sk3:runs/math_se_v3_3_dense_llama8b/
  preds_{eval,test}.csv` (19,849 held-out rows, eval AUC .7944 / test .7939); the
  Qwen-scored a01–a14 matrix; the lint V battery; the raw dump plumbing.
- **Reused:** the raw-data plumbing (`raw_dump/Posts.xml`) and the OLD A axes as
  **seed concepts** for the fresh Gemma bank (32 criteria, split/replaced/extended
  — see the build-2 section). The old Qwen matrix itself is NOT mixed in, per the
  charge.
- **OVERLAP CHECK (the coordinator's condition for reusing .794), by `answer_id`:**

  | comparison | overlap | share of the new 13,001-row population |
  |---|---:|---:|
  | new population ∩ old dense HELD-OUT rows (19,849) | **103** | **0.79%** |
  | new population ∩ whole old v3.3 population (99,722) | **605** | **4.65%** |

  So **.794 cannot be reused as T** — the populations are essentially disjoint,
  which is exactly what un-binarizing does (v3.3 required accepted ∧ score ≥ 3 vs
  score ≤ 0 and deleted every answer scoring 1–2; the new y's are within-question
  contrasts over all answers). Both math.SE dense arms are therefore retrained.
  Descriptive by-product on the 103 shared rows (small n, never a headline): the
  OLD dense model's probabilities score **.782** against the new accepted-verdict y
  and **.748** against the new vote-score y.
- **Built, and why:** the un-binarized rebuild itself (both y's), the 32-criterion
  Gemma bank, a 28-feature V module on the answer body, and two dense arms — all
  because the population and the judge both genuinely change.

### 3. AoPS curation
- **Existed:** `v_features_same_approach.parquet` (the V .706 lexicon stack), the
  `v_features_from_tfidf.py` extractor, and a grouped Llama-3.1-8B dense arm WITH
  row-level predictions (`runs/aops_same_approach_dense_llama8b/preds_{eval,test}.csv`).
- **Reused:** (a) the **nine lexicon regexes are ported verbatim** into
  `datasets/math/aops/va/v_features.py`, along with `log_len`, `n_display_math`
  and `latex_density`; (b) the **dense arm is reused, not retrained** — its
  predictions verified row-aligned with `split_full/{eval,test}.csv` (judgement
  and problem match elementwise), giving eval .7739 / test .7879 / **pooled T
  .7806** on this build's rows; (c) the problem-grouped split itself.
- **Built, and why:** ONLY the A bank (44 criteria) — the cell has never had one.
  The A/V population was re-scoped to the union of the dense arm's eval+test rows
  (**5,202 rows / 606 problems after dropping 488 duplicate bodies**, pos-rate
  .6734) precisely so T is same-rows by construction at **zero GPU cost**; the
  AoPS dense arm has been REMOVED from the dense chain.
- **Caveat recorded:** the published V .706 also used `p_ed` (a learned
  out-of-fold editorial-register score) and two LLM-judge fields
  (`is_correct`, `has_answer`). Those are not a deterministic surface channel, so
  they are excluded here — this build's V and the published .706 are **not the
  same instrument** and must not be differenced.

### 4. homepage curation
- **Existed:** the census 14 news-values criteria (`analysis/run_A_newsvalues.py`
  `RUBRICS`, the .593 bank) and the LLM-mined 14 (`news_newmetrics.py`, .531);
  seven 70B-judged A score matrices over a 4,400-row snapshot-grouped sample; the
  clean_v9 population; a snapshot-grouped dense sweep (`runs/homepage_newsworthiness_
  groupsplit_sweep_llama8b/`) behind the T .824-provisional number.
- **Reused:** the **census criteria concepts verbatim** (all 14, only rephrased
  into the anchored house form, each `gepa_revision` recording the change), the
  frozen clean_v9 population, and its `snapshot_id` definition.
- **Built, and why:** (a) the **outlet map**, because no shipped CSV carries
  `outlet` and the cell is specified outlet-held-out — recovered from the raw IA
  captures without re-running the labelling pipeline; (b) a 23-feature V module;
  (c) an outlet-held-out dense arm, because the existing sweep left **no
  row-level predictions** (only `training_run.log` + `validation_metrics.csv`), so
  there is nothing to reuse for a same-rows T, and because its split was
  snapshot-grouped rather than outlet-held-out. The prior T .824 is carried as
  provisional context only.

---

## BUILD 1 RESULT — reddit-jokes community A/V ledger (T pending)

`methods/taste_decomposition/results/jokes_community_ledger.json`.
n = 16,000, pos-rate .496, 50 topic groups, V = 27 features, A = 47 criteria,
NA rate .2757, sklearn 1.7.2.

| quantity | value | group bootstrap 95% CI |
|---|---:|---|
| V_lin | **.5843** | [.5759, .5937] |
| V_nl (seed-mean) | .6326 | — |
| **A_lin** | **.7158** | [.7065, .7256] |
| VA_lin | **.7169** | [.7076, .7266] |
| VA_nl (seed-mean of {0,1,2}) | **.7345** | seed-0 [.7245, .7431] |
| Δ_interact = VA_nl − VA_lin | **+.0176** | [+.0131, +.0209], P(>0) = 1.00 |
| V_interact = V_nl − V_lin | +.0482 | — |
| T | pending (dense arm training) | |

**Headline: the mature Gemma bank moves the articulated channel from the
May-2026 floor harness's VA .564† to VA_lin .717** — a +.153 jump, squarely in
the .10–.15 band the dagger flag predicted for floor-vs-mature banks, and the
single largest instrument correction in the wave. The floor numbers should be
retired for this cell.

**Δ_interact is positive and tight, but it is SURFACE nonlinearity, not a tacit
combination rule.** The design note's rollout rule is explicit: when Δ_interact
> 0, decompose via the V-only interaction gain; a large `V_nl − V_lin` alongside
length-feature-dominated pairs marks surface nonlinearity (the Style Invitational
pattern) and routes to Layer 2(b)/Track B, not to a tacit-combination claim. Here
**V_interact (+.0482) is 2.7× Δ_interact (+.0176)** — the GBM's gain is coming
from the deterministic surface block, so this cell is a Layer-2(b) referral.

Instrument checks:
- **Extended anchor battery, K = 50 per class: pos .8718 > neg .7671 > scrambled
  .2422, ordering holds; pos-vs-neg AUC .685, coherent-vs-scrambled AUC .966.**
  That is a far stronger known-label contrast than the CW-community cell managed
  at the same K (.562), so the bank is certified on both halves of the check.
- **Shard 3 is recorded INVALID**: its 3-row blinded draw never ordered
  pos > neg > scrambled in 4 independent draws (pos .923/.778/.645/.575 vs neg
  .943/.911/.932/.825; the scrambled row stayed far below throughout). Its rows
  are retained in the headline readout — a re-draw cannot change temperature-0
  item scores — and `scaleupC_layer1.py` now emits a leave-that-shard-out
  sensitivity readout beside every ledger. This is the expected failure mode of a
  3-row check on a cell where one top-quartile joke barely separates from one
  bottom-quartile joke, and is exactly why the K ≥ 50 battery is the binding gate.
- 2 of 47 criteria collapsed to near-constant ("Quoted speech and its turns do
  work", "Coined or altered expression stays recoverable") — recorded, not
  removed.
- Top univariate criteria: point recoverable on one careful reading (.592),
  point survives as a quotable formulation (.588), reveal is present and
  locatable (.587), closing beat shaped for emphasis (.582), self-contained
  (.576). The bank's discriminating end is **legibility and reveal architecture**,
  not transgression or target choice.

---

## SCHEDULING DECISION (2026-08-07 18:30) — seed-42-first

Stacking the chained Gemma scorer and a LoRA training job on the single permitted
GPU costs each about half its solo throughput: the jokes dense arm measured
**~2.2 h per seed** (1,606 steps) under sharing, so 4 cells × 3 seeds would be
~24 GPU-hours of training alone. The dense chain was therefore restarted in
**two passes**: pass 1 runs seed 42 for every cell, pass 2 adds seeds 1 and 2.
This guarantees every cell has a T rather than leaving the late cells with none,
and the `RUN_DONE` sentinel makes both passes resumable and non-duplicating.
Precedent for a single honest seed on a cell exists in the frozen standard
itself (`run_dense_standard_v4.sh` gives patents claim-fell seed 42 only).

Handover mechanics, in case this is interrupted: the old chain's WRAPPER SHELLS
were killed (outer first, then inner — the standing rule) while its in-flight
`train_reward_model.py` was deliberately left alive; `scripts/scaleupC_supervise2.sh`
waits for that orphan, writes its `RUN_DONE` if `best_model` exists, and only then
starts the new two-pass chain (`logs/scaleupC_dense_chain2.log`). GPU 1 carries
exactly two of this wave's processes and nothing else (the patents seed-2 job
visible on the box is on GPU 3 and belongs to another chain — not touched).

Ledgers are produced as their inputs land, not all at the end. **AoPS needs no
dense training at all** (T reused), so its ledger completes as soon as its Gemma
scoring finishes.

---

## BUILD 1 — FINAL LEDGER (with T)

`methods/taste_decomposition/results/jokes_community_ledger.json` (canonical run
on sk3; a local re-run agrees to ≤.001 — small liblinear/platform float
differences, sk3 is the quoted one).

| quantity | value |
|---|---:|
| V_lin | .5852 |
| V_nl (seed-mean) | .6336 |
| **A_lin** | **.7157** |
| **VA_lin** | **.7169** |
| **VA_nl** (seed-mean {0,1,2}) | **.7321** |
| **T** (dense standard, clean eval, seed 42, n_eval 1,663) | **.7470** |
| | test .7236 (n 1,500) |
| Δ_total = T − VA_lin | **+.0301** |
| Δ_interact = VA_nl − VA_lin | **+.0152** |
| V_interact = V_nl − V_lin | +.0485 |
| **Δ_beyond = T − VA_nl** | **+.0149** |

**The headline finding of build 1: the reddit-jokes residual essentially closes.**
The cell entered this wave looking like the widest gap in the humor row —
floor-harness VA .564† against a provisional ungrouped T .824p, an apparent
band of **+.26**. With a mature Gemma A bank and a topic-grouped dense arm on
the identical population, the band is **+.030 (Δ_total)** and the part eligible
to be called taste is **+.015 (Δ_beyond)**. Both halves of the old gap were
instrument artifacts: the articulated channel was under-measured by the floor
harness (+.153) and the dense channel was over-measured by an ungrouped split
(−.077 from .824p to .747).

Reading discipline for this cell:
- Δ_beyond +.015 sits at the same order as the VA_nl seed spread and well inside
  the T seed uncertainty (only seed 42 has landed; seeds 1 and 2 are queued in
  pass 2). **Do not quote a taste residual for reddit jokes until the 3-seed T is
  in** — the honest present statement is "the residual is at most a few AUC
  points, and may be zero".
- Δ_interact +.0152 is positive with a tight group bootstrap, but V_interact
  (+.0485) is 3.2× larger, so the nonlinear gain lives in the deterministic
  surface block. Per the design note's rollout rule this is SURFACE nonlinearity
  → Layer 2(b) / Track B referral, **not** a tacit-combination-rule claim.
- Invalid-shard sensitivity (drop shard 3, n = 13,994): V .5826 / A .7150 /
  VA .7149 / VA_nl(seed0) .7319 — every quantity moves by less than .003, so the
  failed 3-row anchor draw does not carry the result.

---

## BUILD 2 — math.SE Gemma bank scored (2026-08-07 20:20)

13,001 items x 32 criteria x 6 shards = 416,032 judge calls, plus 4,800 battery
calls. Overall NA rate ~.24 per shard.

Instrument gates:
- Per-shard blinded anchors: shards 0-3 valid on the FIRST draw
  (pos .790/.740/.583/.840 vs neg .654/.483/.321/.759, scrambled .000
  throughout); shards 4 and 5 failed their first draw and passed on the second
  (pos .821 vs neg .808; pos .904 vs neg .783). **No shard is invalid.**
- **Extended battery, K = 50 per class: pos .7399 > neg .6592 > scrambled .0007,
  ordering holds; pos-vs-neg AUC .583, coherent-vs-scrambled AUC 1.000.**
  The scrambled gate is perfect — the judge is certainly reading the mathematics.
  The known-label contrast is real but weak (.583, below the jokes bank's .685),
  which is the honest signature of the accepted-verdict y: on a single pair, the
  accepted answer is often not the visibly better-written one.

The two y's are read off this one matrix and reported separately, never merged.

---

## BUILD 2 RESULT — math.SE ACCEPTED-VERDICT ledger

`methods/taste_decomposition/results/mathse_accepted_verdict_ledger.json`.
n = 13,001 answers / 4,960 questions, pos-rate .3815, V = 28, A = 32,
NA rate .2397.

| quantity | value |
|---|---:|
| V_lin | .5875 |
| V_nl (seed-mean) | .5909 |
| A_lin | .6250 |
| **VA_lin** | **.6320** |
| **VA_nl** | **.6320** |
| **T** (dense standard, clean eval, seed 42, n_eval 1,300) | **.6319** |
| | test .6558 (n 1,300) |
| Δ_total | **−.0001** |
| Δ_interact | **.0000** (group bootstrap −.0002 [−.0059, +.0057]) |
| V_interact | +.0034 |
| **Δ_beyond** | **−.0001** |

**The residual is exactly zero.** The 8B dense reader, given the raw text, finds
nothing about which answer the asker accepted that 32 named criteria plus 28
surface features do not already carry — and the GBM finds no interaction either
(Δ_interact .0000, V_interact +.0034; both nulls, unlike every other cell in the
wave). This is the cleanest fully-articulable cell in the whole decomposition
programme.

**Why this matters for the old math.SE number.** The published binarized cell
reads V .565 / VA .673 / T .794, a band of **+.12**. That band does not survive
un-binarization. The old y fused two signals and then censored the middle
(positive = accepted AND score ≥ 3, negative = score ≤ 0, everything scoring 1-2
deleted), which manufactures an easy, partly popularity-driven contrast. Asking
the honest within-question question — *which of these answers did the asker
accept?* — collapses the band to zero. The two cells are also nearly disjoint in
rows (0.79% overlap with the old dense arm's held-out set), so this is not a
re-estimate of the same quantity; it is a different, cleaner question about the
same site.

Top articulated criteria: names the real obstacle (.569), transitions carry a
stated licence (.555), omissions are recoverable (.555), hypotheses stated and
seen to hold (.548), quantifiers and domains explicit (.547). The discriminating
end is **argument hygiene**, not pedagogy or elegance. Two criteria collapsed to
near-constant ("Uncertainty pinned to a step", "Pitched to the asker's evident
level") — recorded, not removed.

Caveat: T is seed-42 only so far (pass 2 adds seeds 1 and 2); the eval/test
spread (.6319 / .6558) is ~.024, larger than |Δ_beyond|, so the correct present
statement is "no detectable residual", not "residual = 0.000".

---

## BUILD 3 RESULT — AoPS curation ledger (the cell's FIRST A-bank stack)

`methods/taste_decomposition/results/aops_curation_ledger.json`.
n = 5,202 forum solutions / 606 problems (all dense-held-out), pos-rate .6734,
V = 24 deterministic features, A = 44 criteria, NA rate .2285,
**0 criteria collapsed**.

| quantity | value |
|---|---:|
| V_lin | .7026 |
| V_nl (seed-mean) | .7084 |
| **A_lin** | **.7691** |
| **VA_lin** | **.7712** |
| **VA_nl** | **.7705** |
| **T** (REUSED grouped Llama-3.1-8B arm, same rows) | **.7806** |
| | eval .7739 / test .7879 |
| Δ_total | **+.0095** |
| Δ_interact | **−.0007** (group bootstrap −.0022 [−.0125, +.0085]) |
| V_interact | +.0058 |
| **Δ_beyond** | **+.0101** |

**The cell's first articulated stack lands at VA .771 against a dense .781 — a
residual of one AUC point.** Two things follow. First, the AoPS same-approach
label is very nearly fully articulable: 44 named criteria plus 24 surface
features recover essentially everything an 8B reader gets from the text.
Second, **A alone (.769) already matches the dense model (.781) to within .012**,
and beats the deterministic V channel (.703) by .066 — so this is a genuine
articulated-channel win, not a surface artifact. Δ_interact is a clean null.

Instrument gates: extended battery at K = 50 per class gives pos .5712 >
neg .4663 > scrambled .0006, ordering holds, pos-vs-neg AUC .619,
coherent-vs-scrambled AUC .990. All six shards' 3-row anchors are recorded in the
ledger.

Top articulated criteria: the whole question is answered (.666), a substantive
attempt at the demand (.647), every assertion has visible support (.641), stands
without the surrounding discussion (.632), stated claims are discharged (.625),
compressed steps are reconstructible (.625). The signal is **completeness and
self-containment**, not elegance or method choice — consistent with the label
being "did this post actually work the problem the way a solution does".

Reuse note (per the coordinator directive): NO dense training was run for this
cell. T is the pre-existing grouped Llama-8B arm's own held-out predictions,
verified row-aligned with `split_full`, so it is same-rows by construction. The
V channel ports the published lexicon regexes verbatim but excludes `p_ed`,
`is_correct` and `has_answer`, so **this V (.703) and the published V .706 are
not the same instrument** despite the near-identical value — do not difference
them.

---

## BUILD 2 RESULT — math.SE VOTE-SCORE ledger, and the multi-y contrast

`methods/taste_decomposition/results/mathse_vote_score_ledger.json`.
n = 11,629 answers (the rows where the within-question median split is defined)
/ 4,960 questions, pos-rate .5015. Same scored matrix as the accepted-verdict
cell, same V, same 32 criteria — **only y differs.**

| quantity | value |
|---|---:|
| V_lin | .5714 |
| V_nl | .5750 |
| A_lin | .6143 |
| **VA_lin** | **.6225** |
| **VA_nl** | **.6242** |
| **T** (dense standard, clean eval, seed 42, n_eval 1,163) | **.6608** |
| | test .6466 (n 1,163) |
| Δ_total | **+.0383** |
| Δ_interact | +.0017 (group bootstrap +.0023 [−.0037, +.0084]) — NULL |
| V_interact | +.0036 |
| **Δ_beyond** | **+.0366** |

### The contrast this build was for

| math.SE y (same 13,001-row scored matrix, same bank, same judge) | VA_lin | VA_nl | T | **Δ_beyond** |
|---|---:|---:|---:|---:|
| **accepted verdict** — which answer the ASKER accepted | .6320 | .6320 | .6319 | **−.0001** |
| **vote score** — above/below the question's median VOTE | .6225 | .6242 | .6608 | **+.0366** |

**Un-binarizing separates two preference signals that the old cell fused, and
they behave completely differently.** The asker's accept decision is *fully
articulable* — 32 named criteria plus 28 surface features exhaust what an 8B
reader can extract. The crowd's vote is *not*: the dense model finds +.037 of
AUC in the text that the articulated bank does not name, on the very same items,
scored by the very same criteria. Whatever the crowd is rewarding beyond the
asker's own judgement is the part of this cell that is still tacit.

Δ_interact is a null on both y's, so neither residual is a tacit combination rule
over articulated criteria; on the vote-score y the whole of Δ_total survives as
Δ_beyond and the cell clears the Layer-3 gate (> .02) while the accepted-verdict
cell does not.

Both cells' top criteria are the same argument-hygiene family (names the real
obstacle, omissions are recoverable, central reason extractable, hypotheses
stated and seen to hold), so the difference is not that the bank suits one y and
not the other — the bank is equally on-target for both and simply runs out of
signal sooner against votes.

Caveat: T is seed-42 only (pass 2 adds seeds 1 and 2); the eval/test spread here
is .0142, well under Δ_beyond, so the vote-score residual is robust to it, while
the accepted-verdict null is stated as "no detectable residual".

---

## BUILD 4 — Gemma scoring done; dense split had to be rebuilt

The homepage A bank scored cleanly (6 shards x 12,998 items x 14 criteria =
181,972 judge calls + a K = 50 battery), but the dense arm **failed in 4 seconds
on its first attempt**: `train_reward_model.py::get_or_create_fixed_split` hard-
requires train/eval/test within ±2 percentage points of 80/10/10, and a
whole-outlet 6/1/1 assignment over 8 outlets of ~1,700 rows each lands at
**.7370/.1317/.1313** — rejected. This is a structural consequence of the cell's
grouping unit, not a bug in the split builder.

Fix (`datasets/news-homepages/fix_dense_split.py`): keep the outlet-held-out
design and keep every dense row inside the already-scored 12,998-row population,
but take a SUBSET sized so the ratios come out right — eval = one held-out
outlet trimmed to E rows, test = a second held-out outlet trimmed to E, train =
the remaining six outlets trimmed to 8E in total, removing WHOLE SNAPSHOTS in
stable-hash order so the snapshot container is never split. WSJ is excluded from
being a held-out outlet (only 61 of its captures resolve; it is the thinnest
fold) but still contributes to train.

Result: **eval = latimes (1,201), test = guardian (1,198), train = {bbc, cnn,
nytimes, reuters, washingtonpost, wsj} (9,590); fractions .7999/.1002/.0999;
per-split pos-rates .5037/.5025/.4953.** Every dense row is a scored row, so T's
eval rows stay a subset of the A/V population and FREEZE CHANGE 2 still holds.
The A/V population itself (12,998 rows) was NOT touched — no Gemma work was
wasted.

### BUILD 4 — the anchor battery FAILS the coherence gate (load-bearing finding)

Extended battery, K = 50 per class, on the homepage census bank:

| anchor class | mean A |
|---|---:|
| known positive (top-half placement) | .5414 |
| known negative (bottom-half) | .4980 |
| **scrambled word salad** | **.5776** |

pos-vs-neg AUC **.574**; **coherent-vs-scrambled AUC .387 — below chance.**
All six shards also failed their first 3-row draw; four eventually passed on a
re-draw, two never did.

Compare the other three banks in this wave, same battery, same K:

| bank | pos | neg | scram | pos-vs-neg | coherent-vs-scrambled |
|---|---:|---:|---:|---:|---:|
| reddit jokes | .8718 | .7671 | .2422 | .685 | **.966** |
| math.SE | .7399 | .6592 | .0007 | .583 | **1.000** |
| AoPS | .5712 | .4663 | .0006 | .619 | **.990** |
| **homepage census** | .5414 | .4980 | **.5776** | .574 | **.387** |

**The 14 news-values criteria score scrambled word salad HIGHER than real
headlines.** That is not a weak instrument, it is a different kind of instrument:
these criteria are entity/keyword detectors ("is a head of state named?", "is
there a large number?", "is there a violence word?") and a scramble preserves the
bag of words that triggers them. The three craft banks all read *structure* and
collapse to ~0 on salad; this one does not read at all.

Consequences, binding on every number this cell produces:
1. The homepage A channel must be described as a **news-values lexical profile**,
   not as an articulated-criteria reading instrument, and it is not comparable to
   the other A banks in the grid on that axis.
2. The weak-instrument flag on this cell is now **measured**, not asserted.
3. Any Δ_beyond from this cell is a statement about how much the dense reader adds
   over a bag-of-news-values profile — a weaker and different claim than the same
   quantity in the other three cells.

Also corrected while building: the battery crashed the first time with
`ValueError: Input contains NaN` because a scrambled headline can legitimately
draw NA on all 14 criteria, leaving an anchor row with no mean.
`run_battery` now drops all-NA anchor rows and reports how many
(`n_anchor_rows_all_NA_dropped`) rather than failing — a fix that matters for any
future small bank.

---

## WAVE-C LEDGER SUMMARY (all quantities grouped-OOF on the cell's own unit)

| cell | n | groups | V_lin | A_lin | VA_lin | VA_nl | T | Δ_total | Δ_interact | **Δ_beyond** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit-jokes community | 16,000 | 50 topics | .5852 | .7157 | .7169 | .7321 | .7470 | +.0301 | +.0152 | **+.0149** |
| math.SE accepted verdict | 13,001 | 4,960 questions | .5875 | .6250 | .6320 | .6320 | .6319 | −.0001 | .0000 | **−.0001** |
| math.SE vote score | 11,629 | 4,960 questions | .5714 | .6143 | .6225 | .6242 | .6608 | +.0383 | +.0017 | **+.0366** |
| AoPS curation | 5,202 | 606 problems | .7026 | .7691 | .7712 | .7705 | .7806 | +.0095 | −.0007 | **+.0101** |
| homepage curation (outlet-held-out) | 12,998 | 8 outlets | .4566 | .5979 | .5490 | .5562 | .4322 ✗ | n/a | +.0072 | **n/a** |
| homepage curation (snapshot-grouped, secondary) | 12,998 | 1,229 snapshots | .5772 | .6420 | .6572 | .7012 (seed 0) | — | — | — | — |

✗ = the homepage outlet-held-out T is BELOW CHANCE on eval (.4322) while its test
outlet reads .7361 — the two held-out outlets disagree in sign, so no residual is
computable for that cell. See the build-4 section.

Every T is same-rows by construction. All five are FIRST-FIT cells (no
reproduction gate; the linear leg IS the first fit). Δ_interact CIs are
group-level bootstraps. Dense T is seed-42 for the four trained cells (pass 2 for
seeds 1 and 2 is running); AoPS T is the reused pre-existing arm.

### What the wave changes

1. **Two of the four "gap" cells turn out to have essentially no gap.**
   math.SE accepted-verdict is **−.000** and AoPS curation is **+.010** — with a
   real A bank, the articulated channel reaches the dense model. reddit-jokes is
   **+.015**, also small, and its apparent +.26 band was an artifact of a floor
   harness on one side and an ungrouped split on the other.
2. **The one real residual is a CROWD signal.** math.SE vote-score keeps
   **+.037** on the same items, same bank, same judge as the accepted-verdict
   cell that keeps nothing. Un-binarizing was what made that contrast visible;
   the old fused y hid it.
3. **Δ_interact is a null or a surface effect everywhere.** Three of five cells
   have Δ_interact within ±.002 of zero; the one clearly positive case
   (reddit-jokes, +.0152) has a V-only interaction gain 3.2× larger, so it is
   surface nonlinearity, not a tacit combination rule. No cell in this wave
   supports a "tacit combination of articulated criteria" claim.
4. **A-bank quality is now measurable, and one bank fails.** The K = 50
   coherent-vs-scrambled gate reads .966 / 1.000 / .990 for the three craft banks
   and **.387 for the homepage news-values census** — the census criteria are
   lexical detectors, not reading instruments.

### Homepage, read with care

Under the registry's OUTLET-HELD-OUT design the deterministic V channel scores
**.4566 — below chance** — and adding it to A *hurts* (VA_lin .5490 < A_lin
.5979). That is the signature of outlet-specific surface conventions: each
outlet's headline furniture is its own dialect, so V learned on six outlets
transfers negatively to a seventh. Under the secondary snapshot-grouped readout
the same matrices give V .5772 / A .6420 / VA .6572 — close to the published
census .593. Report the outlet-held-out numbers as the headline (it is the
registry's specification and the harder generalisation), the snapshot-grouped
ones as the comparability row, and never difference them.

---

## BUILD 4 RESULT — homepage curation ledger, and why its T cannot be quoted

`methods/taste_decomposition/results/homepage_curation_ledger.json`.
n = 12,998 headlines / 8 outlets / 1,229 snapshots, pos-rate .5006,
V = 23 features, A = 14 census criteria.

| quantity | OUTLET-HELD-OUT (headline design) | snapshot-grouped (secondary) |
|---|---:|---:|
| V_lin | **.4566** | .5772 |
| V_nl (seed-mean) | .4909 | — |
| A_lin | **.5979** | .6420 |
| VA_lin | **.5490** | .6572 |
| VA_nl | **.5562** | .7012 (seed 0) |
| Δ_interact | +.0072 | — |
| V_interact | +.0343 | — |

Dense arm (outlet-held-out, train = 6 outlets, eval = latimes, test = guardian):
**eval AUC .4322 (n 1,201), test AUC .7361 (n 1,198).**

### T IS NOT QUOTABLE FOR THIS CELL

The dense-standard canonical readout is the clean-eval AUC, and here it is
**.4322 — below chance** — while the test split on a *different* held-out outlet
is **.7361**. The two held-out outlets disagree not in magnitude but in SIGN. A
model selected on a below-chance eval split is not a measurement of anything, so
the arithmetic Δ_total = −.117 and Δ_beyond = −.124 that the ledger records are
**artifacts and must never be quoted as a residual**. They are retained in the
JSON only so the failure is auditable.

What this actually establishes — and it is a real result, not a null:
**homepage spatial placement does not transfer across outlets.** Everything about
the cell points the same way:
1. V alone is **below chance** (.4566) out-of-outlet, and adding V to A *hurts*
   (VA_lin .5490 < A_lin .5979). Each outlet's headline furniture is its own
   dialect, so surface features learned on six outlets invert on a seventh.
2. The dense reader, which has the most capacity to fit outlet-specific layout
   conventions, is the most damaged by the transfer (.4322 / .7361).
3. The A bank fails the coherent-vs-scrambled gate (.387), so even the
   articulated channel here is a lexical profile rather than a reading.
4. Under snapshot grouping — where train and test share outlets — everything
   behaves normally (V .5772, A .6420, VA .6572, VA_nl .7012) and lands near the
   published census .593.

The honest statement for the paper: *the journalism curation cell has a working
within-outlet instrument and no cross-outlet instrument.* The registry's
outlet-held-out .593 census number should be read as an optimistic
within-snapshot figure, and the T .824 provisional should be retired for this
cell — it came from a snapshot-grouped sweep on a different population, and the
outlet-held-out dense arm built here shows what happens when the harder
generalisation is actually demanded.

Top census criteria (snapshot-comparable, univariate): hard news rather than soft
(.605), part of the day's top-tier running story (.599), large scale of people or
stakes (.576), elite political actor is a central subject (.560). These are the
Galtung-and-Ruge staples, and they do carry real within-outlet signal.

### Next step if this cell is to be rescued
Train the dense arm snapshot-grouped (matching the secondary readout and the
existing sweep) and report outlet-held-out only as a leave-one-outlet-out
robustness table over all 8 outlets, rather than betting the whole T on one
held-out outlet. That is a scoped follow-up, not part of this brief.


---

## FINAL STATE — 2026-08-08 01:45

**All five ledgers written** to `methods/taste_decomposition/results/`:
`jokes_community_ledger.json`, `mathse_accepted_verdict_ledger.json`,
`mathse_vote_score_ledger.json`, `aops_curation_ledger.json`,
`homepage_curation_ledger.json` (plus `<cell>_va_nl_oof_{seed0,mean3}.npy` per
cell, and `homepage_curation_ledger.NO_T.json.bak`, the superseded T-less draft,
kept rather than deleted).

Score matrices: `outputs/va_gemma_banks_scaleupC/` (4 banks x shards + meta +
`anchor_battery.json`), mirrored local and on sk3.

Still running unattended on sk3 GPU 1, and safe to leave:
- `scripts/run_scaleupC_dense_chain.sh` **pass 2** — seeds 1 and 2 for
  jokes / mathse-accepted / mathse-vote / homepage
  (`logs/scaleupC_dense_chain2.log`). Every ledger currently carries a seed-42 T;
  when pass 2 lands, re-run `scaleupC_layer1.py --cell <cell>` to fold in the
  3-seed mean. The RUN_DONE sentinel makes this resumable.
- Nothing else. The Gemma scoring chain is finished for all four banks.

Known follow-ups, scoped but NOT part of this brief:
1. 3-seed T refresh for the four trained cells (above).
2. Homepage: snapshot-grouped dense arm + leave-one-outlet-out robustness table
   (the current outlet-held-out T is unusable).
3. Layer 2(b) referral for reddit-jokes (its Δ_interact is surface, not tacit).
4. math.SE vote-score is the only cell in this wave clearing the Layer-3 gate
   (Δ_beyond +.037 > .02) and is the natural next closure-campaign candidate.
