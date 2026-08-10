# Layer-3 articulation closure — PRESS / JOURNALISM VERDICT cell (editorial pickup)

Date: 2026-08-08/10. Protocol: the FROZEN preregistration
`notes/2026-08-05__layer3-closure-prereg.md` — FREEZE DECLARATION (2026-08-06) plus
FREEZE ADDENDUM 1 (B-side missing mass; stacked increment), ADDENDUM 2 (Track-B
upstream-factor mode; MIXED flag), ADDENDUM 3 (MIXED-channel decomposition pass) and
ADDENDUM 4 (position-in-container Track-B prior). Worked precedents:
`notes/2026-08-09__peer_completion.md` (peer revealed + peer curation) and
`notes/2026-08-06__closure_nc_responded.md` (N&C responded, and its DECISION-1
split-width precedent).

Code + artifacts: `methods/taste_decomposition/closure/press_verdict/`.
Machinery reused verbatim from `closure/peer_revealed/` and `closure/maps_hw_si/`
(`closure_core.py`, `stage1_slice.py`, `harness_maps.py`, `run_fleet.py`, `species.py`,
`audit.py`, `arbiter.py`, `score_gemma_maps.py`, `readout.py`, `decompose_round.py`,
`mixed_parents.py`, `discount_cumulative.py`, `jackknife.py`, `build_splits.py`,
`gpu_runner.sh`); what is new is listed in §0.3.

---

## 0. Terms, cell, and what is reused vs new

### 0.1 Terms, spelled out (standing rule)

**press verdict cell** = press-release **editorial pickup**: y = 1 iff the release was
written up by **≥ 3 distinct tracked news outlets** ("consensus coverage"), y = 0 for a
topic-matched release with **zero** tracked pickups. **V** = the 88-column reconstructed
deterministic surface-feature bank; **A** = the 40-rubric Gemma-judged articulated
criterion bank; **VA_nl** = HistGradientBoosting aggregation of the V+A matrix, mean over
seeds {0,1,2}; **VA_lin** = its logistic counterpart; **T** = the dense readout
(Llama-3.1-8B LoRA reward model on raw release text), always the **same-rows** value on
that model's own held-out rows; **Δ_beyond** = T − VA_nl, the unarticulated residual;
**Δ_r** = Δ_beyond after r rounds of mining; **ε** = .005, the frozen per-round
saturation threshold; **AUC** = area under the ROC curve; **FIT+MINE / MONITOR** = the
closure splits; **M** = the mining slice (dense-held-out rows inside FIT+MINE);
**HONEST population** = every dense-held-out row (M ∪ MONITOR), the population on which
T and VA are both out-of-sample; **Track A** = the quality-criterion miner (k_A = 15
scored criteria per round); **Track B** = the suspected-spurious-channel miner
(k_B = 10 scored channels per round); **alone-AUC** = the AUC of a single judged
channel's raw 0–10 score, no fitting; **joint B model** = HistGB on the B channels only;
**stacked increment** = AUC(logistic stack of joint-B + X) − AUC(joint-B), the
stratification-free discount; **MIXED channel** = a Track-B channel whose conjectured
upstream parent plausibly causes real quality as well (ADDENDUM 2);
**Good-Turing missing mass M̂** = f1/N over proposal *species*; **LOPO** =
leave-one-proposer-out jackknife; **P** = sealed-fleet size; **GEPA** = the
prompt-iteration pass required before any final quoted phrasing — **not applied here**,
so every number carries a pre-GEPA flag.

### 0.2 The cell, and the three structural facts that govern every number

| fact | value | consequence |
|---|---|---|
| population | 2,956 rows (1,478 pos / 1,478 topic-matched neg), **556 companies**, grouped by company | every split, fold and bootstrap is company-grouped |
| dense split | company-grouped stable-hash 80/10/10 → train 2,351 (511 companies) / **eval 288 (36)** / **test 317 (9)** | the HONEST population is **605 rows in 45 companies** — thin, and the campaign's binding constraint |
| dense selection | select-on-eval (dense-standard, **no** deviation) | **TEST is the selection-free half here** — the mirror of N&C, whose chain selected on test |
| concentration | 9 companies carry **93.9 %** of the HONEST rows (about_fb 130, prnewswire 123, tesla 84, corporate_target 64, news_airbnb 60, spglobal 39, salesforce 28, pcmag.com 26, news_walgreens 14) | a leave-one-company-out jackknife is the honest width statistic, and it is wide |
| sklearn | Layer-1 numbers produced under **scikit-learn 1.9.0**; `GroupKFold` fold *assignments* move across releases (documented LANDMINE in `press_verdict_layer1.py`) | the whole campaign runs in a pinned 1.9.0 scratch venv; `cells.sklearn_guard()` asserts it every script |

### 0.3 What is NEW here

1. `cells.py` — the press cell adapter. The A cache
   (`results/press_verdict_pr_A_k3_scores_CACHE.npz`) *is* the population definition and
   its `ids` order is used verbatim; V is rebuilt from
   `press_verdict_v_features_recon.py`; texts come from the deconfounded parquet through
   the same `clean_text` the V bank read.
2. `fetch_dense.py` — assembles T from the three sk3 seeds. The `preds_*.csv` files carry
   **no row key**, so the join is positional against `split/{eval,test}.csv` and the
   script **asserts** that the `group` and `judgement` columns match element-for-element
   for all three seeds and both splits before writing anything.
3. `round0.py` — the split-width diagnostic, the state-0 bank under both A-imputation
   conventions, T on every population, the same-rows anchor, the swap baseline and the
   leave-one-company-out jackknife, all in one pass and all **before round 1 ran**.
4. `census.py` — the incoming-bank concept census (L0→L5) with two sealed blind judges
   and an authored anchor battery.
5. `applicability_probe.py` — new to this cell and forced by the census (§2.2).
6. `company_structure.py` — the between-company variance decomposition and the
   within-company pair-concordance readout (the N&C §7.3 enlargement, adapted).
7. `position_line.py` — FREEZE ADDENDUM 4's observed-covariate line, built on
   `press_release_date` (85.5 % coverage) rather than on a judged channel.
8. Corpus-matched planted probes for the press auditor (`audit.py`), four authored pairs
   for a corporate press release, two drawn per round by stable sha256 of the round tag.
9. Track-B **MODE 4** in the harness: the corpus's named upstream priors (PR-agency vs
   in-house drafting, organisation size/prominence, newswire distribution tier, embargo
   and timing conventions), alongside ADDENDUM 4's MODE 3 rewritten for this container
   (position in the organisation's release stream, and in the calendar).

---

## 1. Two decisions recorded BEFORE round 1

### DECISION 1 — the A block is Layer 1's constant-0.5 fill, not the closure median impute

The A matrix is **applicability-gated**: each of the 40 rubrics carries an `applicable`
bit, and an inapplicable cell has no judged level. Layer 1's production pipeline and the
A-layer's own published gate **both** fill such a cell with the constant 0.5. The closure
protocol's `clean_fit` instead median-imputes inside FIT+MINE. Both were fitted at round 0,
and the gap is large and one-directional:

| state-0 bank | features | VA_nl HONEST (n=605) | VA_nl MONITOR (n=312) | MONITOR seed spread |
|---|---:|---:|---:|---:|
| **PRIMARY — Layer-1 constant-0.5** | 126 | **.7296** | .7446 | .0074 |
| sensitivity — closure median impute | 125 | .7009 | .7051 | .0201 |

Median-imputing **erases the applicability pattern and costs the bank .029 of held-out
AUC**, which would inflate Δ_beyond by the same amount. The primary is therefore both the
faithful choice (it is the matrix this cell's Layer 1 actually has) and the conservative
one. Mined criteria added in later rounds are NaN-coded and median-imputed by `clean_fit`,
matching how each block was built — recorded, not silently mixed.

### DECISION 2 — the saturation statistic is read on HONEST, with MONITOR alongside

The freeze fixes MONITOR ⊂ dense-held-out. Applied here (median-of-sha256 cut over the 45
dense-held-out companies, the `maps_hw_si` coarse-group adaptation) that gives
MONITOR = 312 rows / 23 companies, mining slice M = 293 rows / 22 companies.

The N&C campaign's DECISION 1 escaped a thin MONITOR by reading the saturation statistic
on a larger, VA-honest MONITOR_FULL. **That escape is not available here and the width
diagnostic says so:** MONITOR_FULL (hash ≥ .80 over all 556 companies) is only **413 rows /
108 companies** — barely larger than MONITOR, because companies are very unevenly sized —
and it costs T-honesty. So the frozen MONITOR is kept, and the **HONEST population
(605 rows, 45 companies, pos-rate .501) is the primary readout for both the level and the
round-over-round gain**, exactly as the peer campaigns did; MONITOR (312 rows,
pos-rate .420) is reported every round alongside.

---

## 2. ROUND 0

### 2.1 The incoming bank's concept census

Cheapest decisive test first; TF-IDF used only to shortlist candidate duplicate pairs
within one register (bank rubric vs bank rubric); identity decided by two sealed blind
judges, never by a similarity threshold.

| level | instrument | count |
|---|---|---:|
| L0 | rubrics delivered | **40** |
| L1 | distinct names (normalised, exact) | **40** |
| L2 | columns surviving the frozen degeneracy screen (fit on FIT+MINE only) | **37** |
| L3 | value clusters after collapsing \|Pearson r\| ≥ .98 columns | **37** |
| L5 | **effective concepts** after blind pairwise adjudication (strict: both judges SAME) | **36** |
| L5′ | loose rule (either judge says SAME) | 35 |

Instrument health: judge raw agreement **.975**; both judges **4/4** on the anchor battery
(two authored paraphrase SAME pairs, two authored DIFFERENT pairs). The three columns the
degeneracy screen drops are *Outlet/beat targeting and personalization*,
*Funder/partner visibility and co-branding compliance*, and *CONSORT adherence* — all
essentially never applicable to a corporate press release. The single adjudicated merge is
*Multimedia assets: availability, quality and usability* with *Publish-ready multimedia
assets provided*.

**This is the least redundant bank in the programme** — 40 → 36 effective concepts, a 10 %
collapse, against peer's −65 % and N&C's −21 %, with max column \|r\| = .78 and **no**
column pair at .90.

**And it is also the least informative, by a wide margin.** Alone-AUC over the 37 surviving
columns, computed on FIT+MINE only: **max .527, min .472, median .496**, median absolute
deviation from chance **.0069**, and **zero** columns at ≥ .55 or ≤ .45. N&C's bank had one
column at ≥ .55 and peer's best reached .607. Every single press rubric is at chance on its
own.

**Register.** The bank is written in journalism-ethics / PR-practitioner language and was
coverage-selected as 40 k-means medoids over a 309-rubric pool built for news and science
communication generally; the corpus is corporate press releases and the outcome is
editorial pickup. Several rubrics (CONSORT adherence, primary-outcome effect sizes, apology
and acceptance of responsibility, M&A/SPAC disclosure) are off-register for most items,
which is exactly why the bank is applicability-gated — and §2.2 is about what that gating
turns out to be doing.

### 2.2 Where the bank's signal actually lives — the applicability probe

The census leaves an arithmetic problem. Thirty-seven rubric columns are each at chance
(max .527), yet Layer 1 reports A_lin .669 / A_nl .674. A bank of null columns that
aggregates to .67 is carrying its signal somewhere other than in its judged levels, and
this A matrix has an obvious candidate: it is applicability-gated, and Layer 1 turns an
inapplicable cell into the constant 0.5 — so the **missingness pattern**, i.e. *which of
40 news-and-science-communication rubrics a judge thinks even apply to this release*, is
silently a 40-bit description of the release's genre.

`applicability_probe.py` splits the A block three ways and fits the frozen closure spec to
each (`applicability_probe.json`):

| block | features | VA_nl HONEST (n=605) | VA_nl MONITOR |
|---|---:|---:|---:|
| **A_mask_only** — the applicability bits, no judged level at all | 38 | **.7322** | .6951 |
| A_levels_only — judged levels, median-imputed so the mask is erased | 37 | .6705 | .6356 |
| A_layer1_const05 — the A block as Layer 1 has it (mask + levels fused) | 38 | .7160 | .7092 |
| A_mask + A_levels as two separate blocks | 75 | .7260 | .6951 |
| V only | 88 | .6918 | .7082 |
| **V + mask (no judged level anywhere)** | 126 | **.7282** | .7467 |
| *(for reference)* **the whole V + A primary bank** | 126 | **.7296** | .7446 |

Three readings, in ascending order of how uncomfortable they are.

**The applicability mask ALONE reaches .7322 on the honest rows — at or above the entire
126-feature V+A scorecard (.7296), .040 above V alone, and .062 above the judged levels
with the mask removed.**

**Swapping the whole judged A block for nothing but its 38 applicability bits changes the
scorecard by .0014** (V+mask .7282 vs V+A .7296). On this cell the forty Gemma-judged
rubric *levels* are worth about one-tenth of one AUC point over the bare fact of which
rubrics a judge thought applied.

**Individual mask bits out-predict every judged level.** The strongest applicability bit
is *Clear communication of uncertainty and preliminary status* (applicable to 49 % of
releases, alone-AUC **.578** on FIT+MINE), followed by *Crisis response strategy* (.556,
15 %) and *Specific corrective/preventive actions* (.550, 13 %) on the positive side, and
*Quote quality* (.451, 58 %) and *Headline effectiveness* (.455, 39 %) on the negative
side — against a best judged level of .527. The count of applicable rubrics is at chance
(.511, mean 14.0 of 40 applicable), so it is **which** rubrics apply, not how many:
science/uncertainty and incident/crisis releases get picked up, quote-driven and
headline-driven promotional releases do not.

Whatever the press A bank is measuring, it is overwhelmingly *which kind of release this
is* — a genre fingerprint — and not *how well the release satisfies an articulated
criterion*.

This is a Track-B object sitting inside the A block, and it is the single most important
structural fact about this cell. It is reported here rather than acted on: the frozen
protocol does not permit re-routing a Layer-1 bank column mid-campaign, so the primary
bank keeps the mask (DECISION 1) and the discount readouts below carry this finding as
the reason the bank's own number should not be read as "articulated quality".

### 2.3 The round-0 anchor, and a correction to the dispatched Δ


The campaign was dispatched on **Δ_beyond = +.0486 = T .7497 − VA_nl .7011**. That
subtraction is not same-rows: the **T** term is the mean of the three dense seeds' AUCs on
the 288 **dense-eval** rows, while the **VA_nl** term is Layer 1's pooled grouped-OOF AUC
over **all 2,956** rows. Round 0's first job was to put both on the same rows.

**Same-rows anchor, Layer-1 fitting protocol** (`press_verdict_samerows_anchor.json`).
Taking Layer 1's own saved OOF prediction vector and restricting it to the 605
dense-held-out rows:

| quantity | value |
|---|---:|
| Layer-1 VA_nl, pooled over all 2,956 rows (as dispatched) | .7011 |
| **Layer-1 VA_nl, same OOF predictions restricted to the 605 HONEST rows** | **.7442** |
| T on the 605 HONEST rows (mean of 3 seed AUCs) | **.7508** |
| **Δ_beyond, same rows, Layer-1 protocol** | **+.0066** |

**Same-rows anchor, closure protocol** (VA refit on FIT+MINE only, `round0.py`):

| population | n | T | VA_nl | **Δ_beyond** |
|---|---:|---:|---:|---:|
| **HONEST (dense-held-out)** | 605 | .7508 | .7296 | **+.0212** |
| MONITOR | 312 | .7360 | .7446 | **−.0086** |
| mining slice M | 293 | .7712 | .7112 | +.0600 |
| eval only (selection-contaminated half) | 288 | .7497 | .7198 | +.0299 |
| test only (**selection-free half**) | 317 | .7525 | .7384 | +.0141 |
| *sensitivity: median-impute bank, HONEST* | 605 | .7508 | .7009 | *+.0499* |

Three things follow, and they reframe the campaign.

1. **The dispatched +.0486 is a cross-population artifact.** On the same 605 rows the
   residual is **+.0066** under Layer 1's own fitting protocol and **+.0212** under the
   closure protocol. The dense-held-out rows are a subpopulation on which the articulated
   bank does markedly *better* than average (.7442 vs .7011 pooled) — the same
   AMENDMENT-1 phenomenon N&C hit (.781 vs .724), here large enough to consume 60–86 % of
   the dispatched residual before a single criterion was mined.
2. **The residual is not distinguishable from zero at this cell's width.** Leave-one-company-out
   jackknife over the 45 dense-held-out companies: pooled Δ +.0212, jackknife mean +.0213,
   **SE .0250**, range [+.0093, +.0383], pseudo-CI **[−.028, +.070]**; the most influential
   company is **prnewswire** (dropping it moves Δ to +.0383). Under the median-impute
   sensitivity the same jackknife gives +.0499 ± .0239, CI [+.003, +.097]. **So the sign of
   the "is there a residual" answer depends on an imputation convention, and neither
   answer clears its own jackknife comfortably.**
3. **MONITOR's Δ is negative (−.0086).** On 312 rows in 23 companies the articulated
   scorecard *beats* the dense model. That is within noise, but it is the direction that
   matters: this cell has no residual to speak of before mining begins.

### 2.4 Company structure — the press analogue of the N&C docket floor

`company_structure.json`. **36.5 %** of the label variance sits *between* companies; 410 of
556 companies are entirely positive (50) or entirely negative (360), covering 628 rows that
carry no within-company pair at all and can only be predicted by company-level information.

Within-company pair concordance on the HONEST rows (9,815 (positive, negative) pairs inside
the 8 companies that have both labels), read against the pooled numbers computed with the
same score vector:

| readout | T | VA_nl | Δ |
|---|---:|---:|---:|
| pooled, HONEST (seed-ensemble dense vector) | .7744 | .7296 | +.0448 |
| **within-company, HONEST** | **.6777** | **.6461** | **+.0316** |
| within-company, FULL population (bank only) | — | .6583 | — |

Conditioning on the issuing organisation costs the dense model .097 and the bank .083, so
the residual survives at about 70 % of its pooled size — but both instruments lose a
quarter of their edge over chance, which is the honest size of the company floor here.

### 2.5 The position line (FREEZE ADDENDUM 4, observed covariates)

`press_verdict_position_line.json`. Release dates recovered for **85.5 %** of rows;
variables built are within-company release rank and rank fraction, company size, calendar
time (days since corpus start, year, month, ISO weekday — the embargo/timing convention the
brief names) and an id-rank control. **No position variable enters V, A, any judge prompt,
or the closure curve.**

| variable | alone-AUC, full 2,956 | alone-AUC, HONEST 605 | direction |
|---|---:|---:|---|
| **company_size** | **.696** | **.605** | **corpus-construction artifact / label-coupled** |
| within_company_rank | .595 | .499 | upstream-plausible but partial (order taken over the sampled population) |
| id_rank_pct | .484 | .535 | corpus-construction ordinal (control) |
| month / dow / year / days_since_start | .463–.498 | .446–.506 | upstream-plausible |
| joint position model (all 8) | .625 | — | — |
| **joint, artifact-free (6)** | **.644** | **.495** | — |
| calendar only (4) | .610 | — | — |
| within-company order only (2) | .667 | — | — |

**The load-bearing finding is that this cell's strongest position variable is an artifact of
how the population was built.** The population keeps *all* of a company's ≥3-outlet releases
and only a topic-matched sample of its zero-pickup ones, so a company's row count is a direct
function of its pickup rate — and label rate duly climbs across company-size quintiles
(.203 → .438 → .571 → .692 → .626). company_size must never be read as an upstream "company
prominence" effect; it is the sampling design showing through.

Net of that, ADDENDUM 4's prior is answered for this cell: an ordinal channel **does** exist
on the full population (within-company order alone .667) but **vanishes on the honest rows**
(artifact-free joint .495). And discounting position *raises* the residual, as in every other
cell in the programme: stratifying Δ on position deciles moves it +.0448 → **+.0539** (all
variables) / **+.0500** (artifact-free) / **+.0511** (stratified on year).

### 2.6 Swap baseline

Pair algebra on the HONEST population (91,506 (positive, negative) pairs; w₊ = .774 of them
ordered correctly by the dense model):

| quantity | round 0 |
|---|---:|
| C₊ = P(bank concordant \| dense correct) | **.8048** |
| C₋ = P(bank concordant \| dense wrong) | **.4711** |
| Spearman ρ(VA_nl, dense) | .5760 |

C₋ = .471 is the load-bearing baseline: on the 23 % of pairs the dense model orders
backwards the bank is right slightly *less* than half the time, so on this cell the bank and
the dense model are closer to agreeing-and-erring together than they are on peer or N&C.

---

## 3. ROUND 1

Sealed dual-track fleet at the freeze's full target: **P = 6 across 3 families**
(Claude Opus + Claude Sonnet as sealed subagents, gpt-5.6-luna ×2 via `codex exec`
effort=high in read-only scratch working directories, GLM-5.2 ×2 on the two live Lite
keys with `thinking` enabled, budget_tokens = 2048 / max_tokens = 32000).
**All 12 sealed slots returned parseable, distinctly named sets: 0 parse failures,
0 retries, 0 key rotations, 150 proposals (90 A / 60 B).** Prompts were 169–172 KB with
12 distinct stable-hash slice orderings; the slice was the 60 highest
|dense percentile − scorecard percentile| rows inside M (median |rank gap| .476).

### 3.1 Instrument health

| | round 1 |
|---|---|
| fleet | P = 6 / 3 families, 12/12 slots ok |
| proposals | 150 (90 A / 60 B) |
| species (A / B) | 48 / 50 |
| Good-Turing M̂ (A / B) | **.433** [LOPO .347–.533] / **.717** [LOPO .70–.82] |
| cross-proposer recapture (A / B) | .188 / .140 |
| species named by ≥ 2 families (A / B) | 5 / 5 |
| routing | 14 A / 11 B, **6 MIXED** |
| misrouting | **1/25 = 4.0 %** |
| planted probes | **4/4 separated** |
| disputes → arbiter | 1 |
| anchors | pos 2.881 / neg 2.559 / scrambled 0.943; **coherent-vs-scrambled .9252 PASS**; pos-vs-neg **.6232** |
| collapse gate | **0 collapsed**; overall NA .039; 0 rows all-NA (interrupted-generation gate clear) |
| judge calls | 77,650 (2,956 rows + 150 anchors × 25 criteria), Gemma-4-31B offline batch |

The A/B asymmetry the N&C campaign found reproduces exactly: the quality space is
partially converged (M̂ = .43) while the spurious space is not (M̂ = .72). The pos-vs-neg
anchor AUC of **.623** is the highest in the programme for a first round, so the
instrument is reading these releases well — it is the *criteria* that will turn out not
to help.

**The arbitrated dispute.** The Track-A proposer offered *"Distinctive quotable voice
rather than interchangeable corporate template"* — an explicitly authored plea that
newsworthiness includes voice. The blind auditor called it incidental; the Opus arbiter
upheld the auditor and routed it to B as MIXED:

> the instruction disclaims content by fiat ("judge only manner of expression; a release
> with no facts at all can still score 10") and its zero anchor is an explicit list of
> boilerplate features, so a judge scoring it measures inverted boilerplate and house
> register rather than newsworthiness.

It then came back as the **strongest channel in the round's spurious map (.584)** — the
audit's call was load-bearing, and it is the same "score-the-form" failure mode the N&C
campaign caught four times.

### 3.2 The round-1 curve step — the criteria made the scorecard WORSE

T is quoted throughout as the **mean over the three dense seeds of the AUC** (the same
convention VA_nl uses): T_HONEST = **.7508**, T_MONITOR = **.7360**. (`readout.py`'s
internal `Delta_beyond_*` fields instead use the seed-ENSEMBLE probability vector, whose
AUC is .7744 on HONEST because averaging denoises; those fields therefore run ~.024 high
and are not quoted.)

| round | bank feats | VA_nl HONEST | gain HONEST | VA_nl MONITOR | gain MONITOR | **Δ_beyond HONEST** | Δ_beyond MONITOR |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | 126 | .7296 | — | .7446 | — | **+.0212** | −.0086 |
| r1 | 140 | .7225 | **−.0071** | .7336 | **−.0110** | **+.0283** | +.0024 |

Gain CIs (company-level paired bootstrap): HONEST **[−.0514, +.0295]**, P(>0) = .355;
MONITOR **[−.0959, +.0120]**, P(>0) = .108. **Fourteen audited quality criteria from a
six-proposer, three-family fleet moved the scorecard down, not up, on both readouts.**
Under the frozen *signed* saturation reading (`gain < ε`, the reading the N&C campaign
resolved in favour of the prereg text) **round 1 is sub-ε: saturation flag 1 of 2.**

Per-criterion, the round-1 Track-A alone-AUCs top out at **.548** (*Serious dramatic
incident with significant stakes*, *Breadth of public or consumer impact*) and
**.547** (*Strong time-sensitive news peg*) — i.e. the fleet's best new quality criterion
is about as predictive as the incoming bank's best rubric, and neither clears .55.

### 3.3 Swap readout

| | r0 | r1 | Δ |
|---|---:|---:|---:|
| C₊ = P(bank concordant \| dense correct) | .8048 | .8045 | **−.0003** |
| C₋ = P(bank concordant \| dense wrong) | .4711 | .4407 | **−.0304** |
| Spearman ρ(VA_nl, dense) | .5760 | .5956 | +.0196 |

**No swap signature** by the strict test (C₊ did not rise). But the shape is worth
naming: ρ with the dense model rose .020 while C₋ fell .030 — the round-1 criteria moved
the bank *toward* the dense model's ordering and the movement was concentrated on the
pairs the dense model gets **wrong**. That is the swap mechanism with the C₊ leg missing,
and it is the cleanest available explanation for a negative closure gain.

### 3.4 The round-1 spurious map and the discount

| alone AUC (HONEST) | mixed | channel | conjectured upstream parent |
|---:|:--:|---|---|
| **.584** | YES | Distinctive quotable voice vs interchangeable corporate template | surface-only (arbiter-routed from A) |
| **.552** | YES | Reactive rebuttal framing naming an external antagonist | a reactive trigger on the production side |
| **.547** | YES | Presupposes an earlier installment in a series | **position in the run of releases this organisation issues** |
| .518 | YES | Forwarded / republished internal communication markers | republishing distribution practice |
| .505 | | Capture integrity: extraction residue, mid-sentence truncation | the corpus scraping and extraction pipeline |
| .483 | | Character-encoding corruption (mojibake) density | encoding of the source page |
| .472 | | Calendar and embargo markers | submission timing and news-cycle convention |
| .461 | | Consumer marketing call-to-action | product-marketing practice |
| .446 | | Recurring data-bulletin format | recurring distribution/editorial deal |
| **.438** | YES | Quantitative data & index dump | compliance and reporting requirements |
| **.414** | YES | Formal wire-service dateline and ticker format | the distribution tier the release went out on |

Nothing was dropped by the degeneracy screen. Two structural points. **(a) The fleet found
the position channel unprompted-but-prompted**: ADDENDUM 4's MODE 3 produced *"presupposes
an earlier installment in a series"* at .547, the third-strongest channel, tagged to
position in the organisation's release stream — the first time a proposer fleet in this
programme has named an ordinal channel, which is exactly what Addendum 4 was written to
fix. **(b) The two strongest anti-predictive channels are distribution-tier fingerprints**
(wire-service dateline/ticker .414, quantitative index dump .438): a release that looks
like it went out on a paid newswire in the standard financial-data format is markedly
*less* likely to be written up by three outlets.

| readout, HONEST n = 605 | ALL B (11 ch.) | STRICT, mixed dropped (5 ch.) |
|---|---:|---:|
| spurious-alone AUC (HistGB / linear) | **.6037 / .6106** | .5302 / .4628 |
| pooled Δ (ensemble-T basis, as `readout.py` computes it) | +.0519 | +.0519 |
| decile-stratified Δ_adj | **+.0519** | +.0526 |
| **stacked: dense increment over B alone** | **+.1592** [+.103, +.234] | — |
| stacked: bank increment over B alone | +.1075 | — |
| **stacked: dense increment over B + bank** | **+.0576** | — |

Spurious-alone is **.604**, below the freeze's .65 matched-sampling trigger, so the
decile estimator is primary and matched sampling is not run this round. The discount is
**null**: Δ_adj = +.0519 against a pooled +.0519, i.e. eleven named nuisance channels
explain none of the residual — the same negative result N&C got three rounds running.

---

## 4. ROUND 2 — decomposition-first

FREEZE ADDENDUM 3 applies from round 2 on this cell: round 1 produced six MIXED Track-B
channels, which are the object Addendum 3 names (unlike the humor batch, which had to
pick bank criteria off a SHAP screen because no mined MIXED channels existed yet). Parents
were ranked by |alone-AUC − .5| on **FIT+MINE only** — a design decision never reads
MONITOR — and the top three were decomposed by a sealed Opus decomposer into one
candidate-real and one surface component each:

| parent (r1) | FIT+MINE alone-AUC | candidate-real component | surface component |
|---|---:|---|---|
| Distinctive quotable voice vs interchangeable corporate template | .539 | On-record candour with concrete disclosure in attributed statements | Extent of departure from templated corporate register |
| Formal wire-service dateline and ticker format | .466 | Checkable specifics of a dated, concrete event | Extent of newswire dateline, ticker and boilerplate formatting |
| Presupposes an earlier installment in a series | .533 | Substantive change of state beyond a restated standing position | Extent of un-recapped prior-installment presupposition |

The three parents are **retired from the readouts and recorded, not deleted**
(`press_verdict_retired_channels.json`). Per Addendum 3 the components count toward the
round's budgets, so round 2's scored set is **12 fleet-A species + 3 candidate-real
components + 7 fleet-B species + 3 surface components = 15 A / 10 B**.

### 4.1 Instrument health

| | round 2 |
|---|---|
| fleet | P = 6 / 3 families, **12/12 slots ok**, 150 proposals (90 A / 60 B), 0 parse failures |
| species (A / B) | 66 / 40 |
| Good-Turing M̂ (A / B) | **.589** / **.450** |
| cross-proposer recapture (A / B) | .18 / **.33** |
| routing | 13 A / 12 B, **9 MIXED** |
| misrouting | **2/25 = 8.0 %** |
| planted probes | **4/4 separated** |
| disputes → arbiter | **2, both → B, both MIXED, both upholding the blind auditor** |

**The two tracks' missing masses cross over between rounds 1 and 2** — Track A rises
.433 → .589 while Track B falls .717 → .450 and its recapture more than doubles
(.14 → .33). The spurious space is converging as the quality space *diverges*: by round 2
the fleet is still producing new "quality" species at a high rate (66 species from 90
proposals) while, as §4.2 shows, buying nothing with them. That combination — high species
turnover, no AUC — is the signature peer curation showed at its cap, and it points the
same way: what is left here is not unnamed criteria.

**Both arbitrated disputes are the same failure mode as round 1's**, which makes it a
replicated finding rather than an anecdote. *Tangible public calls to action together with
exact logistics* → B (mixed): "rewards consumer call-to-action mechanics (dates, pricing,
URLs), a marker of promotional launch copy rather than evidence a journalist would write
it up". *Distinctive executive voice with personality* → B (mixed): "scores register and
authorial fingerprint … a tell about who produced the release rather than its news value".
**Three rounds of dispute, three times the fleet proposed authorial voice or promotional
mechanics as newsworthiness, three times the blind auditor caught it and the frontier
arbiter upheld the catch.** On this cell the miner's characteristic error is to mistake
*how the release is written* for *whether it is news*.

All three **candidate-real** components were routed to A and all three **surface**
components to B by the blind auditor, with no dispute on any of the six — the
decomposition pass separated cleanly on this cell, unlike peer revealed where the surface
component of the trend family came back *stronger* than its parent.

Scoring health, round 2: 77,650 judge calls; anchors pos 2.774 / neg 2.487 / scrambled
1.290; **coherent-vs-scrambled .8829 PASS**; pos-vs-neg .5956; NA .013; 0 rows all-NA;
**one collapse — `B05 Corpus Duplication / Repetition`**, a Track-B channel the corpus
simply does not vary on. As in the N&C campaign, a collapsed channel tests the auditor but
cannot test the discount, so round 2 contributes 13 A and 11 usable B channels.

### 4.2 THE CURVE

T is the **mean over the three dense seeds of the AUC**: T_HONEST = **.7508**,
T_MONITOR = **.7360**. The bank is the primary (Layer-1 constant-0.5) A block plus every
A-routed criterion accepted so far. `press_verdict_curve.json`.

| round | bank feats | VA_nl HONEST | gain HONEST | VA_nl MONITOR | gain MONITOR | **Δ_beyond HONEST** | Δ_beyond MONITOR |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | 126 | .7296 | — | .7446 | — | **+.0212** | −.0086 |
| r1 | 140 | .7225 | **−.0071** | .7336 | −.0110 | **+.0283** | +.0024 |
| **r2** | **153** | **.7415** | **+.0190** | .7346 | +.0009 | **+.0093** | +.0014 |

Gain CIs (company-cluster paired bootstrap): r1 HONEST [−.0514, +.0295] P(>0) = .355;
**r2 HONEST [+.0018, +.0445] P(>0) = .989**. On MONITOR: r1 [−.0959, +.0120] P = .108;
r2 [−.0340, +.0475] P = .626.

**Signed sub-ε flags on the primary statistic: [YES, no] → trailing run 0. Saturation is
NOT declared; the campaign is live at the cap boundary, two rounds in.**

Two things are worth stating plainly.

**(i) The stopping rule is non-monotone here too — for the third independent time.** Round
1 was sub-ε (indeed negative) and round 2 then delivered the largest closure step in the
campaign (+.0190, CI excluding zero). Had round 1's flag been allowed to run with a second
sub-ε round, the rule would have declared a taste bound of +.0283 that round 2 pushed down
to +.0093. Peer curation showed exactly this (its rule fired retrospectively at round 2 and
four more rounds closed a further .011); N&C showed the mirror. **The frozen rule is a
stopping heuristic, not a saturation proof**, and the cap must always be reported with it.

**(ii) After two rounds the residual is +.0093 and the best bank so far is round 2.** With
a leave-one-company-out jackknife SE of ~.025 on this cell (§2.3), a residual of one AUC
point is not distinguishable from zero by any margin. **Quote the best bank so far:
Δ_beyond = +.0093 on HONEST (n = 605), 153 features, round 2.**

### 4.3 Swap readout

| step | ΔC₊ | ΔC₋ | Δρ | swap signature? |
|---|---:|---:|---:|---|
| r0 → r1 | −.0003 | **−.0304** | +.0196 | no (C₊ flat, C₋ down — the swap shape with the C₊ leg missing) |
| r1 → r2 | **+.0208** | **+.0128** | +.0224 | **no — both rose** |

Round 2 raised concordance on the pairs the dense model gets right *and* on the pairs it
gets wrong, which is the signature of genuine independent signal rather than dense
imitation. So the one round that closed anything is also the round that closed it
honestly.

### 4.4 What the round-2 criteria actually are — and a sign-contradicting result

The strongest new Track-A criteria by |alone-AUC − .5| on HONEST:

| alone AUC | criterion | direction |
|---:|---|---|
| **.389** | **Independent evidentiary grounding** | **strongly ANTI-predictive** |
| **.401** | **Primary-source artifact released with the announcement** | **strongly ANTI-predictive** |
| .591 | Accountable disclosure | + |
| .580 | On-record candour with concrete disclosure in attributed statements *(decomposition candidate-real component)* | + |
| .580 | Non-routine change | + |
| .443 | Editorially usable completeness | − |

**The two strongest quality criteria the fleet has produced in two rounds both point the
wrong way.** A press release that supplies independent evidentiary grounding (.389) or
ships a primary-source artifact alongside the announcement (.401) is markedly *less* likely
to be written up by three or more outlets — each about as far from chance as the best
positive criterion, and further from chance than anything in the incoming 40-rubric bank.
This is the sign-contradicting case the prereg parked as report-only (open question (a) of
AMENDMENT 2), and on this cell it is not a curiosity but the substantive result: the
articulated-quality direction and the editorial-pickup direction are partly **opposed**.
Read with §2.2 — where the bank's whole contribution turned out to be a genre fingerprint —
the picture is consistent: what gets picked up is a *kind of story*, and the marks of
careful evidentiary practice are, if anything, negatively associated with it.

### 4.5 Round-2 spurious map (the round's own 11 usable channels)

| alone AUC (HONEST) | mixed | channel | conjectured upstream parent |
|---:|:--:|---|---|
| **.595** | YES | Extent of departure from templated corporate register | *surface carrier of the r1 "distinctive voice" parent* |
| **.579** | YES | Extent of un-recapped prior-installment presupposition | *surface carrier of the r1 position parent* |
| .566 | YES | Named-individual personal-voice signature on the dateline | who inside the organisation drafted it |
| .563 | YES | Defensive legal posturing | crisis management / legal involvement |
| .557 | YES | Distinctive executive voice with personality | surface-only (arbiter-routed from A) |
| .513 | YES | Internal-comms repackaging | PR strategy, bypassing wires |
| .490 | YES | Consumer purchase-mechanics and promotional offer detail | which internal function produced it |
| .488 | YES | Tangible public calls to action with exact logistics | surface-only (arbiter-routed from A) |
| .461 | | Encoding or extraction damage | content-ingest / export pipeline |
| **.413** | YES | Formal wire-service dateline / distribution conventions | **distribution tier: paid newswire vs own newsroom** |
| **.403** | | **Extent of newswire dateline, ticker and boilerplate formatting** | *surface carrier of the r1 wire parent* |
| — | | *Corpus duplication / repetition* | **collapsed at the score gate** |

**Decomposition worked, and it purified in both directions.** Every one of the three
surface components came back *stronger than its parent*: templated-register departure
.595 vs parent .539, un-recapped presupposition .579 vs .547, newswire formatting .403
(|AUC−.5| = .097) vs .414 (.086). The candidate-real components also carried signal
(on-record candour .580 in Track A). So on this cell the MIXED parents really were
mixtures, and splitting them raised the resolution of both halves — the outcome
ADDENDUM 3 was written to produce, and a cleaner result than peer revealed got, where the
surface half absorbed essentially all of the parent.

**The distribution-tier family replicates across both rounds and is the map's most stable
finding**: the r1 wire-dateline channel (.414) and its r2 decomposed surface carrier
(.403) and the r2 re-naming (.413) all say the same thing in the same direction — a
release that looks like it went out in standard paid-newswire livery is *less* likely to
be picked up by three outlets.

| discount readout, HONEST n = 605 | round 1 (11 ch.) | round 2 (12 ch.) |
|---|---:|---:|
| spurious-alone AUC (HistGB, HONEST) | .6037 | **.6327** |
| spurious-alone AUC (linear, MONITOR) | .6106 | .6739 |
| decile-stratified Δ_adj | +.0519 | **+.0394** (pooled, same basis: +.0329) |
| **stacked: dense increment over B alone** | +.1592 [+.103, +.234] | **+.1335 [+.073, +.242]**, P(>0) = 1.00 |
| stacked: bank increment over B alone | +.1075 | +.0992 |
| **stacked: dense increment over B + bank** | +.0576 | **+.0440** |

Spurious-alone is still under the freeze's .65 matched-sampling trigger on the HONEST
HistGB readout (.633), so the decile estimator remains primary. As in every other cell,
**the discount does not remove the residual — it slightly raises it** (+.0394 adjusted vs
+.0329 pooled on the same basis).

### 4.6 Missing mass, both tracks

| round | track | S_obs | f1 | f2 | **M̂** | LOPO jackknife | recapture | species ≥ 2 families |
|---|---|---:|---:|---:|---:|---|---:|---:|
| r1 | A | 48 | 39 | 5 | **.433** | [.347, .533] | .188 | 5 |
| r1 | B | 50 | 43 | 5 | **.717** | [.700, .820] | .140 | 5 |
| r2 | A | 66 | 53 | 9 | **.589** | [.533, .667] | .182 | 10 |
| r2 | B | 40 | 27 | 8 | **.450** | [.420, .580] | **.325** | 7 |

**The two tracks cross over.** Track B converges hard (M̂ .717 → .450, recapture
.14 → .33, families-in-agreement 5 → 7): six independent proposers are increasingly naming
the *same* nuisance channels, which is why the spurious map replicates across rounds.
Track A does the opposite (M̂ .433 → .589, S_obs 48 → 66): the fleet keeps minting new
"quality" species — 66 species from 90 proposals in round 2 — while the closure gain over
two rounds nets to **+.0119** of AUC.

That combination is diagnostic, and peer curation is the precedent: **high species
turnover with no AUC is evidence that what remains is not unnamed criteria.** At P = 6
there is still a ~.59 chance a seventh proposer names an unseen Track-A species, so the
claim remains "not discoverable by this miner at this fleet size", never "not
articulable" — but on this cell the miner is not running out of *ideas*, it is running out
of ideas *that predict anything*.

### 4.7 Cumulative discount at 22 channels

`press_verdict_r2_cumulative_discount.json`. Accumulated nuisance set after two rounds:
**22 named channels, 15 of them MIXED, 3 MIXED parents retired by the decomposition pass**
(recorded in `press_verdict_retired_channels.json`, never deleted). All quantities in this
table are computed on the seed-ENSEMBLE dense vector, which is the basis `readout.py` and
`discount_cumulative.py` share, so the pooled Δ shown is +.0329 rather than the
declared-convention +.0093; read the columns against each other, not against §4.2.

| readout, HONEST n = 605 | ALL B (22 ch.) | STRICT, mixed dropped (7 ch.) |
|---|---:|---:|
| spurious-alone AUC (HistGB / linear) | **.6498 / .6315** | .5667 / — |
| pooled Δ | +.0329 | +.0329 |
| **decile-stratified Δ_adj** (primary; alone-AUC .650 sits just under the .65 trigger) | **+.0502** | +.0397 |
| matched-sampling Δ_adj (secondary, caliper .02) | +.0446 (224 pairs) | +.0569 (246 pairs) |
| **stacked: dense increment over B + bank** | **+.0424** [+.0153, +.0792], P(>0) = .9985 | +.0451 [+.0197, +.0798], P(>0) = .999 |
| stacked: dense increment over B alone | +.1201 | — |
| stacked: bank increment over B alone | +.0873 | — |

MONITOR agrees in direction: pooled +.0253, decile +.0299, matched +.0673.

**Every discount raises the residual, at both ends of the MIXED band and under all three
estimators** — the fourth cell in the programme to give that answer. The stratification-free
stacked increment is the number to quote when the nuisance set is this large and this
MIXED-heavy: **the dense model adds +.042 [+.015, +.079] over the 22-channel joint nuisance
model AND the 153-feature bank together**, and the ALL/STRICT band there is only .0027 wide
(against .0105 on deciles and .0123 on matched sampling).

Note what this does *not* say. The joint nuisance model reaches **.650** — above the
articulated bank's own .7415? No: below it. But it is above every single quality criterion
the fleet produced, and it is built from 22 channels explicitly defined as *not*
newsworthiness. Together with §2.2 (the bank's contribution is a genre mask) and §4.4 (its
two strongest new criteria anti-predict), the consistent reading of this cell is that
**editorial pickup is only weakly a function of anything an expert would call release
quality** — which is also why there is so little residual for the dense model to hold.

---

## 5. WHERE THE CAMPAIGN STANDS — best bank so far, and the saturation verdict

### 5.1 The quotable state

Best bank so far is **round 2** (153 features). Refit alone and jackknifed
(`press_verdict_r2_jack_state.json`, `jack_state.py`):

| readout | n | T | VA_nl | **Δ_beyond** |
|---|---:|---:|---:|---:|
| **HONEST (dense-held-out)** | 605 | .7508 | .7415 | **+.0093** |
| MONITOR | 312 | .7360 | .7346 | **+.0014** |
| eval only (dense chain **selected on eval** — contaminated half) | 288 | — | — | +.0223 |
| **test only (SELECTION-FREE half)** | 317 | — | — | **+.0007** |

Leave-one-company-out jackknife over the 45 dense-held-out companies: mean **+.0094**,
**SE .0313**, range **[−.0102, +.0256]**, pseudo-CI **[−.052, +.071]**. One of the 45
leave-one-out values is negative; the most influential company is **tesla** (84 rows,
89 % positive), whose removal flips Δ to **−.0102** — the bank beating the dense model.

**Read those two rows together.** On the selection-free half of the honest population the
mined 153-feature scorecard matches the dense model to within **seven ten-thousandths of an
AUC point**, and the pooled residual of +.0093 is a third of its own jackknife SE.

Swap at the round-2 bank: C₊ = .8253, C₋ = .4535, ρ = .618. Within-company pair
concordance at the round-2 bank (9,815 pairs, ensemble basis): T .6777 vs VA_nl .6500,
Δ = +.0277 against a pooled ensemble-basis +.0329 — the residual survives conditioning on
the issuing organisation at ~84 % of its pooled size, but both numbers are inside the
jackknife's noise.

### 5.2 Saturation verdict

**NOT saturated by the frozen rule, and the campaign is live.** Signed sub-ε flags are
[YES (r1), no (r2)] → trailing run 0 of the required 2; the cap is 5 and two rounds have
run. The honest verdict has three parts:

1. **The rule has not fired, but the residual has effectively closed anyway.** Δ_beyond
   went +.0212 (r0) → +.0283 (r1) → **+.0093 (r2)**, and +.0007 on the selection-free
   half. There is no longer a residual of a size this cell's width could resolve.
2. **The rule is non-monotone here, for the third independent time in the programme.**
   Round 1's negative gain would, with one more sub-ε round, have declared a bound of
   +.0283 that round 2 halved-and-then-some. Any plateau quoted off this rule must carry
   the cap and the full curve.
3. **Remaining mass is real but is not the binding constraint.** Track-A M̂ = .589 at
   P = 6, so a seventh proposer would very likely name unseen quality species. The
   odds-form bound on what further proposal buys, taking the last realised HONEST gain
   (+.0190) and M̂ = .589, is **R̂ ≈ +.019 × .589/(1−.589) ≈ +.027** — larger than the
   residual itself, i.e. **the estimator's own answer is that mining could in principle
   overshoot zero**, which is another way of saying this cell has nothing left to close.
   The species form is never quoted (f2 = 9).

### 5.3 What this cell is, in one paragraph

Press/journalism editorial pickup behaves like **peer curation, not peer revealed**. The
articulated scorecard reaches the dense model; there is essentially no taste residual. But
unlike peer curation — where the reason was outcome noise against a genuinely
quality-shaped construct — here the reason is that **the construct is only weakly about
release quality at all**. Three independent lines say so: the incoming 40-rubric bank has
zero columns above .527 alone and contributes through its **applicability mask** (a genre
fingerprint, .7322 alone, worth all but .0014 of the whole bank); the fleet's two strongest
new criteria in round 2 **anti-predict** pickup (independent evidentiary grounding .389,
primary-source artifact .401); and the most stable channel in the spurious map across both
rounds is **distribution tier** (paid-newswire livery, .403–.414, anti-predictive).
Editorial pickup is largely a question of *what kind of story this is and how it reached the
newsroom*, and both the scorecard and the dense model are reading that, not craft.

---

## 6. Caveats that travel with every number here

1. **Pre-GEPA.** No criterion phrasing was GEPA-iterated. All alone-AUCs and closure gains
   are lower bounds on what the same *concepts* could achieve with tuned phrasing, so the
   residual is an upper bound on taste.
2. **The dispatched Δ_beyond +.0486 must not be re-quoted.** It compares T on 288 dense-eval
   rows against VA_nl pooled over all 2,956. Same-rows it is **+.0066** (Layer-1 fitting
   protocol) or **+.0212** (closure protocol) at round 0, and **+.0093** at the round-2 bank.
3. **The residual's sign depends on an imputation convention** (§1, DECISION 1). Under the
   closure-standard median impute the round-0 residual is +.0499 rather than +.0212; the
   campaign runs on the Layer-1 constant-0.5 bank because that is the matrix this cell's
   Layer 1 actually has, and because it is the conservative choice.
4. **Width is the binding constraint.** The HONEST population is 605 rows in **45**
   companies, **9 of which carry 93.9 %** of the rows; the jackknife SE on Δ is .025–.031.
   No Δ of one to five AUC points is resolvable on this cell, in either direction.
5. **MONITOR is 42 % positive** (against .501 on HONEST) and holds 23 companies; it is
   reported every round but the primary statistic is HONEST (DECISION 2).
6. **The dense chain selected on EVAL**, so eval-only readouts are the contaminated half
   here and test-only is the clean one — the mirror of the N&C cell. Both are reported.
7. **company_size is label-coupled by construction** (§2.5) and must never be read as an
   upstream prominence effect.
8. **sklearn is pinned to 1.9.0.** `GroupKFold` fold assignments move across releases on
   this cell (documented LANDMINE); every script asserts the version.
9. **Closure-split levels are protocol-specific** and not comparable to Layer-1
   Δ_beyond; only round-over-round changes and the same-rows honest levels are quotable.
10. **Two rounds, not five.** The campaign is incomplete against the frozen cap; §5.2's
    verdict is "residual closed, rule not fired", not "saturated".

---

## 7. Ledger and artifacts

| item | count |
|---|---:|
| Gemma-4-31B judge calls (rounds 1–2) | **155,300** (2 × 77,650) |
| sealed proposer slots | **24** (6 proposers × 2 tracks × 2 rounds), **0 parse failures, 0 retries** |
| decomposition components authored + scored | 6 (3 MIXED parents retired) |
| blind auditors | 2 (fresh per round); planted probes **8/8 separated** |
| arbitrated disputes | **3, all → B, all MIXED, all upholding the blind auditor** |
| blind census judges | 2 (agreement .975, anchors 4/4 each) |
| GPU | sk3 GPUs 3 and 1 via `gpu_stack.sh`; co-tenants recorded by PID in the shared ledger and never signalled; every claim released |
| CPU | laptop only, pinned scikit-learn 1.9.0 scratch venv, for every fit, slice, readout and discount |

**Code (new):** `cells.py`, `fetch_dense.py`, `round0.py`, `census.py`,
`applicability_probe.py`, `company_structure.py`, `position_line.py`, `curve.py`,
`jack_state.py`, `after_scoring.sh`, `gpu_stack.sh`, plus the press-specific probe pool in
`audit.py`, the MODE 3/MODE 4 Track-B brief in `harness_maps.py` and the press persona in
`score_gemma_maps.py`. Everything else copied unchanged from `closure/peer_revealed/` and
`closure/maps_hw_si/`.

**Round-0 artifacts:** `press_verdict_r0_context.json`, `press_verdict_r0_preds.npz`,
`press_verdict_samerows_anchor.json`, `census{,_stage1,_blind_packet}.json`,
`census_verdicts_judge{A,B}.json`, `applicability_probe.json`, `company_structure.json`,
`press_verdict_position_line.json`, `press_verdict_position.npz`,
`press_verdict_splits.json`, `press_verdict_population.csv`,
`press_dense_preds_3seed.csv` + `.report.json`.

**Round artifacts:** `press_verdict_r{1,2}_{slice,proposals_fleet,species,audit_prompt,
audit_key,audit_verdicts,arbiter,routing_final,scores.npz,score_report,results}.json`,
`press_verdict_r2_{newparents,parents_used,cumulative_discount,jack_state}.json`,
`press_verdict_retired_channels.json`, `press_verdict_curve.json`, logs
`r{1,2}_stage1.log`, `codex_r{1,2}.log`, `glm_r{1,2}.log`, `readout_r{1,2}.log`,
`cumdisc_r2.log`, `jack_r2.log`.

**Sealed prompts and raw proposer/auditor/arbiter/decomposer outputs:** session scratchpad
`press_verdict/press_verdict_r{1,2}/`. **sk3 mirror:**
`/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/press_verdict/`.

---

## 8. STRICT LIST — the terminal line for this cell

> **Press / journalism verdict (editorial pickup, ≥3 tracked outlets), Layer-3 closure,
> rounds 0–2 of a capped 5, pre-GEPA.** Same-rows honest T = **.7508** (n = 605,
> 45 companies; mean over 3 dense seeds of the AUC). Best bank so far = round 2,
> 153 features, VA_nl = **.7415** → **Δ_beyond = +.0093**, jackknife SE **.031** over
> 45 held-out companies (CI [−.052, +.071]), and **+.0007 on the selection-free test half**.
> The dispatched +.0486 was a cross-population comparison and is **retracted**: same-rows it
> is +.0066 (Layer-1 protocol) / +.0212 (closure protocol) before mining.
> **Verdict: the residual is closed at this cell's resolution; the frozen stopping rule has
> not fired (signed flags [YES, no], trailing run 0 of 2, cap 5).** Remaining Track-A mass
> M̂ = .589 at P = 6 / 3 families; Track-B M̂ = .450 and falling. Discounting the 22-channel
> nuisance set raises rather than lowers the residual under all three estimators
> (stacked dense increment over B + bank **+.042 [+.015, +.079]**).
> **Structural caveat that governs the cell:** the incoming 40-rubric A bank has no column
> above .527 alone and its entire contribution is the **applicability mask** (.7322 alone;
> swapping all 40 judged levels for the 38 mask bits costs .0014), so this cell's "articulated
> scorecard" is largely a genre detector — and the fleet's two strongest mined criteria
> (independent evidentiary grounding .389, primary-source artifact .401) **anti-predict**
> pickup.
