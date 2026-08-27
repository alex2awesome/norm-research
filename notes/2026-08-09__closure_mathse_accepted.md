# Layer-3 articulation closure — math.SE ACCEPTED-VERDICT cell (the asker's own choice)

Cell: math.StackExchange answers, **y = 1 iff the person who ASKED the question marked
this answer as the accepted one**, 0 = a non-accepted answer on the same question.
Un-binarised v2 rebuild — NOT the old binarised "accepted AND score ≥ 3 vs score ≤ 0,
Qwen-scored A" cell, whose published V .565 / VA .673 / T .794 are a different
population, a different y and a different judge and are **never** differenced here.

Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + addenda 1–4.
Queue: `notes/2026-08-09__full_sweep_queue.md`, LANE B first cell, GPU 6.
Campaign dir: `methods/taste_decomposition/closure/mathse_accepted/`.
Sibling cell (SAME rows, SAME A/V matrix, different y):
`notes/2026-08-10__closure_mathse_vote.md`. **The two y's are never merged and never
differenced**; the vote cell's numbers appear below only as a named contrast.

## Terms, spelled out on first mention (standing rule)

| term | what it means here |
|---|---|
| **V** | the 28 deterministic lint features (`v_*`) computed on the answer body |
| **A** | the 32-criterion Gemma-4-31B-judged rubric bank (GEPA-phrased), shared with the vote cell |
| **VA_lin / VA_nl** | the articulated instrument: V+A fit linearly / by gradient boosting (HistGradientBoosting, frozen grid), grouped out-of-fold; VA_nl = mean over fit seeds {0,1,2} |
| **T** | the dense arm: Llama-3.1-8B LoRA reward model reading raw text; T = MEAN OVER DENSE SEEDS of the held-out AUC, never the AUC of the seed-averaged prediction |
| **Δ_beyond** | T − VA_nl: the part of the asker's choice the articulated bank does not reach |
| **Δ_r** | the closure curve: Δ after r rounds of active mining |
| **FIT+MINE / MONITOR** | the closure split; MONITOR lives inside the dense-held-out rows and is never read by any proposer |
| **M (mining slice)** | the dense-held-out half of FIT+MINE — dense scores there are honest |
| **HONEST** | M ∪ MONITOR = every dense-held-out row (2,600 rows). Identical row set to the master ledger's **E** |
| **Track A / Track B** | the dual design: A proposes quality-relevant criteria that could close the gap; B proposes suspected-SPURIOUS predictive channels used only to DISCOUNT |
| **MIXED channel** | a Track-B channel whose conjectured upstream parent plausibly causes real quality too; decomposed (FREEZE ADDENDUM 3) rather than routed to one side |
| **alone-AUC** | a single criterion's held-out AUC on its own |
| **swap pair (C₊, C₋)** | P(bank orders a discordant pair correctly \| dense does) and \| dense does not) |
| **missing mass** | fleet-based Good–Turing estimate of the criterion species the miner has not yet found |

## 0. Why this cell is in the campaign at all — recorded BEFORE any closure number

The 2026-08-06 FREEZE roster routes cells with matched Δ_beyond ≤ .02 to the
**map-focused dual track** (Track-B emphasis, Track A still run) rather than excluding
them. This cell's dispatched Δ_beyond is **−.0001** (Layer-1 ledger: T .6319 = dense
EVAL AUC at seed 42, VA_nl .6320 = pooled GroupKFold OOF over all 13,001 rows). So it
enters as a **map-focused cell by the frozen roster** — a decision fixed before any
closure statistic existed, and recorded in `round0.py`'s `gate_rule` field.

There is no seed gate to wait on: dense seeds {42, 1, 2} were all trained and scored
before this campaign opened. At three seeds the dispatch-convention numbers are

| reading | T | − VA_nl_mean3 (.6320) |
|---|---|---|
| EVAL mean (the ledger's own convention) | .6295 | **−.0025** |
| eval + test pooled | .6375 | +.0055 |
| TEST only (the selection-free half) | .6458 | +.0138 |

all below the .02 full-closure trigger, all consistent with "articulable at dispatch".

**But the master ledger's E-row full-grid arm reads T .6439 vs VA_nl .5737 — a +.070
gap on 2,600 rows.** Reconciling that against the −.0001 dispatch figure is exactly
what round 0 is for, and §2 does it.

## 1. Protocol adaptations, recorded before any mining slice was built

### 1.1 Alignment gate — PASSED EXACTLY

Registry landmine 2026-08-10: `*_va_nl_oof_*.npy` are keyed in **bank item_ids order**.
On this cell `isfinite(ys["accepted_verdict"])` is a **no-op** (13,001 of 13,001), so the
row order is `meta["item_ids"]` verbatim — unlike the sibling vote cell, which drops
1,372 median-tie rows and runs on a different 11,629-row order. `cells.load()` asserts
the no-op, and the gate refuses an array of the wrong length.

```
AUC(y, mathse_accepted_verdict_va_nl_oof_seed0.npy in assembled order) = 0.6317782528854139
published ledger nonlinear.VA["0"].auc                                  = 0.6317782528854139
abs diff = 0.0        GATE_PASS = true        shuffled counterfactual = .5041
```

`mathse_accepted/oof_alignment_gate.json`. Seed 0 only; the mean3 array reads .6347 and
is never the gate.

### 1.2 Splits — `mathse_accepted_splits.json`

Group key = `question_id`; no question straddles two dense splits (asserted).

| | rows | questions | pos rate |
|---|---|---|---|
| population | 13,001 | 4,960 | .3815 |
| dense train | 10,401 | 3,968 | — |
| dense held-out (**HONEST**) | 2,600 | 992 | — |
| **MONITOR** (⊂ held-out) | **1,326** | 500 | .3771 |
| **M** (mining slice, ⊂ held-out) | **1,274** | 492 | — |
| FIT+MINE | 11,675 | 4,460 | .3820 |

**Salt, recorded not silent.** The dense arm's own 80/10/10 is a stable sha256 hash on
the same key, so the closure cut hashes `sha256("mathse-accepted-closure|" + question_id)`.
Collision check: an unsalted cut would have put **506** of the 992 held-out questions in
MONITOR versus the salted **500** — the two salts do not collide, and the salt is kept
anyway because that could only be known after the fact.

**HONEST is the master ledger's E.** 2,600 rows / 992 questions / pos .3815 matches
`vat_fullgrid_mathse_accepted_verdict.json`'s `n_E` / `n_groups_E` / `pos_rate_E`
exactly. That is what makes §2's reconciliation a same-rows statement.

### 1.3 No imputation fork

A-block NA rate .2397 (the applicability gate firing). This cell's Layer-1 linear leg
uses `SimpleImputer(median, add_indicator)` inside each fold — the same median-impute
convention `closure_core.clean_fit` applies — so there is no const-0.5 fork here.
(The vote cell measured the `add_indicator` difference at −.0038 to Δ, i.e. the frozen
protocol if anything slightly understates the residual; same matrix, same conclusion,
not re-run.)

### 1.4 Readout tiers, declared in advance

* **TIER 1, GOVERNING — pooled AUC on MONITOR.** The tier the dispatch quantity lives
  on, so the only tier on which dispatch and curve are commensurable.
* **TIER 2, SECONDARY — n-weighted within-QUESTION AUC.** y is "the asker picked THIS
  answer among the answers to their own question", so this tier matches the
  y-definition exactly. Reported every round, never substituted.
* **TIER 3, DIAGNOSTIC — eval-only / test-only / HONEST same-rows level.**

Closure-split Δ levels are protocol-specific and NOT comparable to the Layer-1
Δ_beyond (prereg AMENDMENT 1); only round-over-round changes and the same-rows honest
level are quotable.

### 1.5 sklearn

Layer-1 ledger under scikit-learn 1.8.0; this campaign runs 1.7.2. GroupKFold fold
assignments move across releases, so Layer-1 LEVELS are not byte-reproducible here and
the campaign's own round-0 anchor is the baseline the curve is measured from. The
alignment gate is version-free (it reads a stored vector).

### 1.6 Reuse, declared

Machinery is the sibling vote cell's, adapted and diffed: `cells.py` (y key, no-op keep
mask, construct string), `oof_alignment_gate.py` (published constant), `build_splits.py`
(salt), `fetch_dense.py` (dense dir), `harness_maps.py` (MODE 3 wording — the vote
version said "already had accepted answers", which would leak this cell's outcome word,
and now says "already had several answers"). The 8192-context LaTeX rule and the
bank-matched HEAD-3000 + TAIL-2000 truncation in `score_gemma_maps.py` carry over
unchanged.

## 2. ROUND 0 — the honest residual, and the reconciliation the brief asked for

`mathse_accepted/mathse_accepted_r0_context.json`. T = mean over dense seeds {42,1,2}
of the AUC.

| tier / population | n | T | VA_nl | **Δ₀** |
|---|---|---|---|---|
| **TIER 1 GOVERNING — MONITOR** | 1,326 | .6443 | .6376 | **+.0067** |
| TIER 1 — HONEST (= E rows) | 2,600 | .6375 | .6147 | +.0229 |
| TIER 1 — mining slice M | 1,274 | .6308 | .5908 | +.0401 |
| TIER 1 — eval only | 1,300 | .6295 | .5995 | +.0300 |
| TIER 1 — test only (selection-free) | 1,300 | .6458 | .6297 | +.0161 |
| **TIER 2 — MONITOR, within-question** | 1,326 | .6241 | **.6673** | **−.0433** |
| TIER 2 — HONEST, within-question | 2,600 | .6287 | .6369 | −.0082 |

Question-cluster paired bootstrap of Δ₀ on MONITOR: **[−.0154, +.0498]**, p(Δ>0) = .84
(computed on the seed-mean score vector, so centred high relative to the T-based +.0067;
read for WIDTH). Leave-one-question-out jackknife over the 500 MONITOR questions:
SE **.0164**, range [+.0011, +.0086] — no single question drives it.

Per-seed T on MONITOR: .6514 / .6281 / .6534 (spread .025 — wider than the vote cell's
.012, and wider than Δ₀ itself).

### 2.1 The master-ledger +.070 does not survive a same-rows VA refit

On the **same 2,600 E rows**, three different VA fits give three different levels:

| VA fit | AUC on E / HONEST | Δ against T (seed 42, .6439) |
|---|---|---|
| master ledger full-grid OOF arm (`VA_nl_E_mean`) | **.5737** | +.0702 |
| master ledger full-fit-at-E reference (`VA_nl_fullfit_at_E`) | .6159 | +.0280 |
| **closure protocol** (refit on FIT+MINE, predict E) | **.6147** | **+.0292** |

The closure refit reproduces the full-grid arm's own full-fit reference to **.0012** and
sits **+.041 above** its OOF arm. So the honest same-rows residual on this cell is
**+.023 at three dense seeds** (+.029 against seed 42 alone), **not +.070**: two thirds
of the master-ledger gap is the E-arm's VA fit, not taste. On the governing MONITOR tier
the residual is **+.007**, and on the tier that matches the y-definition the articulated
bank is **ahead by .043**.

This is the number this campaign mines from. It is quoted with both designs named, per
the standing rule.

### 2.2 What the label is made of (composition audit) — `composition_audit.json`

| | |
|---|---|
| questions with **exactly one** accepted answer | **4,960 of 4,960 (100%)** — zero-accept and multi-accept questions are absent by construction |
| questions with exactly 2 answers | 3,394 (68.4%) — **52.2% of all rows** |
| 3 / 4 / 5+ answers | 746 / 496 / 324 questions |
| clean head-to-head questions (2 answers, 1 accept) | **3,394** (6,788 rows) |
| **first answer wins the clean head-to-head** | **.560** |

The structure is cleaner than the vote cell's: every question contributes exactly one
positive, there are no ties to drop (the vote cell discarded 1,372 median-tie rows), and
half the corpus is a two-way choice. In those two-way choices the asker picks the
earlier-arriving answer 56% of the time.

### 2.3 Swap baseline

| population | w₊ | **C₊** | **C₋** | ρ(bank, dense) |
|---|---|---|---|---|
| MONITOR | .655 | **.757** | **.410** | .523 |
| HONEST | .649 | .732 | .397 | .499 |

Where the dense model orders a discordant pair correctly the bank agrees 76% of the
time; where the dense model is wrong the bank is well below chance. The bank has
independent signal to lose.

## 3. ROUND 0 — concept census of the incoming bank

`census_stage1.json`, `census.json`.

| level | count |
|---|---|
| L0 criteria delivered | 32 |
| L1 distinct names | 32 |
| L2 after the frozen degeneracy screen (FIT+MINE only) | **32** |
| L3 value clusters at \|r\| ≥ .98 | **32** |
| L5 after blind pairwise adjudication (strict: both judges SAME) | **32** |

Max off-diagonal |Pearson r| = **.567**; fraction of column pairs at |r| ≥ .90 = **0**;
collapse L0→L5 = **0.0%**; zero merge edges under either the strict or the loose rule;
all four planted anchors passed for both judges.

**Judge reuse, declared.** The blind adjudication packet built here is **byte-identical**
to the vote cell's (49 shortlisted pairs + 4 anchors, same bank text, same register), so
per the standing reuse rule the two sealed Sonnet verdict files were reused rather than
re-dispatched; criterion-text identity is y-independent. The **hive-mind caveat travels**:
both judges were claude-sonnet-5 instances agreeing 49/49, which is weaker evidence of
distinctness than cross-family agreement would be. The value-side evidence
(max |r| = .567, zero pairs ≥ .90) is family-free and points the same way.

Per-criterion alone-AUC on FIT+MINE under **this** y: max **.567** ("Names the real
obstacle"), median .528, min .501, nothing below .50, 3 criteria ≥ .55 — the same
"many weak, near-independent, same-signed indicators" shape the vote cell showed, at
slightly lower amplitude.

Top V features by alone-AUC on this y: `v_inline_math_delims` .586, `v_word_count` .576,
`v_log_len` .574, `v_sentence_count` .565, `v_latex_cmd_count` .562 (`v_type_token_ratio`
.442 is again the only sub-.50 column). **Length and LaTeX are more predictive of the
asker's accept than of the crowd vote** (.586/.576 here vs .558/.542 there) — and they
are already bank columns, so Track B cannot discount them off Δ without discounting them
off VA too.

## 4. ROUND 0 — the answer-position covariate (FREEZE ADDENDUM 4). **Bigger here than on the vote cell.**

`position_line.json`, `position_matched.json`. Every variable here is an OBSERVED
covariate from the population file: never added to V or A, never judged by any LLM,
never fitted into anything that feeds the closure curve.

### 4.1 The first-answer advantage, side by side with the sibling y

| answer_position | n | **accept rate (this cell)** | vote-cell label rate |
|---|---|---|---|
| **0 (first)** | 4,933 | **.503** | .629 |
| 1 | 4,950 | .402 | .450 |
| 2 | 1,583 | **.207** | .411 |
| 3 | 827 | **.131** | .326 |
| 4 | 330 | .091 | .309 |
| 5 | 159 | .101 | .331 |
| 6 | 88 | .045 | .239 |
| 7 | 45 | **.000** | .195 |
| 8+ | 86 | .023 | .266 |

The brief's hypothesis is confirmed and then some: the accept y's arrival-order gradient
is **far steeper in the tail** than the vote y's. Crowd votes keep flowing to late
answers (.24–.33 at positions 4–8+); the asker essentially never comes back to accept one
(.00–.10). Accepting is a **one-shot, early, single-person act**; voting is a continuing
crowd process.

Alone-AUCs (single covariate, no text at all), on this cell:

| covariate | pooled AUC (full) | within-question | ρ with **T** | ρ with **VA_nl** |
|---|---|---|---|---|
| `is_first` | **.598** | .567 | **+.085** | −.046 |
| `answer_position` | .351 (= .649 inverted) | .404 (= .596 inv.) | −.133 | +.024 |
| `position_pct` | .420 (= .580 inv.) | .404 | −.070 | +.052 |
| `n_answers` | .350 (= .650 inv.) | **.5000** | −.156 | −.050 |
| `answer_year` | .496 | .457 | +.016 | +.046 |
| **joint position model** (grouped OOF, 6 vars) | **.6754** | .593 | **+.148** | **+.008** |

Two readings.

**(i) A no-text model beats the dense text model on the pooled tier.** The six-variable
ordinal reads **.6754** pooled (.6600 on HONEST) against the dense arm's T = .6375 on
HONEST / .6443 on MONITOR. On the vote cell the equivalent model read .654 and sat
*below* T. Here arrival order alone out-predicts an 8B model that read every word.

**(ii) `n_answers` is NOT structurally neutralised on this y.** It reads .350 pooled
(= .650 inverted) — questions that drew many answers are questions where any given
answer is less likely to be the accepted one, which is arithmetic, not a preference.
Within question it is exactly .5000, as it must be. **This is a real difference from the
vote cell**, whose within-question median split makes `n_answers` uninformative pooled
too. Consequence, fixed now: on this cell the pooled tier carries a large
denominator-arithmetic component that the within-question tier does not, which is part
of why TIER 2 and TIER 1 disagree in sign at round 0.

**(iii) The dense arm reads arrival order; the fitted bank does not.** ρ(joint position,
T) = **+.148** against ρ(joint position, VA_nl) = **+.008**. Same signature as the vote
cell, stronger.

### 4.2 How much of the residual arrival order accounts for — **effectively all of it**

Ensemble convention throughout (a stratified or matched readout needs ONE score vector,
so these use the seed-mean probability, whose pooled AUC is the ENSEMBLE figure and is
higher than T = mean of per-seed AUCs). **Standing rule for this cell, fixed here:** read
the shrinkage WITHIN the ensemble instrument; never difference an ensemble-based Δ_adj
against the T-based Δ₀.

**HONEST** (2,600 rows; pooled ensemble T .6491, VA .6147, Δ = +.0345):

| discount | T_adj | VA_adj | **Δ_adj** | share absorbed |
|---|---|---|---|---|
| none (pooled, ensemble) | .6491 | .6147 | **+.0345** | — |
| decile strata, within-question order vars only | .6413 | .6209 | +.0205 | 41% |
| decile strata, joint position model | .6385 | .6191 | +.0193 | 44% |
| strata on raw `answer_position` | .6364 | .6244 | +.0119 | 65% |
| exact strata on `is_first` | .6457 | .6222 | +.0235 | 32% |
| **matched sampling on the joint position score** (935 pairs) | .6439 | .6449 | **−.0011** | **103%** |

**MONITOR** (1,326 rows; pooled ensemble Δ = +.0176):

| discount | **Δ_adj** | share absorbed |
|---|---|---|
| none (pooled, ensemble) | **+.0176** | — |
| exact strata on `is_first` | +.0045 | 74% |
| **matched sampling on the joint position score** (463 pairs) | **−.0281** | >100% (bank ahead) |

Matched on arrival order, **the articulated bank is level with the dense model on HONEST
and ahead of it on MONITOR**. The vote cell's matched readout absorbed 55% on MONITOR and
25% on HONEST; here it absorbs the lot.

### 4.3 Matched sampling is the estimator of record from round 0

The freeze arms matched sampling once a nuisance channel's alone-AUC exceeds .65. The
joint position model reaches **.6754 pooled / .6600 on HONEST** before a single Track-B
channel has been proposed, so matched sampling governs this cell from round 0 onward —
same trigger as the vote cell, fired harder.

### 4.4 Stacked increment (FREEZE ADDENDUM 1, stratification-free), on HONEST

| stack | AUC | increment over position alone |
|---|---|---|
| position family alone | .6591 | — |
| position + **dense** | .7079 | **+.0488** |
| position + **bank** | .7029 | **+.0437** |

Conditional on everything the arrival-order family knows, the dense arm adds only
**+.0051 more** than the articulated bank does. The vote cell's equivalent gap was
**+.0219**. This is the same conclusion the matched readout reaches, by an estimator that
does not degenerate as the nuisance set grows: on the accept y there is very little left
for the dense model that the bank cannot also reach once arrival order is held fixed.

### 4.5 The contrast that makes this a localisation — length and LaTeX do the OPPOSITE

`length_stratification.json`, ensemble convention throughout.

| stratifier | Δ_adj on MONITOR | Δ_adj on HONEST | direction |
|---|---|---|---|
| (none, pooled) | +.0176 | +.0345 | — |
| `v_log_len` deciles | +.0159 | **+.0443** | Δ grows on HONEST |
| `v_latex_density` deciles | +.0206 | +.0358 | Δ grows |
| `v_n_display_math` deciles | **+.0250** | **+.0432** | Δ grows |
| length × LaTeX 4×4 | +.0228 | **+.0450** | Δ grows |
| joint arrival-order model | **+.0193** | **+.0193** | **Δ shrinks** |
| matched on arrival order | **−.0281** | **−.0011** | **Δ shrinks past zero** |

Same signature as the vote cell: stratifying on length or LaTeX costs the BANK more than
the dense model (those columns are *in* the bank), so it raises the residual; arrival
order is the only channel tested that lowers it. Two opposite signatures is what makes
the position result a localisation rather than a conditioning artifact.

### 4.6 Round-0 spurious map — measured, no proposer had spoken

| channel | alone-AUC on HONEST | kind |
|---|---|---|
| **joint arrival-order model** (fitted, 6 vars) | **.660** | OBSERVED covariate |
| `n_answers` | .356 (= .644 inverted) | OBSERVED covariate (denominator arithmetic; exactly .500 within question) |
| `answer_position` | .371 (= .629 inverted) | OBSERVED covariate |
| `is_first` | **.580** | OBSERVED covariate |
| `position_pct` | .441 (= .559 inverted) | OBSERVED covariate |
| `v_inline_math_delims` / `v_word_count` / `v_log_len` | .586 / .576 / .574 (FIT+MINE) | **ALREADY IN THE BANK** |
| `answer_year` | .488 | OBSERVED covariate |

## 5. Where round 0 leaves this cell

The dispatched "fully articulable (−.0001)" headline survives contact with the closure
protocol, and the master ledger's +.070 E-row gap does not. The honest same-rows residual
is **+.023 (HONEST) / +.007 (MONITOR, governing)**, it is **negative on the tier that
matches the y-definition (−.043 on MONITOR)**, and **arrival order absorbs essentially
all of what is left** — matched Δ_adj −.0011 on HONEST and −.0281 on MONITOR, with the
dense arm adding only +.005 over the bank once the position family is stacked in.

Rounds 1+ therefore run as the frozen **map-focused dual track**: Track A is still run at
full k=15 across the sealed P=8 fleet (a null there is evidence, not an omission), and
Track B carries the weight — its job on this cell is to find the **textual fingerprint**
of the arrival-order channel that the observed covariate proves is there. The sibling
cell's round-1 central negative is the standing warning: four proposers across two
families conjectured "presupposes sibling answers exist", Gemma scored it corpus-wide,
and it read **.492 — chance**. Whatever carries arrival order into the dense model, the
obvious fingerprint is not it.

## 6. Rounds

### Declared steers, fixed before round 1 and held CONSTANT across all rounds

Prereg AMENDMENT 2 requires proposal shape to be fixed in advance and any steer to be
recorded, not silent.

* **Track A** keeps the frozen interaction-shaped steer verbatim ("composite /
  interaction criteria are encouraged; this instruction is held constant across all
  rounds").
* **Track B MODE 3** (FREEZE ADDENDUM 4) is instantiated as the answer's ORDER UNDER ITS
  QUESTION, with examples of the *shape* of an arrival-order fingerprint. The sibling
  cell's wording said "arriving after the question already had **accepted** answers";
  that word is this cell's outcome, so it was changed to "already had **several**
  answers". Proposers are told they cannot see the actual position and must propose a
  fingerprint scorable from text alone.
* **Track B MODE 4** is instantiated as this corpus's upstream priors: answerer standing
  on the site, relationship to the reader, typographic/markup habit, and the kind of
  question the answer is attached to.
* **Round 0's arrival-order finding is not shown to any proposer**, and neither is the
  sibling cell's round-1 negative. The steer is structural (the freeze's own
  addendum-4 language, specialised to this container), not a hint about what was
  measured.

### Round-1 disagreement slice, built and sealed

`mathse_accepted_r1_slice.json`: 60 rows drawn inside M, 30 `dense_high_card_low` +
30 `dense_low_card_high`, label-blind, carrying text and both percentile ranks only
(median |gap| .703). The §4 mechanism is visible in the slice itself: mean
`answer_position` is **1.07** among the rows the dense model ranks far above the
scorecard and **1.73** among the rows the scorecard ranks far above the dense model
(population mean 1.11). The proposers are not told this.

Round-1 fitting state entering the round: FIT+MINE n = 11,675, 60 features,
VA_lin OOF .6329, VA_nl OOF per seed [.6313, .6308, .6294].

### Fleet check before round 1 (recorded, per the freeze's degradation clause)

Smoke-tested immediately before dispatch: Codex leg `gpt-5.6-luna` LIVE (in-spec reply);
GLM key A LIVE (glm-5.2, 1.0 s); GLM key B **LIVE** (1.6 s) — both z.ai credentials
answering under the revived Lite plan. Fleet runs at the current standard **P = 8 across
3 families** (claude ×3 sealed subagents, gpt-5.6-luna ×3 via the Codex companion,
glm-5.2 ×2), TIER S, 16 sealed prompts with 16 distinct row orderings.

### Round 1 — fleet

**Full target fleet, no degradation: P = 8 across 3 families on BOTH tracks**
(claude-opus ×2 salts, claude-sonnet, gpt-5.6-luna ×3 via the Codex companion, glm-5.2 ×2),
**16/16 slots returned and parsed, 200 proposals** (120 Track A, 80 Track B). GLM key B hit
`1302 rate_limit` on its Track-B leg and cleared on attempt 2 under the frozen patient
retry stack, so the round is **not** recorded as degraded. TIER S throughout;
`n_directed_excluded = 0`.

### Round 1 — the Track-B species merge, run BEFORE the audit this time

The sibling cell's round 1 discovered that `species.py`'s bge-cosine clustering at τ = .79
**under-merges the arrival-order family**, inflating Good–Turing f₁ and letting the
channel fall out of the scored set; and because its second judge arrived after the audit
had already been keyed to the pre-merge pool, that campaign had to score the LOOSE
selection while quoting the STRICT one for mass. **This campaign ran the blind pairwise
merge first**, so one selection is both scored and the figure of record. That is the only
process change, and it is recorded here rather than left silent.

The τ-only Track-B table showed exactly the predicted fragmentation: "Reply-position
framing" (4 proposers) sitting beside singleton shards *Explicit references to existing
thread*, *Named cross-reference to other users or answers*, *Names other site participants
by handle or first name*, *Reply-order language*, *Thread-aware addendum language*,
*References to prior answers or comments*, *Explicit mention of multiple own answers*.

120 cross-proposer pairs at cos ≥ .55 plus 2 planted anchors, two sealed blind judges,
strict rule (both judges SAME = a merge edge), fixed before the verdicts were read.
Both judges' anchors **passed**; raw agreement **110/120 = .917**.

| Track-B accounting | τ-only (embedding) | **STRICT, 2 judges (figure of record)** |
|---|---|---|
| species S_obs | 56 | **39** |
| singletons f₁ | 44 | **23** |
| doubletons f₂ | 7 | 7 |
| **Good–Turing missing mass** | .550 | **.2875** |
| cross-proposer recapture | .214 | **.410** |
| species named by ≥ 2 families | 6 | **11** |
| merge edges | — | 41 |

The τ-only .550 is **RETIRED** — it overstates remaining B-side mass by 91% relative to
the strict figure, purely through f₁ inflation. Track A is unmerged and quoted at
**M̂ = .333** (S_obs 55, f₁ 40, LOPO jackknife [.305, .390]).

**The merge promotes the arrival-order channel from a 4-proposer shard to the single
largest species in the pool**: `Reply-position framing`, **7 of 8 proposers, 9 members**.
Post-merge Track-B scored set:

| id | proposers | channel |
|---|---|---|
| **B01** | **7** | **Reply-position framing** |
| B02 | 7 | Personal voice: sign-offs, exclamations, emoticons, flourishes |
| B03 | 6 | Total verbosity and length |
| B04 | 5 | Density of external citations and tool references |
| B05 | 4 | Second-person coaching addressed at the asker |
| B06 | 3 | Era-marked conventions |
| B07 | 3 | Custom TeX macro preamble and heavy display-math apparatus |
| B08 | 3 | Markup and display conventions |
| B09 | 3 | Future-reader framing |
| B10 | 2 | Flat declarative terseness versus visible hedging |

### Round 1 — routing audit and the boundary this cell draws

Audit built on the MERGED selection: 29 items (25 criteria + 4 planted corpus-matched
probes), fresh blind Sonnet-class auditor, probe draw chained from round 1.

| | |
|---|---|
| misrouting rate | **4/25 = .16** |
| **planted probe pass** | **4/4 = 1.00** |
| disputes → arbiter | 4 |
| final routing | **A = 11, B = 14 (11 flagged MIXED)**, `arbiter_present: true` |

**All four disputes land on one boundary, and the frontier arbiter upheld the blind
auditor 4/4 — every one moved A → B, every one flagged MIXED:**

| id | criterion | ruling |
|---|---|---|
| A01 | Affirmation of correct work before correction | → B (mixed) — an etiquette/ordering convention |
| A02 | Excessive meta-commentary or apologies | → B (mixed) — boilerplate and rhetorical register |
| A04 | Conversational warmth / patient register | → B (mixed) — a tone and community-style marker |
| A12 | Socratic prompting and interactivity | → B (mixed) — a pedagogical delivery convention |

This is prereg **open question (b)** — "the nuisance-vs-merit boundary for fluency-like
channels is a substantive decision to be made explicitly per cell" — being decided
explicitly for math.SE accept, and it lands in the **same place** the sibling vote cell's
arbiter put it: **pedagogical/social REGISTER is nuisance; pedagogical SUBSTANCE is
merit.** Two campaigns, two different y's, two independent fresh auditors and arbiters,
one boundary. Worth noting that the accept y is the one where a warmth-is-merit reading
would have been most defensible — the *asker* is a person being addressed directly — and
the arbiter still declined it.

The consequence for round 1 is stated plainly: the fleet proposed 15 A-side species and
only **11** survived routing, while the nuisance set grew to 14 of 25. On a map-focused
cell that is the expected shape, but it does mean Δ₁'s A-side is being moved by 11
criteria, not 15.

### Round 1 — Gemma scoring

Corpus-wide pass on **GPU 6** (LANE B; ledger CLAIM-STACKED at 182,632 MiB free, 0% util,
no compute apps, verified immediately before the claim). `gpu_stack_runner.sh` gained a
`LANE_GPU` pin so the runner waits for its own lane instead of migrating onto another
lane's card. 13,001 rows + 150 anchor texts × 25 criteria = **328,775 prompts**, offline
batch, `--max-model-len 8192` (the LaTeX ≈2 chars/token rule inherited from the sibling
cell — the text is never shortened, the context is raised), bank-matched HEAD-3000 +
TAIL-2000 truncation, anchors K = 50/class.

### Round 1 — READOUT

`mathse_accepted_r1_results.json`. **Instrument health first:** anchors K = 50/class,
coherent-vs-scrambled AUC **.9948** (gate passes decisively), pos-vs-neg .568, overall NA
rate **.007**, **0 of 25 criteria collapsed**, no all-NA rows. 328,775 prompts, 41 min of
one B200, GPU 6 released rc=0.

#### The curve

Bank 60 → **71** features (11 A-routed criteria join).

| tier | Δ₀ | **Δ₁** | round-1 VA_nl gain | 95% CI | p(gain>0) |
|---|---|---|---|---|---|
| **MONITOR (governing)** | +.0176 | **+.0186** | **−.0009** | [−.0090, +.0073] | .41 |
| HONEST | +.0345 | **+.0327** | **+.0017** | [−.0041, +.0070] | .73 |

(Δ here is the ensemble-convention pooled figure the readout carries, so it is compared
only against itself round-over-round; the T-based Δ₀ of §2 is +.0067 / +.0229.)

**Round 1 bought nothing on either tier, and this time the two tiers agree.** Both gains
sit inside ε = .005 and both bootstrap CIs straddle zero. Under the frozen signed reading
**round 1 is sub-ε #1**, and — unlike the sibling vote cell, where MONITOR and HONEST
disagreed in sign and the null had to be read as MONITOR's ±.032 width talking — here the
two independent populations return the same near-zero answer.

**Why the null, mechanically.** The mined Track-A criteria are, for the first time in this
programme's math.SE work, **not better than the bank they were meant to beat**:

| alone-AUC (HONEST) | mined criterion |
|---|---|
| .562 | Proportionate method |
| .555 | Sanity-check discipline |
| .548 | Edge-case completeness |
| .539 | Returns the exact object the question asked for, in the asked format |
| .532 | Register matched to the asker's evident level |

against the incoming 32-rubric bank's best of **.567** ("Names the real obstacle"). On the
vote cell the top mined criterion (.598) beat the incoming best (.573) and *still* added
nothing once fitted — a pure redundancy result. Here the fleet did not even clear the
bank's own ceiling. Both readings point the same way: the 32-criterion GEPA bank already
spans what a P=8 three-family fleet can name about this corpus.

**Swap check: clean.** C₊ .7322 → .7334, C₋ .3972 → .4000, ρ(bank, dense) .4990 → .4979,
`swap_signature: false`. The round did not buy rank agreement by inheriting dense errors.

#### The Track-B map — and the same central negative as the sibling cell

Alone-AUC on HONEST, with each channel's strongest rank correlation against the V block:

| alone-AUC | mixed | max\|ρ\| with V | closest V column | channel |
|---|---|---|---|---|
| **.557** | yes | **.89 → ALREADY ARTICULATED** | `v_log_len` | Total verbosity and length |
| **.540** | yes | **.81 → ALREADY ARTICULATED** | `v_n_display_math` | Markup and display conventions |
| **.537** | yes | **.82 → ALREADY ARTICULATED** | `v_second_person` | Second-person coaching addressed at the asker |
| .533 | yes | .60 | `v_second_person` | Supportive and patient register *(arbiter-rerouted)* |
| .526 | yes | **.83 → ALREADY ARTICULATED** | `v_n_display_math` | Custom TeX macro preamble / display-math apparatus |
| .518 | yes | .20 | — | Affirmation of correct work before correction *(arbiter-rerouted)* |
| **.516** | no | .30 | — | **Reply-position framing** |
| .513 | yes | .44 | — | Future-reader framing |
| .512 | no | .42 | — | Personal voice: sign-offs, exclamations, emoticons |
| .503 | yes | .38 | — | Density of external citations and tool references |
| .503 | yes | .27 | — | Socratic prompting and interactivity *(arbiter-rerouted)* |
| .502 | yes | .35 | — | Excessive meta-commentary or apologies *(arbiter-rerouted)* |
| .497 | no | .12 | — | Era-marked conventions |
| .494 | yes | .48 | — | Flat declarative terseness versus visible hedging |

**The arrival-order fingerprint reads .516 against the observed covariate's .660.** This
is the round's most important result and it **replicates the sibling cell's central
negative on a different y**: there the judged fingerprint "presupposes sibling answers
exist" read **.492** while the observed covariate read .614; here "Reply-position framing"
— named by **7 of 8 sealed proposers across 3 families**, the single largest species in
the pool — reads **.516** while the observed covariate reads **.660**. Two campaigns, two
different labels, one conclusion: **whatever carries arrival order into the dense model,
it is not a text-visible reply-position register that a well-resourced fleet can name.**
The channel is real and large in the covariate and nearly absent in its conjectured trace.

**The rest of the map is a mirror.** Every channel with real alone-AUC is one the
articulated instrument already owns (ρ = .89 / .83 / .82 / .81 with V columns). Dropping
the MIXED channels collapses the whole nuisance model from .590 to **.510** on HONEST
(.612 → .521 on MONITOR) — as on the vote cell, **all** of the mined nuisance set's
predictive power lives in channels whose upstream parent plausibly causes real quality too,
and mostly in ones already inside the bank. Spurious-alone stays below the .65
matched-sampling trigger for the MINED set; the OBSERVED position family, which does
exceed it, keeps its matched readout from §4.

#### Discount table

| stratifier | Δ_adj HONEST | Δ_adj MONITOR |
|---|---|---|
| none (pooled, ensemble) | +.0327 | +.0186 |
| joint B model, ALL 14 channels (decile / quintile) | **+.0386** | +.0197 |
| joint B model, STRICT (11 MIXED dropped) | +.0323 | +.0196 |
| *(round 0)* matched on the OBSERVED arrival-order family | **−.0011** | **−.0281** |

Discounting on the *mined* nuisance set does not move Δ — it slightly **raises** it,
because those channels are bank columns in judged clothing. Only the *observed*
arrival-order family moves it, and that moves it past zero. The two rows are doing
different jobs and are never differenced.

#### Stacked increment (stratification-free)

| stack | HONEST | MONITOR |
|---|---|---|
| joint B alone | .5905 | .6119 |
| B + dense | .6506 (**+.0602**) | .6608 (**+.0489**) |
| B + bank | .6209 (+.0304) | .6421 (+.0302) |
| B + dense + bank | .6548 | .6640 |
| **dense increment over B + bank** | **+.0340** [.040, .078 for the raw dense-over-B] | **+.0219** |
| bank increment over B + dense | +.0042 | +.0032 |

Read against the round-0 stacked readout on the OBSERVED position family (dense +.0488 vs
bank +.0437, a gap of only +.005), this says the residual that survives the *mined*
nuisance set is larger than the one that survives the *observed* arrival-order family.
That is the expected ordering — the mined set does not contain arrival order in any
useful strength — and it is the quantitative form of the round's central negative.

#### Missing mass at round 1

| track | S_obs | f₁ | f₂ | **M̂** | LOPO jackknife | recapture | ≥2 families |
|---|---|---|---|---|---|---|---|
| A (τ-only, unmerged) | 55 | 40 | 7 | **.333** | [.305, .390] | .273 | 7 |
| **B (strict blind merge)** | **39** | **23** | **7** | **.2875** | — | **.410** | **11** |

Both tracks are better covered than the sibling cell's round 1 (A .533, B .483 there),
which is what a P = 8 fleet buys over P = 6. Neither is saturated on species count, so the
sub-ε reading is a curve observation, not yet a plateau claim.

### Round 2 — running as a full sealed PROPOSING round

Registered before launch: the stopping rule needs **two consecutive sub-ε PROPOSING
rounds** (2026-08-08 addendum: decomposition rounds do not count). Round 1 is sub-ε #1, so
round 2 runs as another full sealed P = 8 fleet with **all steers held constant** (prereg
AMENDMENT 2), slice rebuilt on the round-1 bank (71 features). Round 1's findings are not
shown to any proposer.

**Round-2 fleet: 16/16 slots, 200 proposals, P = 8 across 3 families, no degradation**
(no GLM rate-limit this round). Slice: 60 rows in M, median |gap| .716; entering state
FIT+MINE 11,675 rows / 71 features, VA_nl OOF .6378.

**Blind Track-B merge** (same discipline as round 1, again run BEFORE the audit): 120
cross-proposer pairs + 2 anchors, two sealed judges, anchors **pass**, raw agreement
**115/120 = .958**, 44 strict merge edges.

| Track-B accounting | τ-only | **STRICT, 2 judges (record)** |
|---|---|---|
| species S_obs | 44 | **36** |
| singletons f₁ | 28 | **21** |
| doubletons f₂ | 7 | 6 |
| **Good–Turing missing mass** | .350 | **.2625** |
| cross-proposer recapture | .36 | **.417** |

Track A (unmerged, τ-only): S_obs 47, f₁ 39, f₂ 1, **M̂ = .325**, recapture .17 — the A-side
mass is essentially unchanged from round 1 (.333), i.e. a second independent P=8 fleet
found about as much new A-species territory as the first, and no more.

Post-merge Track-B scored set, with the arrival-order channel again surfacing at the top
of the pool under a different name — **B05 "Prior-answer dependence", 6 of 8 proposers**
(round 1's was "Reply-position framing", 7 of 8): B01 Markup density (7), B02 Visible
revision markers (7), B03 External-resource citation and tool appeals (7), B04 Sheer bulk
and display-mathematics density (6), **B05 Prior-answer dependence (6)**, B06 Author
personality / conversational asides (5), B07 Named participants and @handles (3), B08
Era-specific conventions (3), B09 Elementary-exercise vs research-register vocabulary (3),
B10 Hint register and deliberate incompleteness (2).

**Round-2 routing audit**: 29 items, fresh blind auditor, **probes 4/4**, misrouting
**6/25 = .24**, 6 disputes → frontier arbiter **upheld the auditor 6/6, all A → B** (5 of
6 flagged MIXED). Final routing **A = 9, B = 16 (15 MIXED)**.

The arbiter's round-2 reasons are worth recording because they are a *sharper* version of
round 1's boundary, and they are mostly about **instrument construction rather than
construct**: three of the six were rejected because the scoring instruction's discriminating
axis was not the property the proposer named — a composite gated on a property of the
QUESTION rather than the answer (A15, A14); anchors that make *address/stance* dominant so
an impersonal answer scores 0 however substantive it is (A10, A12); an instruction under
which "a confidently asserted wrong answer scores 10" (A05); and one where the axis is
*placement* of the decisive step rather than its presence (A09). Round 1's misrouting was
.16 with the disputes clustered on register-vs-substance; round 2's is .24 with the
disputes clustered on **the criterion text not measuring what its name claims**. That the
rate went UP as the fleet reached further is itself informative: the second round's Track-A
proposals are more elaborate composites, and elaboration is where mis-specification enters.

**Round-2 Gemma pass**: launched on lane GPU 6. The card acquired a co-tenant vLLM
(PID 994306, ~170 GB) between rounds; the lane-pinned runner waits for headroom on its own
card rather than migrating, and the co-tenant is never touched.

### Round 2 — READOUT

Instrument health: anchors K = 50/class, coherent-vs-scrambled **.9990**, pos-vs-neg .611,
NA rate **.002**, **0 of 25 criteria collapsed**. 328,775 prompts, 46 min, GPU 7 released
rc = 0.

#### The curve — SATURATION FIRES

Bank 71 → **80** features (9 A-routed criteria join).

| tier | Δ₀ | Δ₁ | **Δ₂** | round-2 gain | 95% CI | p(gain>0) |
|---|---|---|---|---|---|---|
| **MONITOR (governing)** | +.0176 | +.0186 | **+.0157** | **+.0028** | [−.0076, +.0131] | .71 |
| HONEST | +.0345 | +.0327 | **+.0308** | **+.0020** | [−.0049, +.0094] | .71 |

Both gains are sub-ε (ε = .005) on both populations, both CIs straddle zero. Round 1 was
sub-ε #1 and **round 2 is sub-ε #2** — and, critically, **both were full sealed P = 8
PROPOSING rounds**, so the 2026-08-08 addendum's requirement that a decomposition round
cannot supply a sub-ε count is satisfied without needing to invoke it. **The frozen
stopping rule fires at round 2.**

**Caveat that travels with the round-2 gain: the swap signature fired.** C₊ +.0050,
C₋ −.0037, ρ(bank, dense) +.0140 → `swap_signature: true`. Round 2's (tiny) gain was
bought partly by moving the bank toward the dense model's ordering *including on pairs the
dense model gets wrong*. Round 1's swap was clean. Given the gain is +.0028 and inside
noise the practical consequence is negligible, but the direction is recorded and the
round-2 increment should not be quoted as clean articulation gain.

Mined Track-A criteria again fail to clear the incoming bank's ceiling: best **.565**
("Proportionality of method") against the 32-rubric bank's **.567**. Two independent P = 8
fleets, 240 Track-A proposals, and neither produced a single criterion that out-scores the
best rubric already in the bank.

#### The Track-B map — the arrival-order negative replicates a THIRD time

| alone-AUC | mixed | max\|ρ\| with V | channel |
|---|---|---|---|
| **.557** | yes | **.82 → ALREADY ARTICULATED** | B04 Sheer bulk and display-mathematics density |
| .539 | yes | .24 | A12 Discovery narration paired with respect for the asker *(arbiter-rerouted)* |
| .530 | yes | **.75 → ALREADY ARTICULATED** | B01 Markup density |
| .530 | yes | .41 | A15 Bounded personal question answered in the asker's terms *(rerouted)* |
| .527 | yes | .38 | B09 Elementary-exercise versus research-register vocabulary |
| .518 | yes | .35 | A14 Elementary question closed in a handful of lines *(rerouted)* |
| .515 | yes | .49 | A05 Confidence without hedging *(rerouted)* |
| .515 | yes | **.73 → ALREADY ARTICULATED** | A10 Second-person address with an executable next step *(rerouted)* |
| **.510** | yes | .28 | **B05 Prior-answer dependence** |
| .506 | yes | .53 | B06 Author personality / conversational asides |
| .505 | yes | .66 | B02 Visible revision markers |
| .503 | yes | .39 | B03 External-resource citation and tool appeals |
| .493 | yes | .22 | B07 Named participants, @handles and priority acknowledgements |
| .489 | no | .34 | B08 Era-specific conventions |
| .462 | yes | .34 | B10 Hint register and deliberate incompleteness |

**B05 "Prior-answer dependence" reads .510.** Named by 6 of 8 sealed proposers, under a
different name and a different instruction from round 1's "Reply-position framing" (.516),
in a fleet that could not see round 1. Together with the sibling vote cell's .492, that is
**three independent corpus-wide measurements, across two labels and two campaigns, all
putting the judged textual fingerprint of arrival order at chance** — against an observed
arrival-order covariate that reads .660 on the same rows.

Dropping the MIXED channels again collapses the nuisance model to chance (.591 → .512 on
HONEST, .606 → .517 on MONITOR).

#### Discount and stacked increment

| stratifier | Δ_adj HONEST | Δ_adj MONITOR |
|---|---|---|
| none (pooled, ensemble) | +.0308 | +.0157 |
| joint B model, all 16 channels | +.0332 | +.0153 |
| joint B model, STRICT (15 MIXED dropped) | +.0277 | +.0180 |
| *(round 0)* matched on the OBSERVED arrival-order family | **−.0011** | **−.0281** |

| stack | HONEST | MONITOR |
|---|---|---|
| joint B alone | .5910 | .6063 |
| B + dense | .6485 | .6568 |
| B + bank | .6194 | .6397 |
| B + dense + bank | .6533 | .6618 |
| **dense increment over B + bank** | **+.0339** | **+.0221** |
| bank increment over B + dense | +.0048 | +.0050 |

Both quantities are essentially unchanged from round 1 (+.0340 / +.0219), i.e. **a second
full mining round moved neither the residual nor the nuisance map.**

#### Missing mass at round 2

| track | S_obs | f₁ | f₂ | **M̂** | recapture | ≥2 families |
|---|---|---|---|---|---|---|
| A (τ-only) | 47 | 39 | 1 | **.325** | .17 | — |
| B (strict merge) | 36 | 21 | 6 | **.2625** | .417 | 7 |

Round 1 → round 2: A **.333 → .325**, B **.2875 → .2625**. Two independent P = 8 fleets
returned nearly identical mass estimates. That stability is the honest reading of the
plateau: **the miner is not running out of species — it is running out of species that
matter.** The A-side pool still has ~⅓ of its mass unfound, and the two rounds' worth of
found mass bought +.0028 and −.0009 of MONITOR AUC.

## 7. TERMINAL VERDICT — saturated at round 2, MAPPED not closed

> **math.SE accepted-verdict (the asker's own choice), Layer-3 dual-track closure —
> TERMINAL at round 2 of a capped 5, by the frozen stopping rule (two consecutive sub-ε
> PROPOSING rounds, both full sealed P = 8 fleets across 3 families).**
>
> **The honest residual is small and the master ledger overstates it threefold.** On the
> master ledger's own E rows, refitting VA under the closure protocol moves it from .5737
> to .6147 — reproducing the full-grid arm's own full-fit reference to .0012 — so the
> same-rows residual is **+.0229 at three dense seeds (+.0292 vs seed 42)**, not +.070. On
> the governing MONITOR tier Δ₀ = **+.0067**, and on the within-question tier that matches
> the y-definition the articulated bank is **ahead by .043**. The dispatched Layer-1
> headline ("fully articulable, −.0001") survives contact with the closure protocol.
>
> **Mining did not move it.** Δ curve on MONITOR: **+.0176 → +.0186 → +.0157** (ensemble
> convention); VA_nl gains **−.0009** then **+.0028**, both sub-ε, both CIs straddling
> zero, on both MONITOR and HONEST. 400 sealed proposals from two independent P = 8 fleets
> across 3 families produced 20 bank-joining criteria and no measurable closure. The
> mechanism is redundancy at the ceiling: **neither fleet produced a single Track-A
> criterion out-scoring the best rubric already in the 32-criterion bank** (best mined .562
> then .565 vs incoming .567), and the round-0 census found that bank to be the least
> degenerate in the programme (L0 = L5 = 32, max |r| = .567, zero pairs ≥ .90).
>
> **What the residual IS, is arrival order — and it is not text-visible.** A no-text
> six-variable model of where an answer sits in its question's arrival order reads
> **.6754 pooled / .6600 on HONEST, ABOVE the dense model's T (.6375)**; accept rate falls
> .503 → .402 → .207 → … → .000 by position 7, a far steeper tail than the sibling
> vote cell's. ρ(position, dense) = **+.148** against **+.008** for the fitted bank.
> **Matched sampling on it removes the entire residual and then some: Δ_adj = −.0011 on
> HONEST and −.0281 on MONITOR**, while length and LaTeX strata move Δ the other way
> (+.0450) — a localisation, not a conditioning artifact. Conditional on the position
> family, the dense arm adds only **+.005** more than the bank.
> **But three independent corpus-wide Gemma measurements put the judged textual
> fingerprint of that channel at chance** — .516 (round 1, "Reply-position framing",
> 7/8 proposers), .510 (round 2, "Prior-answer dependence", 6/8 proposers, a fleet blind to
> round 1), and .492 on the sibling vote cell. The channel is large in the covariate and
> absent in every fingerprint a well-resourced sealed fleet can name.
>
> **Every mined nuisance channel with real signal is one the bank already owns**
> (ρ = .89 `v_log_len`, .82 `v_n_display_math`, .82 `v_second_person`); dropping the MIXED
> channels collapses the nuisance model to .510 / .512 in both rounds. Missing mass is
> stable across the two fleets (A .333 → .325; B .2875 → .2625 strict), so the plateau is
> **"not discoverable by this miner at P = 8 across 3 families"**, not "nothing left to
> find".
>
> **Verdict: MAPPED, not closed.** The cell's dense-model edge over the articulated bank is
> small, is concentrated in the pooled (cross-question) comparison rather than the
> within-question one the label encodes, and is accounted for by an exogenous arrival-order
> prior that the dense model carries and the fitted bank does not — a prior with no
> nameable textual trace.

### Claim discipline attached to this verdict

1. **Levels are protocol-specific** (prereg AMENDMENT 1). The closure protocol fits VA on
   FIT+MINE and reads MONITOR; Layer 1 fits pooled GroupKFold OOF over all 13,001 rows;
   the master ledger's E arm does a third thing. §2.1 differences them explicitly and by
   name — never quote across designs without doing the same.
2. **Ensemble vs T convention.** Δ₀ = +.0067 / +.0229 is the T-based (mean-of-per-seed-AUC)
   figure. The round-over-round curve (+.0176 → +.0186 → +.0157) and every discount table
   are the ENSEMBLE-based figures, because a stratified or matched readout needs one score
   vector. **Never difference the two.**
3. **MONITOR is thin** — 1,326 rows / 500 questions, Δ bootstrap half-width ≈ .033,
   per-seed T spread .025 (wider than Δ₀ itself). No single round's gain is individually
   significant; the curve is read as a curve.
4. **Round 2's gain carries a swap signature** (`swap_signature: true`); round 1's does not.
5. **The concept census reused the sibling cell's two sealed judges** (byte-identical
   packet) and both were claude-sonnet-5 — same-family agreement, not a cross-family
   certification.
6. **`answer_position` is an observed covariate**, never in V, never in A, never judged,
   never fitted into the closure curve; it enters only the discount readouts.
7. **Pre-GEPA.** Mined criteria are fidelity-phrased but not GEPA-iterated; the freeze
   requires a GEPA pass before any final quoted number from a *positive* closure result.
   This cell's result is a null, so the pass would only make the null harder to overturn —
   recorded, not run.
8. **Lane deviation, recorded**: round 2's Gemma pass ran on GPU 7, not lane card GPU 6,
   because GPU 6 was held by a co-tenant vLLM (PID 994306, 169,524 MiB, 100% util) that was
   never touched. Ledger carries an explicit LANE-DEVIATION NOTE; GPU 7 was ledger-released
   and idle at claim time and was released rc = 0 immediately after.

## 8. Artifact index

All under `methods/taste_decomposition/closure/mathse_accepted/` unless stated.

| artifact | what |
|---|---|
| `RUNBOOK`-equivalent | this note; the machinery is the sibling cell's, diffed in §1.6 |
| `cells.py`, `oof_alignment_gate.py` / `.json` | loader + the mandatory registry gate (PASS, abs diff 0.0) |
| `build_splits.py`, `mathse_accepted_splits.json`, `mathse_accepted_population.csv` | salted FIT+MINE / MONITOR split + collision check |
| `fetch_dense.py`, `mathse_accepted_dense_preds.csv` | 3-seed dense probabilities, positional join asserted |
| `round0.py`, `mathse_accepted_r0_context.json`, `mathse_accepted_r0_preds.npz` | round-0 baseline, tiers, swap, jackknife |
| `census.py`, `census_stage1.json`, `census_blind_packet.json`, `census_verdicts_judge{A,B}.json`, `census.json` | L0→L5 concept census (judges reused from the vote cell, declared) |
| `position_line.py` / `.json`, `position_matched.py` / `.json`, `mathse_accepted_position.npz` | FREEZE ADDENDUM 4 arrival-order audit + matched discount |
| `length_stratification.py` / `.json` | the length/LaTeX contrast control |
| `composition_audit.json` | label composition (1 accept per question, 68% two-answer) |
| `mathse_accepted_r{1,2}_slice.json` | the sealed disagreement slices |
| `mathse_accepted_r{1,2}_proposals_fleet.json`, `fleet_{codex,glm}_r{1,2}.log` | 200-proposal pools + fleet transcripts |
| `species.py`, `species_merge.py`, `mathse_accepted_r{1,2}_species.json` (+ `.PREMERGE`), `*_bmerge_{packet,key,judgeA,judgeB}.json` | species, two-tier guard, blind pairwise Track-B merge |
| `audit.py`, `arbiter.py`, `mathse_accepted_r{1,2}_{audit_prompt,audit_verdicts,arbiter_raw,routing_final}.json` | blind routing audit + planted probes + arbiter |
| `score_gemma_maps.py`, `gpu_stack_runner.sh` (LANE_GPU pin), `launch_score.sh` | corpus-wide Gemma-4-31B offline batch, 8192 ctx |
| `mathse_accepted_r{1,2}_{scores.npz,score_report.json,results.json}` | per-round scores, instrument health, full readouts |
