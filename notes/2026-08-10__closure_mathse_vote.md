# Layer-3 articulation closure — math.SE VOTE-SCORE cell (within-question crowd vote)

Cell: math.StackExchange answers, y = raw vote score strictly ABOVE the median answer
score on its own question (ties at the median dropped). Un-binarised v2 rebuild.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + all four freeze addenda.
Production pipeline: `methods/taste_decomposition/closure/code_v3/RUNBOOK.md` (§12 shape).
Worked example: `notes/2026-08-09__peer_completion.md`.
Campaign dir: `methods/taste_decomposition/closure/mathse_vote/`.

## BOTTOM LINE (round 0, at THREE dense seeds, before any mining)

The dispatched residual was **Δ_beyond = +.0366** (T .6608, dense EVAL at seed 42, vs
VA_nl .6242). The gate is now in at three seeds and it **PASSES with room**: Δ_gate =
**+.0467**. Round 0 then finds three things that change how that residual should be read.

1. **Under the closure protocol the residual is +.0136 on MONITOR** (T .6597, VA_nl .6460),
   question-cluster bootstrap **[−.003, +.062]**, p(Δ>0) = .96. The protocol refits the bank
   on FIT+MINE, where it reaches .6460 rather than .6242. Levels are protocol-specific by
   prereg AMENDMENT 1 — but here that clause moves the residual by about two thirds.
2. **On the tier that matches the y-definition there is no residual at all.** Within
   question, on MONITOR, the articulated bank is **ahead** of the dense model
   (.6639 vs .6536, Δ = **−.0103**).
3. **A large share of the residual is the FIRST-ANSWER ADVANTAGE.** A model that sees no
   text at all — only where an answer sits in its question's arrival order — reads
   grouped-OOF AUC .654 pooled. The dense score correlates with `is_first` at **+.089**;
   the fitted bank at **−0.00007**. Conditioning on arrival order shrinks the residual
   under every instrument tried (on MONITOR, matched sampling takes it from +.030 to
   **+.014**, a 55% cut), while conditioning on length or LaTeX *grows* it — so this is a
   localisation, not a generic conditioning artifact. It is a large share, **not all**:
   see the seed-count correction in §4.3.

## Terms, spelled out on first mention (standing rule)

| term | what it means here |
|---|---|
| **V** | the 28 deterministic lint features (`v_*`) computed on the answer body |
| **A** | the 32-criterion Gemma-4-31B-judged rubric bank (a01–a32), GEPA-phrased |
| **VA_lin / VA_nl** | the articulated instrument: V+A fit linearly / by gradient boosting (HistGradientBoosting, frozen grid), grouped out-of-fold, VA_nl = mean over fit seeds {0,1,2} |
| **T** | the dense arm: Llama-3.1-8B LoRA reward model reading raw text; T = MEAN OVER DENSE SEEDS of the held-out AUC (never the AUC of the seed-averaged prediction) |
| **Δ_beyond** | T − VA_nl: the part of the crowd's preference the articulated bank does not reach |
| **Δ_r** | the closure curve: Δ after r rounds of active mining |
| **FIT+MINE / MONITOR** | the closure split; MONITOR lives inside the dense-held-out rows and is never read by any proposer |
| **M (mining slice)** | the dense-held-out half of FIT+MINE — dense scores there are honest |
| **Track A / Track B** | the dual design: A proposes quality-relevant criteria that could close the gap; B proposes suspected-SPURIOUS predictive channels that are used only to DISCOUNT |
| **MIXED channel** | a Track-B channel whose conjectured upstream parent plausibly causes real quality too; decomposed (FREEZE ADDENDUM 3) rather than routed to one side |
| **alone-AUC** | a single criterion's held-out AUC on its own |
| **swap pair (C₊, C₋)** | P(bank orders a discordant pair correctly \| dense does) and \| dense does not) |
| **missing mass** | fleet-based Good–Turing estimate of the criterion species the miner has not yet found |

## ROSTER SUPERSESSION, recorded

The 2026-08-06 FREEZE DECLARATION's roster line reads "math.SE excluded (user)". That
exclusion was written when the only math.SE cell in the programme was the OLD BINARISED
one (accepted AND score ≥ 3 vs score ≤ 0, Qwen-scored A). This campaign runs the NEW
un-binarised vote-score cell on an explicit later instruction (user, this session). The
old cell's published V .565 / VA .673 / T .794 are a different population, a different
y and a different judge; they are **never** differenced against anything here.

## 1. What this cell is, and why it was worth opening

Of the five scale-up-wave-C builds, math.SE vote-score is the **only** one that cleared
the Layer-3 gate (`notes/2026-08-08__scaleupC_builds.md`):

| scale-up-C cell | n | VA_nl | T | Δ_beyond |
|---|---|---|---|---|
| math.SE **vote score** | 11,629 | .6242 | .6608 | **+.0366** |
| math.SE accepted verdict | 13,001 | .6320 | .6319 | −.0001 |
| AoPS curation | — | — | — | +.010 |
| jokes / homepage | — | — | — | (below gate) |

The two math.SE cells share ONE scored A/V matrix and one Gemma pass. They differ only
in y. The accepted-verdict y — a single asker's choice — is **fully articulated** by the
bank; the vote-score y — the crowd's within-question ranking — is not. That contrast is
the reason this cell is interesting, and it is the reason every readout below is kept
strictly separate from the accepted-verdict cell.

## 2. Protocol adaptations, recorded BEFORE any mining slice was built

### 2.1 Alignment gate — PASSED EXACTLY

Registry landmine 2026-08-10: `*_va_nl_oof_*.npy` are keyed in **bank item_ids order**,
not population/join order; for the math.SE cells that means the KEPT-SUBSET order (bank
`item_ids` filtered by `isfinite(ys["vote_score"])`, 11,629 of 13,001). The mandatory
gate is asserted inside `cells.load()` on every call and cannot be skipped by accident:

```
AUC(y, mathse_vote_score_va_nl_oof_seed0.npy in assembled row order) = 0.624849045069194
published ledger nonlinear.VA["0"].auc                               = 0.624849045069194
abs diff = 0.0            GATE_PASS = true
shuffled counterfactual  = .5007   (what a misaligned join would have read)
```

`closure/mathse_vote/oof_alignment_gate.json`. The gate binds on seed 0 only; the mean3
array reads .6272 and is **never** the gate (mean3's AUC is not the mean of per-seed
AUCs). A length assertion also refuses the accepted-verdict cell's 13,001-row array.

### 2.2 Splits

`closure/mathse_vote/mathse_vote_splits.json`. Group key = `question_id`; no question
straddles two dense splits (asserted).

| | rows | questions | pos rate |
|---|---|---|---|
| population | 11,629 | 4,960 | .5015 |
| dense train | 9,303 | 3,820 | — |
| dense held-out (HONEST) | 2,326 | 1,140 | — |
| **MONITOR** (⊂ held-out) | **1,124** | 554 | .5009 |
| **M** (mining slice, ⊂ held-out) | **1,202** | 586 | — |
| FIT+MINE | 10,505 | 4,406 | .5016 |

**Salt, recorded not silent.** The dense arm's own 80/10/10 is a stable sha256 hash on
the SAME key. A second unsalted cut on the same key is not independent of the first, so
the closure cut hashes `sha256("mathse-vote-closure|" + question_id)`. The collision
check is reported: an unsalted cut would have put 562 of the 1,140 held-out questions in
MONITOR versus the salted 554, so the two salts do not in fact collide — the salt is kept
anyway because that could only be known after the fact.

### 2.3 No imputation fork on this cell

press_verdict had to choose between the Layer-1 constant-0.5 fill and the closure
standard's median impute. This cell's Layer-1 linear leg already uses
`SimpleImputer(median, add_indicator)` inside each fold — the same convention
`closure_core.clean_fit` applies — so there is no const-0.5 vs median fork here. The
A-block NA rate is .2396 (the applicability gate firing).

There is still one real difference: Layer 1 used `add_indicator=True`, so it also gave the
model a 0/1 "the judge said this criterion does not apply here" column per rubric, and
`closure_core` does not. With a .2396 NA rate that is not a rounding detail, and dropping
it could only weaken the articulated bank — which would INFLATE Δ. Measured
(`na_indicator_check.json`, MONITOR, otherwise-identical closure protocol):

| bank | pooled AUC | within-question |
|---|---|---|
| VA_nl, frozen closure protocol | **.6460** | **.6639** |
| VA_nl + 31 NA-indicator columns | .6422 | .6528 |
| the NA pattern **alone** (nonlinear) | .5725 | — |
| VA_lin, frozen protocol / + indicators | .6311 / .6319 | — |

Adding the indicators makes the bank slightly **worse**, not better: the boosted model
already recovers the applicability pattern from the imputed values, and the extra 31
columns cost more in variance than they return. `imputation_cost_to_Δ = −.0038` — the
frozen protocol, if anything, slightly *understates* the residual. The objection is
closed and the frozen spec stands unchanged.

### 2.4 Readout tiers, declared in advance

* **TIER 1, GOVERNING — pooled AUC on MONITOR.** The Layer-1 gate quantity lives on the
  pooled tier, so it is the only tier on which the gate and the curve are commensurable.
* **TIER 2, SECONDARY — n-weighted within-QUESTION AUC.** y is a within-question median
  split, so this tier matches the y-definition. Reported every round, never substituted.
* **TIER 3, DIAGNOSTIC — eval-only / test-only / HONEST same-rows level.**

Closure-split Δ levels are protocol-specific and NOT comparable to the Layer-1
Δ_beyond (prereg AMENDMENT 1); only round-over-round changes and the same-rows honest
level are quotable.

### 2.5 Scoring view matched to the bank

The A bank was judged on a deterministic HEAD-3000 + TAIL-2000 middle omission at 5,000
source chars, with the question TITLE (400 chars) prepended and the question body absent.
`score_gemma_maps.py` was patched to apply the identical truncation, so a round's mined
criteria and the incoming bank are answering about the same document. The bank's own
scale is 0 / 0.5 / 1.0 and mined criteria are 0–10; both are standardised by the fitter,
and the difference is recorded rather than harmonised.

### 2.6 sklearn

The Layer-1 ledger was produced under scikit-learn 1.8.0; this campaign runs 1.7.2, and
GroupKFold fold assignments move across releases. Consequence, stated: Layer-1 LEVELS are
not byte-reproducible here, which is why the closure protocol's own round-0 anchor — not
the Layer-1 number — is the baseline the curve is measured from. Every round runs under
the one version, asserted and recorded in each results JSON. The alignment gate is
version-independent (it reads a stored vector).

## 3. ROUND 0 — concept census of the incoming bank

`closure/mathse_vote/census_stage1.json`, `census.json`.

| level | count |
|---|---|
| L0 criteria delivered | 32 |
| L1 distinct names | 32 |
| L2 after the frozen degeneracy screen (FIT+MINE only) | **32** |
| L3 value clusters at \|r\| ≥ .98 | **32** |

**This is the least degenerate bank the programme has censused.** Max off-diagonal
\|Pearson r\| = .557; the fraction of column pairs at \|r\| ≥ .90 is exactly **0**; nothing
was dropped by the degeneracy screen. For contrast, the patents bank collapsed to ONE
concept and the mined banks in the A-bank degeneracy audit ran 54–68% degenerate. The
32 criteria are a GEPA-revised re-derivation with eleven genuinely new axes and five
deliberate splits of older ones, and the split halves genuinely separated.

Per-criterion alone-AUC on FIT+MINE is uniformly small and uniformly positive: max .573,
median .532, min .505, **nothing below .50**, 3 criteria ≥ .55. The bank is a large
number of weak, near-independent, same-signed indicators — which is exactly the shape
that makes VA_nl (.624) so much larger than any of its parts.

Blind pairwise adjudication (49 shortlisted pairs + 4 planted anchors, two sealed
judges): **L5 = 32 = L0**. Zero merge edges under either the strict rule (both judges
SAME) or the loose rule (either judge SAME); collapse 0.0%. All four anchors passed for
both judges.

> **Hive-mind caveat, carried.** Both sealed judges were claude-sonnet-5 instances and
> they agreed on 49/49 pairs. Perfect agreement between two samples of one model is
> weaker evidence of concept distinctness than agreement between two families would be.
> The census verdict is therefore quoted as "no duplicate pair survived a blind
> same-family adjudication", not as a cross-family certification. The L2/L3 value-side
> evidence (max |r| = .557, zero pairs ≥ .90) is family-free and points the same way.

Top A criteria by alone-AUC (FIT+MINE):

| alone-AUC | applicability | criterion |
|---|---|---|
| .573 | .92 | Names the real obstacle |
| .556 | .95 | Central reason is extractable |
| .553 | .80 | Omissions are recoverable |
| .549 | .75 | Hypotheses stated and seen to hold |
| .545 | .87 | Load-bearing step not waved through |

### 3.1 The V census — four of the brief's five upstream priors are ALREADY IN THE BANK

This cell's V block is not a reconstructed hand bank nobody would re-propose; it is 28
NAMED surface features, and the priors a Track-B proposer would reach for first are
already inside the articulated instrument:

| upstream prior for math.SE votes | where it already lives |
|---|---|
| LaTeX density | **ALREADY IN V** — `v_latex_density`, `v_latex_cmd_count`, `v_n_display_math`, `v_inline_math_delims` |
| answer length | **ALREADY IN V** — `v_log_len`, `v_word_count`, `v_sentence_count`, `v_avg_sentence_words` |
| confident register / formatting habits | **PARTIAL** — `v_hedging`, `v_first_person`, `v_second_person`, `v_imperative_hint`, `v_meta_edit`, `v_list_marker_count`, `v_paragraph_count` |
| question popularity spillover | **STRUCTURALLY NEUTRALISED** — see §4 |
| answer timing / first-answer advantage | **NOT in any bank** — audited in §4 |

Top V features by alone-AUC: `v_inline_math_delims` .558, `v_n_display_math` .547,
`v_word_count` .542, `v_latex_cmd_count` .542, `v_log_len` .541 (and `v_type_token_ratio`
.462, the only sub-.50 feature in either block).

**Consequence for Track B, fixed now.** A spurious channel that is a monotone function of
a `v_*` column cannot be discounted off Δ without also being discounted off VA — it is a
channel the ARTICULATED instrument already owns. Round-1 discount readouts must say so
instead of double-counting. Length and LaTeX are therefore not available to this cell as
"the dense model is just reading length" explanations: the bank reads length too.

## 4. ROUND 0 — the answer-position covariate (FREEZE ADDENDUM 4). **THE BIG ONE.**

`closure/mathse_vote/position_line.json`, `position_matched.json`,
`position_strat_monitor.json`, `paired_2answer_readout.json`. Every variable here is an
OBSERVED covariate from the population file. None is ever added to V or A, judged by any
LLM, or fit into anything that feeds the closure curve.

### 4.1 The first-answer advantage is enormous

| answer_position | n | label rate | mean raw score | accepted rate |
|---|---|---|---|---|
| **0 (first)** | 4,587 | **.629** | 4.22 | .527 |
| 1 | 4,552 | .450 | 3.33 | .417 |
| 2 | 1,217 | .411 | 3.94 | .222 |
| 3 | 706 | .326 | 3.10 | .137 |
| 4 | 246 | .309 | 4.64 | .098 |
| 5 | 130 | .331 | 3.65 | .108 |
| 6 | 71 | .239 | 4.08 | .056 |
| 7 | 41 | .195 | 4.78 | .000 |
| 8+ | 79 | .266 | 5.42 | .025 |

Alone-AUCs (a single covariate, no text at all):

| covariate | pooled AUC | within-question AUC | ρ with **T** | ρ with **VA_nl** |
|---|---|---|---|---|
| `is_first` | **.601** | **.601** | **+.089** | **−.00007** |
| `answer_position` | .381 (= .619 inverted) | .355 (= .645 inverted) | −.105 | −.016 |
| `n_answers` | **.5026** | **.5000** | −.003 | −.023 |
| `answer_year` | .462 | .452 | −.014 | +.054 |
| **joint position model** (grouped OOF, 6 vars) | **.654** | **.643** | +.106 | +.019 |

Two things to read off this table.

**(i) The y-definition works exactly as designed.** `n_answers` reads .5026 pooled and
.5000 within-question — a question-level covariate cannot move a within-question median
split. The whole "question popularity spillover" family is structurally matched away by
the label, not merely uncorrelated by luck. Era effects are only partly protected
(`answer_year` .462, because answers to one question do spread over time).

**(ii) A no-text model reads .654 pooled and .614 on the dense-held-out rows**, against
the dense text model's .660 on those rows. The two are on different populations and are
not one measurement; what matters is that a six-variable ordinal with no access to a
single word lands within .05 of an 8B model that read all of them.

### 4.2 The dense arm reads arrival order; the articulated bank does not

ρ(`is_first`, dense) = **+.089** against ρ(`is_first`, VA_nl) = **−0.00007**. To three
decimal places the articulated instrument has *literally no* correlation with whether an
answer arrived first, and the dense model has a real one.

**It is not that the bank's raw columns are blind to arrival order — the FITTED
instrument is.** Individually several bank columns correlate with `is_first` about as
strongly as the dense score does: `v_linebreak_count` −.147, `v_log_len` −.138 (first
answers are SHORTER), `v_type_token_ratio` +.131, and on the A side "Engages the asker's
position" +.114, "Delivers the requested kind of thing" −.094. The y-optimal combination
nets the arrival-order component to zero — the length channel and the engagement channel
point opposite ways — while the dense model's end-to-end objective keeps it.

### 4.3 How much of the residual that accounts for — and a SEED-COUNT CORRECTION

**Correction, recorded.** An earlier pass of this audit ran at ONE dense seed and reported
that arrival order absorbed 55% of the residual on HONEST and all of it on MONITOR. That
reading does not survive the third seed and is **retracted**. Two things changed:

* the seed-mean dense score is a stronger instrument than any single seed, and the extra
  strength is *not* arrival-order-related, so the absorbed FRACTION falls;
* a stratified or matched readout needs ONE score vector, so these tables are necessarily
  computed on the seed-mean probability, whose pooled AUC is the **ensemble** figure
  (MONITOR .6763) and is higher than **T = mean of per-seed AUCs** (MONITOR .6597).
  At one seed the two coincided, which is why the discrepancy did not show up before.

**Standing rule for this cell, fixed here:** read the shrinkage WITHIN the ensemble
instrument; never difference an ensemble-based Δ_adj against the T-based Δ₀.

Ensemble-convention discounts on **MONITOR** (the governing population):

| discount | T_adj | VA_adj | **Δ_adj** | share absorbed |
|---|---|---|---|---|
| none (pooled, ensemble) | .6763 | .6460 | **+.0303** | — |
| strata on raw `answer_position` | — | — | +.0246 | 19% |
| decile-stratified, joint position model | — | — | +.0209 | 31% |
| decile-stratified, within-question order vars only | — | — | +.0201 | 34% |
| exact strata on `is_first` | — | — | +.0192 | 37% |
| **matched sampling on the joint position score** (444 pairs) | .6284→ | .6599→ | **+.0135** | **55%** |

Same table on **HONEST** (2,326 rows):

| discount | **Δ_adj** | share absorbed |
|---|---|---|
| none (pooled, ensemble) | **+.0413** | — |
| strata on raw `answer_position` | +.0359 | 13% |
| decile, joint position model | +.0331 | 20% |
| decile, within-question order vars only | +.0325 | 21% |
| exact strata on `is_first` | +.0339 | 18% |
| matched sampling (931 pairs) | +.0311 | 25% |

Stacked increment (FREEZE ADDENDUM 1, stratification-free, grouped-OOF logistic stack on
HONEST):

| stack | AUC | increment over position alone |
|---|---|---|
| position family alone | .6117 | — |
| position + **dense** | .7051 | **+.0935** |
| position + **bank** | .6833 | **+.0716** |

Conditional on everything the arrival-order family knows, the dense arm still adds
**+.0219 more** than the articulated bank does. At one seed that gap was +.0036 and the
honest earlier reading was "the dense edge is almost entirely arrival order"; at three
seeds it is **"arrival order is a large part of the dense edge, and something else
survives it."** The something-else is what rounds 1–5 are for.

### 4.4 Matched sampling is armed from round 0

The freeze triggers matched sampling once a nuisance channel's alone-AUC exceeds .65. The
joint position model reaches **.654 pooled** *before a single Track-B channel has been
proposed*, so matched sampling is the estimator of record on this cell from round 0
onward. It is also the instrument that absorbs the most (55% on MONITOR) — consistent with
decile strata being a coarser control on a channel whose dominant component is binary.

### 4.4b The head-to-head readout, which shows the mechanism directly

73% of this cell's rows sit in questions with exactly two kept answers, so the label there
is a clean head-to-head. Fraction of pairs each instrument orders correctly, split by
whether the FIRST answer won:

| | MONITOR (552 pairs) | HONEST (1,134 pairs) |
|---|---|---|
| share where the **first answer won** | .607 | .604 |
| **T** overall | .6685 | .6702 |
| **VA_nl** overall | .6612 | .6508 |
| **T** where the first answer WON | **.7075** | **.6818** |
| **T** where the first answer LOST | **.6083** | **.6526** |
| **VA_nl** where the first answer WON | .6448 | .6321 |
| **VA_nl** where the first answer LOST | **.6866** | **.6793** |

The dense model is ~.10 better on MONITOR pairs where the first answer won than where it
lost; the bank tilts the *other* way by ~.04. That asymmetry is the signature of a model
carrying an arrival-order prior, and it is why conditioning on `is_first` removes a third
to a half of the residual — it removes exactly the subgroup where the prior pays.

### 4.5 The contrast that makes this a localisation — length and LaTeX do the OPPOSITE

`length_stratification.json`, ensemble convention throughout.

| stratifier | Δ_adj on MONITOR | Δ_adj on HONEST | direction |
|---|---|---|---|
| (none, pooled) | +.0303 | +.0413 | — |
| `v_log_len` deciles | **+.0399** | **+.0497** | Δ GROWS |
| `v_latex_density` deciles | +.0346 | +.0442 | Δ GROWS |
| length × LaTeX 4×4 | **+.0413** | **+.0515** | Δ GROWS |
| joint arrival-order model | **+.0209** | +.0331 | **Δ SHRINKS** |
| matched on arrival order | **+.0135** | +.0311 | **Δ SHRINKS MOST** |

Stratifying on length or LaTeX costs the BANK more than the dense model — unsurprising,
since those columns are *in* the bank — so it raises the residual. Arrival order is the
only channel tested that lowers it. The two signatures are opposite, which is what makes
the position result a localisation rather than a conditioning artifact.

### 4.6 What the articulated instrument is made of

Bank ablations on MONITOR, refit under the closure protocol:

| instrument | pooled AUC | within-question AUC |
|---|---|---|
| **T** (dense, mean-of-seed-AUCs convention) | .6597 | .6536 |
| dense seed-mean score, pooled (ENSEMBLE, never quoted as T) | .6763 | — |
| **VA_nl** (V + A, full) | .6460 | **.6639** |
| VA_nl with the 15 surface `v_*` columns dropped | .6351 | — |
| **A only** (32 rubrics) | .6272 | .6142 |
| **V only** (28 lint features) | .5949 | .6034 |

(i) The 15 surface columns are worth only **+.011** of the bank's pooled AUC — the
articulated instrument is mostly rubric, not lint. (ii) On the within-question readout the
full bank (.664) sits far above either block alone (.614 / .603) *and* above the dense
model (.654): V and A interact strongly on the comparison the label actually encodes.

### 4.7 The round-0 spurious map — MEASURED, not proposed

`round0_spurious_map.json` (alone-AUC on HONEST, ranked by distance from chance). No
proposer had spoken when this was built.

| channel | alone-AUC | kind |
|---|---|---|
| joint arrival-order model (fitted) | **.614** | OBSERVED covariate |
| `position_pct` | .397 (= .603 inverted) | OBSERVED covariate |
| `answer_position` | .404 (= .597 inverted) | OBSERVED covariate |
| `is_first` | **.587** | OBSERVED covariate |
| `v_inline_math_delims` | .565 | **ALREADY IN THE BANK** |
| `v_latex_cmd_count` | .555 | **ALREADY IN THE BANK** |
| `v_word_count` / `v_log_len` / `v_n_display_math` | .551 / .550 / .550 | **ALREADY IN THE BANK** |
| `answer_id_rank_pct` | .472 | OBSERVED covariate |

The map's whole top band is arrival order; everything below it that a Track-B proposer
would name is a column the articulated instrument already owns.

## 5. ROUND 0 — the closure baseline, and a protocol warning

`closure/mathse_vote/mathse_vote_r0_context.json`, recomputed at **three dense seeds**
after the gate landed (`refresh_round0_3seed.sh`).

### 5.1 The round-0 residual on all tiers

T is the mean over dense seeds {42, 1, 2} of the AUC. Per-seed T on MONITOR:
.6570 / .6550 / .6670 (spread .012).

| tier / population | n | T | VA_nl | **Δ₀** |
|---|---|---|---|---|
| **TIER 1 GOVERNING — MONITOR** | 1,124 | .6597 | .6460 | **+.0136** |
| TIER 1 — HONEST (M ∪ MONITOR) | 2,326 | .6597 | .6347 | +.0251 |
| TIER 1 — mining slice M | 1,202 | .6601 | .6238 | +.0363 |
| TIER 1 — eval only | 1,163 | .6709 | .6551 | +.0158 |
| TIER 1 — test only | 1,163 | .6483 | .6139 | +.0344 |
| **TIER 2 — MONITOR, within-question** | 1,124 | .6536 | **.6639** | **−.0103** |
| TIER 2 — HONEST, within-question | 2,326 | .6625 | .6511 | +.0114 |

Question-cluster paired bootstrap of Δ₀ on MONITOR: **[−.0027, +.0622]**, p(Δ>0) = **.965**
(the bootstrap uses the seed-mean score vector, so it is centred high relative to the
T-based +.0136 and is read for WIDTH, not level). Leave-one-question-out jackknife over
the 554 MONITOR questions: SE **.0154**, range [+.0116, +.0151] — no single question
drives it.

### 5.2 Three warnings that travel with every number in this campaign

1. **The closure-protocol level is NOT the dispatched level.** The dispatched +.0366
   compares the dense EVAL AUC at seed 42 against the Layer-1 VA_nl fitted by pooled
   GroupKFold OOF over all 11,629 rows. The closure protocol refits VA on FIT+MINE and
   reads it on MONITOR, where the bank scores **.6460** rather than .6242. That is prereg
   AMENDMENT 1 operating as written — but here it moves the residual from +.037 to +.014.
2. **On the tier that matches the y-definition there is no residual at round 0.** Within
   question on MONITOR the bank is **ahead** by .010. The dense arm's pooled advantage
   comes from cross-question ordering, which is not the comparison the label encodes.
3. **MONITOR is thin.** 1,124 rows / 554 questions gives a Δ bootstrap width of ±.032.
   A round would have to buy more than ~.016 of MONITOR AUC to be individually
   distinguishable from noise, against an ε of .005. Round-over-round differences are read
   as a curve, never as individually significant steps.

### 5.2b What the label is made of (composition audit, round 0)

| | |
|---|---|
| questions with exactly 2 kept answers | 4,246 of 4,960 (**85.6%**) — **73.0% of all rows** |
| questions with 4 kept answers | 518 (10.4%) |
| raw score, median / p10 / p90 | 2 / 0 / 8 |
| winner−loser raw-score gap in 2-answer questions | median **2**, IQR [1, 4] |
| **2-answer questions decided by a gap of ≤ 1 vote** | **35.5%** |
| rows dropped as median ties | 1,372 of 13,001 (10.6%) |

Three quarters of this cell is a head-to-head between two answers to one question, and a
third of those turn on a single vote. That is the cell's noise ceiling talking.

### 5.3 Swap baseline

| population | w₊ | **C₊** | **C₋** | ρ(bank, dense) |
|---|---|---|---|---|
| MONITOR | .657 | **.743** | **.442** | .483 |

Where the dense model orders a discordant pair correctly the bank agrees 74% of the time;
where the dense model gets it wrong the bank is *below* chance (.442). The bank is not
merely tracking the dense model's errors, and it has independent signal to lose.

## 6. GATE — the 3-seed dense T

### 6.1 How the gate was unblocked

The scaleupC dense chain 2 (`sk3:logs/scaleupC_dense_chain2.log`, PID 3162146, GPU 1)
was, at the start of this campaign, on `jokes_community` seed 2 of its "1 2" pass, with
`mathse_accepted_verdict` seeds 1–2 queued ahead of `mathse_vote_score` — an ETA of about
**ten and a half hours** for the gate quantity. Rather than idle the campaign for that
long, `mathse_vote_score` seeds 1 and 2 were run **into the canonical output directories**
on a stacked GPU (GPU 4, behind a 12-day-old idle co-tenant vLLM at 115 GB / 0% util,
68 GB free, ledger CLAIM posted, co-tenant never touched). The chain's own resume logic is
a `RUN_DONE` sentinel test, so when it reaches those cells it will SKIP them and re-score
— no duplicated GPU work, no write collision, and the chain's ledger entries stay correct.
Recipe byte-identical (same `methods/dense/run_dense_standard_scaleupC.sh`, same
`CELLS=` spec, `SEEDS="1 2"`).

### 6.2 The decision rule, and the provisional-start policy — RECORDED BEFORE THE NUMBER

Gate quantity, in the ledger's own convention (so it is commensurable with the dispatched
+.0366): **Δ_gate = mean over dense seeds {42, 1, 2} of the EVAL AUC − VA_nl_mean3
(.6242101942973988)**. Proceed iff Δ_gate > .02; otherwise STOP at round 0 and the seed
verdict is the terminal result. The eval+test and test-only versions are reported beside
it and are never substituted for the rule.

**Provisional-start policy, fixed before seeing seed 1.** Seeds run sequentially, so a
2-seed mean exists ~50 minutes before the 3-seed one. If the 2-seed Δ clears the gate with
a comfortable margin (> .03, i.e. more than the .0142 same-seed eval/test width above the
threshold) the round-1 proposer fleet starts on the 2-seed evidence, **and no round-1
readout is quoted until the 3-seed gate is in hand.** If the 2-seed Δ is marginal
(.02–.03) the fleet waits for seed 2. This buys wall time without letting a provisional
number decide anything.

**How the provisional was read WITHOUT spending any GPU.** Re-running
`score_eval_dense_v4.py` on seed 1 while seed 2 trains on the same device would put a
second Llama-3.1-8B on GPU 4 and risk OOM-ing my own job — declined. But the trainer
already writes `rm_out_seed*/validation_metrics.csv`, the per-checkpoint AUC on the
selection split, and selection *is* on eval, so the best row of that file is the same
quantity the scorer reports (seed 42: trainer best .6621 vs scorer .6608, a −.0013 offset
from the reload).

| seed | trainer best val AUC (eval split) | scorer eval AUC |
|---|---|---|
| 42 | .6621 (step 1048) | **.6608** |
| 1 | **.6736** (step 1048) | *(pending the scorer)* |

Seed 1 came in **above** seed 42. The 2-seed provisional Δ is ≈ (.6608 + .672)/2 − .6242 =
**≈ +.042**, comfortably past the .03 provisional bar, so the round-1 proposer fleet was
started at 03:24 PDT while seed 2 trained. **No round-1 readout is quoted until the 3-seed
gate lands** (§6.3).

**Slice basis, recorded.** The round-1 mining slice was sealed on the seed-42 dense score
— the only one on disk when the fleet started. The slice is a proposal-generation device
and is never a readout; rounds ≥ 2 draw theirs on the 3-seed mean.

### 6.3 The gate result — **PASS**

`gate_3seed.json`. All three seeds trained under the byte-identical dense-standard recipe;
seed 2 finished at 04:12 PDT and `score_eval_dense_v4.py` wrote all three into
`eval_pass_results.json`.

| seed | EVAL AUC | TEST AUC |
|---|---|---|
| 42 | .6608 | .6466 |
| 1 | .6705 | .6451 |
| 2 | .6815 | .6531 |
| **mean** | **.6709** | **.6483** |

| reading | T | − VA_nl_mean3 (.6242) | verdict |
|---|---|---|---|
| **GATE — 3-seed EVAL mean (ledger convention)** | .6709 | **+.0467** | **PASS** (> .02) |
| eval + test pooled | .6597 | +.0355 | pass |
| test only (the selection-free half) | .6483 | +.0241 | pass |

The gate clears on **all three** readings, including the conservative selection-free one,
and the 3-seed EVAL mean came in **above** the dispatched seed-42 figure (+.0467 vs
+.0366) — seeds 1 and 2 both beat seed 42 on eval. The campaign proceeds.

Note the eval/test asymmetry is systematic, not noise: every seed scores higher on eval
than on test (+.014, +.025, +.028), which is the selection-on-eval signature. TEST is the
honest half and its +.0241 is the number to quote if only one is quoted.
## 7. Rounds

**Fleet check before round 1** (recorded, per the freeze's degradation clause). Codex
leg `gpt-5.6-luna` LIVE (smoke call returned in-spec). GLM key A LIVE (`glm-5.2`, 1.6 s).
GLM key B **429 Too Many Requests** at smoke time. `run_fleet.py`'s key rotation makes the
`glm_b` slot fall back to key A, so the round runs at **P = 6 slots across 3 families with
5 distinct credentials**, above the freeze floor of P ≥ 4 / ≥ 2 families.

**Declared steers, fixed before round 1 and held CONSTANT across all rounds** (prereg
AMENDMENT 2 requires proposal shape to be fixed in advance and any steer to be recorded,
not silent):

* Track A keeps the frozen interaction-shaped steer verbatim ("composite / interaction
  criteria are encouraged; this instruction is held constant across all rounds").
* Track B MODE 3 is instantiated for this cell as the answer's ORDER UNDER ITS QUESTION,
  with examples of the *shape* of an arrival-order fingerprint ("as the other answer
  notes", "an alternative approach", "just for completeness" versus language that engages
  the raw question from scratch). Proposers are told they cannot see the actual position
  and must propose a fingerprint scorable from text alone.
* Track B MODE 4 is instantiated as this corpus's upstream priors: answerer standing on
  the site, relationship to the reader, typographic/markup habit, and the kind of question
  the answer is attached to.
* Round 0's arrival-order finding is **not** shown to any proposer. The steer is
  structural (the freeze's own addendum-4 language, specialised to this container), not a
  hint about what round 0 measured.

**Round-1 disagreement slice, built and sealed** (`mathse_vote_r1_slice.json`, 60 rows,
30 `dense_high_card_low` + 30 `dense_low_card_high`, drawn inside M; label-blind, carries
text + both percentile ranks only). The mechanism from §4 shows up in the slice itself:
mean `answer_position` is **0.63** among the rows the dense model ranks far above the
scorecard and **1.57** among the rows the scorecard ranks far above the dense model
(population mean 1.06). The proposers are not told this.

Round-1 fitting state entering the round: FIT+MINE n = 10,505, 60 features,
VA_lin OOF .6200, VA_nl OOF per seed [.6231, .6214, .6228].

### Round 1 — fleet

Full target fleet, no degradation: **P = 6 across 3 families on BOTH tracks**
(claude-opus, claude-sonnet, gpt-5.6-luna ×2 via the Codex companion, glm-5.2 ×2),
12/12 slots returned and parsed, **150 proposals** (90 Track A, 60 Track B). GLM hit
`1302 rate_limit` twice on the Track-B legs and cleared on attempt 3 under the frozen
retry stack; both GLM slots landed, so the round is **not** recorded as degraded.

### Round 1 — a species-machinery failure, caught and repaired inside the freeze

`species.py` clusters a round's proposals by bge-large cosine at τ = .79 and then selects
the top k by cross-proposer support. On this round's Track B that shortcut **under-merged
the single most important channel on the cell**. Four proposers across two families
independently named the answer-arrival-order fingerprint:

| proposer | channel name | its own `upstream_parent` tag |
|---|---|---|
| claude_opus | "Supplementary framing presupposing an already-populated answer set" | position in the answer arrival order |
| claude_sonnet | "Presupposes or names sibling answers on this question" | position in the answer-arrival order |
| codex_luna_a | "Answer-Stream Awareness" | position in the answer-entry stream |
| codex_luna_b | "Reply-aware framing" | position in the entry stream |

The embedding put all four in **separate singleton species**. The consensus-first
selection rule then saw four coin-flips instead of one four-proposer species, and the
channel **missed the scored set entirely** — the Track-B species table was 36 singletons
out of 38, and the top-10 was decided largely by stable hash.

**Why this had to be fixed BEFORE missing-mass accounting, not after.** Good–Turing
missing mass is f₁/N — the share of the pool sitting in singleton species. Splitting one
real channel into four shards moves four proposals from a single 4-member species into
four singletons, which inflates f₁ directly. The τ-only Track-B figures were
S_obs = 51, f₁ = 46, **M̂ = .767** on N = 60; that number is not an estimate of unfound
spurious channels, it is largely a measurement of the clusterer's own fragmentation. Any
B-side remaining-mass claim computed on it would have been wrong in a known direction
(too high). The merged species table is therefore the figure of record and the τ-only one
is kept beside it, labelled, as `good_turing_PREMERGE_tau_only`.

The repair is not a deviation; it is the freeze's own rule. The FREEZE DECLARATION says
concept identity is decided by **"full-recall blind pairwise (NEVER embedding-τ across
registers)"** — the cosine was only ever a shortlist device. `species_merge.py` applies
that rule to the Track-B pool: cross-proposer pairs at cos ≥ .55 are shortlisted, and two
sealed blind judges (plus planted SAME/DIFFERENT anchors) decide identity, exactly as the
round-0 concept census does. Selection then re-runs unchanged on the merged species, and
the τ-only species table, its Good–Turing figures and the pre-merge selection are all kept
beside the merged ones rather than overwritten.

Merge rule, fixed before the verdicts were read: a merge edge requires **both** sealed
judges to say SAME (the strict rule the round-0 census used), with two planted anchors per
judge. Fallback recorded in advance: if only one judge returns, the merge runs on that
judge alone and every downstream number carries a SINGLE-JUDGE flag.

### Round 1 — the merge result: the correction fires, and it is large

**Judge B returned late** (after the session-limit reset killed its first dispatch and a
re-dispatch overran its bounded window), so both the pre-registered STRICT rule and the
fallback LOOSE rule can be reported. Both judges' planted anchors passed. Raw agreement on
the 120 adjudicated pairs: **96/120 = .80**.

| Track-B accounting | τ-only (embedding) | **STRICT, 2 judges** | LOOSE, judge A only |
|---|---|---|---|
| species S_obs | 51 | **41** | 37 |
| singletons f₁ | 46 | **29** | 25 |
| doubletons f₂ | 2 | 7 | 6 |
| **Good–Turing missing mass** | **.767** | **.483** | .417 |
| cross-proposer recapture | .098 | .293 | .324 |
| species named by ≥ 2 families | 3 | 10 | 10 |
| merge edges | — | 14 | 23 |

**Figure of record for B-side missing mass this round: M̂ = .483** (strict, both judges —
the same rule the round-0 census used). The loose .417 is the sensitivity. The τ-only .767
is **RETIRED**: it overstates the remaining mass by 59% relative to the strict figure
purely through f₁ inflation, which is exactly why the merge had to precede the accounting.

**Where the two judges disagree is itself the finding.** Their 24 disagreements concentrate
on the arrival-order family. Both agree SAME on the two core pairs — {"Presupposes or names
sibling answers", "Answer-Stream Awareness"} and {"Cross-answer and cross-user
referencing", "Explicit Reference to Other Answers"} — but split on whether those two
clusters, plus "Reply-aware framing" and "Supplementary framing presupposing an
already-populated thread", are one channel or several. So the strict rule yields **two
2-member arrival-order species** where the loose rule yields **one 4-member species**. The
channel survives either way; only its granularity is contested.

**Which selection was scored, and why — recorded, not silent.** The strict and loose
top-10 Track-B sets overlap **9 of 10**. The single difference:

| | channel |
|---|---|
| LOOSE only (scored this round) | *Presupposes or names sibling answers on this question* |
| STRICT only (not scored this round) | *LLM-Style Templated Boilerplate* |

The **loose selection is the one that was scored**, because judge B arrived after the
blind routing audit and the arbiter had already adjudicated that exact 29-item pool —
blind ids, planted probes and arbiter rulings are all keyed to it, and re-selecting after
seeing the merge outcome would both break the audit chain and be a post-hoc choice. The
strict-only channel *LLM-Style Templated Boilerplate* is therefore unscored in round 1 and
carries forward into round 2's pool automatically. The consequence is stated plainly: the
round-1 map measures the arrival-order channel at the loose granularity and does not
measure the boilerplate channel; the strict two-judge merge governs the missing-mass
accounting only.

### Round 1 — routing audit### Round 1 — routing audit

Audit prompt rebuilt on the merged selection: 29 items (25 criteria + 4 planted
corpus-matched probes), fresh blind Sonnet-class auditor.

| | |
|---|---|
| misrouting rate | **3/25 = .12** |
| **planted probe pass** | **4/4 = 1.00** |
| final routing | A = 14, B = 11 (9 flagged MIXED) |
| disputes → arbiter | 3 |

**The probes work.** Both of the brief's named corpus-matched pairs were classified
correctly by a blind auditor that had never seen them: "Generalizes the method beyond the
specific numbers" → quality_relevant, "Answer length" → incidental; and the two authored
pairs likewise ("Answers the question that was actually asked" → quality_relevant,
"Confident, unhedged register" → incidental).

**All three disputes land on the same boundary**, and it is the one the prereg's own open
question (b) parked as "a substantive decision to be made explicitly per cell, not by
default routing":

| id | criterion | proposed | blind auditor |
|---|---|---|---|
| A01 | Conversational warmth and approachability | A (quality) | incidental |
| A03 | Hint-based or Socratic guidance | A (quality) | incidental |
| B02 | Hint-only pointer style vs full derivation | B (nuisance) | quality_relevant |

Three separate proposers and one blind auditor disagree about whether *pedagogical
register* — warmth, Socratic hinting, hint-vs-derivation — is merit or nuisance on a
crowd-voted maths site. Note A03 and B02 are near-inverses of each other and were routed
to OPPOSITE tracks by their proposers, with the auditor flipping both. That is not noise;
it is the cell's genuine nuisance-vs-merit boundary surfacing on its first round, and it
goes to the frontier arbiter with provenance visible, as the freeze requires.

### Round 1 — the arbiter, and the boundary this cell actually draws

The frontier arbiter (provenance visible by design) **upheld the blind auditor 3/3**:

| id | criterion | ruling | mixed |
|---|---|---|---|
| A01 | Conversational warmth and approachability | → **B** (nuisance) | yes |
| A03 | Hint-based or Socratic guidance | → **B** (nuisance) | yes |
| B02 | Hint-only pointer style vs full derivation | → **A** (merit) | no |

The distinction it drew is sharp and worth keeping: **pedagogical REGISTER is nuisance,
pedagogical SUBSTANCE is merit.** Warmth of tone and the rhetorical mode of asking guiding
questions score how the answer *sounds*; how much of the derivation is actually delivered
versus gestured at scores what the answer *contains*. Two criteria that a proposer had
routed to opposite tracks (A03 and B02 are near-inverses of each other) both moved, in
opposite directions, onto that line. This is prereg open question (b) — "the
nuisance-vs-merit boundary for fluency-like channels is a substantive decision to be made
explicitly per cell" — being decided explicitly, on the record, for math.SE.

Final round-1 routing: **A = 14, B = 11 (9 MIXED)**, `arbiter_present: true`. The recovered
arrival-order channel B04 routes to **B**, as expected.

### Round 1 — LANDMINE: LaTeX tokenises far denser than prose

The first Gemma launch died 750 prompts into rendering with
`VLLMValidationError: maximum context length is 4096 tokens ... your prompt contains at
least 4097 input tokens`. Cause: the bank-matched item view is up to 5,000 source
characters, and on this corpus that is **~2 characters per token** (LaTeX macros, braces,
symbols) against ~4 on the press/peer cells, so the longest maths answers render past
4,096. Shortening the text was not an option — the truncation is matched to the incoming A
bank on purpose (§2.5) — so the context was raised instead: `--max-model-len 8192`,
`--gpu-mem 0.60`. Recorded in `score_gemma_maps.py` as a standing note for any
LaTeX-bearing cell. 294,475 prompts (11,629 rows + 150 anchor texts × 25 criteria).

### Round 1 — READOUT

`mathse_vote_r1_results.json`. Instrument health first: anchors K=50/class,
coherent-vs-scrambled AUC **.9998** (gate passes decisively), pos-vs-neg .610, overall NA
rate .011, no all-NA rows, no interrupted generation, **1 of 25 criteria collapsed**.

#### The curve

| tier | Δ₀ | **Δ₁** | round-1 VA_nl gain | 95% CI | p(gain>0) |
|---|---|---|---|---|---|
| **MONITOR (governing)** | +.0136 | **+.0209** | **−.0073** | [−.022, +.007] | .17 |
| HONEST | +.0251 | **+.0119** | **+.0131** | [+.002, +.024] | .99 |

**The two tiers disagree in sign, and the governing one says round 1 bought nothing.** On
MONITOR the bank went from .6460 to .6388 — mining made it slightly *worse*, so Δ grew.
On HONEST it went .6347 → .6478 and Δ shrank. This is exactly the §5.2 warning firing:
MONITOR is 1,124 rows with a Δ bootstrap half-width of ±.032, and a −.007 move sits well
inside it. Under the frozen signed reading, **round 1 is the first of the two consecutive
sub-ε rounds** that trigger saturation.

**Why a null, when the mined criteria are individually strong?** The five best mined
Track-A criteria all beat the best criterion in the incoming 32-rubric bank
(.573 "Names the real obstacle"):

| alone-AUC (HONEST) | mined criterion |
|---|---|
| **.598** | Proportional Completeness |
| .591 | Dissolves the confusion behind the question |
| .576 | Complete task resolution |
| .573 | Self-Contained Support |
| .570 | Full derivation to closure vs. bare pointer/hint |

So the round did not fail to find good criteria — it found criteria the bank already
spans. That is the redundancy signature, not a mining failure, and it is the substantive
content of a sub-ε round on a bank whose round-0 census showed 32 distinct, near-orthogonal
concepts already in place.

**Swap check: clean.** C₊ .733 → .749, C₋ .429 → .437, ρ(bank, dense) .481 → .498,
`swap_signature: false`. The round did not buy rank agreement with the dense model by
inheriting its errors.

#### The Track-B map — and the round's most important negative

Alone-AUC on HONEST, with each channel's strongest rank correlation against the V block
(the cell-specific annotation added because four of the brief's five upstream priors are
already bank columns):

| alone-AUC | mixed | max\|ρ\| with V | channel |
|---|---|---|---|
| **.570** | yes | **.74 → ALREADY ARTICULATED** | Site-markup fluency vs plaintext ASCII maths |
| **.545** | yes | **.90 → ALREADY ARTICULATED** | Response Volume |
| .534 | yes | .45 | Direct engagement with the asker's specific work |
| .533 | yes | **.74 → ALREADY ARTICULATED** | Direct Reader Coaching |
| .505 | no | .60 | Conversational warmth and approachability |
| .505 | yes | .26 | Cross-answer and cross-user referencing |
| .502 | no | .24 | Subject-matter altitude |
| .501 | yes | .11 | Answerer's disclosed professional identity |
| .499 | yes | .24 | Hint-based or Socratic guidance |
| **.492** | yes | .18 | **Presupposes or names sibling answers on this question** |
| .484 | yes | .42 | Self-correction and uncertainty markers |

**The recovered arrival-order channel scores .492 — chance.** This is the round's most
important result and it cuts against the obvious reading of round 0. The *observed*
covariate arrival order reads .614 on these same rows; the *judged textual fingerprint* of
it — "does this answer presuppose that sibling answers already exist" — reads nothing at
all, and correlates only .18 with the V block. Four proposers across two families
conjectured that fingerprint independently, a Gemma pass measured it corpus-wide, and it
is not there.

So the dense model's +.089 correlation with `is_first` is **not** carried by explicit
sibling-answer references. Either the fingerprint is subtler than any proposer's phrasing,
or the dense arm is reading arrival order through some correlate none of this round's
channels names. Round 2's directed decomposition of the MIXED arrival-order parent is the
obvious next probe, and the freeze's ADDENDUM-3 machinery is the right tool.

**The rest of the map is a mirror.** The only channels with real alone-AUC are the ones
the articulated instrument already owns — markup fluency (ρ=.74 with V), response volume
(ρ=.90, essentially `v_word_count`), reader coaching (ρ=.74). They cannot be discounted off
Δ without being discounted off VA too, which is precisely what §3.1 fixed in advance.

#### Discount and stacked increment

| joint Track-B model | MONITOR | HONEST |
|---|---|---|
| spurious-alone AUC, all 11 channels (HistGB) | .6245 | .6057 |
| spurious-alone AUC, STRICT (9 MIXED channels dropped) | **.5011** | **.4922** |

Dropping the MIXED channels collapses the whole Track-B model to chance: **every bit of
the nuisance set's predictive power lives in channels whose upstream parent plausibly
causes real quality too**, and mostly in ones already inside the bank. Spurious-alone stays
below the .65 matched-sampling trigger, so decile stratification remains the estimator for
the mined set (the *observed* position family, which does exceed .65, keeps its matched
readout from §4.4).

Stratification-free stacked increment:

| stack | HONEST | MONITOR |
|---|---|---|
| joint B alone | .6057 | .6245 |
| B + dense | .6794 (**+.0737**) | .6820 (**+.0575**) |
| B + bank | .6540 (+.0483) | .6562 (+.0318) |
| B + dense + bank | .6888 | .6870 |
| **dense increment over B + bank** | **+.0348** | **+.0307** |
| bank increment over B + dense | +.0094 | +.0050 |

Conditional on every named nuisance channel *and* the full articulated bank, the dense arm
still adds **+.031 to +.035**, while the bank adds only +.005 to +.009 over the dense arm.
That is the residual round 2 has to attack.

#### Missing mass at round 1

| track | S_obs | f₁ | **M̂** | LOPO jackknife |
|---|---|---|---|---|
| A | 57 | 48 | **.533** | [.480, .600] |
| B (strict merge) | 41 | 29 | **.483** | — |

Both tracks are far from saturated on species count, which is the expected shape at round 1
and is the reason the sub-ε reading is a *curve* observation, not yet a plateau claim.

### Round 2 — FREEZE ADDENDUM 3 directed decomposition (design fixed BEFORE the decomposer ran)

Coordinator GO this session, on all three of round 1's flagged decisions. Round 2 is the
Addendum-3 decomposition pass, not a fresh proposal round — recorded as a scope choice.

**Why decomposition and not more proposing.** Addendum 3's remedy for a MIXED parent is to
SPLIT it. Round 1 supplied the trigger: the fleet's conjectured textual fingerprint of
answer-arrival order read **.492 — chance** — while the *observed* arrival-order covariate
reads .614 on the same rows. One fingerprint was not enough, so the parent is split rather
than re-proposed.

**Parents selected, and why** (`mathse_vote_r2_parents_used.json`):

| parent | alone-AUC | \|ρ\| with V | components | reason |
|---|---|---|---|---|
| **Presupposes or names sibling answers** | .492 | .18 | **3** | the coordinator-directed headline parent; gets one candidate-real and **two distinct** surface fingerprints precisely because one was not enough |
| Site-markup fluency vs plaintext ASCII maths | .570 | **.74** | 2 | real alone-AUC but already spanned by the bank's lint — is any of it craft? |
| Response Volume | .545 | **.90** | 2 | essentially `v_word_count` in judged clothing — same question |
| Direct Reader Coaching | .533 | **.74** | 2 | same |
| Direct engagement with the asker's specific work | .534 | .45 | 2 | the one MIXED channel with real signal that is NOT already articulated |

For the headline parent the decomposer is told explicitly that the "does it reference other
answers" fingerprint has already been measured and found empty, and is steered to two
genuinely different surface components — how the answer OPENS and orients to the raw
question (the *first*-answer end), and its SCOPE POSTURE (whole answer vs supplement,
alternative route, special case). It cannot see position; it scores text.

**Scored set** = 11 decomposition components + the one channel the round-1 STRICT
two-judge merge selected that the loose (scored) selection omitted, which inherits here
with its strict-only provenance carried in the record. Parents are **retired** from the
readout once their components are scored — recorded, not deleted, as Addendum 3 requires.

`decompose_r2.py` is cell-specific and says so in its docstring: the stock
`decompose_round.py` builds its prompt around a SHAP interaction screen ("this criterion
carries its weight only with character count / capitalisation / handle count"), which is
not why these parents are MIXED here.

**Round-2 scored set** (`mathse_vote_r2_species.json`), 12 criteria:

| id | side | component | parent |
|---|---|---|---|
| A01 | candidate-real | Route choice named and mathematically motivated in context | sibling-answer parent |
| **B01** | surface | **Extent of from-scratch setup in the opening lines** | sibling-answer parent |
| **B02** | surface | **Scope posture: fragment or supplement versus whole solution** | sibling-answer parent |
| A02 | candidate-real | Notation carries the argument, judged independently of typography | markup fluency |
| B03 | surface | Extent of LaTeX machinery versus ASCII orthography | markup fluency |
| A03 | candidate-real | Every load-bearing step supplied, once, nothing restated | response volume |
| B04 | surface | Extent of sheer text volume, elaboration, repeated explanation | response volume |
| A04 | candidate-real | Names a specific pitfall and explains why it fails | reader coaching |
| B05 | surface | Extent of second-person address, reassurance, imperatives | reader coaching |
| A05 | candidate-real | Lands the exact object asked, fully specialised to it | engagement with asker's work |
| B06 | surface | Extent of verbatim echo and attribution of the asker's material | engagement with asker's work |
| B07 | inherited | LLM-Style Templated Boilerplate | *(strict-only, round 1)* |

B01 and B02 are the two new arrival-order fingerprints, and neither re-proposes the retired
explicit-reference one: B01 targets the **first**-answer end (does the answer orient itself
to the raw question from scratch), B02 the **later**-answer end (does it read as the whole
solution or as a supplement / alternative route / special case).

**A probe-draw bug, found and fixed before round 2's audit ran.** `audit.probes_for` picks
2 of the 4 planted pairs by stable hash of the round tag, and the freeze's stated purpose is
that "a fresh auditor each round never audits the same planted pair as the previous
auditor". With only 4 pairs it can collide — and it did: round 2's plain-hash draw was
round 1's pair set over again. The draw now chains, excluding the previous round's ACTUAL
draw (a first attempt that banned the previous round's *plain-hash* draw still repeated at
r2→r3, so the ban is computed by replaying the chain from round 1). Round 1's realised draw
is unchanged. Round 2 consequently exercises the brief's OTHER named pair — "identifies the
actual error in the asker's approach" vs "contains LaTeX display blocks" — so across rounds
1–2 all four pairs, and both of the brief's named pairs, are used.

### Round 2 — routing audit: the decomposition splits cleanly

| | round 1 | **round 2** |
|---|---|---|
| misrouting rate | 3/25 = .12 | **0/12 = .00** |
| planted probe pass | 4/4 | **4/4** |
| disputes → arbiter | 3 | **0** |
| final routing | A=14 / B=11 (9 mixed) | A=5 / B=7 (4 mixed) |

**Every candidate-real component routed to A and every surface component to B, with a
fresh blind auditor agreeing on all twelve and no arbiter needed.** That is the cleanest
possible evidence that the Addendum-3 split did its job: round 1's pool produced 12%
misrouting and three disputes clustered on the register/substance boundary, and decomposing
the parents on exactly that boundary removed the ambiguity entirely. It also means the
round-2 A-side and B-side are, by construction, the two halves of the same five channels —
which is what makes the Δ₂ readout interpretable as "how much of each MIXED parent was
craft".

---

## ADDENDUM — 2026-08-08, **REGISTERED PRE-Δ₂** (branch decision, not result-contingent)

**Provenance of the timing claim.** Written while `mathse_vote_r2_results.json` did not yet
exist: the check `ls mathse_vote_r2_results.json` returned *pending* at **2026-08-08
11:24:11 PDT**, with round-2 Gemma scoring at ~2% of 141,348 prompts on GPU 6. No Δ₂ number
of any kind had been computed, seen, or estimated when this branch was fixed. Coordinator
decision, this session.

**The distinction this resolves.** Round 2 is a **DECOMPOSITION** round, not a
**PROPOSING** round: it split five MIXED parents into components and ran no sealed
proposer fleet. The stopping rule's intent is *"proposing exhausted"*, and a decomposition
round cannot evidence that.

**(a) Counting.** If Δ₂'s gain is sub-ε it **is** recorded as sub-ε #2, per the frozen
rule's letter. But **no plateau language ships** — not in this note, not in the strict
list, not in any downstream table — until a **full sealed proposer fleet round** has also
come back sub-ε. Saturation, if it is ever declared on this cell, requires a sub-ε
*proposing* round in the count.

**(b) Round 3 runs as a full SEALED fleet regardless of Δ₂'s sign**, with the proposer
count raised from P = 6 to **P = 6–8**. Basis: today's A-side leave-out recovery audit
(`notes/2026-08-08__aside_recovery_audit.md`), whose dose-response fits put rediscovery at
**~70% at P ≈ 6–7 and ~80% at P ≈ 8** (beta-binomial; the zero-inflated fit with asymptote
π = .88 reaches 70% at P ≈ 7, 80% at P ≈ 10). That audit is explicit that raising P is
**"the only route that does"** keep the estimator valid, at a cost of ~2–6 extra sealed
calls per round.

**(c) TWO-TIER RULE (design note §8 addendum), binding.** If a taxonomy-DIRECTED coverage
sweep is also run, it lives in a **separate tier** and **NEVER feeds the Good–Turing
estimator**. The audit measures the directed arm at +.10 mean target cosine but records
that it *"breaks proposal-independence: directed rounds are OUT of the Good-Turing/Chao1
estimator"* and carries category-level bank visibility (weak unsealing). So:

* **Tier S (sealed)** — the only tier that feeds species counts, f₁/f₂, Good–Turing
  missing mass, LOPO jackknife, and cross-proposer recapture.
* **Tier D (directed)** — coverage only. Its criteria may be scored and may join the bank
  through the ordinary blind audit, but they are excluded from every estimator quantity,
  and any table reporting mass must state which tier it counts.

**If Δ₂ is above ε** the sub-ε count **resets to zero** and round 3 proceeds as the fleet
round anyway.

This addendum is the governing text for rounds 3+; where it and the round-1/2 write-ups
differ, this wins.

**Implementation, done now so round 3 cannot start non-compliant.**
`harness_maps.PROPOSERS` is raised to **P = 8 across 3 families** (claude ×3, gpt-5.6-luna
×3, glm-5.2 ×2), eight distinct salts so every slot still reads its own independently
ordered slice; extra slots are the same models under fresh salts, exactly as
`codex_luna_a/b` and `glm_a/b` already were. `run_fleet.py`'s codex default now covers the
third luna slot. `harness_maps.TIER = "S"`, and `species.py` **enforces** the two-tier rule
rather than merely documenting it: any proposal carrying `tier == "D"` is dropped from the
Good–Turing pool, clustering is re-run on the sealed subset, and the emitted block records
`tier` and `n_directed_excluded`. Verified as a no-op on round 1 (S_obs 57/51, f₁ 48/46,
M̂ .533/.767 all reproduce exactly), so the guard changes nothing where no directed tier
exists.

**LANDMINE, found by that very regression check and now guarded.** `species.py` is *not*
idempotent once a round has been merged: re-running it rebuilds the τ-only clustering and
**silently overwrote** round 1's merged `species.json` — the file the blind audit, the
arbiter and a 294,475-prompt Gemma pass were all keyed to. It was caught by diffing against
the `SINGLEJUDGE` backup and restored; integrity was then re-verified three ways (species
ids == scored `crit_ids`, species ids == routing ids, and all criterion names matching by
id, B04 included). `species.py` now **refuses** to overwrite a species file that carries a
`b_merge`, or whose round is already scored or routed, unless `--force` is passed. Any
future worker re-running a completed round hits the refusal instead of a silent
id↔criterion remap.

---

### Round 2 — recovery diagnosis (2026-08-09): the GPU job finished, the LOCAL waiter died

Reported missing overnight as "chain did not complete". Diagnosed before touching a GPU,
and the diagnosis changed the action:

| check | finding |
|---|---|
| scorer / runner PIDs on sk3 | none — the job had exited |
| `mathse_vote_r2_scores.npz` on sk3 | **present, 255,016 bytes** |
| `mathse_vote_r2_score_report.json` | **present** |
| GPU ledger | `RELEASE rc=0` at 2026-08-08T19:57:07Z |
| scored matrix | **(11,629 × 12)** — full population, all 12 criteria |
| anchors | 150 × 12 present |

So the Gemma pass **completed cleanly on GPU 6 and released properly**; what died with the
session limit was my LOCAL waiter-and-readout chain. This was a **harvest, not a relaunch**
— no GPU was claimed, GPU 7 was left free for the other lane, and nothing was recomputed.
Worth stating as an ops lesson: "results file absent locally" is not evidence the compute
failed, and the ledger's own RELEASE line settled it in one look.

Round-2 instrument health, better than round 1's on every axis:

| | round 1 | **round 2** |
|---|---|---|
| coherent-vs-scrambled AUC (gate) | .9998 | **.9571** (pass) |
| pos-vs-neg anchor AUC | .610 | **.679** |
| collapsed criteria | 1 / 25 | **0 / 12** |
| overall NA rate | .011 | **.0022** |
| all-NA rows / interrupted generation | 0 / no | 0 / no |

**Two readout bugs surfaced and were fixed**, both the same root cause: `readout.py`
assumed every round has a sealed fleet, and indexed `species["tracks"]` for Good–Turing and
for `n_species`. A DECOMPOSITION round has no fleet and so no species pool — `tracks` is
absent *by design*. Both sites now record that explicitly rather than crashing, which is
also the two-tier rule doing its job: decomposition components are not independent
proposals and must never enter the estimator, so round 2's standing mass estimate is
carried forward from the last sealed-fleet round rather than recomputed.

### Round 2 — **Δ₂ READOUT**

`mathse_vote_r2_results.json`.

#### The curve

| tier | Δ₀ | Δ₁ | **Δ₂** | round-2 gain | 95% CI | p(gain>0) |
|---|---|---|---|---|---|---|
| **MONITOR (governing)** | +.0136 | +.0209 | **+.0363*** | **+.0013** | [−.0053, +.0075] | .64 |
| HONEST | +.0251 | +.0119 | **+.0284*** | **−.0003** | [−.0050, +.0045] | .44 |

\* Δ₂ levels are on the ENSEMBLE convention the round readouts use (dense seed-mean score
vector); the Δ₀/Δ₁ column entries are the T-convention numbers from §5.1 and §7. Compare
the GAINS across rounds, not the levels across columns — §4.3's standing rule.

**Round 2 is sub-ε on both tiers, and this time the tiers agree.** |gain| ≤ .0013 against
ε = .005, with both CIs straddling zero and p(gain>0) of .64 / .44 — as close to a flat
round as this instrument can report. Round 1's tiers disagreed in sign; round 2's do not.

**Counted as sub-ε #2 per the frozen rule's letter — and NO plateau language ships**, per
the pre-Δ₂ addendum above. Round 2 was a decomposition round, not a proposing round; a full
sealed fleet must also come back sub-ε before saturation can be claimed. Round 3 runs.

#### ⚠ The SWAP SIGNATURE fired

| | C₊ | C₋ | ρ(bank, dense) |
|---|---|---|---|
| entering round 2 | — | — | — |
| **Δ over the round** | **+.0039** | **−.0089** | **+.0156** |
| `swap_signature` | | | **TRUE** (round 1: false) |

C₊ rose while C₋ *fell*: the round made the bank agree with the dense model slightly more
where the dense model is right, and slightly **worse** where it is wrong, with rank
correlation up .016. That is exactly the pattern the freeze defines as buying agreement by
inheriting the dense model's errors. It did not move Δ (gain +.0013), so nothing is
contaminated in the curve — but a decomposition round producing the swap signature is worth
carrying: the candidate-real components may be tracking *what the dense model responds to*
rather than what the crowd rewards. Flagged for round 3's readout to watch.

#### The decomposition worked — one new fingerprint has signal

Track-B components, alone-AUC on HONEST, with V-block overlap:

| alone-AUC | \|ρ\| with V | component | parent |
|---|---|---|---|
| .559 | **.83 ALREADY** | Extent of LaTeX machinery vs ASCII orthography | markup fluency |
| .546 | **.92 ALREADY** | Extent of sheer text volume, elaboration, repetition | response volume |
| .538 | **.83 ALREADY** | Extent of second-person address, reassurance, imperatives | reader coaching |
| .531 | .52 | Extent of verbatim echo of the asker's material | engagement |
| **.520** | .32 | **Extent of from-scratch setup in the opening lines** | **arrival order** |
| **.447** | **.25** | **Scope posture: fragment/supplement vs whole solution** | **arrival order** |
| .502 | .10 | LLM-Style Templated Boilerplate | *(inherited, strict-only)* |

**Scope posture reads .447 — i.e. .553 inverted, and it is NOT already articulated
(ρ = .25).** Round 1's single conjectured arrival-order fingerprint ("does the answer
reference sibling answers") measured **.492, chance**; splitting the parent produced a
component that does carry signal, in the direction "answers that present themselves as the
whole solution rather than a supplement do better". That is Addendum 3 earning its place:
the parent was not empty, it was *mixed at the wrong grain*.

The three surface components of the already-articulated parents behave exactly as §3.1
predicted — .54–.56 alone-AUC and ρ = .83–.92 with the bank's own lint. They are not new
information about the cell; they are the bank's length-and-markup columns wearing a judge's
clothing.

**The inherited strict-only channel is empty.** "LLM-Style Templated Boilerplate" reads
**.502** with ρ = .10. That retires the round-1 worry that scoring the loose selection
rather than the strict one might have cost something: the one channel the strict merge
would have bought instead of the arrival-order channel carries nothing.

Track-A candidate-real components are individually strong — "Notation carries the argument,
judged independently of typography" **.579**, "Every load-bearing step supplied, once"
.573, "Lands the exact object asked" .572 — all above the incoming bank's best rubric
(.573) and above their own surface twins. They still bought no fitted gain, which is the
same redundancy story as round 1.

#### Discount table

| joint Track-B model | MONITOR | HONEST |
|---|---|---|
| spurious-alone AUC, all 7 channels (HistGB) | **.6208** | .6140 |
| spurious-alone AUC, STRICT (MIXED dropped) | **.5880** | .5780 |
| spurious-alone AUC, linear, OOF on FIT+MINE | .6152 | .5830 |

Below the .65 matched-sampling trigger, so decile stratification remains the estimator for
the mined set (the *observed* arrival-order family keeps its matched readout from §4.4).
Note the contrast with round 1: there, dropping the MIXED channels collapsed the nuisance
model to **.501**; here it only falls to **.588**, because the decomposition deliberately
produced non-mixed surface components that carry real signal. The nuisance set is now
made of channels that stand on their own.

Stratification-free stacked increment:

| stack | MONITOR | HONEST |
|---|---|---|
| joint B alone | .6208 | .6140 |
| B + dense | .6797 (+.0589) | .6807 (+.0667) |
| B + bank | .6514 (+.0306) | .6535 (+.0395) |
| B + dense + bank | .6846 | .6878 |
| **dense increment over B + bank** | **+.0332** | **+.0343** |
| bank increment over B + dense | +.0049 | +.0071 |

Essentially unchanged from round 1 (+.0307 / +.0348). **Two rounds of mining — 150 sealed
proposals and a directed decomposition of five MIXED parents — have not moved the dense
model's conditional advantage at all.** That is the quantity round 3 has to attack, and it
is the reason the sub-ε count is not yet a plateau.

#### Missing mass

Not estimated this round, by design: a decomposition round runs no sealed fleet, so there
is no species pool, and decomposition components are excluded from Good–Turing by the
two-tier rule. The standing estimate remains round 1's — Track A M̂ = .533, Track B
M̂ = .483 (strict).

### Round 3 — sealed fleet at P = 8: the dose-response prediction lands

Full sealed fleet, **16/16 slots, 200 proposals** (120 Track A, 80 Track B), P = 8 across
3 families (claude ×3, gpt-5.6-luna ×3, glm-5.2 ×2), eight distinct salts, TIER S only. No
degradation, no retries needed on the GLM legs this round. Bank entering the round:
**79 features** (V 28 + A_base 32 + A_round1 14 + A_round2 5).

**Raising P from 6 to 8 did what the recovery audit's dose-response said it would**, and
the effect is large on both tracks:

| | round 1 (P = 6) | **round 3 (P = 8)** |
|---|---|---|
| Track A — proposals / species / f₁ | 90 / 57 / 48 | 120 / 53 / **34** |
| **Track A Good–Turing M̂** | **.533** | **.283** |
| Track A cross-proposer recapture | .158 | **.360** |
| Track B — proposals / species / f₁ | 60 / 51 / 46 (τ) | 80 / 47 / **29** |
| **Track B M̂ (pre-merge, τ-only)** | .767 | **.362** |
| Track B recapture | .098 | **.380** |

Track-A missing mass **almost halves** (.533 → .283) and recapture more than doubles. This
is the audit's beta-binomial curve (~70% recovery at P≈6–7, ~80% at P≈8) showing up as a
direct measurement on this cell rather than a projection, and it is the strongest available
evidence that the round-1/2 sub-ε readings were not simply an under-powered miner. It also
means the pre-merge τ-only B figure is, for the first time, close to the post-merge one —
with more proposers the embedding has more genuine repeats to find, so it fragments less.

#### Round 3 — Track-B merge, and the arrival-order channel at full consensus

Strict two-judge blind pairwise merge (both judges returned this time; anchors pass for
both; raw agreement **102/122 = .836**; 35 merge edges).

| Track-B accounting | τ-only | **strict merge** |
|---|---|---|
| species S_obs | 47 | **45** |
| singletons f₁ | 29 | **28** |
| **Good–Turing M̂** | .362 | **.350** |
| cross-proposer recapture | .380 | .378 |
| species named by ≥ 2 families | — | 10 |

Note how little the merge moves anything now (47 → 45 species) against round 1's 51 → 41.
That is the P = 8 effect again: with eight proposers the embedding has enough genuine
repeats to find that it no longer shatters channels into singletons, so the τ-only and
post-merge figures nearly coincide. The blind-pairwise step is still run — it is the
freeze's identity rule, not an optimisation — but on this round it is confirming the
clusterer rather than rescuing it.

**B01 is the arrival-order channel, named by ALL EIGHT proposers (P = 8, 8 members)** —
"Cross-reference to other answers or users", "Explicit cross-reference to other answers or
comments", "Language presupposing that sibling answers already exist" and five more, merged
into one species. It is the highest-consensus channel of the entire campaign. Round 1's
version of this channel was a four-way singleton pile-up that the τ clusterer lost; at P=8
it is unmissable.

Selected Track B: B01 cross-reference to other answers (P=8), B02 diagnosing the asker's
posted attempt (P=4), B03 display-math volume share (P=4), B04 vintage of notation/markup
(P=4), B05 dense structured display (P=3), B06 orthographic slips (P=3), B07 answer
length/verbosity (P=3), B08 hint-mode withholding (P=3), B09 direct second-person address
(P=3), B10 revision-trace markers (P=3). **Zero singletons in the scored set** — round 1
had five.

#### Round 3 — routing audit

Misrouting **3/25 = .12**, planted probes **4/4** (this round drew the brief's *other*
named pair, per the probe-chaining fix), final A = 12 / B = 13 (10 MIXED), 3 disputes.

The three disputes land on the **same boundary as round 1's**, which is itself a finding —
two independent sealed fleets, two fresh blind auditors, one recurring fault line:

| id | criterion | proposed | blind auditor |
|---|---|---|---|
| A05 | Notational Discipline and Consistency | A (quality) | incidental |
| A08 | Invites reader dialogue or explicitly requests feedback | A (quality) | incidental |
| A14 | Sustained second-person address to this asker | A (quality) | incidental |

A08 and A14 are the pedagogical-register channels the round-1 arbiter already ruled
nuisance; A05 is the sharper case, since notational discipline sits exactly on the
register/substance line that round 2's decomposition was built to split. Sent to the
frontier arbiter with provenance visible.

#### Round 3 — the arbiter overturns the auditor for the first time

| id | criterion | auditor | **arbiter** | mixed |
|---|---|---|---|---|
| A05 | Notational Discipline and Consistency | incidental | **A — OVERTURNED** | no |
| A08 | Invites reader dialogue / requests feedback | incidental | B — upheld | no |
| A14 | Sustained second-person address to this asker | incidental | B — upheld | yes |

Rounds 1 and 2 saw the arbiter uphold the blind audit 3/3 and 0/0; this is the first
reversal in the campaign, and it sharpens the line the round-1 arbiter drew rather than
contradicting it. Its reasoning: A05's *object is the mathematics itself* — whether symbols
are standard and used consistently — so it is substance, whereas A08 and A14 score
direct-address moves and how much second-person voice runs through the text, which is
register. **"Notation as argument" is merit; "notation as typography" is not** — exactly
the split round 2's decomposition was built around (its A02 component,
"Notation carries the argument, judged independently of typography", was the round's
strongest at .579 alone-AUC). Two sealed fleets, two fresh auditors and two arbiters have
now converged on the same boundary from different directions.

Final round-3 routing: **A = 13, B = 12 (9 MIXED)**, `arbiter_present: true`.

*(round-3 scoring on GPU 7, 294,475 prompts → Δ₃ to follow)*

## 7b. TERMINAL LINE (current — gate PASSED, round 1 complete, first sub-ε round on the board)

> **math.SE vote-score (within-question crowd vote), Layer-3 closure.** The 3-seed gate
> **PASSES** at Δ_gate = **+.0467** (EVAL mean .6709 vs VA_nl .6242; +.0241 on the
> selection-free TEST half). Under the frozen closure protocol the residual is far smaller
> than the dispatched +.0366: **Δ₀ = +.0136 on MONITOR** (bootstrap [−.003, +.062]), and on
> the within-question tier that matches the y-definition the articulated bank is **ahead**
> by .010. **Round 1 bought nothing on the governing tier** — VA_nl gain **−.0073**
> (CI [−.022, +.007]), so **Δ₁ = +.0209**, the first of the two consecutive sub-ε rounds
> the stopping rule needs; on HONEST the same round gained +.0131 and Δ fell to +.0119, and
> the two tiers disagreeing in sign is MONITOR's ±.032 width talking. The null is a
> REDUNDANCY result, not a mining failure: the five best mined criteria all out-score the
> best rubric in the incoming 32-criterion bank (top .598 vs .573) yet add nothing once
> fitted, and the swap check is clean (`swap_signature: false`).
>
> The cell's dominant nuisance is the FIRST-ANSWER ADVANTAGE — label rate .63 at position 0
> vs .45 at position 1, a no-text arrival-order model reading .654 pooled, ρ with the dense
> score +.089 against **−0.00007** for the fitted bank, and matched sampling on it removing
> 55% of the MONITOR residual while length/LaTeX move it the other way. **But round 1's
> central negative is that this channel is not text-visible as conjectured**: the judged
> fingerprint "presupposes sibling answers exist", named independently by four proposers
> across two families and measured corpus-wide by Gemma, reads **.492 — chance**. Whatever
> carries arrival order into the dense model, it is not explicit sibling-answer reference.
> Every Track-B channel with real alone-AUC is one the bank already owns (markup fluency
> ρ=.74 with V, response volume ρ=.90), and dropping the MIXED channels collapses the whole
> nuisance model to .501. Conditional on all named nuisance channels *and* the full bank,
> the dense arm still adds **+.031 to +.035**.

## 8. State at hand-off, and how to resume

Everything below is on disk and the campaign is resumable by a fresh worker from
`RUNBOOK.md` + `DISPATCH.md` alone.

| stage | state |
|---|---|
| alignment gate / dense seeds / **3-seed gate** | PASS / all three / **PASS, Δ_gate = +.0467** |
| round 0 (baseline, census, arrival-order audit, discounts, ablations, sensitivities) | **complete at 3 seeds** |
| round 1 — sealed fleet P=6 → merge → audit → arbiter → score → readout | **complete**; gain −.0073 MONITOR (**sub-ε #1**), Δ₁ = +.0209 |
| round 2 — ADDENDUM-3 directed decomposition (12 criteria) | **complete**; gain **+.0013** MONITOR (**sub-ε #2**), Δ₂ = +.0363; **swap signature TRUE** |
| round 3 — sealed fleet **P=8**, 200 proposals, 16/16 slots | **complete**; M̂ A .533→**.283**, B .483→**.350** |
| round 3 — Track-B strict two-judge merge | **complete**; 35 edges, agreement .836, **B01 = arrival order at P=8** |
| round 3 — audit + arbiter | **complete**; misrouting .12, probes 4/4, A=13 / B=12 (9 mixed); **first arbiter reversal** (A05 notational discipline → merit) |
| round 3 — Gemma scoring, 294,475 prompts, GPU 7 (pinned) | **running**, ~45 min at last check |
| **round 3 — Δ₃ readout** | **chained and armed** (`b4z9e72gb`): pulls scores → `readout.py --round 3` → `mathse_vote_r3_results.json` |
| plateau language | **BLOCKED** until a sealed-fleet round returns sub-ε (pre-Δ₂ addendum). Round 3 IS that round — Δ₃ decides. |

## Caveats that travel with every number here

1. **Levels are protocol-specific.** The closure protocol fits VA on FIT+MINE and reads
   it on MONITOR; the Layer-1 ledger fits pooled GroupKFold OOF over all 11,629 rows.
   The two are not differenceable (prereg AMENDMENT 1). Only round-over-round changes and
   the same-rows honest level are quotable from this campaign.
2. **MONITOR is thin** (1,124 rows / 554 questions; Δ bootstrap half-width ≈ .034). No
   single round's gain is individually significant against that width; the curve is read
   as a curve.
3. **One-third of the label turns on a single vote** (2-answer questions, gap ≤ 1). The
   noise ceiling is low and every residual should be read against it.
4. **The judge saw the question TITLE only**, never the question body. Criteria about fit
   to the asker are decided from title + answer alone, for the bank and for every mined
   criterion alike (the truncation is matched).
5. **sklearn 1.7.2 here vs 1.8.0 for the Layer-1 ledger.** GroupKFold assignments move
   across releases, so Layer-1 levels are not byte-reproducible under this run; the
   campaign's own round-0 anchor is the baseline. The alignment gate is version-free.
6. **The concept census used two same-family judges** (both claude-sonnet-5). Perfect
   agreement between two samples of one model is not a cross-family certification.
7. **`answer_position` is an observed covariate, not a judged channel.** It is never in
   V, never in A, never scored by any LLM, and never fitted into the closure curve. It
   enters only the discount readouts. The textual FINGERPRINT of arrival order is what
   Track B is asked to name and Gemma to score — that is a round-1 measurement, not a
   round-0 one.
8. **Round-1 Track-B granularity is judge-contested.** The two sealed judges agree 80% and
   both pass anchors, but they split on whether the arrival-order family is one channel or
   two. Missing mass is quoted under the strict rule (M̂ = .483); the SCORED set is the
   loose selection, which differs from the strict one by exactly one channel (it carries
   the sibling-answer channel and omits an LLM-boilerplate channel). Neither the scored map
   nor the mass estimate should be quoted without that sentence.
9. **Arrival order is a MIXED channel, not a clean nuisance.** Exposure time is pure
   nuisance, but whoever answers first may also be the person who found the problem easy,
   and a later answerer writes knowing what has already been said. Under FREEZE ADDENDUM
   3 that means it is reported in both the discounted and undiscounted readouts, and its
   textual components get decomposed rather than routed wholesale to one side.

## Appendix — artifact index

All under `methods/taste_decomposition/closure/mathse_vote/` unless stated.

| artifact | what |
|---|---|
| `RUNBOOK.md` | the campaign's own runbook, incl. every cell-specific deviation |
| `cells.py` | cell loader (kept-subset row order, A/V blocks, texts, observed covariates, dense) |
| `oof_alignment_gate.py` / `.json` | the mandatory registry-2026-08-10 gate; PASS at abs_diff 0.0 |
| `build_splits.py`, `mathse_vote_splits.json`, `mathse_vote_population.csv` | salted FIT+MINE / MONITOR split |
| `fetch_dense.py`, `mathse_vote_dense_preds.csv` | per-seed dense probabilities, positional join asserted |
| `round0.py`, `mathse_vote_r0_context.json`, `mathse_vote_r0_preds.npz` | round-0 baseline, tiers, swap, jackknife, gate-uncertainty |
| `census.py`, `census_stage1.json`, `census_blind_packet.json`, `census_verdicts_judge{A,B}.json`, `census.json` | L0→L5 concept census of A and V |
| `position_line.py`, `position_line.json`, `mathse_vote_position.npz` | FREEZE ADDENDUM 4 arrival-order audit |
| `position_matched.py`, `position_matched.json` | matched-sampling discount on the position family |
| `position_strat_monitor.json` | the same decile discounts read on MONITOR (the governing population) |
| `paired_2answer_readout.json` | head-to-head accuracy on 2-answer questions, split by whether the first answer won |
| `round0_spurious_map.json` | measured round-0 spurious map (observed covariates + V columns) |
| `na_indicator_check.py`, `na_indicator_check.json` | imputation sensitivity: NA-indicator columns vs the frozen median-impute |
| `gate_3seed.json` | the 3-seed gate arithmetic |
| `mathse_vote_r{1,2}_{scores.npz,score_report.json,results.json}` | per-round Gemma scores, instrument health, and full readouts |
| `decompose_r2.py`, `mathse_vote_r2_parents_used.json` | the ADDENDUM-3 directed decomposition (cell-specific; parent selection recorded pre-hoc) |
| `length_stratification.py`, `length_stratification.json` | length / LaTeX stratification + bank ablations |
| `harness_maps.py`, `run_fleet.py`, `species.py`, `audit.py`, `arbiter.py`, `readout.py`, `discount_cumulative.py` | the per-round machinery (inherited from `press_verdict/`, patched where recorded) |
| `species_merge.py`, `*_bmerge_packet.json`, `*_bmerge_judge{A,B}.json`, `*_species.PREMERGE.json` | blind pairwise Track-B species merge (the freeze's identity rule) and the pre-merge τ-only state it replaces |
| `mathse_vote_r1_species.{PREMERGE,SINGLEJUDGE,STRICT_TWOJUDGE}.json` | the three species states: τ-only, loose (scored), strict (missing-mass of record) |
| `launch_score.sh` | one-command sync + stacked-GPU launch of a round's Gemma pass |
| `DISPATCH.md` | the sealed-agent dispatch card (proposers / auditor / arbiter) |
| `score_gemma_maps.py`, `gpu_stack_runner.sh` | corpus-wide Gemma-4-31B scoring on sk3, offline batch, stacked-GPU claim |
| sk3 `datasets/math-stackexchange/v2_va/dense_standard_mathse_vote_score/` | the dense arm (3 seeds) |
| `fleet_codex_r1.log`, `fleet_glm_r1.log`, `mathse_vote_r1_proposals_fleet.json` | round-1 fleet transcripts and the 150-proposal pool |
| `gate_and_refresh.sh`, `gate_and_refresh.log` | the chained gate computation + 3-seed round-0 refresh |
| sk3 `logs/mathse_vote_seeds12_gpu4.log` | the accelerated seed-1/2 run that unblocked the gate |
