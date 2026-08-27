# Layer-3 articulation closure — AoPS CURATION cell (same approach as the editorial)

Cell: solutions posted to the Art of Problem Solving competition forums, **y = 1 iff the
forum solution takes substantially the SAME SOLUTION APPROACH as the canonical
editorial / AoPS-wiki write-up of that competition problem**. Not a preference, not a
vote, not an acceptance: an approach-identity match against a reference the model never
sees. The editorial solution is never in the text (`build_provenance.json`).

Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + addenda 1–4.
Queue: `notes/2026-08-09__full_sweep_queue.md`, LANE B second cell, GPU 6.
Campaign dir: `methods/taste_decomposition/closure/aops_curation/`.
Reference campaigns whose machinery this one inherits and diffs:
`../mathse_accepted/` (notes/2026-08-09__closure_mathse_accepted.md) and
`../mathse_vote/` (RUNBOOK/DISPATCH). Those two cells' numbers appear below only as
named contrasts; the three y's are never merged and never differenced.

## Terms, spelled out on first mention (standing rule)

| term | what it means here |
|---|---|
| **V** | the 24 deterministic lint features (`v_*`) computed on the SOLUTION BODY ONLY (the problem statement is stripped first, so statement length/LaTeX cannot leak into a "solution style" feature) |
| **A** | the 44-criterion Gemma-4-31B-judged rubric bank (GEPA-phrased, olympiad-write-up register, 0.0/0.5/1.0 + NA scale) |
| **VA_lin / VA_nl** | the articulated instrument: V+A fit linearly / by HistGradientBoosting (frozen grid), grouped out-of-fold; VA_nl = mean over fit seeds {0,1,2} |
| **T** | the dense arm: Llama-3.1-8B LoRA reward model reading problem + solution. ONE arm, ONE seed, REUSED not retrained |
| **Δ_beyond** | T − VA_nl: the part of the approach-match the articulated bank does not reach |
| **Δ_r** | the closure curve: Δ after r rounds of active mining |
| **FIT+MINE / MONITOR** | the closure split, salted stable-hash on `problem` |
| **M (mining slice)** | on this cell **M = FIT+MINE in full** — see §1.1 |
| **HONEST** | the FULL 5,202-row population = the master ledger's **E** rows |
| **Track A / Track B** | the dual design: A proposes quality-relevant criteria that could close the gap; B proposes suspected-SPURIOUS predictive channels used only to DISCOUNT |
| **MIXED channel** | a Track-B channel whose conjectured upstream parent plausibly causes real quality too; decomposed (FREEZE ADDENDUM 3) rather than routed to one side |
| **alone-AUC** | a single criterion's held-out AUC on its own |
| **swap pair (C₊, C₋)** | P(bank orders a discordant pair correctly \| dense does) and \| dense does not) |
| **missing mass** | fleet-based Good–Turing estimate of the criterion species the miner has not yet found |

## 0. Why this cell is in the campaign, and what round 0 has to settle

The 2026-08-06 FREEZE roster routes cells with matched Δ_beyond ≤ .02 to the
**map-focused dual track** (Track-B emphasis, Track A still run at full k) rather than
excluding them. This cell's dispatched Δ_beyond is **+.0101** (Layer-1 ledger:
T .7806 pooled over the dense arm's own held-out rows, VA_nl .7705 pooled GroupKFold OOF)
— **the smallest starting residual in the sweep**. It enters as a map-focused cell by the
frozen roster, a decision fixed before any closure statistic existed.

**The stopping rule cannot terminate this cell at round 0.** Registered before any
number was read (`round0.py` `stopping_rule_note`): per the coordinator's brief and the
2026-08-08 addendum, a sub-ε round 0 is **not a round**; at least one full sealed
PROPOSING round runs whatever Δ₀ reads. Prereg AMENDMENT 2 is the precedent — the pilot's
round-1 null would have been declared a taste bound had the rule allowed stopping at one
sub-ε round.

## 1. Protocol adaptations, recorded before any mining slice was built

### 1.1 The structural fact that sets this cell apart from every other closure cell

`datasets/math/aops/build_va_population.py` defined the A/V population **as the dense
arm's held-out set** — `split_full/eval` ∪ `split_full/test` of the reused
`runs/aops_same_approach_dense_llama8b` arm — precisely so that T would be same-rows by
construction at zero GPU cost (reuse-first directive 2026-08-07). Consequences:

* **every one of the 5,202 rows is dense-held-out**, so HONEST = the full population =
  the master ledger's E rows (n_E 5,202 / 606 groups / pos .6734, asserted in code);
* the FREEZE's "MONITOR must live inside the dense-held-out rows" is satisfied by *any*
  cut, so the split reverts to the **prereg's base 80/20 rule** rather than the
  .50-within-held-out rule the math.SE cells needed (applying that rule here would throw
  away half the fitting data for no honesty gain — decision recorded in
  `build_splits.py` before any split existed);
* **M (mining slice) = FIT+MINE in full** — dense scores are honest on every row, so
  there is no M-vs-FIT+MINE distinction on this cell.

### 1.2 One dense seed — the convention caveat that does NOT apply here

The dense arm is a single reused LoRA run. `dense_prob` sits in the population file
itself, so there is no fetch/join step and `fetch_dense.py` is deleted from this
campaign. Because there is exactly **one score vector**, the math.SE cells' governing
caveat — "T = mean of per-seed AUCs vs AUC of the seed-mean; never difference the two" —
**collapses**: the two are the same number here. Every Δ in this note, curve or discount
or matched, is on one convention and may be differenced freely.

The cost of one seed is the other side of the same coin: there is **no dense-seed spread
to report**, so the seed-noise band that widened math.SE-accepted's read (spread .025,
wider than Δ₀ itself) is simply unmeasured here rather than small. Recorded as a
limitation, not a strength.

### 1.3 Alignment gate — PASSED EXACTLY

Registry landmine 2026-08-10: `*_va_nl_oof_*.npy` are keyed in **bank item_ids order**.
On this cell `isfinite(ys["same_approach"])` is a no-op (5,202/5,202) and
`population.csv.gz` is written in that same order — `cells.load()` asserts it row-for-row
on both the judgement and the problem column.

```
AUC(y, aops_curation_va_nl_oof_seed0.npy in assembled order) = 0.7689588189522913
published ledger nonlinear.VA["0"].auc                        = 0.7689588189522913
abs diff = 0.0     GATE_PASS = true     shuffled counterfactual = .5020
```

`oof_alignment_gate.json`. Seed 0 only; the mean3 array reads .7735 and is never the gate.

### 1.4 The item view is the BANK's item view, reproduced exactly

The A bank was scored on
`PROBLEM: <statement[:1500]>\n\nFORUM SOLUTION:\n<body under a deterministic HEAD-3000 +
TAIL-2000 middle omission at 5,000 chars>`
(`datasets/va_gemma_banks/score_scaleupC_banks.py::build_aops_curation`). Note the
truncation applies to the **body only** and the statement prefix is added after. The
sibling cells' `score_gemma_maps.py` applies a whole-view HEAD/TAIL cut, which here would
have shown the mined criteria a **different document** from the bank's. `cells.item_view()`
is the single definition, `build_splits.py` writes it into the population CSV, and the
scorer consumes it unchanged. Max realised view: 6,570 chars; median 1,195.

**8192-token context** on the Gemma pass (inherited LaTeX rule: LaTeX tokenises at ~2
chars/token against ~4 for prose, so a 6.5K-char view can exceed a 4,096 window). The
text is never shortened; the context is raised.

### 1.5 Scale mismatch between bank and mined criteria, recorded

The A bank is judged on 0.0 / 0.5 / 1.0 + NA; mined criteria are judged 0–10, the
programme standard on every closure cell (math.SE, press and CW banks are 0/0.5/1 too).
The mismatch is a monotone rescaling absorbed by the StandardScaler / GBM. Recorded, not
corrected, and identical to the sibling cells' handling.

### 1.6 Splits — `aops_curation_splits.json`

Group key = `problem`; no problem straddles the dense eval/test halves (asserted).

| | rows | problems | pos rate |
|---|---|---|---|
| population = **HONEST** = E | 5,202 | 606 | .6734 |
| **FIT+MINE** (= M) | **4,267** | 499 | .6644 |
| **MONITOR** | **935** | 107 | .7144 |

**Salt, recorded not silent.** The dense arm's own 80/10/10 was *not* a plain hash cut —
it was `build_va_population.py::stable_hash_bucket_map`, a greedy size+prevalence
balancer ordered by sha1(problem). The closure cut hashes
`sha256("aops-curation-closure|" + problem)`; the collision check reports that an
unsalted sha256 cut would have put 121 problems in MONITOR against the salted 107, with
Jaccard **.086** between the two sets — the salts do not collide. MONITOR draws
575 eval / 360 test rows, so it is not an artifact of either dense half.

**MONITOR is thin and prevalence-shifted.** 935 rows / 107 problems, pos .714 against
FIT+MINE's .664. Problem groups run 1 to 106 solutions (median 4), so a 20% cut on
GROUPS is not a 20% cut on rows. The Δ bootstrap half-width on MONITOR is ≈ .043 (below);
no single round's gain will be individually significant and the curve is read as a curve.

### 1.7 COLLAPSE GATE — now ENFORCED, not flagged (coordinator ruling 2026-08-09)

Inherited from the jokes_community round-2 finding: `score_gemma_maps.py` *flags*
collapse at modal_frac > .98, but the historic `closure_core.clean_fit` only dropped a
column when fewer than 5 rows sat off the mode — at n = 5,202 a criterion at modal .988
leaves 62 off-modal rows and sails through. The flag was recorded and not acted on.

It is now acted on **inside `clean_fit`**, so the gate applies to **every** refit in this
campaign — round-0 baseline, every round's incoming blocks, the Track-B joint model,
every ablation — not to a single post-hoc regate. Each `fit_block` result carries a
`collapse_gate` record naming what it dropped.

Effect at round 0: the gate drops **`v_list_marker_count`** (modal share .9843, 67
off-modal rows) from V; all 44 A criteria pass; two further V columns
(`v_imath_tag_count`, `v_quote_block_count`) were already caught by the off-modal screen.
The bank enters round 1 at **65 features** (was 66 ungated). Δ₀ moved by ≤ .0002 on every
tier, i.e. the dropped column was near-constant and carried nothing — but the gate is in
force for the rounds, which is where jokes measured it mattering.

### 1.8 Swap-signature algebra of record (coordinator ruling 2026-08-09)

The adverse pattern is **dC₊ > 0 together with dC₋ ≤ 0** — the bank moving toward the
dense model's ordering *including on pairs the dense model gets wrong*. A **rising C₋
means the bank gained independent signal and is FAVOURABLE**, and must not trip the flag.
The historic strict `dC₋ < 0` left dC₋ == 0 unclassified; `readout.py` now carries the
ruling's `<=` and states the rule in the output.

### 1.9 Wait loops test PARSEABILITY (coordinator ruling 2026-08-09)

Inherited from the jokes half-written-verdicts race in `audit.py finalize`.
`waitfile.py` polls for a file that exists, is byte-stable across one poll interval, and
parses as JSON with the required keys — never for mere existence.

### 1.10 sklearn

The Layer-1 ledger was produced under scikit-learn 1.8.0; this campaign runs 1.7.2.
GroupKFold fold assignments move across releases, so Layer-1 LEVELS are not
byte-reproducible here and the campaign's own round-0 anchor is the baseline the curve is
measured from. The alignment gate is version-free (it reads a stored vector).

## 2. ROUND 0 — the honest residual

`aops_curation_r0_context.json`. Bank state entering: **65 features** (24 V → 21 after
the screens, 44 A). A-block NA rate **.2285**.

| tier / population | n | T | VA_nl | **Δ₀** |
|---|---|---|---|---|
| **TIER 1 GOVERNING — MONITOR** | 935 | .7504 | .7392 | **+.0112** |
| TIER 1 — HONEST (= E rows) | 5,202 | .7806 | .7710 | +.0096 |
| TIER 1 — mining slice M (= FIT+MINE) | 4,267 | .7858 | .7760 | +.0098 |
| TIER 1 — eval only | 2,510 | .7739 | .7756 | **−.0017** |
| TIER 1 — test only (selection-free) | 2,692 | .7879 | .7663 | **+.0215** |
| **TIER 2 — MONITOR, within-problem** | 805 | .6863 | .6991 | **−.0128** |
| TIER 2 — HONEST, within-problem | 4,322 | .6861 | .6974 | −.0113 |

Problem-cluster paired bootstrap of Δ₀ on MONITOR: **[−.0308, +.0556]**, p(Δ>0) = .70.
On HONEST: [−.0106, +.0315], p = .81. Leave-one-problem-out jackknife over the 107
MONITOR problems: SE **.0220**, range [+.0037, +.0238] — no single problem drives it
(most influential: 2013_USAMO#2, without which Δ₀ = +.0238).

**The eval/test split is the widest thing in the table and it points the "wrong" way.**
The dense chain selected on EVAL, so TEST is the selection-free half — and the residual
is **−.002 on eval and +.022 on test**, i.e. the *selection-free* half shows the *larger*
gap. That is the opposite of a selection artifact (selection would inflate eval), and it
is a 2.4-point swing on two equal-sized problem-disjoint halves of one trained model. It
is the natural yardstick for how much Δ₀ can move, and it is **twice Δ₀ itself**.

**On the tier that removes the between-problem component the articulated bank is
AHEAD by .011–.013.** Only 310 of 606 problems are label-mixed (274 are all-positive,
22 all-negative), so TIER 2 runs on 4,322 of 5,202 rows; that composition is reported in
`label_composition` so the tier's n is never a surprise.

### 2.1 The master ledger does NOT overstate this cell (contrast math.SE-accepted)

On the **same 5,202 E rows**, three VA fits:

| VA fit | AUC on E | Δ against T (.7806) |
|---|---|---|
| master ledger full-grid OOF arm (`VA_nl`) | .7705 | +.0101 |
| master ledger full-fit-at-E reference | .7735 | +.0071 |
| **closure protocol** (refit on FIT+MINE, grouped-OOF inside it) | **.7710** | **+.0096** |

All three agree to **.003**. This is the cleanest same-rows reconciliation in the
programme so far, and it is a real contrast with the sibling accepted-verdict cell, where
the same exercise moved VA from .5737 to .6147 and cut the apparent residual by two
thirds. Here **the dispatched +.0101 survives the closure protocol essentially unchanged**.

### 2.2 Swap baseline

| population | w₊ | **C₊** | **C₋** | ρ(bank, dense) |
|---|---|---|---|---|
| MONITOR | .749 | **.851** | **.407** | .676 |
| HONEST | .779 | **.873** | **.412** | .719 |

Where the dense model orders a discordant pair correctly the bank agrees 85–87% of the
time — the highest C₊ in the programme — and where the dense model is wrong the bank is
well below chance. ρ(bank, dense) = .68–.72, also the highest seen. The two instruments
are close to being the same ranking, which is the pair-level form of Δ₀ = +.010.

## 3. ROUND 0 — concept census of the incoming bank

`census_stage1.json`, `census.json`.

| level | count |
|---|---|
| L0 criteria delivered | 44 |
| L1 distinct names | 44 |
| L2 after the frozen degeneracy screen + ENFORCED collapse gate (FIT+MINE only) | **44** |
| L3 value clusters at \|r\| ≥ .98 | **44** |
| L5 after blind pairwise adjudication (strict: both judges SAME) | **44** |
| L5′ loose rule (either judge SAME) | **44** |

Max off-diagonal |Pearson r| = **.848**; fraction of column pairs at |r| ≥ .90 = **0**;
collapse L0→L5 = **0.0%**; **zero merge edges under either the strict or the loose rule**.
68 pairs adjudicated (in-register TF-IDF ≥ .30, plus every pair sharing a rubric
`origin`, plus the top-40 cosine tail) + 4 planted anchors; both judges passed **4/4**
anchors and agreed **68/68 = 1.000**.

Perfect agreement on a 68-pair packet is itself worth flagging: it means no pair in this
bank was even arguable, not that the judges were unusually discerning. **The hive-mind
caveat travels** — both judges were claude-sonnet-5 instances, so same-family agreement,
not cross-family certification. The value-side evidence (max |r| = .848, zero pairs at
≥ .90, L3 = 44) is family-free and points the same way.

Per-criterion alone-AUC on FIT+MINE: max **.667** ("The whole question is answered"),
median **.555**, min .443, **23 of 44 at ≥ .55**. That is a substantially stronger bank
than either math.SE cell (max .567 / .573 there) — the olympiad write-up rubric is
genuinely informative about whether a solution takes the editorial's route.

V-block alone-AUCs are the surprise: **`v_numeral_density` .680 — higher than any A
criterion** — then `v_boxed` .622, `v_alpha_share` .366 (= .634 inverted),
`v_answer_stmt` .566, `v_hide_block` .563, `v_latex_density` .558. Solutions that are
dense in *numerals* and end in a *boxed answer* match the editorial approach; solutions
that are dense in *prose letters* do not. This is a real structural fact about the corpus
(AMC/AIME-style numeric problems have one canonical route; olympiad proof problems have
many) and it means **surface form is already carrying a large share of the articulated
instrument** — which the Track-B discount readouts must not double-count.

Ablations on MONITOR: VA_nl full **.7392**, V-only .6936, A-only .7201, and VA_nl with
the 16 surface `v_*` columns removed .7316 — so the surface block is worth **+.008** on
top of the rubric, and the rubric is worth **+.046** on top of surface. On the
within-problem tier the ordering flips: A-only **.7034** beats VA_nl full .6991 and
V-only .6374, i.e. **the surface block earns its keep only between problems**.

## 4. ROUND 0 — the position line (FREEZE ADDENDUM 4). **Observed ordinals, and a clean negative.**

`position_line.json`, `position_covariates.json`, `position_matched.json`. The brief
asked for post position within thread AND problem-thread age, and for observed ordinals
rather than text fingerprints. Both were **recovered**, not inferred:
`build_position_covariates.py` joins `datasets/math/aops/forum_solutions.parquet` (the
raw crawl behind this corpus) on sha1(problem + whitespace-normalised `body_noquote`) at
**100.0% coverage, 5,202/5,202**, key asserted unique. `post_canonical` matches only
4,662/5,202, so `body_noquote` is the join column. 618 topics, 1,760 posters.

Every variable here is an OBSERVED covariate: never added to V or A, never judged by any
LLM, never fitted into anything that feeds the closure curve.

### 4.1 The two axes

| variable | pooled AUC | within-problem | ρ with **T** | ρ with **VA_nl** |
|---|---|---|---|---|
| `post_number` (true thread ordinal) | .430 (= .570 inv.) | .552 | −.198 | −.209 |
| `sol_rank` (rank among the problem's solutions) | .421 (= .579 inv.) | .552 | −.257 | −.269 |
| `is_first_solution` | .535 | .508 | +.185 | +.192 |
| `position_pct` | .536 | .552 | +.075 | +.059 |
| `thread_age_days` | .506 | .552 | −.028 | −.052 |
| `years_after_contest` | .541 | .560 | +.041 | +.036 |
| `post_year` | .531 | .560 | +.044 | +.046 |
| **joint position model** (grouped OOF, 7 vars) | **.6332** | .5615 | **+.346** | **+.327** |
| *context:* `n_sols_group` | .376 (= .624 inv.) | .500 | −.370 | −.373 |
| *context:* `topic_num_views` | .370 (= .630 inv.) | .501 | −.377 | −.373 |
| *context:* `poster_n_posts` | .576 | .562 | +.158 | +.152 |

Label rate by rank within the problem: **.807** at rank 0, .723, .704, .716, .737, .696,
.659, .602, .625, and **.616** at 8+. A real first-solution advantage of about 19 points,
decaying over the first eight posts and then flat — far shallower than the accepted-verdict
cell's .503 → .000 collapse. Thread age is **U-shaped**, not monotone: .755 same-day,
dipping to **.570** at ~15 hours, then climbing back to **.742** in the oldest quintile
(median 13 years). Solutions written in the contest rush and solutions written a decade
later both match the editorial; the ones written in the first day or two of argument do
not.

### 4.2 Three findings that break the sibling cells' pattern

**(i) The no-text position model does NOT beat the dense model — it is not close.**
.6332 pooled against T = .7806. On math.SE-accepted the equivalent model read **.6754
against T .6375**, i.e. arrival order out-predicted an 8B model that read every word.
Here the dense model is 15 AUC points clear of everything the position family knows.

**(ii) The dense arm does NOT read arrival order more than the fitted bank does.**
ρ(joint position, T) = **+.346** against ρ(joint position, VA_nl) = **+.327** — a gap of
.019. On math.SE-accepted the same pair was **+.148 vs +.008**, an 18× ratio; on the vote
cell the same signature appeared. **Here the two instruments read the position family
essentially identically**, which is exactly the condition under which discounting it
cannot move Δ.

**(iii) Discounting on position does not shrink Δ — it slightly raises it.**

| discount (HONEST, pooled Δ = +.0096) | T_adj | VA_adj | **Δ_adj** |
|---|---|---|---|
| joint position model, deciles | .7580 | .7458 | **+.0122** |
| within-problem position vars only | .7593 | .7497 | +.0096 |
| position + context, deciles | .7537 | .7427 | +.0110 |
| `post_number` deciles | .7729 | .7614 | +.0115 |
| `thread_age_days` deciles | .7677 | .7570 | +.0107 |
| matched on the joint position score (1,699 pairs) | .7572 | .7446 | +.0127 |
| exact strata on `is_first_solution` | .7742 | .7652 | +.0090 |

On MONITOR the matched readout is the one exception (Δ_adj +.0038 on 266 pairs) and it is
the thinnest estimator in the table; every stratified reading on MONITOR sits at
+.011–.014, i.e. **unchanged**. Compare math.SE-accepted, where matched sampling on
arrival order removed **103%** of the residual.

**Matched sampling is NOT armed on this cell.** The freeze arms it once a nuisance
channel's alone-AUC exceeds .65; the joint position model reads **.6332**, below the
trigger. Decile stratification is therefore the estimator of record here and the matched
row above is a **declared sensitivity** — recorded in `position_matched.json`
(`TRIGGER_STATUS`) before the numbers were read. Both math.SE cells fired this trigger at
round 0; this one does not.

### 4.3 The length/LaTeX contrast — no localisation either

`length_stratification.json`. Δ_adj on HONEST: `v_log_len` deciles +.0119,
`v_latex_density` +.0115, `v_n_display_math` +.0117, length×LaTeX 4×4 +.0154, against
pooled +.0096. On MONITOR: +.0110 / +.0141 / +.0133 / +.0123 against +.0112.

On the math.SE cells the length/LaTeX strata *raised* Δ while arrival order *lowered* it,
and that opposition is what made the position result a localisation. **Here every
stratifier — position, thread age, length, LaTeX, joint — leaves Δ within ±.006 of
pooled.** There is no channel yet found on which the residual concentrates, and no
channel on which it dissolves.

### 4.4 Stacked increment (FREEZE ADDENDUM 1, stratification-free)

| stack | HONEST | MONITOR |
|---|---|---|
| position family alone | .6198 | .5606 |
| position + **dense** | .7767 (**+.1569**) | .7395 (**+.1789**) |
| position + **bank** | .7690 (**+.1492**) | .7274 (**+.1668**) |
| position + dense + bank | .7911 | .7528 |
| **dense increment over position + bank** | **+.0221** | **+.0254** |

Conditional on everything arrival order and thread age know, the dense arm adds only
**+.008 more than the articulated bank does** (.1569 vs .1492) — the same conclusion the
stratified table reaches, by an estimator that does not degenerate as the nuisance set
grows.

### 4.5 A neighbouring-outcome observation, reported and never modelled

`thanks_received` (crowd approval on the post) reads alone-AUC **.428 pooled** (= .572
inverted) and .448 within problem: **more-thanked solutions are LESS likely to take the
editorial's approach.** That is a coherent story — the forum thanks the clever alternative,
not the canonical route — and it is precisely why this y is not a quality preference. It
is a NEIGHBOURING OUTCOME: never a feature, never a stratifier, never in any model here.
Likewise `match_sim` / `match_kind` in the crawl are the *crawler's* problem-statement
matching scores (how confidently a forum topic was matched to a contest problem), NOT the
solution-vs-editorial similarity that produced y — that comes from
`approach_verdicts.jsonl`. Both carried in the covariate file solely so the record can say
they exist and were never used.

### 4.6 Round-0 spurious map — measured, no proposer had spoken

| channel | alone-AUC (HONEST) | kind |
|---|---|---|
| `v_numeral_density` | **.680** (FIT+MINE) | **ALREADY IN THE BANK** |
| `joint position model` (fitted, 7 vars) | **.633** | OBSERVED covariate |
| `topic_num_views` | .370 (= .630 inv.) | OBSERVED, topic-level |
| `n_sols_group` | .376 (= .624 inv.) | OBSERVED, group-level |
| `v_boxed` | .622 (FIT+MINE) | **ALREADY IN THE BANK** |
| `sol_rank` | .421 (= .579 inv.) | OBSERVED covariate |
| `poster_n_posts` (author standing) | .576 | OBSERVED, upstream |
| `v_latex_density` | .558 (FIT+MINE) | **ALREADY IN THE BANK** |
| `thread_age_days` | .506 | OBSERVED covariate |

Unlike the math.SE cells, the strongest single channel in the round-0 map is **already a
bank column** and the group-level "how many people attacked this problem / how much
traffic did it draw" family (.62–.63 inverted, exactly .500 within problem) is a genuine
between-problem effect that this y — a match against an external reference, not a
within-group split — is **not** structurally protected from. Both facts constrain what
Track B can honestly discount, and both are recorded here before round 1.

## 5. Where round 0 leaves this cell

The dispatched "+.0101, smallest residual in the sweep" survives the closure protocol
intact (§2.1: three VA fits on the same rows agree to .003), sits at **+.0112 on the
governing MONITOR tier** with a bootstrap half-width of .043, is **negative (−.013) on
the within-problem tier**, and — unlike both math.SE cells — is **not** accounted for by
arrival order: the position family reads .633 against T .7806, the two instruments
correlate with it near-identically (+.346 vs +.327), and discounting on it leaves Δ
unchanged or slightly higher.

Rounds 1+ therefore run as the frozen **map-focused dual track**: Track A at full k = 15
across the sealed P = 8 fleet (a null there is evidence, not an omission), Track B at
k = 10 with the FREEZE ADDENDUM 4 position instruction instantiated for this container
(post order in the AoPS thread; thread age; era). The declared steers are fixed here and
held CONSTANT across all rounds (prereg AMENDMENT 2):

* **Track A** keeps the frozen interaction-shaped steer verbatim.
* **Cell-specific hard constraint, both tracks (new here):** *no reference-comparison
  criteria.* The Gemma judge sees the problem and one post and nothing else — no
  editorial, no other post. A criterion of the form "resembles the standard treatment" is
  unscorable on this cell and is discarded at collection. This is stated in both prompts
  because the construct itself is a match against an unseen document and a proposer would
  otherwise reach for it immediately.
* **Track B MODE 3** is instantiated as the post's ORDER IN ITS AoPS TOPIC THREAD and the
  thread's AGE, with examples of the *shape* of such a fingerprint ("as above", "here is
  another way", "Solution 2", era-marked markup). Proposers are told they cannot see the
  actual position and must propose a text-scorable fingerprint.
* **Track B MODE 4** is instantiated as this corpus's upstream priors: poster standing on
  the forum, audience (fellow competitors now vs future searchers), AoPS markup habit,
  and the kind of problem the post is attached to.
* **Round 0's findings are shown to no proposer**, and neither are the sibling cells'.

**Registered expectation, before any round-1 number exists:** the obvious Track-B
fingerprint on this corpus is "announces itself as an alternative" ("another way",
"Solution 2"). On this y that is not obviously spurious — a post that announces a
different route plausibly *is* on a different route — so it is a MIXED channel by
construction and the FREEZE ADDENDUM 3 decomposition pass is expected to fire on it. The
blind audit and the frontier arbiter decide, not this note.

## 6. Rounds

### Round 1 — fleet

Smoke-tested immediately before dispatch: Codex leg `gpt-5.6-luna` LIVE (in-spec reply);
GLM key A LIVE (glm-5.2, 1.1 s); GLM key B LIVE (1.1 s). Fleet runs at the current
standard **P = 8 across 3 families** (claude-opus ×2 salts, claude-sonnet,
gpt-5.6-luna ×3 via the Codex companion, glm-5.2 ×2), TIER S, **16 sealed prompts with 16
distinct row orderings** (97,720 chars Track A / 101,776 chars Track B).

Round-1 disagreement slice `aops_curation_r1_slice.json`: 60 rows drawn inside
M, 30 `dense_high_card_low` + 30 `dense_low_card_high`, label-blind, carrying the item
view and both percentile ranks only (median |gap| .639). Entering fitting state:
FIT+MINE 4,267 rows / 65 features, VA_lin OOF .7642, VA_nl OOF per seed
[.7715, .7735, .7723].

**Round-1 fleet: 16/16 slots returned and parsed, 200 proposals (120 Track A, 80 Track
B), P = 8 across 3 families, NO degradation.** GLM key B's Track-A slot hit two
`IncompleteRead` failures truncating at **exactly the same 15,993 bytes** — a
deterministic gateway cutoff rather than random flakiness, worth recording as an endpoint
signature — and cleared on attempt 3 under the frozen patient retry stack (prompt served
from cache, 38,272 cached tokens, so the weekly GLM cost was one full-price call not
three). Only 6 of 80 Track-B channels were tagged `surface-only`: the fleet almost always
conjectured an upstream parent for what it named.

Round-1 slice `aops_curation_r1_slice.json`: 60 rows in M, median |gap| .639. Entering
state: FIT+MINE 4,267 rows / 65 features, VA_lin OOF .7642, VA_nl OOF per seed
[.7715, .7735, .7723].

### Round 1 — the blind merges, run BEFORE the audit, on BOTH tracks

Two sealed judges per track, strict rule (both judges SAME = a merge edge), 2 planted
anchors per packet, **all anchors passed on both tracks**.

| | Track A | Track B |
|---|---|---|
| proposals | 120 | 80 |
| species, tau-only (embedding) | 83 | 54 |
| **species, STRICT 2-judge (record)** | **87** | **45** |
| f₁ / f₂ | 67 / 12 | 27 / 11 |
| **Good–Turing M̂, tau-only** | .500 | .500 |
| **Good–Turing M̂, STRICT (record)** | **.5583** | **.3375** |
| cross-proposer recapture | .230 | .400 |
| species named by ≥ 2 families | 6 | 13 |
| merge edges (strict) | 33 | 35 |
| pairs adjudicated | 150 | 120 |

**This is why the brief made both tracks blind-merge, and it is a result in its own
right.** At tau the two tracks looked identical — .500 and .500. Under the freeze's actual
identity rule they diverge sharply, and **in opposite directions**: the embedding
UNDER-merged Track B (54 → 45, mass .500 → .3375, the familiar f₁-inflation the B-side
merge was invented to remove) and **OVER-merged Track A** (83 → 87, mass .500 → .5583).
The sibling math.SE campaigns merged only Track B and quoted Track A at tau, so their
A-side mass was reported on a rule their B-side mass had already been shown to violate.
On this cell the corrected figures say the A-side pool is **substantially less** covered
than the B-side (.56 vs .34), which is the reverse of what the tau numbers suggested.

The Track-B merge promoted the arrival-order channel to the largest species in the pool:
**B01 "Inter-post referencing and sequence markers", 8 of 8 proposers**. The §5 registered
expectation ("the obvious Track-B fingerprint here is *announces itself as an
alternative*") is confirmed — a full-fleet consensus channel.

### Round 1 — routing audit, and a planted probe that failed

**The first blind auditor scored 3/4 on the planted battery**, routing P03 "Argues the
step a competent reader would stop at" (quality-relevant) as incidental. Both math.SE
campaigns recorded 4/4 every round, so the freeze names no remedy. One was registered in
`audit_readmission_rule.json` **before the replacement's verdicts existed**, on the
precedent of this cell's own Layer-1 Gemma anchor battery, which marks a shard invalid and
redraws when the pos/neg ordering fails (`aops_curation_ledger.json` anchor_reports:
shards 4 and 5 each retried once). An auditor is a judge; a judge that fails its blinded
known-label anchors is not admitted. Auditor #1's verdicts and routing are retained
verbatim as `*_AUDITOR1_FAILED_PROBE.json`, never deleted, and the rule fixes in advance
what happens if the replacement also fails (degraded-audit flag, both routings reported,
arbiter adjudicates every auditor-vs-auditor disagreement; no third draw).

**The probe battery earned its keep.** Auditor #2 passed **4/4**. The two auditors agreed
on 27 of 29 items (`audit_auditor_agreement.json`), and **both disagreements ran the same
way — auditor #1 calling a quality-relevant item incidental**, the identical bias the
probe caught. The failure was systematic, not a coin flip, and a 3/4 battery detected it.

Routing of record (auditor #2): **misrouting 5/25 = .20**, 5 disputes to the frontier
arbiter, which upheld the auditor on **4 of 5** (A09, A02, A06, A08 → B, three flagged
MIXED) and **overturned it on A07** ("Post questions or disowns the argument it posts" →
A: reading whether a post asserts, hedges or disowns its own argument is an epistemic
judgement about the mathematics, not a register marker). Final: **A = 11, B = 14, 6
MIXED**.

### Round 1 — Gemma scoring

Corpus-wide pass on **GPU 6** (lane card; ledger CLAIM-STACKED at 182,632 MiB free, 0%
util, no compute apps verified immediately before the claim; RELEASE rc = 0 after 24 min).
5,202 rows + 150 anchor texts × 25 criteria = **133,800 prompts**, offline batch,
`--max-model-len 8192`, the bank's own item view consumed unchanged.

Instrument health: anchors K = 50/class, **coherent-vs-scrambled .9789** (gate passes),
pos-vs-neg .6986, overall NA rate **.0012**, **0 of 25 criteria collapsed**, no all-NA
rows.

### Round 1 — READOUT

`aops_curation_r1_results.json`. Bank 65 → **76** features (11 A-routed criteria join).

| tier | Δ₀ | **Δ₁** | round-1 VA_nl gain | 95% CI | p(gain>0) |
|---|---|---|---|---|---|
| **MONITOR (governing)** | +.0112 | **−.0004** | **+.0116** | [−.0014, +.0258] | .954 |
| HONEST | +.0096 | **+.0086** | **+.0010** | [−.0049, +.0067] | .644 |

**Round 1 is NOT sub-ε on the governing tier** (+.0116 against ε = .005), so the stopping
rule does not begin counting and **round 2 is running**. On MONITOR the residual is now
**−.0004 — fully closed**. On HONEST nothing happened.

**The two tiers disagree, and the thin-MONITOR caveat is the reason to distrust the
bigger number, not the smaller one.** MONITOR is 935 rows / 107 problems with a Δ
bootstrap half-width of ≈ .043; a +.0116 gain sits well inside that. HONEST is 5,202 rows
and says +.0010. Registered reading, before round 2: **the honest interpretation of round
1 is "no measurable closure, with a MONITOR-sized fluctuation that happens to point
down"** — the same shape the sibling vote cell hit when its two populations disagreed in
sign. The curve is read as a curve.

**The swap signature FIRED** under the ruling's algebra (dC₊ = +.0027, dC₋ = **−.0047**,
dρ = +.0075 → adverse). Part of the MONITOR gain was bought by moving the bank toward the
dense model's ordering *including on pairs the dense model gets wrong*. The round-1
increment must not be quoted as clean articulation gain.

**Track A cleared more than either math.SE fleet managed.** Best mined criterion **.654**
("Method sophistication proportionality to problem"), then .616 ("Post questions or
disowns the argument it posts" — the arbiter's one overturn), .595 ("Exploits the
multiple-choice option list"), .583, .583. Against an incoming A-bank best of .667 on
FIT+MINE, the fleet came close without clearing it — but unlike math.SE, where two fleets
produced nothing above the bank's ceiling, several mined criteria here land in the bank's
top decile, and one of them is corpus-specific in a way no rubric-writer had reached
(**the multiple-choice option list**: AMC/AIME problems supply answer choices, and
exploiting them is a route the editorial write-up never takes).

#### The Track-B map — and the arrival-order negative does NOT replicate here

Alone-AUC on HONEST (inverted where the channel predicts downward):

| alone-AUC | mixed | channel | conjectured parent |
|---|---|---|---|
| **.649** | yes | Algebraic step transparency *(arbiter-rerouted A09)* | surface-only |
| **.617** | yes | Answer-choice anchoring | assessment format |
| .616 (= .384 inv.) | no | Colloquial and emotional register | poster's practice / audience |
| **.599** | no | Markup scaffolding | typographic habit |
| .589 (= .411 inv.) | no | Validation-seeking and uncertain posture | poster confidence |
| **.582 (= .418 inv.)** | no | **Inter-post referencing and sequence markers (8/8 proposers)** | **position in the entry stream** |
| .552 | no | Density of display mathematics and align environments | LaTeX fluency |
| .531 | no | Legacy autocorrupted LaTeX and era-marked markup | calendar era |
| .528 | yes | Absence of unconventional-approach markers | surface-only |

**This is the round's most important result and it breaks the programme's three-times
replicated central negative.** On math.SE the judged textual fingerprint of arrival order
read .516, .510 and .492 — chance — against observed covariates of .660 and .614. Here the
fingerprint reads **.582 inverted against an observed `sol_rank` of .579 inverted and a
joint observed position model of .633**. The text-visible trace and the observed ordinal
are **the same size**. Posts that reference earlier posts are less likely to take the
editorial's approach, and a sealed fleet can name that in text.

Why this cell and not those: on math.SE the channel had to be inferred from tone, whereas
AoPS threads carry explicit, conventionalised sequence furniture — "Solution 2", "another
way", "shorter than the above". The arrival-order channel is **lexicalised** in this
corpus and merely **pragmatic** in the other. That is a claim about corpora, not about
model capability, and it is the first positive fingerprint result in this line.

#### Discount table and stacked increment

**Matched sampling is now ARMED**: the mined joint-B model reads **.7231 on HONEST /
.7135 on MONITOR**, far above the freeze's .65 trigger (the *observed* position family at
round 0 read .633 and did not arm it). Decile stratification and matched sampling are both
reported from here on.

| estimator | Δ_adj HONEST | Δ_adj MONITOR |
|---|---|---|
| none (pooled) | +.0086 | −.0004 |
| decile strata, joint B, all 14 channels | **+.0325** | +.0041 |
| decile strata, joint B, STRICT (6 MIXED dropped) | +.0177 | +.0074 |
| **matched sampling, joint B, all 14** (1,534 / 241 pairs) | **+.0284** | **+.0187** |
| matched sampling, joint B, STRICT (1,545 pairs) | +.0252 | — |

`aops_curation_r1_cumulative_discount.json`. Discounting the mined nuisance set **raises**
Δ on both estimators and both populations, as on both math.SE cells and for the same
reason: the strongest B channels are ones the articulated bank already owns, so
conditioning on them costs the bank more than it costs the dense arm. Note the two
estimators disagree on MONITOR in the direction that matters for reading round 1 — decile
stratification leaves Δ at +.004 while matched sampling puts it at **+.019**, i.e. the
matched estimator says the MONITOR closure seen in the raw curve does not survive holding
the nuisance set fixed. That is a second reason, independent of the bootstrap width and
the swap signature, to read round 1 as no measurable closure.

| stack | HONEST | MONITOR |
|---|---|---|
| joint B alone | .7231 | .7135 |
| B + dense | .7811 (+.0580) | .7501 (+.0366) |
| B + bank | .7707 (+.0476) | .7494 (+.0359) |
| B + dense + bank | .7908 | .7614 |
| **dense increment over B + bank** | **+.0201** | **+.0120** |
| bank increment over B + dense | +.0096 | +.0113 |

Conditional on all 14 named nuisance channels, the dense arm adds **+.010 more than the
bank does on HONEST and +.001 more on MONITOR** — on the governing tier the two
instruments are indistinguishable once the nuisance set is held.

### Round 2 — running as a full sealed PROPOSING round

Registered before launch: round 1 was **not** sub-ε on the governing tier, so the stopping
rule has not begun counting. Round 2 runs as another full sealed P = 8 fleet with all
steers held constant (prereg AMENDMENT 2), slice rebuilt on the round-1 bank (76
features, VA_nl OOF per seed [.7702, .7708, .7728], median |gap| .634). Round 1's findings
are shown to no proposer.

## 7. Artifact index

All under `methods/taste_decomposition/closure/aops_curation/` unless stated.

| artifact | what |
|---|---|
| this note | the campaign record; machinery inherited from `../mathse_accepted/`, diffed in §1 |
| `cells.py`, `oof_alignment_gate.py` / `.json` | loader + the mandatory registry gate (PASS, abs diff 0.0) |
| `build_splits.py`, `aops_curation_splits.json`, `aops_curation_population.csv` | salted 80/20 FIT+MINE / MONITOR + collision check; the population CSV carries the BANK's item view |
| `build_position_covariates.py`, `aops_curation_position_covariates.csv`, `position_covariates.json` | 100%-coverage recovery of the TRUE thread ordinals from the raw crawl |
| `closure_core.py` | frozen fitting spec + the ENFORCED collapse gate (§1.7) |
| `waitfile.py` | parseability-not-existence wait (§1.9) |
| `round0.py`, `aops_curation_r0_context.json`, `aops_curation_r0_preds.npz` | round-0 baseline, tiers, swap, jackknife, master-ledger reconciliation |
| `census.py`, `census_stage1.json`, `census_blind_packet.json` | L0→L3 concept census + the blind judge packet |
| `position_line.py` / `.json`, `position_matched.py` / `.json`, `aops_curation_position.npz` | FREEZE ADDENDUM 4 audit + the declared matched sensitivity |
| `length_stratification.py` / `.json` | the length/LaTeX contrast control + surface/rubric ablations |
| `harness_maps.py`, `run_fleet.py`, `aops_curation_r1_slice.json` | sealed dual-track proposer harness (AoPS MODE 3/4, no-reference-comparison rule) + the non-Claude legs |
| `species.py`, `species_merge.py`, `audit.py` (AoPS-matched probe pairs), `arbiter.py` | species + two-tier guard, blind pairwise Track-B merge, blind routing audit, arbiter |
| `score_gemma_maps.py`, `gpu_stack_runner.sh`, `launch_score.sh` | corpus-wide Gemma-4-31B offline batch, 8192 ctx, LANE_GPU=6 pin |
| `readout.py` | per-round readout, with the swap algebra of record (§1.8) |
