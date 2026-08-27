# jokes_community — Layer-3 articulation-closure campaign (LANE A, cell 1)

Cell: **reddit-jokes community** — r/Jokes posts, y = crowd upvote quartile (top 25% vs
bottom 25% of raw score *inside* a `length_bin × format × topic` stratum; the middle 50%
was dropped upstream). 16,000 posts, 50 LDA topics, pos-rate .496.

Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + Addenda 1–4.
Queue: `notes/2026-08-09__full_sweep_queue.md` (LANE A, GPU 5).
Code + artifacts: `methods/taste_decomposition/closure/jokes_community/`.
Instrument provenance: `notes/2026-08-08__scaleupC_builds.md` BUILD 1.

Abbreviations spelled out, per the standing rule. **V** = 27 deterministic label-blind
surface features. **A** = the 47-criterion GEPA-phrased bank scored one criterion at a
time by Gemma-4-31B on the {1.0, 0.5, 0.0, NA} scale. **VA_lin / VA_nl** = grouped-OOF
logistic / HistGradientBoosting aggregation of the same V+A matrix, seed-mean over
{0,1,2}. **T** = the dense standard (Llama-3.1-8B LoRA, 3 seeds 42/1/2). **Δ_beyond** =
T − VA_nl. **MONITOR** = the frozen decision population. **HONEST** = every dense-held-out
row. **Track A** = candidate-real criteria mined to close the residual. **Track B** =
suspected-spurious channels mined to discount it. **Good-Turing missing mass** = the
share of the species pool represented by singletons, i.e. the estimated mass of concepts
the fleet has not yet named. **LOPO** = leave-one-proposer-out.

---

## 0. What was already on disk (nothing rebuilt)

| artifact | value |
|---|---|
| Layer-1 ledger `results/jokes_community_ledger.json` | V_lin .585 / A_lin .716 / VA_lin .717 / VA_nl .7321 (seeds .7322/.7325/.7317) / T .747 (seed 42 eval) |
| Master-ledger E rows `results/vat_fullgrid_jokes_community.json` | n_E 3,163 · T .7469 (**seed ENSEMBLE**) · VA_lin .6643 · VA_nl .6888 · VAT_nl .7375 · V3 .7411 |
| A/V matrix | `outputs/va_gemma_banks_scaleupC/jokes_community_shard{0..7}.npz` |
| Dense standard | `datasets/humor/reddit_jokes/dense_standard/` on sk3, **all three seeds present** (eval .7470/.7507/.7508, test .7236/.7254/.7283) |

**Two T conventions exist for this cell and are never mixed in one figure.** The master
ledger quotes the **seed-ensemble** AUC (.7469 at the 3,163 E rows). This campaign quotes
the programme's **mean-over-seeds-of-the-AUC** convention (.7377 on the same rows), which
is what VA_nl's own seed-mean convention requires.

---

## 1. Design decisions taken BEFORE any proposal was read

**Splits** (`build_splits.py`, salt `"jokes-community-closure|"`).
MONITOR ⊂ dense-held-out, as the freeze requires: within the 10 dense-held-out topics,
`sha256(salt + topic) ≥ .50` → MONITOR.

| split | rows | topics | pos-rate |
|---|---:|---:|---:|
| FIT+MINE | 14,083 | 44 | .4962 |
| — of which mining slice M (dense-held-out) | 1,246 | 4 | — |
| MONITOR | 1,917 | 6 | .4945 |
| HONEST = M ∪ MONITOR | 3,163 | 10 | — |

Salt-collision check: the unsalted cut would have put 5 of the 10 held-out topics in
MONITOR, the salted cut puts 6 — the two cuts differ, so the dense arm's split salt does
not collide with the plain one. Group overlap between FIT+MINE and MONITOR: **0**.

**Granularity caveat, registered here rather than discovered later.** This is the
coarsest-grouped cell in the programme: MONITOR holds only **6** grouping units, so a
group-cluster bootstrap over MONITOR is coarse by construction. Every readout therefore
prints the item-level band beside the group band; **the group band is the quoted one**
and the item band is never substituted for it.

**Readout tiers, declared before any round.**
TIER 1 (governing, the stopping rule reads it) = pooled AUC on MONITOR.
TIER 2 (secondary) = n-weighted within-**topic** AUC — y is a quartile split taken inside
`length_bin × format × topic` strata, so the within-topic readout is the one that matches
the y-definition. Reported every round, never substituted for TIER 1.
TIER 3 (diagnostic) = eval-only / test-only / HONEST same-rows level.

**Roster tier.** The FREEZE DECLARATION's roster runs the *full* dual-track where matched
Δ_beyond > .02 and the *map-focused* dual-track (Track-B emphasis, Track A still run) on
the rest. This cell's Layer-1 Δ_beyond is **.0149** and its round-0 closure-protocol
Δ_beyond is **+.0143** — below the line. Both tracks run at full budget (k_A = 15,
k_B = 10, sealed fleet P = 8), and the **spurious map is the headline** while the Track-A
closure curve is reported in full. That ordering is fixed now, before any proposal.

**What the fleet is never told.** y strata on `length_bin × format × topic`, so raw
length and coarse format are already partly matched out of the label. Telling a proposer
would be a design steer outside the freeze; the fact is used only when the map is
interpreted — a length-family channel that still carries alone-AUC here is carrying
residual *within*-stratum variation, and must be read that way.

**FREEZE ADDENDUM 4 on a corpus with no sibling container.** An r/Jokes post has no
container of siblings. The honest ordinal is **position in the subreddit's own posting
stream** — which era of the forum's conventions, running gags and repost cycle the item
comes from. `build_covariates.py` recovers `created_utc` for **13,772 / 16,000 (86.1%)**
rows by an exact `sha1(title + " " + selftext)[:20]` join against the raw scrape;
unmatched rows are carried as NaN and dropped from each readout, never imputed. The raw
`score` column is deliberately **not** carried — it is what y is defined from.
`era_line.py` is the observed-covariate audit; the Track-B MODE 3 brief asks the fleet for
the *textual fingerprint* of that ordinal, so the two can be compared exactly as the
math.SE cell compared its arrival-order fingerprint against the observed ordinal.

**Alignment gate** (`oof_alignment_gate.py`, mandatory inside `cells.load()`):
AUC(y, `jokes_community_va_nl_oof_seed0.npy` in assembled bank-item_ids order) =
.7321856098790323 = the ledger's `nonlinear.VA["0"].auc`, **abs diff 0.0**. PASS.
Shuffled counterfactual .5015. sklearn here is **1.7.2**, not the ledger's 1.8.0, so every
number the campaign quotes is recomputed under one version (fold *assignments* move
across releases; the gate reads a stored vector and is version-independent).

---

## 2. Round 0 — the residual the campaign starts from

`jokes_community_r0_context.json`. Bank state 0 = V + A, 74 features after the
degeneracy screen, grid picks 15/15/15, VA_nl seed spread on MONITOR **.00016**.

| population | n | topics | T (mean-of-AUCs) | VA_nl | Δ_beyond |
|---|---:|---:|---:|---:|---:|
| **MONITOR (TIER 1)** | 1,917 | 6 | .7421 | .7278 | **+.0143** |
| HONEST | 3,163 | 10 | .7377 | .7235 | +.0142 |
| mining slice M | 1,246 | 4 | .7311 | .7179 | +.0132 |
| eval only | 1,663 | 5 | .7495 | .7423 | +.0072 |
| test only | 1,500 | 5 | .7257 | .7039 | +.0218 |

TIER 2 (within-topic): MONITOR Δ = **+.0126**, HONEST Δ = +.0130.

Bands on the MONITOR Δ: topic-cluster bootstrap **[+.0139, +.0396]**, p(>0) = 1.00
(bootstrap mean +.0247 — the mean sits above the point estimate because resampling 6
topics changes the composition an AUC is computed over; read the band for width, not for
a level). Item-level band [+.0021, +.0456], p(>0) = .985. Leave-one-topic-out jackknife:
mean +.0145, SE .0080, range [+.0090, +.0196], most influential topic t09.

**The split-half caveat that governs how this campaign should be read.** eval-only Δ is
+.0072 and test-only Δ is +.0218 — a gap of .0146, i.e. *as large as the residual itself*,
between two equal-sized disjoint topic-grouped halves of the same models. The dense chain
selected on eval, so test is the selection-free half and reads the *higher* Δ here. Any
round-over-round movement smaller than ~.015 on this cell is inside that noise, and the
ε = .005 stopping rule is being applied to a quantity whose split-half spread is 3× ε.
This is recorded now, before rounds, so it cannot be discovered post hoc.

Swap baseline (MONITOR): w₊ .752, C₊ .8206, C₋ .4466, Spearman(bank, dense) .628.
(HONEST: C₊ .8243, C₋ .4263, ρ .646.)

---

---

## 2b. Round 0 — FREEZE ADDENDUM 4 observed-covariate line (`era_line.json`)

The container is the subreddit's posting stream; the ordinal is `created_utc`.
**The ordinal is real.** Every T in this file is the seed-ENSEMBLE AUC (the stratified and
stacked readouts need one score column) — labelled in the file, never mixed with the
campaign's mean-of-AUCs T.

| covariate | alone-AUC (full) | alone-AUC (HONEST) | within-topic | ρ with dense |
|---|---:|---:|---:|---:|
| `created_utc` / `era_rank_pct` | **.5925** | .5959 | .5929 | +.099 |
| `post_year` | .5881 | .5888 | .5884 | +.097 |
| `post_month_of_y` | .5011 | .5137 | .5016 | +.001 |
| `post_hour_utc` | .4870 | .4915 | .4864 | +.013 |
| `post_dow` | .5061 | .5106 | .5070 | +.015 |
| joint era model (grouped OOF) | .5912 | .5912 | .5923 | +.108 |

Label rate by year is monotone: 2015 .353 → 2016 .386 → 2017 .481 → 2018 .528 →
2019 .575 → 2020 .571 (n = 1,406 / 2,703 / 2,962 / 2,779 / 3,130 / 792). Mean dense score
tracks it (.446 → .537), i.e. **the dense model reads the era**. Submission-timing
covariates with no craft component (hour, day of week) are flat — the channel is *era*,
not *time-of-day visibility*.

This is the **third** cell in which the Addendum-4 position-in-container family lands
(patents claim-ordinal .754, code repo-recency, now r/Jokes era .592) and the first on a
corpus whose items have no sibling container at all.

Discounts on HONEST (decile-stratified): pooled Δ .0233 → **Δ_adj .0200** stratifying on
the joint era model, **Δ_adj .0214** on raw era rank. Stacked increment: era alone .5902,
era + dense .7560 (**dense increment +.1658**), era + bank .7372 (bank increment +.1469).
So era is a real but small channel that removes ~14% of the pooled residual and leaves
the dense arm's advantage over the named era family almost untouched.

Coverage confound, reported not hidden: matched rows have label rate .485, unmatched .562.
Unmatched rows are dropped from every era readout, never imputed.

---

## 3. Round 1

**Fleet: 16/16 slots, P = 8, 3 families, 200 proposals (120 A / 80 B).** No degradation.
`glm_b` took one HTTP 429 and succeeded on retry 2; both GLM keys were smoke-tested live
before the round (thinking enabled, budget 512, ~2–3 s round-trip). Sixteen distinct
slice orderings, one per (slot × track) salt.

Track-B tagging by the fleet: 43 of 80 channels tagged `mixed` (their conjectured upstream
parent plausibly causes real quality too), 13 tagged `surface-only`.

### 3.1 Species and missing mass — and a two-directional merge finding

The freeze's identity rule is **blind pairwise adjudication, never embedding-τ**. This
campaign extends the strict merge to **both tracks** (the mathse_vote reference ran it on
Track B only) because missing mass is reported on both sides here. Two sealed blind
judges (Sonnet + Opus) adjudicated 240 cross-proposer shortlisted pairs; a pair merges
only if **both** say SAME. Both planted identity anchors passed for both judges (2/2, 2/2).

| track | τ-only S_obs | **merged S_obs** | direction | τ-only mass | **merged mass (record)** | LOPO band |
|---|---:|---:|---|---:|---:|---|
| A | 47 | **70** | τ **over**-merged | .283 | **.425** | [.410, .524] mean .457 sd .044 |
| B | 49 | **38** | τ **under**-merged | .500 | **.363** | [.343, .414] mean .373 sd .025 |

**The τ shortcut fails in both directions inside one round of one cell.** The mathse_vote
campaign documented τ *under*-merging Track B; here τ under-merges B (49 → 38) *and*
over-merges A (47 → 70) — 50 merge edges survived strict adjudication on A and 42 on B.
That is a direct argument for the freeze's choice of blind pairwise over embedding-τ, and
it means any τ-only mass number is biased in a direction that depends on the track.
`merged_jackknife.py` computes the leave-one-proposer-out band on the merged species (the
figure of record); `species.py`'s τ-only jackknife is kept beside it, never in place of it.

**Read of the mass.** At P = 8 the sealed fleet has left an estimated **.425 (A)** and
**.363 (B)** of the concept space unnamed. Cross-proposer recapture is .271 (A) / .237 (B)
— most species are still named by one proposer only, so this is a wide-open concept space
on both sides, not a saturated one.

### 3.2 The scored set (post-merge, top-k by cross-proposer support)

Track A (15): transgression carried by a mechanism not shock (P=7); platform in-group
meta-humour about the joke's own reception (P=6); compression/economy (P=6); retroactive
reinterpretation (P=4); reads as freshly minted rather than a circulated chestnut (P=4);
textual completeness (P=4); punchline closure (P=4); transformed cultural allusion (P=4);
topical/political specificity (P=4); natural phonetic transformation (P=3); nerdy/academic
wordplay (P=3); clean semantic double-use (P=3); orthographic error as accident vs
deliberate device (P=3); vivid concrete imagery (P=2); read-aloud cadence (P=2).

Track B (10): **datable topical reference fixing the item to one forum era (P=8/8 — every
sealed proposer named it)**; forum/platform meta-reference (P=7); reliance on shock or
taboo subjects (P=7); orthographic noise (P=6); presupposed familiarity / variant / remix
/ knowing-repost framing (P=6); external-source trace (P=5); verbosity and layout load
(P=5); cleanliness of commitment to one delivery format (P=3); explicit external
attribution line (P=2); audience-management meta-framing (P=1).

**The fleet independently converged on the Addendum-4 channel.** B01 — the era fingerprint
— was named by **all eight** sealed proposers across all three families, and the observed
`created_utc` ordinal independently reads .592. Unlike math.SE round 1, where the
conjectured arrival-order fingerprint scored at chance (.492) against an observed ordinal
of .614, this cell's fingerprint and its observed ordinal are about to be measured against
each other on the same rows — that comparison is the round's headline.

### 3.3 Blind routing audit + arbiter — the craft/circulation boundary is porous here

Fresh blind Sonnet-class auditor, 29 items (25 selected + 4 planted from 2 of the 4
corpus-matched probe pairs, drawn by a hash chained off each round's realised draw).

* **Planted probes 4/4.** The auditor called "setup commits to one misleading reading" and
  "premise is specific rather than generic" quality-relevant, and "question-and-answer
  riddle template" and "shouting typography" incidental — exactly as planted.
* **Misrouting 6/25 (.24)**, every one of them Track A → incidental. Zero Track-B
  proposals were pulled toward quality.
* Frontier arbiter (provenance visible by design) upheld **5 of 6** re-routes and reversed
  one. Final: **A = 10, B = 15, of which 12 mixed**.

| id | criterion | auditor | arbiter | mixed |
|---|---|---|---|---|
| A03 | Compression and economy | incidental | **A (reversed)** | — |
| A05 | Reads as freshly minted rather than a circulated chestnut | incidental | B | yes |
| A06 | Textual completeness and legibility | incidental | B | yes |
| A09 | Topical/political specificity | incidental | B | no |
| A11 | Nerdy or academic wordplay | incidental | B | no |
| A13 | Orthographic error as accident vs deliberate device | incidental | B | yes |

**This is a substantive cell finding, not a fleet failure.** Prereg AMENDMENT 2 parked
the nuisance-versus-merit boundary as "a substantive decision to be made explicitly per
cell, not by default routing". On a humour corpus the boundary runs straight through the
middle of what comedy writers call craft: whether a joke is *fresh rather than a
circulated chestnut*, whether it is *topical*, whether it is *nerdy*, and whether a
misspelling is a *device or a typo* are all simultaneously craft judgements and markers of
where the item came from and when. Five of the six re-routes are exactly that family, and
the arbiter's one reversal (compression/economy) is the one case where the criterion text
made the *reader-facing* property, not the circulation marker, the thing being scored.
The consequence for this cell is structural: **12 of 15 Track-B channels are MIXED**, so
the Addendum-2 sensitivity band (ALL_B vs STRICT discount) is unusually wide here and both
ends must be quoted.

### 3.4 Scoring pass

Gemma-4-31B, offline batch vLLM, sk3 **GPU 5** (lane-pinned claim, released rc=0),
16,000 rows × 25 criteria + a 150-text anchor battery = **403,750 prompts**, ~45 min.
Item view matched to the A bank exactly (`JOKE:\n"<text>"`, no truncation).

Anchors (K = 50/class, in the same batch): pos 3.650 / neg 3.365 / scrambled 1.538;
pos-vs-neg AUC .577; **coherent-vs-scrambled AUC .914, PASS**. Overall NA .0086. Rows
NA on every criterion: **0** (interrupted-generation gate clear). Collapse gate: **2 of 25**
collapsed (A02 platform in-group meta-humour, modal .988; B09 explicit attribution line,
modal .984) — both are genuinely rare features in this corpus, and B09 is dropped by the
degeneracy screen in the cumulative discount.

### 3.5 TRACK A — one round of mining closed the residual

TIER 1, MONITOR (n = 1,917, 6 topics). Bank 74 → 84 features.

| quantity | entering r1 | after r1 | change |
|---|---:|---:|---:|
| VA_lin MONITOR | .7083 | .7210 | +.0128 |
| **VA_nl MONITOR** | **.7278** | **.7448** | **+.0170** |
| VA_nl MONITOR within-topic (TIER 2) | .7298 | .7458 | +.0160 |
| VA_nl HONEST | .7235 | .7398 | +.0163 |
| VA_nl OOF FIT+MINE | .7304 | .7452 | +.0148 |

Gain CI on MONITOR: topic-cluster bootstrap **[+.0104, +.0238]**, p(>0) = 1.00; item-level
band [+.0068, +.0281], p = .999. Seed spread of the new bank on MONITOR .0027.

**The gain is 3.4× ε and larger than the entire round-0 residual.** Δ_beyond on MONITOR
goes from **+.0143 to −.0027** on the campaign's mean-of-AUCs T convention (T = .7421), and
from +.0241 to **+.0070** on the seed-ensemble convention `readout.py` uses (T = .7519).
On HONEST, +.0233 → +.0071 (ensemble). Either way this is a **~70–100% close of the taste
residual in a single proposing round**.

The strongest new criteria, alone-AUC on HONEST (the incoming bank's best univariate
criterion was .592):

| alone-AUC | new Track-A criterion |
|---:|---|
| **.682** | **Read-aloud cadence** |
| .630 | Punchline closure / resolution |
| .592 | Clean semantic double-use |
| .592 | Retroactive reinterpretation |
| .569 | Transgression carried by a mechanism, not by shock alone |
| .557 | Compression and economy |
| .552 | Vivid concrete imagery |

**Read-aloud cadence is the headline finding on the A side.** A single mined criterion —
prosody, the rhythm of the joke as spoken — out-predicts every one of the 47 criteria in a
GEPA-iterated, expert-authored humour bank built from the r/StandUpWorkshop rubric
hierarchy and the parsed joke-craft literature. It is not an exotic construct; it is one
of the most standard things a comedy writer would say. The bank missed it because the
bank was built from written-craft rubrics.

**Swap check (honest caveat).** C₊ .8243 → .8472 (+.0230), C₋ .4263 → .4227 (−.0036),
Spearman(bank, dense) .646 → .688. `swap_signature = true`: C₊ rose while C₋ fell, so part
of the gain is the bank inheriting the dense model's ordering rather than independently
getting pairs right. The asymmetry is large (+.0230 vs −.0036), so most of the movement is
genuine, but the round did buy some rank agreement and that must travel with the number.

### 3.6 TRACK B — the spurious map, and the Addendum-4 fingerprint fails

15 channels scored, none dropped by the degeneracy screen in the round readout (B09 is
dropped in the cumulative discount as collapsed). Alone-AUC on HONEST, ordered by |AUC−.5|:

| alone-AUC | mixed | ρ with V block | channel | conjectured parent |
|---:|:--:|---:|---|---|
| **.386** | ✓ | .19 | Reads as freshly minted rather than a circulated chestnut | surface-only |
| .596 | ✓ | .43 | Textual completeness and legibility | surface-only |
| .566 | ✓ | .29 | Cleanliness of commitment to one delivery format | poster's practice |
| .546 | ✓ | .24 | Orthographic error as accident vs deliberate device | surface-only |
| .537 | ✓ | .10 | Nerdy or academic wordplay | surface-only |
| .450 | — | .25 | Datable topical reference fixing the item to one forum era | submission timing / era |
| .450 | ✓ | .29 | Orthographic noise | editing / proofreading |
| .481–.501 | mostly ✓ | ≤.20 | shock/taboo, forum-meta, external-source, topical specificity, attribution, audience-management, retelling-familiarity | various |
| .501 | ✓ | **.80** | Verbosity and layout load — **ALREADY ARTICULATED** (the V block owns it) | delivery-format habit |

Only **1 of 15** channels duplicates the V block at ρ ≥ .70, so on this cell the mined
nuisance set is genuinely new information rather than a restatement of the surface bank.

**The Addendum-4 fingerprint fails against its own observed ordinal**
(`jokes_community_r1_fingerprint_vs_observed.json`, matched-timestamp HONEST rows,
n = 2,728; observed era ordinal AUC **.5959**):

* The era channel B01 was named by **8 of 8** sealed proposers across all three families —
  the strongest fleet consensus of the round.
* Its judged score correlates with the actual timestamp at **ρ = −.034**. Wrong sign,
  essentially zero.
* **No** Track-B channel exceeds |ρ| = .058 against the observed ordinal. The largest is
  "cleanliness of commitment to one delivery format" at ρ = +.058.
* The observed ordinal still adds **+.0226** on top of all 15 named channels stacked, and
  the 15 channels add +.0999 over the ordinal — they are measuring almost disjoint things.

This **replicates and strengthens the math.SE round-1 result** (conjectured arrival-order
fingerprint .492 against an observed ordinal of .614). There, one could argue the fleet had
simply not named the channel well. Here the fleet named it unanimously, tagged the correct
upstream parent, and *still* produced a text channel orthogonal to the real ordinal.
**Two cells now say the same thing: LLM proposers can name a position-in-container channel
and cannot see it in the text.** That is an argument for keeping observed ordinals as
covariate lines rather than expecting Track B to recover them — and it is the reason
Addendum 4 exists.

### 3.7 Discount (freeze-triggered matched sampling)

Joint spurious-alone AUC on MONITOR = **.699** (HistGB) / .688 (linear), **above the .65
trigger**, so matched sampling runs alongside decile stratification
(`jokes_community_r1_cumulative_discount.json`).

| band | n channels | joint-B (HONEST) | pooled Δ | decile Δ_adj | matched Δ_adj |
|---|---:|---:|---:|---:|---:|
| ALL_B, HONEST | 14 | .696 | +.0071 | **+.0043** | −.0127 (1,106 pairs) |
| ALL_B, MONITOR | 14 | .697 | +.0070 | **+.0069** | +.0195 (666 pairs) |
| STRICT no-mixed, HONEST | 3 | .547 | +.0071 | **+.0047** | +.0007 (1,409 pairs) |
| STRICT no-mixed, MONITOR | 3 | .544 | +.0070 | **+.0051** | +.0269 (855 pairs) |

The decile estimator is stable across both ends of the Addendum-2 mixed band and both
populations: **Δ_adj ≈ +.004 to +.007**. The matched estimator swings from −.013 to +.027
across the four cells — with only 666–1,409 pairs over 5–6 topics it is too noisy to
adjudicate here, and the decile reading is the quoted one, with the matched range reported
as the sensitivity it is.

Stratification-free stacked increment (HONEST): joint-B .6963, +dense .7525, +bank .7444,
+both .7641. **Dense increment over B alone +.0559; over B plus the enriched bank +.0197.**
Bank increment over B plus dense +.0116.

### 3.8 Stopping-rule state after round 1

Gain on MONITOR **+.0170 ≫ ε = .005**. This was a PROPOSING round, and it is not sub-ε, so
**the stopping clock is at zero**: round 2 is required. Cap is 5.

### 3.9 GEPA phrasing pass, collapse gate and sign re-audit (`gepa_phrasing.py`)

Run before any round-1 number is quoted as final, per the freeze. All three screens are
applied together because they interact.

**A defect found and fixed.** `score_gemma_maps.py` FLAGS collapse (modal > .98) but
`closure_core.clean_fit` only drops a column when fewer than 5 rows sit off the mode — at
n = 16,000 a criterion at modal .988 leaves 190 off-modal rows and sails through. The flag
was being *recorded and not acted on*, in this campaign and in the pipeline it was
inherited from. `gepa_phrasing.py regate` now enforces it.

| id | fidelity | modal | NA | alone-AUC (FIT+MINE) | sign verdict | quotable |
|---|---:|---:|---:|---:|---|:--:|
| A14 Vivid concrete imagery | .835 | .155 | .000 | .546 | ok | ✓ |
| **A15 Read-aloud cadence** | **.724** | **.368** | **.005** | **.671** | **ok** | **✓** |
| A04 Retroactive reinterpretation | .710 | .561 | .015 | .598 | ok | ✓ |
| A12 Clean semantic double-use | .704 | .574 | .010 | .599 | ok | ✓ |
| A07 Punchline closure | .665 | .620 | .000 | .641 | ok | ✓ |
| A01 Transgression by mechanism | .665 | .461 | .090 | .573 | ok | ✓ |
| A03 Compression and economy | .617 | .585 | .006 | .555 | ok | ✓ |
| A10 Natural phonetic transformation | .594 | .734 | .043 | .516 | ok | ✓ |
| A08 Transformed cultural allusion | .502 | .852 | .000 | .494 | sign_null_band | ✗ GEPA-targeted |
| A02 Platform in-group meta-humour | .346 | .988 | .000 | .496 | sign_null_band | ✗ **COLLAPSED** |

* **Read-aloud cadence passes the phrasing pass outright** — fidelity .724, modal .368,
  NA .005, no rephrasing indicated. Its number is quotable as it stands, and the pass is
  recorded as run rather than skipped.
* Sign band: Hanley-McNeil SE at the null = .00487, so the two-sided trigger sits at
  .4903. Both sub-chance criteria (A08 .494, A02 .496) are **inside** the band →
  `sign_null_band`, kept, **0 sign-contradiction triggers**.
* GEPA-targeted for rephrasing at terminal: **A08** (modal .852) and A02 (excluded anyway).

**Collapse-gated headline (the quotable one).** Excluding A02, the bank is 83 features:

| | MONITOR | HONEST |
|---|---:|---:|
| entering r1 | .7278 | .7235 |
| after r1, with collapsed A02 | .7448 | .7398 |
| **after r1, collapse-gated** | **.7453** | **.7403** |

**Gain +.0175** [group CI **+.0114, +.0234**], p(>0) = 1.00 — *larger* than the ungated
+.0170, because A02 was pure noise. Δ_beyond MONITOR collapse-gated: **−.0032** (campaign
mean-of-AUCs T .7421) / **+.0066** (ensemble T .7519). The correction strengthens the
result rather than weakening it, and both figures are kept side by side.

---

## 3b. Round 2 (proposing round, mining complete; scoring on GPU 5)

Slice rebuilt on the **enriched 84-feature bank** (VA_nl OOF FIT+MINE .7452 entering), so
it tracks what the enriched bank still misses. Fleet again **16/16 slots, P = 8, 3
families, 200 proposals**, no degradation.

### 3b.1 Species — the A-side concept space OPENS as the bank closes

| track | τ-only S_obs | merged S_obs | τ mass | **merged mass (record)** | LOPO band | r1 → r2 |
|---|---:|---:|---:|---:|---|---|
| A | 49 | **89** | .317 | **.600** | [.600, .667] mean .630 sd .022 | **.425 → .600** |
| B | 43 | **39** | .412 | **.350** | [.329, .386] mean .368 sd .023 | .363 → .350 |

τ over-merges A and under-merges B **again** (31 A edges / 41 B edges survived strict
adjudication) — the two-directional failure replicates within the cell. Both identity
anchors passed for both judges again.

**The A-side missing mass rose from .425 to .600 while the bank got better.** 72 of 120
A proposals are now singletons. Once the obvious craft channels are in the bank, the fleet
fans out into idiosyncratic long-tail constructs rather than converging — the concept
space *opens* as it is mined. The B side did the opposite and stayed put (.363 → .350):
the nuisance space is small, bounded and near-saturated.

### 3b.2 Track-B replication across rounds — the map is stable

Round 2's Track-B selection reproduces round 1's families almost exactly, with the top two
at unanimous 8/8 consensus:

| r2 rank | channel | P | r1 counterpart |
|---|---|---:|---|
| B01 | Taboo-register intensity | **8/8** | B03 shock/taboo (7) |
| B02 | Forum-aware framing | **8/8** | B04 forum/platform meta (7) |
| B03 | Era anchoring | 6 | B01 datable topical reference (8) |
| B04 | Orthographic noise | 6 | B04 orthographic noise (6) |
| B05 | Repost/familiarity framing (retelling-chain position) | 6 | B05 presupposed familiarity (6) |
| B06 | Sourcing-versus-origination traces | 5 | B06 external-source trace (5) |
| B07 | Template commitment | 4 | B08 delivery-format commitment (3) |

Seven of ten channels are the same families at similar consensus, from an independently
re-ordered slice drawn against a *different* bank. **The spurious map for this cell is
reproducible**, which is the property a map has to have before it is used to discount.

### 3b.3 Audit — misrouting collapses once the craft/circulation line is drawn

Probes **4/4** on the **rotated** pair set (r2 drew pairs 1 and 3, r1 drew 2 and 4 — the
realised-draw chaining works as designed, no auditor saw a repeat). Misrouting
**1/25 (.04)**, down from 6/25 (.24) in round 1. The single dispute (A01 self-referential
platform meta-humour) was routed to B/mixed by the arbiter — the same craft/circulation
boundary as round 1, now the *only* remaining ambiguity. Final **A = 14, B = 11 (8 mixed)**,
down from 12 mixed of 15.

### 3b.4 LANDMINE (new, recorded): file-existence is not file-completeness

`audit.py finalize` first reported B06 with `audit_label: null` and
`route_source: no_verdict_default_proposed`, inflating misrouting to 2/25. The verdicts
file was complete and valid when re-read — the finalize had run against a **partially
written** file, because the wait loop tested `[ -f <file> ]` and fired the instant the
file was created. Re-running finalize on the complete file gave the correct 1/25 and the
correct route. **Any wait-on-agent-output loop in this pipeline must test parseability,
not existence** (`until python3 -c "json.load(open(f))"`), which is what the arbiter wait
now does. Nothing downstream was contaminated: the error was caught before scoring, and a
silently defaulted route would have been a real routing error.

### 3b.5 Round-2 scoring and readout

Gemma-4-31B, GPU 5 lane-pinned, 403,750 prompts, released rc=0. Anchors: pos 2.856 / neg
2.616 / scrambled 2.053; coherent-vs-scrambled **.780 PASS**; NA .004; **0 collapsed**;
0 all-NA rows.

**TRACK A, TIER 1 (MONITOR):** VA_nl .7448 → **.7527**, gain **+.0079**
[group CI **+.0026, +.0109**], p(>0) = .997; item band [+.0019, +.0139]. Within-topic gain
+.0077. HONEST gain +.0095.

**+.0079 > ε = .005, so round 2 is NOT a sub-ε round. The stopping clock stays at zero.**

**The residual is now closed and slightly negative.** Δ_beyond MONITOR: +.0070 → **−.0009**
(ensemble T .7519) and **−.0106** (campaign mean-of-AUCs T .7421). On HONEST the enriched
bank now reads **.7493 against the dense model's .7469** — the articulated instrument has
overtaken the dense standard on this cell.

Best new r2 criterion: **surprise–coherence balance, alone-AUC .667**; then dual-coherence
wordplay .596, concrete image-bearing detail .568, vivid absurd specificity .567.

### 3b.6 SWAP FLAG (coordinator standing request) — C₋ turned positive in r2

| round | dC₊ | dC₋ | dρ | code's `swap_signature` |
|---|---:|---:|---:|:--:|
| 1 | +.0230 | **−.0036** | +.0415 | **true** |
| 2 | +.0096 | **+.0093** | +.0086 | false |

**Flagged as instructed.** Reading it out plainly, because the direction matters: C₋ is
P(bank orders a discordant pair correctly | the *dense model gets it wrong*). C₋ rising
means the bank is improving precisely where the dense model fails — that is **independent
signal, the favourable direction**. The adverse pattern is the one the swap algebra names:
C₊ up while C₋ **falls**, i.e. the bank buying rank agreement by inheriting dense errors.
That is what happened in **round 1** (dC₋ −.0036, signature true), and it is what round 2
**cleared**: r2 raised C₊ and C₋ by almost the same amount (+.0096 / +.0093), which is a
uniform improvement independent of the dense model's ordering.

So: the condition to watch fired, and it fired in the good direction. If the coordinator's
rule was written expecting dC₋ > 0 to be adverse, that is the opposite of what the algebra
implies here, and the r1 row is the one that deserved the caution — as it was flagged.

### 3b.7 Track B round 2 — map replicates, discount goes negative

Joint spurious-alone MONITOR .654 (HistGB), just over the .65 matched-sampling trigger.
Two channels are now **ALREADY ARTICULATED** (template commitment ρ_V .72, verbosity ρ_V
.85). Discounted Δ_adj is **negative on both bands** — ALL_B HONEST −.0091 / MONITOR
−.0043; STRICT HONEST −.0035 / MONITOR −.0009 — consistent with a closed residual.

Stacked increment HONEST: joint-B .6495, +dense .7469, +bank .7493, +both .7672. Dense
increment over B+bank **+.0182**; bank increment over B+dense **+.0203** — the bank now
adds *more* over dense-plus-nuisance than dense adds over bank-plus-nuisance.

**Addendum-4 fingerprint fails a third time.** r2 matched-timestamp HONEST (n = 2,728,
observed ordinal .5959): max |ρ| across all 11 channels = **.052**; the named "Era
anchoring" channel reads **ρ = −.020**. The observed ordinal still adds **+.0391** over all
11 named channels stacked. Three independent measurements now (math.SE r1, jokes r1,
jokes r2) all say the same thing.

### 3b.8 Sign-contradiction re-audit (2 triggers, both resolved KEEP)

The two-sided band sits at .4903 (Hanley-McNeil SE .00487). Two round-2 Track-A criteria
fired the trigger:

| id | alone-AUC | criterion text (abridged) | verdict |
|---|---:|---|---|
| r2 A06 | .386 | "Score how much the text **fails to deliver** either a setup or a payoff. 10 = clearly missing one half entirely" | **KEEP** |
| r2 A05 | .459 | "Score how much the text sets up an obvious punchline and then **deliberately refuses to deliver it**" | **KEEP** |

Both are **defect-naming criteria**: they score the presence of a flaw, so a high score is
by construction a *worse* joke and a sub-chance alone-AUC is the expected polarity, not
evidence of a nuisance channel. The aggregator is sign-free, so nothing is mismodelled.
Recorded as inverted-polarity craft criteria rather than re-routed. (Contrast the peer
pilot, where "Restraint" fired the same trigger and *was* re-routed to nuisance — the
distinction is whether the criterion names a defect on purpose.)

### 3b.9 Cumulative screen state after two rounds

24 mined Track-A criteria: **16 quotable**, 1 collapsed (r1 A02, excluded), 7
GEPA-targeted for rephrasing at terminal (modal > .75), 2 sign-triggered (both KEEP
above). **Read-aloud cadence remains quotable and is still the strongest single mined
criterion in the campaign at .682.**

Carried debt, recorded: `stage1_slice.current_blocks` rebuilds each round's bank from every
A-routed criterion of prior rounds, so the r2 bank still contains the collapsed r1 A02. Its
measured effect at r1 was +.0005 *against* the bank, so the r2 figures are if anything
mildly conservative; the collapse gate is applied cumulatively for the terminal ledger.

## 4. Per-round ledger

| round | VA_nl MONITOR | gain | Δ_beyond MONITOR (campaign T / ensemble T) | probe pass | misroute | spurious-alone | GT mass A / B (merged, strict) |
|---|---:|---:|---|---|---|---:|---|
| 0 | .7278 | — | +.0143 / +.0241 | — | — | observed era .592 | — |
| 1 (raw) | .7448 | +.0170 | −.0027 / +.0070 | 4/4 | 6/25 (.24) | .699 | .425 [.410,.524] / .363 [.343,.414] |
| **1 (collapse-gated, quotable)** | **.7453** | **+.0175** | **−.0032 / +.0066** | 4/4 | 6/25 (.24) | .699 | .425 / .363 |
| **2** | **.7527** | **+.0079** | **−.0106 / −.0009** | 4/4 | 1/25 (.04) | .654 | .600 [.600,.667] / .350 [.329,.386] |
| 3 (**decomposition, TIER D, exempt**) | .7503 | −.0024 | −.0082 / +.0016 | 4/4 | **0/8 (.00)** | .723 cum. | n/a (TIER D contributes no mass) |
| **4 (proposing, SUB-ε #1)** | .7514 | **+.0011** (CI −.0039,+.0030) | −.0093 / +.0004 | 4/4 | 4/25 (.16) | .704 | .475 [.438,.562] / .287 [.257,.357] |

Stopping clock: r1 +.0175 and r2 +.0079 are both **above** ε = .005 → **0 consecutive
sub-ε proposing rounds**. Cap 5. Round 3 = Addendum-3 decomposition (does not count).

**Swap tracking (coordinator standing request).** r1: C₊ +.0230, C₋ **−.0036** (negative,
i.e. the safe direction). The rule to watch: if C₋ turns **positive** in a later round the
bank is buying agreement by inheriting dense errors, and that is flagged immediately.

---

## 3c. Round 3 — FREEZE ADDENDUM 3 decomposition pass (does NOT count toward stopping)

**Scope, recorded rather than silent.** Round 3 runs the decomposition and NOTHING ELSE —
no sealed fleet. That is what makes it unambiguously exempt from the stopping rule: the
registered 2026-08-08 rule exempts non-PROPOSING rounds, and a round with no proposers
cannot be one. Precedent: `mathse_vote/decompose_r2.py`. `decompose_round.py merge`
assumes a fleet ran alongside, so this cell uses `decompose_only_merge.py` instead.

**Two-tier rule applied.** The components are DIRECTED (authored against four named
parents, not independently proposed from the slice), so they are **TIER D**: scored and
auditable, but excluded from every Good-Turing / missing-mass quantity. The round's
species file carries an empty `tracks` block and an explicit `tier: D` marker so nothing
downstream can count them as sealed-fleet species.

**Parent selection** (`mixed_parents.py`, ranked by |alone-AUC − .5| on **FIT+MINE only** —
MONITOR is never read for a design decision; collapse gate applied first). 19 accumulated
MIXED channels available, 0 previously decomposed, top 4 taken:

| parent | round | alone-AUC (FIT+MINE) |
|---|---|---:|
| Textual completeness and legibility | r1 A06 | .599 |
| Cleanliness of commitment to one delivery format | r1 B08 | .568 |
| Era anchoring | r2 B03 | .449 |
| Reads as freshly minted rather than a circulated chestnut | r1 A05 | .378 |

Three are the craft/circulation-confound family §3.3 identified. The fourth is **the
Addendum-4 channel itself** — the ranking put "era anchoring" ahead of the
orthography parent, which is the more informative draw: it asks directly whether the era
channel's signal is topical *craft* or mere datedness, after the fingerprint has failed
against the observed ordinal three times. All four parents are now RETIRED from the
nuisance readouts (recorded in `jokes_community_retired_channels.json`, never deleted).

**The eight components** (4 candidate-real → Track A, 4 surface → Track B):

| parent | candidate-real | surface |
|---|---|---|
| freshness | Genuine comic invention performed by this text | Extent of overlap with stock circulated joke material |
| completeness | Setup supplies exactly what the punch needs to land | Extent of clean mechanical text hygiene |
| format commitment | Economy and timing of information release before the punch | Extent of strict conformity to one recognisable joke template |
| era anchoring | References are load-bearing, not decorative name-dropping | Extent of dated period markers present in the text |

**Blind routing audit: 0/8 misrouted, probes 4/4, zero disputes** — so no arbiter was
needed. A fresh blind auditor, told nothing about which component was which, independently
confirmed every candidate-real / surface split the decomposer drew. That is the cleanest
audit of the campaign (r1 .24, r2 .04, r3 .00) and it is the direct evidence Addendum 3
asks for: these parents really were two separable things wearing one name.

**Scoring:** 8 criteria × 16,000 rows + anchors = 129,200 prompts, GPU 5 lane-pinned,
released rc=0. Best anchor battery of the campaign — pos 7.366 / neg 6.146 / scrambled
0.733, **pos-vs-neg .691**, **coherent-vs-scrambled .9815**, 0 collapsed, NA .009. The
decomposed criteria are sharper instruments than either the bank or the fleet criteria.

### 3c.1 Result — the decomposition worked, and it reinterprets the Addendum-4 channel

Track-A gain on MONITOR **−.0024** [CI −.0035, −.0000]. That is the *expected* shape for a
decomposition round: the four candidate-real components restate information the bank
already absorbed from their parents in rounds 1–2, so they add nothing and cost a little
capacity. **It does not touch the stopping rule.** The round's value is measurement, not
closure, and on that count every one of the eight components separated cleanly:

| parent (alone-AUC as a parent) | candidate-real half | surface half |
|---|---|---|
| Reads as freshly minted (**.386**) | Genuine comic invention performed by this text **.654** | Extent of overlap with stock circulated joke material **.660** |
| Cleanliness of format commitment (.566) | Economy and timing of information release .586 | Extent of strict conformity to one template .586 |
| Textual completeness (.596) | Setup supplies exactly what the punch needs .574 | Extent of clean mechanical text hygiene .542 |
| **Era anchoring (.451)** | **References are load-bearing, not decorative .562** | **Extent of dated period markers .454** |

**Both halves of the freshness parent carry MORE signal than the parent did.** As a single
channel it read .386 (|Δ| = .114); split, its craft half reads .654 and its surface half
.660 (|Δ| = .154 and .160), in opposite directions. That is the cleanest possible
vindication of Addendum 3: the parent was two things cancelling each other out, and
"extent of overlap with stock circulated joke material" is now the **strongest Track-B
channel in the whole campaign**.

**The era channel was never a nuisance channel.** Decomposed, its predictive content sits
almost entirely in the *craft* half — "references are load-bearing, not decorative
name-dropping" .562 — while pure datedness ("extent of dated period markers") reads .454,
mildly *anti*-predictive. **This explains the three-cell fingerprint failure**
(§3.6/§3b.7): when eight of eight proposers named "era anchoring", what their instruction
actually made a judge measure was topical craft, not era. The fingerprint did not fail
because the fleet phrased it badly; it failed because the nameable, judgeable thing in the
neighbourhood of a container ordinal is the craft that co-occurs with it. The observed
ordinal keeps adding signal over every named channel because **no text criterion is
measuring the ordinal at all** — they are all measuring the craft it travels with.

Swap: dC₊ −.0013, dC₋ −.0029, signature false.

### 3c.2 Cumulative discount after three rounds (`jokes_community_r3_cumulative_discount.json`)

Bank 102 features; nuisance set 29 channels with the **four decomposed parents RETIRED**
(recorded in `jokes_community_retired_channels.json`, never deleted) and their surface
components substituted in.

| band | channels | joint-B HONEST | pooled Δ | decile Δ_adj | matched Δ_adj |
|---|---:|---:|---:|---:|---:|
| ALL_B HONEST | 29 | .7227 | −.0007 | +.0054 | +.0019 (1,033 pairs) |
| ALL_B MONITOR | 29 | .7284 | +.0016 | +.0165 | −.0082 (607 pairs) |
| STRICT HONEST | 7 | .6806 | −.0007 | +.0030 | −.0027 (1,126 pairs) |
| STRICT MONITOR | 7 | .6931 | +.0016 | +.0080 | +.0301 (665 pairs) |

The named nuisance set has grown strong enough (joint-B .723 on HONEST) that stratifying
on it costs a lot of resolution, which is why the MONITOR decile figure (+.0165) drifts up
on 5 coarse strata over 6 topics. The stratification-free stacked increment is the
readout to trust as the nuisance set grows, and it is stable and small:
**dense increment over B + bank = +.0149 (HONEST) / +.0165 (MONITOR)** on ALL_B, +.0179 /
+.0192 on STRICT.

---

## 3d. Round 4 — sealed proposing round (mining complete; scoring on GPU 5)

**GLM budget check before committing** (coordinator's condition): both Lite keys answered
in 8 s at the full 2048-token thinking budget with `stop_reason: end_turn`. The window was
not thin, so the round ran at the full **P = 8 / 3 families**; no fallback to P ≥ 6 was
needed. Fleet 16/16 slots, 200 proposals, slice rebuilt on the **102-feature bank**
(VA_nl OOF FIT+MINE .7509 entering).

### 3d.1 Species — the A-side mass turns over

| track | τ-only S_obs | merged S_obs | τ mass | **merged mass (record)** | LOPO band | trajectory |
|---|---:|---:|---:|---:|---|---|
| A | 58 | **77** | .375 | **.475** | [.438, .562] mean .512 sd .035 | .425 → .600 → **.475** |
| B | 39 | **34** | .338 | **.287** | [.257, .357] mean .304 sd .034 | .363 → .350 → **.287** |

τ over-merges A (58 → 77) and under-merges B (39 → 34) for the **third consecutive round** —
the two-directional failure is now a stable property of this cell, not a one-off. Both
identity anchors passed for both judges every round.

**The A-side mass came back down** (.600 → .475) after spiking in r2, so the r2 rise was
not a monotone opening — it was one round's fan-out. The B side keeps contracting
(.363 → .350 → .287) with recapture rising (.237 → .282 → .324): the nuisance space really
is small and is being consumed, while the A space is churning around .42–.60.

### 3d.2 Track-B map — third replication

The same seven families again, with the top two unanimous for the second round running:
register of taboo/shock **8/8**, traces of external sourcing **8/8**, defensive posting
furniture 7, orthographic noise 7, era-dating references 7, meta-platform references 6,
retelling-chain positioning 5. Across three independent fleets reading three
independently re-ordered slices against three different banks, the spurious map for this
cell reproduces essentially unchanged.

### 3d.3 Audit — the craft/circulation boundary bites a third time

Probes **4/4**. Misrouting **4/25 (.16)** — up from r2's .04 — and **every one of the four
was Track A → incidental**, in the same family as rounds 1 and 3: meta-forum
self-awareness, "pre-owned chestnut" provenance, common-knowledge threshold, and
orthographic strain in the wordplay. The frontier arbiter upheld **all four**. Final
**A = 11, B = 14 (9 mixed)**.

That the boundary re-opens whenever the fleet reaches for freshness, provenance, insider
knowledge or orthography — after three rounds and a decomposition pass specifically aimed
at it — is the substantive finding this cell keeps producing: on a humour corpus these are
irreducibly two-sided, and the campaign's answer is not to route them but to decompose
them.

**Scoring:** 25 criteria × 16,000 rows + anchors = 403,750 prompts, GPU 5 lane-pinned
(stacked over an 11.7 GB co-tenant that was left untouched), released rc=0. Anchors
pos 3.618 / neg 3.219 / scrambled 1.500, pos-vs-neg .634, **coherent-vs-scrambled .9374
PASS**, **0 collapsed**, NA .008.

### 3d.4 RESULT — the first SUB-ε PROPOSING round, and it is swap-shaped

**TRACK A, TIER 1 (MONITOR): VA_nl .7503 → .7514, gain +.0011**
[group CI **−.0039, +.0030**], p(>0) = .69; item band [−.0031, +.0051], p = .71.
Within-topic gain +.0006. HONEST gain +.0035.

**+.0011 < ε = .005, and the CI straddles zero.** This is the campaign's **first sub-ε
PROPOSING round**, so the stopping clock advances to **1 of the 2 consecutive rounds the
rule requires.**

### 3d.5 SWAP FLAG — the adverse pattern fired (immediate flag)

| round | C₊ | dC₊ | C₋ | dC₋ | dρ | adverse? |
|---|---:|---:|---:|---:|---:|:--:|
| 1 | .8472 | +.0230 | .4227 | −.0036 | +.0415 | **yes** |
| 2 | .8568 | +.0096 | .4320 | +.0093 | +.0086 | no |
| 3 | .8555 | −.0013 | .4291 | −.0029 | −.0007 | no (C₊ fell) |
| 4 | **.8613** | **+.0058** | **.4259** | **−.0033** | **+.0117** | **YES** |

Per the registry rule of record — adverse = dC₊ > 0 with dC₋ ≤ 0 — **round 4 is adverse**.
And the reading is unusually clean here, because the round's *entire* gain is
swap-shaped: the bank got better on pairs the dense model already orders correctly
(+.0058) and **worse** on the pairs the dense model gets wrong (−.0033), with
Spearman(bank, dense) rising .6955 → .7072, the largest ρ jump since round 1.

**The two readings agree and reinforce each other.** A +.0011 gain whose CI straddles zero
*and* whose sign comes from inherited dense ordering is not a discovery — it is the miner
finding nothing and the aggregator buying a little rank agreement. The sub-ε call is the
right one and the swap flag is the mechanism behind it. Recorded so the terminal verdict
cannot later quote r4's +.0011 as a real increment.

Track-A alone-AUCs are still respectable in isolation (setup-to-payoff economy .617,
coherent absurd detail .611, fresh recombination .594) — they are simply no longer
*additional* to a 102-feature bank that already contains their information.

Track B: joint-B .7038 on HONEST; strongest channel "pre-owned chestnut: the pun arrives
already familiar" **.632** — the freshness/provenance family again, and the third round
running in which it tops the map. Stacked increment (readout of record): dense over
B + bank **+.0169**, bank over B + dense **+.0152**.

Missing mass (strict merged): A **.475**, B **.287**.


---

## 3e. Round 5 — sealed proposing round, the CAP round. STOPPING RULE FIRES.

GLM window re-checked before committing (both keys, 8 s, full 2048 budget, `end_turn`) —
ran at full **P = 8 / 3 families**. Fleet 16/16 slots, **197** proposals (one short slot
recorded: `glm_a` Track A emitted 12 of 15). Slice rebuilt on the 113-feature bank.
Scoring 403,750 prompts, GPU 5, released rc=0; anchors pos-vs-neg .658,
coherent-vs-scrambled .8754 PASS, **0 collapsed**, NA .0027.

Species: τ 47 → merged **85** (A, mass **.564**, LOPO [.559, .657]); τ 45 → merged **39**
(B, mass **.300**, LOPO [.286, .371]). Fourth consecutive round of τ over-merging A and
under-merging B. Both anchors passed for both judges.

Audit: probes **4/4**, misrouting **4/25 (.16)**, all four again Track A → incidental in
the craft/circulation family (long scaffolding, spelling cleanliness, evergreen subject
matter, chestnut familiarity). Arbiter **reversed one** (long scaffolding stays A — its
instruction explicitly disclaims length and asks whether each setup beat earns its place)
and upheld three. Final **A = 12, B = 13 (11 mixed)**.

**RESULT: VA_nl MONITOR .7514 → .7511, gain −.0003** [group CI **−.0063, +.0027**],
p(>0) = .455. Within-topic −.0003, HONEST −.0009.

### STOPPING CONDITION

| round | kind | gain MONITOR | sub-ε? | counts? | clock |
|---|---|---:|:--:|:--:|---:|
| 1 | proposing | +.0175 | no | — | 0 |
| 2 | proposing | +.0079 | no | — | 0 |
| 3 | **decomposition (TIER D)** | −.0024 | yes | **no — exempt** | 0 |
| 4 | proposing | +.0011 | **yes** | yes | **1** |
| 5 | proposing | −.0003 | **yes** | yes | **2** |

**The campaign terminated on the stopping rule itself — two consecutive sub-ε PROPOSING
rounds (r4, r5) — not on the cap.** The cap of 5 was reached in the same round, so both
conditions are satisfied, but the rule fired on its own terms and that is the stronger
statement. The registered decomposition exemption did real work: without it, round 3's
−.0024 would have started the clock a round early and the campaign would have stopped at
r4 on a technicality.

Swap r5: dC₊ +.0003, dC₋ **−.0046**, dρ +.0046 → **adverse by the registry rule** (dC₊ > 0
with dC₋ ≤ 0), and the second adverse round running. On a round whose point estimate is
*negative* this is the expected reading: there is no real gain left, so what little the
aggregator moves is rank agreement with the dense model. Both terminal rounds say the
same thing from different directions.

---

## 4b. Per-round ledger (final)

| round | kind | VA_nl MONITOR | gain | sub-ε | counts | clock | probes | misroute | GT mass A / B (strict merged) |
|---|---|---:|---:|:--:|:--:|---:|---|---|---|
| 0 | baseline | .7278 | — | — | — | 0 | — | — | — |
| 1 | proposing | .7453* | **+.0175** | no | — | 0 | 4/4 | .24 | .425 / .363 |
| 2 | proposing | .7527 | **+.0079** | no | — | 0 | 4/4 | .04 | .600 / .350 |
| 3 | **decomposition (TIER D)** | .7503 | −.0024 | yes | **exempt** | 0 | 4/4 | **.00** | n/a |
| 4 | proposing | .7514 | +.0011 | **yes** | yes | **1** | 4/4 | .16 | .475 / .287 |
| 5 | proposing (cap) | .7511 | −.0003 | **yes** | yes | **2** | 4/4 | .16 | .564 / .300 |

\* collapse-gated. Round 4's +.0011 is **NOT QUOTABLE as an increment** (CI straddles zero
and its sign is entirely swap-shaped — registry ruling).

---

## 7. TERMINAL PACKAGE

### 7.1 Stopping condition

**Terminated on the stopping rule: two consecutive sub-ε PROPOSING rounds (r4 +.0011,
r5 −.0003), clock 2 of 2.** The cap of 5 was reached in the same round, so both conditions
hold, but the rule fired on its own terms. The registered decomposition exemption did real
work here: without it, round 3's −.0024 would have started the clock a round early and the
campaign would have terminated at r4 on a technicality rather than on evidence.

### 7.2 Cumulative collapse gate

One criterion excluded across all rounds: **r1 A02** (platform in-group meta-humour, modal
.988). Rounds 2–5 produced **zero** collapsed criteria. Terminal bank **124 features**
after the gate.

### 7.3 GEPA phrasing pass (stages 1–4, all run)

18 of 51 mined Track-A criteria were GEPA-targeted (modal > .75 or NA > .20). A sealed
label-blind rephraser — shown each criterion's own degeneracy histogram but never y, never
an AUC, never an item — wrote 48 variants over 17 eligible targets (the collapsed one was
not rephrased). All variants **and their incumbents** were rescored on the SAME 600
FIT+MINE probe rows (MONITOR never read for an instrument decision), and a variant was
accepted only on a ≥ .02 probe-fidelity gain.

**15 of 17 targets got an accepted rephrasing**, then were rescored corpus-wide and swapped
into the terminal bank. Largest fixes:

| criterion | incumbent modal | variant modal | fidelity gain |
|---|---:|---:|---:|
| Long scaffolding erected for one terminal pun | 0.743 | 0.398 | +0.229 |
| References are load-bearing, not decorative name | 0.834 | 0.505 | +0.189 |
| Surprise from withholding the expected punchline | 0.780 | 0.517 | +0.156 |
| Wordplay with homophone or near-homophone pivot | 0.783 | 0.413 | +0.137 |
| Specialist-domain knowledge as the hinge | 0.870 | 0.696 | +0.113 |
| Relatability of Frustration | 0.737 | 0.508 | +0.110 |

Two targets kept their incumbents (r5 A01 platform self-awareness; r5 A04 phonetic
double-use, for which the rephraser emitted no variants — recorded). **Read-aloud cadence
was never targeted: it passed at stage 1** (fidelity .724, modal .368, NA .005) and its
.682 stands as authored.

The pass is worth +.0041 on MONITOR on its own (.7511 ungated → .7552 gated + rephrased),
so it is not cosmetic — a sixth of the campaign's total closure came from fixing how
criteria were *worded*, not from finding new ones.

### 7.4 LABELLED-T TERMINAL LEDGER (`jokes_community_TERMINAL_LEDGER.json`)

Both T conventions, always labelled, never differenced against each other.

| | MONITOR (n 1,917 / 6 topics) | HONEST (n 3,163 / 10 topics) |
|---|---:|---:|
| **T_meanAUC** (campaign convention: mean over seeds of the AUC) | **.7421** | **.7377** |
| **T_ensemble** (master-ledger convention: AUC of the seed-mean) | **.7519** | **.7469** |
| VA_nl bank 0 (74 features) | .7278 | .7235 |
| **VA_nl terminal** (124 features, collapse-gated + GEPA) | **.7552** | **.7527** |
| VA_nl terminal, ungated / pre-GEPA | .7511 | .7502 |
| Δ_beyond bank 0 — meanAUC / ensemble | +.0143 / +.0241 | +.0142 / +.0233 |
| **Δ_beyond TERMINAL — meanAUC / ensemble** | **−.0131 / −.0033** | **−.0149 / −.0058** |
| **total closure** | **+.0274** | **+.0291** |

Closure CI on MONITOR (topic-cluster paired bootstrap): **[+.0144, +.0361]**, p(>0) = 1.00.

**Verdict: the residual is closed and slightly negative on both populations and under both
T conventions. The articulated instrument ends above the dense standard.**

### 7.5 Final dual-track map and discount

Terminal nuisance set **56 channels** (four decomposed parents retired, their surface
components substituted). Joint spurious-alone .731 on HONEST.

| band | channels | decile Δ_adj (HONEST / MONITOR) | matched Δ_adj (HONEST / MONITOR) | **stacked dense increment over B + bank** |
|---|---:|---|---|---|
| ALL_B | 56 | +.0102 / +.0268 | +.0049 / +.0523 | **+.0134 / +.0160** |
| STRICT no-mixed | 14 | +.0035 / +.0121 | −.0129 / −.0016 | **+.0149 / +.0162** |

Per the registered readout of record, the **stratification-free stacked increment** is the
quoted discount: with a 56-channel nuisance set, decile stratification has lost resolution
(5 coarse strata over 6 MONITOR topics) and the matched estimator swings ±.05 on 600–1,100
pairs. The stacked figure is stable across both bands and both populations at
**+.013 to +.016** — that is what the dense model still adds over everything nameable,
craft and nuisance together.

Strict two-track merged Good-Turing at terminal: **A .564** [LOPO .559–.657],
**B .300** [.286–.371].

---

## 8. CROSS-CUTTING LINES

**1. Prosody: the residual was an unmined bank, not a taste bound.** `Read-aloud cadence`
(alone-AUC **.682** on HONEST) out-predicts all 47 criteria of a GEPA-iterated
expert-authored humour bank whose best univariate was .592. It is not exotic — it is one of
the first things a comedy writer would say — and the bank missed it because the bank was
built from *written*-craft rubrics for a construct that is spoken. Before this campaign the
cell reported Δ_beyond ≈ +.015 and that number would have been quoted as taste.

**2. Ordinal-craft mechanism (registry-recorded, patents/code follow-up open).** Across
three measurements the fleet's era fingerprint sat at |ρ| ≤ .06 against the observed
`created_utc` ordinal while the ordinal kept adding signal over every named channel
(+.023 r1, +.039 r2). Round 3's decomposition gave the mechanism: split "era anchoring" and
its predictive content is the *craft* half ("references are load-bearing, not decorative",
.562), with pure datedness mildly **anti**-predictive (.454). The nameable, judgeable thing
near a container ordinal is the craft that co-occurs with it — which is why Addendum-4
fingerprints recover ordinals so badly, and why observed ordinals must stay covariate lines
rather than Track-B targets.

**3. Freshness superposition.** The "reads as freshly minted" parent read .386 as a single
channel (|Δ| = .114). Decomposed, its craft half reads .654 and its surface half .660
(|Δ| = .154 and .160) **in opposite directions** — both halves carry more signal than the
parent, because the parent was two things cancelling. "Overlap with stock circulated joke
material" became the strongest Track-B channel of the campaign. This is the cleanest
vindication of Addendum 3 the programme has produced, and it generalises a warning: a MIXED
channel's alone-AUC *understates* both of its components.

**4. Swap trajectory.** dC₋ by round: **−.0036 / +.0093 / −.0029 / −.0033 / −.0046**;
adverse (dC₊ > 0 with dC₋ ≤ 0) in **r1, r4, r5**. The shape is interpretable: r1 was a large
real gain that also bought some rank agreement; r2 was the clean round (both C₊ and C₋ up
by the same amount — pure independent signal); r4 and r5 are adverse *because there was
nothing left to find*, so the only thing the aggregator could move was agreement with the
dense model. The swap readout and the sub-ε readout agree at both terminal rounds, and that
agreement is what makes the termination trustworthy rather than merely arithmetic.

**5. τ-clustering fails in BOTH directions, every round.** Over-merged Track A
(47→70, 49→89, 58→77, 47→85) and under-merged Track B (49→38, 43→39, 39→34, 45→39) in all
four fleet rounds. Any τ-only missing-mass number is biased in a direction that depends on
the track. The freeze's blind-pairwise identity rule is not a refinement here; it is load-
bearing.

**6. The craft/circulation boundary is irreducible on this corpus.** Misrouting ran
.24 / .04 / .00 / .16 / .16, and *every* Track-A→incidental re-route in r1, r4 and r5 came
from one family: freshness, provenance, topicality, insider knowledge, orthography. A
decomposition pass aimed squarely at it did not close it — the fleet reached back into the
same family the very next round. On humour these are genuinely two-sided, and the
programme's answer should be to decompose them, not to route them.

**7. Bank saturated, concept space churning** (the pre-registered terminal language).
A-side Good-Turing ran .425 → .600 → .475 → .564 while VA_nl gained +.0274 and then stopped.
The miner did not run out of concepts — it ran out of concepts that *add* to a 124-feature
bank. B-side mass contracted monotonically (.363 → .350 → .287 → .300) with recapture rising
(.24 → .28 → .32 → .385): the nuisance space really is small and was consumed. Those are
different kinds of exhaustion and the terminal claim distinguishes them.

---

## 9. Claim discipline for anything quoted from this cell

* Quote each T with its convention label. **Never** difference T_meanAUC against T_ensemble.
* Round 4's +.0011 is not an increment (CI straddles zero, swap-shaped).
* The plateau claim is "not discoverable by THIS miner at P = 8 over 4 fleet rounds with
  A-side residual mass ≈ .56", not "no such criteria exist".
* Split-half noise on this cell exceeds ε: round-0 eval Δ +.0072 vs test Δ +.0218. Any
  round-over-round movement under ~.015 sits inside that band; the terminal *level* is
  robust, individual round deltas are not.
* MONITOR holds only 6 topics — group bootstrap bands are coarse by construction and the
  item-level band is never quoted in their place.
