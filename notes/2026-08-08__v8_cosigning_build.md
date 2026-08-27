# V8 — N&C co-signing cell (the regulatory field's VOTE/REVEALED column)

Date: 2026-08-08. Charge: goal item 5, first in order — build the missing vote
column that completes the second field's preference-type triple
(notes/2026-08-08__vat-3xN-decomposition-grid.md, row "Regulatory (N&C)",
cell "co-signing: **UNBUILT (V8)**"). Design contract:
notes/2026-08-05__taste-decomposition-design.md §10.

Status: **BUILT AND COMPLETE.** Population, y, splits, V/A reuse, Layer-1 stack,
grouped-transfer table, `y_nearby` sensitivity arm and the 3-seed dense (T) arm
all landed. **Gate verdict: CONDITIONAL PASS, UNDERPOWERED — do not promote to
a closure campaign (§9b).** Headline: the vote column's articulability is almost
entirely *between*-docket, and it is anti-correlated with the verdict column.

---

## 1. The y definition, and why

### 1.1 What "co-signing" can even mean here

Regulations.gov has no upvote, no like, no reply-count. The *only* act by which
a member of the public endorses **another person's** comment is **adoption**: an
organisation (or an individual) writes a comment, and N other people submit that
same text under their own names. That is co-signing in this field, and it is the
exact structural analogue of the other vote columns in the grid — citations for
peer review, upvotes for creative writing, crowd votes for humor.

**y_cosign_count(c) = the number of documents in c's own docket whose
normalised text is identical to c's** — i.e. how many people signed on.

Binarisations (AUC is the program's readout):

| name | rule | n_pos / 9,520 |
|---|---|---|
| `y_cosign` (PRIMARY) | count ≥ 2 — "did anyone else sign on" | 342 (3.59%) |
| `y_campaign` (secondary) | count ≥ 10 — organised-campaign scale | 132 (1.39%) |
| `y_nearby` (sensitivity) | shipped MinHash near-dup family ≥ 2 | 386, partial coverage |

### 1.2 The form-letter question — decided: mass duplication **IS** the y

The charge flagged form-letter campaigns as the confound, noting that the N&C
*responded* campaign found the campaign/form-letter family ANTI-predictive.
Decision: **do not exclude them; they are the phenomenon.** Three reasons:

1. **Construct fidelity.** The column being filled is "crowd endorsement". In
   this field the crowd's endorsement act *is* adopting someone's text.
   Excluding mass duplication removes the construct, not a nuisance.
2. **The alternative is empty by construction.** "Co-sign count among
   NON-form-letter unique comments" is identically 1 for every row — there is
   no such variable. A high/low split among substantive comments would be a
   *different* construct (comment quality), which is what the verdict and
   curation columns already measure.
3. **The anti-correlation is the scientific payoff, not a problem.** The
   verdict column (responded) and the vote column point in opposite
   directions on the *same text with the same instruments* — see §7. That is
   precisely the cross-y contrast the 3×N grid exists to produce, and it is
   the N&C analogue of peer review's 56× cross-y residual spread.

**The caveat that IS load-bearing** is different from the one anticipated:
predicting co-sign count from text may be **genre detection** — "is this
written in advocacy-campaign register" — rather than a quality judgment. This
is testable and is tested (§6: single-feature AUCs, char-length direction,
V-vs-A ordering) rather than asserted — and it comes back POSITIVE. Every other vote column in the program
carries the same threat (citations↔prestige, upvotes↔posting time), and the
program's answer is the same: nuisance-stratified and within-group readouts.

### 1.3 Two candidate channels were MEASURED and REJECTED as primary

**(a) The `Duplicate Comments` metadata field — DEAD.** Regulations.gov exposes
an agency-populated "Duplicate Comments" column, which reads like the ideal
label. It is not populated: over the full bulk corpus of **11,698,149** comment
rows it is non-null on 4,615,645 and **> 1 on 6,514 rows (0.06%)**. Agencies do
not fill it in. Rejected — measurement artifact, not phenomenon.
Artifact: `sk3:/lfs/skampere3/0/alexspan/outputs/nc_cosign/cosign_scan_stats.json`.

**(b) The shipped near-duplicate cluster mappers — REJECTED AS PRIMARY, kept as
sensitivity.** The upstream pipeline
(`regulations-demo/.../scripts/minhash_comment_deduping.py`) clusters each
docket's submissions by canonical text (exact-text pre-dedup, then MinHash
Jaccard ≥ 0.8 over unique texts, expanded back to every document id) and ships
`public_submission_all_text__dedup_mapper.csv{,.gz}`. Cluster size looks like
exactly the co-sign count. **Two landmines, both caught by ground-truthing
rather than by reading the code:**

- The uncompressed `.csv` sibling is a **stale pre-expansion artifact** (Mar
  5–18) holding roughly one row per unique *text* rather than per submitter.
  A first pass built on it (ice_2019_2020: 29,955 rows vs 47,170 in the `.gz`)
  and additionally **over-merges** through MinHash transitive chaining: it
  assigns document `AMS-NOP-17-0031-37974` to an **8,501-member cluster** when
  that text has **exactly one copy** in the docket.
- The authoritative `.csv.gz` (Mar 30, post-expansion) is structurally sound but
  has **partial coverage that varies by directory**. Ground truth on docket
  `AMS-NOP-17-0031`: the docket holds **47,108** documents including one
  **8,258-member byte-identical** text family; the `.csv.gz` mapper covers only
  **30,661** of them and its largest cluster is **1,879**. Any threshold on it
  is therefore contaminated by *which directories the dedup job finished*.

Because coverage is non-uniform across rows, using it as the primary y would
make the positive rate a function of a pipeline nuisance. It is retained only
as the `y_nearby` sensitivity arm, with its partial coverage stated.

### 1.4 What was built instead

`recount_cosign.py` recomputes the co-sign count **exactly**, from the current
`public_submission_all_text.csv`, over **6,803,623 documents** in the **1,814
dockets** the population touches: normalise (lowercase, collapse whitespace),
md5, count identical texts within docket. Complete and uniform coverage —
**9,521 of 9,524** scored documents located. This is unimpeachable and
auditable; it is also *conservative*, since adoptions that personalise the
template are split off (exact-pos 342 vs gz-near-pos 386, overlap 154). That
conservatism is a false-negative bias at the small end, which attenuates rather
than inflates every AUC below, and the `y_nearby` arm bounds it.

### 1.5 Unit of analysis

**One row per (docket, normalised text).** Co-signers of one template are
duplicate texts carrying an identical y; several of them would be duplicate
rows. Only **1** exact collision existed (the sample is a thin per-docket draw),
resolved by lowest-sha1 doc_id.

**Residual measured, not assumed.** Exact collapse cannot catch co-signers whose
texts differ slightly, so the population was screened for them directly:
**21 pairs** (of 9,520 rows, ≈0.2%) sit in the same docket with text similarity
> .85, several at 1.0 on their first 1,500 characters. Example: three rows in
`APHIS-2015-0093` share an identical 1,500-char prefix yet each carries
`cosign_count` 2 or 3, i.e. one advocacy campaign is being split into several
small exact-duplicate families by a personalised closing. Two consequences,
both bounded:
- *Duplicate-row inflation*: 0.2% of rows — negligible for any AUC here.
- *Under-count of the y*: real, and this is the concrete face of the
  conservatism noted in §1.4. It biases toward false negatives, attenuating
  rather than inflating; `y_nearby` (§6b) is the arm that bounds it.

### 1.6 The `year` landmine — checked, not inherited

The charge warned that `year` is a label leak for the *responded* y in some N&C
jsonls. Checked in this cell's own files: the 27 V extractors
(`aggregate_vat_nc.V_NAMES`) are **purely text-derived and contain no date
field**, and no date column enters V, A, VA, or the dense arm. `posted` is
carried in the population file as metadata for later nuisance work only.

---

## 2. Reuse log (reuse-before-rebuild)

Inventory ran before any build. What was reused verbatim, and what had to be
built:

| component | source | reuse |
|---|---|---|
| Population | `nc_scores_shard0..4.npz` + `nc_scores_unmatched.npz` | **100%** — the identical 9,521-row A/V-scored N&C universe |
| **A bank (198 rubrics, Gemma-4-31B, pre-GEPA)** | same npz | **100% — ZERO new judging.** 99.97% of the scored doc_ids (9,521/9,524) join to the bulk index, so the new y attaches to already-scored comments exactly as the charge's reuse test requires |
| V features (27) | `aggregate_vat_nc.v_features` | 100% — recomputed on the same texts, same extractors |
| Layer-1 estimators | `nc_layer1_stack.py` (`clean_cols`, `linear_oof`, `gbm_oof`, both bootstraps, `outer_folds`) | 100% — imported, not reimplemented, so the cell is comparable to nc responded/outcome/agree |
| Split bucketer | `datasets/patents/build_dense_standard_claimfell.py::stable_hash_bucket_map` | 100% — imported |
| Dense recipe | `methods/dense/run_dense_standard_scaleupC.sh` + `score_eval_dense_v4.py` | 100% — frozen recipe, no flags added |
| **y itself** | — | **BUILT** (§1.4). The only genuinely new measurement, and it is metadata arithmetic — no model, no judge, no label leak |

Net: the cell cost **one CPU text-scan and one dense training run**. No judging
batch was needed, therefore no anchor battery applies to this build (the anchor
rule binds judging batches; the frozen A-bank it reuses carried its own).

---

## 3. Splits

Docket-grouped stable-hash 80/10/10, pos-rate matched via the patents bucketer
(the patents lesson: a row-count-only bucketer let group size correlate with y
and produced a train/eval domain shift).

| split | n | dockets | pos rate |
|---|---|---|---|
| train | 7,617 | 814 | .03597 |
| eval | 951 | 555 | .03575 |
| test | 952 | 445 | .03571 |
| **all** | **9,520** | **1,814** | **.03592** |

Pos rates match to 3 decimal places. No seeded shuffle; sha1 order only.

---

## 4. Layer-1 ledger (CPU, landed 2026-08-08)

`methods/taste_decomposition/nc_cosigning_layer1.py --y cosign` →
`results/nc_cosigning_layer1_cosign.json`. FIRST-FIT cell (no prior V+A stack
of this construction ⇒ no reproduction gate; press-verdict precedent). n=9,520,
342 positives, 1,814 dockets, V=27c A=176c VA=203c after `clean_cols`.

| matrix | linear | nonlinear (mean seeds 0/1/2) | spread | nl − lin |
|---|---|---|---|---|
| V (27 surface) | **.6958** | **.7311** | .0089 | **+.0353** |
| A (176 of 198 rubrics) | .6661 | .7109 | .0107 | +.0448 |
| VA | .6870 | **.7472** | .0086 | **Δ_interact = +.0602** |

Δ_interact docket-level bootstrap (FREEZE CHANGE 3): mean **+.0691**,
95% CI **[+.0413, +.0967]**, P(>0) = 1.000. Second largest Δ_interact among the
N&C cells — responded +.0894, **co-signing +.0602**, agree +.0335,
outcome +.0164.

*Two nonlinear poolings appear in this note and both are reported deliberately:*
`VA_nl` = **mean of the three seeds' AUCs** (.7472, the FREEZE-CHANGE-1 ledger
quantity, used in §4 and every Δ), versus **AUC of the seed-mean OOF
probability** (.7571, the quantity the bootstraps and the grouped-transfer table
in §5 operate on, since those need one score vector). The .010 gap between them
is the usual ensembling gain, not a discrepancy.

**Two orderings that matter, both unusual:**

1. **V beats A** (.6958 vs .6661 linear; .7311 vs .7109 nonlinear). On every
   other N&C cell the 198-rubric quality bank is competitive with or ahead of
   the 27 surface extractors. Here the rubrics lose.
2. **Adding A to V makes the linear fit WORSE** (VA_lin .6870 < V_lin .6958).
   The articulated quality criteria are not merely weaker — as a linear block
   they dilute a signal the surface features already carry.

## 5. Grouped transfer — the finding: the pooled number is docket composition

Layer-2(a) table, mandatory here because this corpus's docket-identity leak is
severe (the design note's warning: identity-alone .86–.92 on the sibling cells).

| readout (nl = AUC of seed-mean OOF probs) | pooled | **within-docket** (156 mixed dockets, pair-weighted) |
|---|---|---|
| V_lin | .6958 | .5647 |
| V_nl | .7393 | .5531 |
| A_lin | .6661 | .5315 |
| A_nl | .7207 | .5396 |
| VA_lin | .6870 | .5594 |
| **VA_nl** | **.7571** | **.5330** |
| **docket identity ALONE** | **.6951** | — |

Docket identity alone scores **.6951** — as much as the entire VA_lin stack
(.6870) and 92% of VA_nl's pooled .7571.

**The within-docket number is weighting-dependent and must NEVER be
point-quoted.** Across weightings of the same 156 mixed dockets, VA_nl reads
pair-weighted **.5330** / n-weighted **.5963** / unweighted-mean **.6509** /
median **.7261**. The spread is not noise, it is composition: **99 of the 156
mixed dockets contain exactly ONE positive**, so their "AUC" is a single-item
rank statistic, and the unweighted mean is dominated by them.

Restricting to dockets where a within-docket comparison is actually powered
resolves the ambiguity in one direction:

| subset | dockets | pos | VA_nl pair-w | VA_nl unweighted | VA_lin pair-w |
|---|---|---|---|---|---|
| all mixed | 156 | 296 | .5330 | .6509 | .5594 |
| ≥3 positives | 30 | 143 | .5094 | .6985 | .5639 |
| ≥5 positives | 11 | 80 | **.4915** | .6316 | .5620 |
| docket n ≥ 50 | 13 | 65 | **.5014** | **.5293** | .5466 |

**Where the comparison is well-powered, both weightings agree and both sit at
chance (.49–.53).** The apparently healthy unweighted .65 is an artifact of the
99 one-positive dockets.

**Reading:** the pooled AUCs are overwhelmingly answering *"is this the kind of
rulemaking that attracts an organised campaign?"* — not *"which comment in this
docket did people sign on to?"* Any co-signing number quoted from this cell must
be quoted with its within-docket companion; the pooled figure alone would be a
docket-composition readout wearing a preference-articulation label.

**Program-convention figure, for the Layer-2 appendix table.**
`layer2_robustness.within_group_auc` uses a different rule from the pair-weighted
one above (groups with ≥ 20 rows, n-weighted). On that exact convention this
cell reads **VA_nl .5704 / VA_lin .5273** over 46 qualifying dockets (2,863
rows). Slotting it beside its sibling verdict cell:

| N&C cell | docket identity alone | within-docket VA_nl (min_n=20, n-weighted) |
|---|---|---|
| responded (verdict) | **.9156** | **.6749** (55 groups) |
| **co-signing (vote)** | **.6951** | **.5704** (46 groups) |

The two columns of the same field fail in *opposite* ways: `responded` has a far
larger docket leak yet the instruments still rank comments inside a docket;
`co-signing` has a smaller leak yet the instruments barely rank inside one. The
vote column's articulability is almost entirely **between**-docket — about which
rulemakings attract campaigns, not which texts get adopted.

**Scope limit (do not over-generalise this):** the *collapse to chance* is
specific to the exact-duplicate channel. On the `y_nearby` channel the
within-docket AUC holds up at ~.62 across every weighting and every powered
subset (§6b). What replicates across both channels is the weaker, still
important claim — **pooled ≫ within-docket, and docket identity alone (~.695)
accounts for most of the pooled number.** Coverage variation cannot explain the
difference, since gz coverage varies by agency-year *directory* and a docket
sits inside one directory, so within a docket the near-dup label is uniform.

## 6. The genre-detection caveat, measured

§1.2 flagged that predicting co-sign count from text may be advocacy-campaign
register detection. Measured rather than asserted (single-feature AUCs, primary y):

| V feature | alone-AUC | direction |
|---|---|---|
| v_kw_attach | .3744 | co-signed comments cite attachments LESS |
| v_caps_ratio | .3807 | less shouting/heading case |
| v_num_density | .3861 | fewer numbers |
| v_sent_count | .3993 | fewer sentences |
| v_char_len | .4182 | **shorter** |
| v_word_len | .4161 | shorter words |
| v_avg_sent_len | .5731 | longer sentences |
| v_first_person | .5920 | **more first-person** |

The profile of a co-signed comment — shorter, fewer numbers, no attachments,
more "I" — is the advocacy form-letter register, exactly. No single feature
dominates (largest |AUC − .5| = .126), so this is a composite register signal,
not one length proxy; and note `char_len` alone is **.4182**, i.e. INVERTED
relative to the usual length-confound direction, so the standard
"longer = better" nuisance story does not apply here.

Combined with §5 and with the large V-only interaction gain (V_nl − V_lin =
+.0353), this cell matches the design note's **SURFACE-nonlinearity signature**
("a large V_nl−V_lin alongside surface-feature-dominated structure marks SURFACE
nonlinearity → route to Layer 2(b)/Track B, not to a tacit-combination claim").
**Δ_interact = +.0602 here must NOT be read as a tacit combination rule over
articulated criteria.**

## 6b. Sensitivity: exact-duplicate vs near-duplicate co-signing

`nc_cosigning_layer1.py --y nearby` → `results/nc_cosigning_layer1_nearby.json`.
n=9,305, 386 positives, 1,772 dockets, identical instruments and protocol.

| quantity | `y_cosign` (exact, PRIMARY) | `y_nearby` (MinHash, partial coverage) |
|---|---|---|
| n / positives | 9,520 / 342 (3.59%) | 9,305 / 386 (4.15%) |
| V_lin | **.6958** | .6249 |
| A_lin | .6661 | **.6428** |
| VA_lin | .6870 | .6469 |
| V_nl | .7311 | .6773 |
| A_nl | .7109 | .6532 |
| VA_nl | .7472 | .6955 |
| Δ_interact [docket CI] | +.0602 [+.041, +.097] | +.0486 [+.024, +.092] |
| docket identity alone | .6951 | .6931 |
| within-docket VA_nl (pair-w, all mixed) | .5330 | .6250 |
| within-docket VA_nl (pair-w, n_pos≥3) | .5094 | .6153 |
| within-docket VA_nl (pair-w, docket n≥50) | .5014 | .6269 |
| char_len alone | .4182 (shorter) | .5272 (slightly longer) |

**Replicates across channels (quotable as structure):**
- Δ_interact is large and positive with heavily overlapping CIs (+.060 / +.049).
- Docket-identity-alone is essentially identical (.6951 / .6931) — as it must be,
  since it is a property of the docket structure, not of the label channel.
- Pooled ≫ within-docket in both.

**Channel-specific (never quote without naming the channel):**
- **The V-vs-A ordering FLIPS.** Exact: V .696 > A .666. Near-dup: A .643 > V .625.
- **The length signature flips** (.418 vs .527).
- **The within-docket collapse is exact-only** (.50 vs .62 on powered subsets).

**Mechanism this suggests:** byte-identical co-signing selects the *purest*
form-letter campaigns — short, templated, register-marked — which is why the 27
surface extractors beat the 198 quality rubrics there and why nothing is left to
rank once you are inside a single docket. Near-duplicate co-signing admits
personalised adaptations, which dilutes the register signature (rubrics recover
the lead) and leaves genuine within-docket ranking signal. The two channels are
measuring adjacent but distinguishable constructs; the primary is the
conservative one.

## 7. The cross-y contrast — the reason the cell was worth building

Same 9,520 texts, same instruments, two different crowds:

| | n | agency responded |
|---|---|---|
| co-signed (y=1) | 342 | **43.9%** |
| not co-signed (y=0) | 9,178 | **79.4%** |

φ = **−.160**. A comment the public signs onto is **~half as likely** to draw an
agency response. The regulatory field's VERDICT column (responded) and its VOTE
column point in **opposite directions on identical text** — which is the whole
point of the 3×N grid, and the N&C analogue of peer review's cross-y residual
spread. It also independently corroborates the earlier campaign finding
(form-letter family anti-predictive for responses) from the other side of the
join, without reusing that analysis.

## 7b. Spot-check of the y (dataset-first protocol)

Read before trusting. The extremes are exactly what the construct predicts:

| doc | count | opening |
|---|---|---|
| `BLM-2018-0001-77516` | **29,794** | "I am writing today in strong opposition to BLM's recent proposal to weaken crucial methane waste protections…" |
| `USCIS-2022-0016-26058` | **23,238** | "The proposed rule is contrary to well-established United States law regarding the right to seek asylum…" |
| `NOAA-NMFS-2023-0028-25182` | **18,097** | "I strongly support the proposed designation of critical habitat for Rice's whales…" |
| `AMS-AMS-15-0044-0002` | 1 | "The National Association of Egg Farmers (NAEF), representing in excess of two hundred farmers…" |
| `AMS-AMS-22-0027-1453` | 1 | "In places like Texas, massive amounts of cow manure are being produced every year…" |

Top counts are textbook advocacy-campaign letters; singletons are individual or
trade-association submissions. Median length: co-signed **1,454** chars vs
singleton **1,827** — consistent with the inverted `v_char_len` AUC in §6.

## 9. Dense arm (T) and the FULL LEDGER

Dense standard, frozen recipe, 3 seeds on sk3 GPU5 (Llama-3.1-8B LoRA r16/a32,
lr5e-5, batch16, max_len1024, 2 epochs, select-on-eval, **no** `class_weight_auto`).
Same-rows by construction: trained on the identical 9,520-unit population and
split. Ledger row = mean-probability ensemble (program convention).

| seed | eval AUC | test AUC |
|---|---|---|
| 42 | .8167 | .7490 |
| 1 | .8175 | .7932 |
| 2 | .8533 | .7050 |
| mean of seed AUCs | .8292 (spread **.0366**) | .7491 (spread **.0882**) |
| **ensemble (ledger row)** | **.8374** | **.7572** |

### The ledger

| quantity | value |
|---|---|
| V_lin / V_nl | .6958 / .7311 |
| A_lin / A_nl | .6661 / .7109 |
| VA_lin / VA_nl | .6870 / **.7472** |
| Δ_interact | **+.0602**, docket bootstrap [+.0413, +.0967], P(>0)=1.000 |
| T (eval / test) | **.8374 / .7572** |
| Δ_total, pooled convention | +.1504 |
| Δ_beyond, pooled convention | +.0902 |

### Same-rows Δ_beyond (stronger than the pooled convention)

Rather than differencing T (951/952 rows) against VA_nl pooled over all 9,520,
the grouped-OOF VA predictions are restricted to **exactly** the dense split's
rows. Every VA prediction used is still out-of-fold. `test` is the clean leg;
`eval` was consumed by checkpoint selection.

| leg | n / pos | VA_lin | VA_nl | T | **Δ_beyond** | paired docket bootstrap | P(Δ>0) | P(Δ>.02) |
|---|---|---|---|---|---|---|---|---|
| eval | 951 / 34 | .7517 | .7814 | .8374 | **+.0561** | [−.0116, +.1178] | .947 | .833 |
| test | 952 / 34 | .6034 | .6931 | .7572 | **+.0642** | [−.0848, +.1973] | .851 | .792 |

**T is pooled-only in this cell.** The 80/10/10 docket-grouped split leaves 8
mixed dockets in eval and **2** in test, so the within-docket collapse
documented for V/A/VA in §5 simply cannot be checked for T (within-docket T
reads .707 eval / .581 test on those 8 and 2 dockets — not interpretable).
Checking it would need a docket-stratified dense design (train on all dockets,
evaluate held-out comments inside seen dockets); that is a follow-up, not part
of this build.

## 9b. GATE VERDICT — **CONDITIONAL PASS, UNDERPOWERED**

Gate applied: the Layer-3 eligibility rule from the design note §4 — *Layer 3
only where Δ_beyond > .02 after Layer 1*. (No reproduction gate applies: this is
a first-fit cell. §11's fused-beats-bank rule is not yet triggered — no fusion
arm exists — though T > VA_nl on both legs, the expected direction.)

- **Point estimate clears on every reading**: +.0902 pooled convention,
  +.0642 same-rows test, +.0561 same-rows eval — all above .02.
- **The interval does not resolve it.** Both paired docket bootstraps straddle
  BOTH zero and the threshold: P(Δ_beyond > .02) = **.79 (test) / .83 (eval)**.
- **The number is noise-dominated.** The 3-seed T spread on test (**.0882**)
  is *larger than the Δ_beyond point estimate itself* (+.0642). With 34
  positives per held-out split this cell cannot resolve a .02 threshold.
  **Never point-quote this Δ_beyond.**

**Verdict: the cell is BUILT and its Layer-1/Layer-2(a) structure is solid, but
it should NOT be promoted to a closure campaign on this evidence.** Two
independent reasons, the second the stronger:

1. *Statistical*: the gate is unresolved at this n (P≈.8, not a pass).
2. *Substantive*: even taking the point estimate at face value, the residual
   lives on a readout that is ~92% docket composition (identity-alone .6951 vs
   pooled VA_nl .7571), and inside powered dockets the articulated instruments
   sit at chance (§5). A Track-A mining round here would discover criteria for
   *"which rulemaking attracts a campaign"*, not *"which text people signed"* —
   it would move the wrong number.

**Routing recommendation**: Layer 2(b) / Track B (spurious-feature mining and
discounting) FIRST — the surface-nonlinearity signature in §6 is exactly its
trigger — and any later Track A evaluated on the **within-docket** readout, not
the pooled one. If the cell is ever taken to closure, prerequisites are (a) a
docket-stratified dense design so T has a within-docket companion, and (b) more
positives per held-out split than 34.

## 10. What this changes for the 3×N grid

The regulatory field now has **all three preference-type columns built** —
verdict (responded), curation (change-made/outcome), vote (co-signing) — making
N&C the **second field after peer review** to support a cross-y decomposition
contrast. The contrast it supplies is sharper than expected, and it is a
*structural* one rather than a difference of residual size:

| | responded (VERDICT) | co-signing (VOTE) |
|---|---|---|
| V_lin vs A_lin | A wins (.624 > .596) | **V wins (.696 > .666)** |
| VA_nl | .7244 | .7472 |
| docket identity alone | .9156 | .6951 |
| within-docket VA_nl (min_n=20, n-wtd) | .6749 | **.5704** |
| Δ_interact | +.0894 | +.0602 |
| agency responds? | — | co-signed **43.9%** vs not-co-signed **79.4%** |

Same corpus, same instruments, same 9,520 texts: the two crowds want opposite
things, and the articulated instruments succeed on the verdict column *within*
a docket while failing there on the vote column. Whatever makes a comment
worth an agency's reply is articulable at the comment level; whatever makes a
comment worth signing is mostly a property of the rulemaking, not the text.

Long-pole queue effect: V8 is done, so the remaining vote-column builds are
**V6 SO votes > V9 tweet labeling > V7 patent forward-cites** in that order.

## 11. Artifacts

| what | where |
|---|---|
| y builder + full rationale in docstring | `datasets/notice-and-comment/cosigning/build_cosign_y.py` |
| population (one row per docket×text, y's, split) | `datasets/notice-and-comment/cosigning/cosign_population.jsonl` |
| build funnel + split/agency/docket audit | `datasets/notice-and-comment/cosigning/cosign_build_stats.json` |
| exact co-sign counts (recomputed) | `datasets/notice-and-comment/cosigning/cosign_counts_exact.json` (+`_stats.json`) |
| gz near-dup join (sensitivity channel) | `datasets/notice-and-comment/cosigning/scored_cosign_join_v2.json` |
| **STALE — DO NOT USE as a label** | `datasets/notice-and-comment/cosigning/scored_cosign_join.json` — the first pass, built on the pre-expansion `.csv` mappers (§1.3). Retained only as the audit trail for the over-merge landmine (never delete data). Any co-sign count must come from `cosign_counts_exact.json`, or from `..._join_v2.json` for the near-dup arm. |
| dense-standard bundle | `datasets/notice-and-comment/cosigning/dense_llama/cosign/{data.csv,split/}` |
| Layer-1 driver | `methods/taste_decomposition/nc_cosigning_layer1.py` |
| Layer-1 result (primary y) | `methods/taste_decomposition/results/nc_cosigning_layer1_cosign.json` |
| Layer-1 result (y_nearby sensitivity) | `methods/taste_decomposition/results/nc_cosigning_layer1_nearby.json` |
| T attach + final ledger | `methods/taste_decomposition/nc_cosigning_attach_T.py` → `results/nc_cosigning_ledger.json` |
| corpus scans (sk3) | `sk3:/lfs/skampere3/0/alexspan/outputs/nc_cosign/` (`scan_cosign.py`, `scan_cosign2.py`, `recount_cosign.py`, logs, `cosign_scan_stats.json`) |
| dense run (sk3) | `sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/cosigning/dense_llama/cosign/`, log `sk3:/lfs/skampere3/0/alexspan/nc_cosign_dense.log` |

Discipline check: no data deleted (append-only; both mapper vintages retained
side by side for the audit trail); `latex/` untouched; no judging batch ran, so
no anchor battery applies to this build; one GPU used (sk3 GPU5) and no process
not started by this build was signalled.
