# Layer-1 nonlinear stack — PILOT on peer-review VERDICT (accept/reject)

Date: 2026-08-05. Status: **exploratory pilot** (design §6 declares the peer-verdict
cell exploratory; the prereg freeze happens after this pilot, before confirmatory
cells). Design spec: `notes/2026-08-05__taste-decomposition-design.md` §0 (ledger)
and §1 (frozen Layer-1 protocol).

Code: `methods/taste_decomposition/layer1_stack.py`.
Machine-readable result: `methods/taste_decomposition/results/peer_verdict_layer1.json`.
OOF predictions of the nonlinear V+A model (seed 0):
`methods/taste_decomposition/results/peer_verdict_va_nl_oof_seed0.npy`.
CPU only, 80 s end-to-end. No GPU, no new judging, nothing killed.

Terminology unpacked on first use, per the spell-out rule:
**V** = verifiable/surface feature block (17 cheap text features);
**A** = articulated criterion block (154 LLM-judged rubric scores, the A bank);
**VA** = V and A concatenated; **lin** = the existing linear aggregation;
**nl** = the nonlinear (gradient-boosting) aggregation of the *same* matrix;
**T** = dense-standard clean-eval AUC (Llama-8B LoRA on raw text);
**Δ_interact** = VA_nl − VA_lin (interactions *of already-articulated criteria*);
**Δ_beyond** = T − VA_nl (the only part eligible to be called taste);
**OOF** = out-of-fold; **AUC** = area under the ROC curve.

---

## 1. Data and protocol actually used

| item | value |
|---|---|
| matrix | `datasets/peer-review/vat_3y/union_scores.npz` (local; not fetched from sk3) |
| raw shape | 14,307 abstracts × (17 V + 154 A) |
| cell rows | 6,030 verdict-labelled abstracts, positive rate .4982 |
| **group column** | **`ntitle`** (normalised paper title) — identical to `aggregate_3y.py::rung_row` |
| groups | 5,999 unique over 6,030 rows |
| feature counts after the degeneracy guard | V 17, A 84 (70 A columns dropped), VA 101 |
| A-bank NA rate | .652, median-imputed exactly as in `aggregate_3y.py::clean_cols` |
| folds | `GroupKFold(5)` on `ntitle`, **the identical fold objects reused by every model** (linear and GBM) |
| linear model | `StandardScaler + LogisticRegression(C=1, max_iter=2000)` |
| nonlinear model | `HistGradientBoostingClassifier`, `max_leaf_nodes ∈ {15,31}`, `learning_rate=.06`, `max_iter=400`, early stopping (`validation_fraction=.1`, `n_iter_no_change=20`), seed 0 |
| grid selection | inner `GroupKFold(3)` **inside each outer train fold only**; no eval-row information ever enters selection |
| readout | pooled AUC over the OOF predictions of all 6,030 rows — the same evaluation population the published linear numbers use |

---

## 2. Hard gate — linear reproduction (PASS)

The mirrored pipeline reproduces the published numbers to machine precision
(differences are 0.0, not merely within the ±.005 tolerance), which confirms that
explicit fold construction is byte-identical to `cross_val_predict` with the same
`GroupKFold(5)`.

| quantity | published (`vat_3y_results.json`) | reproduced | abs diff | gate (±.005) |
|---|---|---|---|---|
| V | .613 (.6128042) | .6128042 | .0000000 | **PASS** |
| A | .683 (.6834697) | .6834697 | .0000000 | **PASS** |
| V+A | .690 (.6896181) | .6896181 | .0000000 | **PASS** |
| n / pos rate | 6,030 / .4982 | 6,030 / .4982 | — | PASS |

---

## 3. Ledger (design §0 names), T = .753

| symbol | value | note |
|---|---|---|
| `V_lin` | **.6128** | published .613 |
| `V_nl` | **.6102** | −.0026 vs linear |
| `A_lin` | .6835 | published .683 |
| `A_nl` | .6813 | −.0022 vs linear |
| `VA_lin` | **.6896** | published .690 |
| `VA_nl` | **.6876** | seed 0 |
| `T` | .753 | dense clean-eval, registry `notes/2026-07-27__vat-run-registry.md` |
| `Δ_total` = T − VA_lin | **+.0634** | the number previously called "the residual" |
| `Δ_interact` = VA_nl − VA_lin | **−.0020** | 95% CI [−.0091, +.0050], P(Δ>0)=.29 |
| `Δ_beyond` = T − VA_nl | **+.0654** | the taste-eligible bound |

**Reading.** For this cell Layer 1 removes *nothing*: the gradient-boosted stack of
the same 101 articulated features does not beat — in fact very slightly trails —
the linear aggregation. Δ_interact is statistically indistinguishable from zero
(paired 2,000× bootstrap over the 6,030 OOF rows; Spearman ρ between linear and
GBM OOF scores = .877). Δ_beyond (+.065) is therefore essentially all of Δ_total
(+.063); the +.06 peer-verdict residual survives the first tightening layer intact
and is not explained by tacit *combination rules* over articulated criteria.

Because Δ_interact ≤ 0, the honest statement is "no detectable interaction gain,
bounded above by about +.005 at 95%", not "interactions are exactly zero".
Δ_beyond > .02, so this cell **qualifies for Layer 3** (articulation closure,
tracks 3A + 3B) under the design's own gating rule.

---

## 4. Sanity and robustness (all four requested checks)

**Seed sensitivity** (`VA_nl`, folds and grid identical, only `random_state` varies —
it drives the early-stopping validation split and the binning subsample):

| seed | VA_nl |
|---|---|
| 0 | .6876 |
| 1 | .6777 |
| 2 | .6828 |
| **spread** | **.0099** (mean .6827) |

The seed spread (.0099) is **five times larger than |Δ_interact| (.0020)**. This is
the load-bearing caveat for the whole programme: at n≈6K a single-seed GBM number
is only good to ±.005, so Layer 1 cannot resolve interaction gains smaller than
about .01 on a cell this size. Using the 3-seed mean instead of seed 0 gives
Δ_interact = −.0069, i.e. the sign of the (null) effect is stable but its magnitude
is not. **Recommendation for the prereg: report VA_nl as the mean over seeds 0–2
plus the spread, not a single seed.**

**Overfit check** (train-fold AUC at the selected grid point vs OOF AUC):

| matrix | mean train-fold AUC | OOF AUC | gap |
|---|---|---|---|
| V (17 feat) | .797 | .610 | .187 |
| A (84 feat) | .726 | .681 | .044 |
| V+A (101 feat) | .785 | .688 | .097 |

Sizeable gaps, largest on the V block — 17 mostly-continuous surface features
(lengths, densities) give a boosted tree ample room to memorise, which is exactly
why the inner CV keeps choosing the smaller `max_leaf_nodes=15` (chosen in 14 of
15 outer folds across the three matrices). The grid is doing its job; the flat
result is not a case of an unregularised model being crippled by variance, it is a
case of the signal in this matrix being close to linear.

**Fold-protocol check.** Confirmed: groups are `ntitle`, the same column
`aggregate_3y.py` passes to `GroupKFold`. Caveat worth carrying: with 5,999 groups
over 6,030 rows, title-grouping is nearly vacuous here (grouped CV ≈ plain 5-fold).
It will bite much harder on cells where the group is coarse (N&C docket, code repo),
so per-cell fold diagnostics should be reported, not assumed.

**NA-handling sensitivity** (non-primary, since the frozen rule is "same matrix"):

| variant | VA_nl |
|---|---|
| primary: cleaned + median-imputed, 101 cols | .6876 |
| HistGBM native NaN on the RAW 171 cols (no impute, no degeneracy drop) | .6887 |
| cleaned 101 cols + 123 rubric-NA indicator columns | .6929 |

Neither variant clears VA_lin = .6896 by more than the seed band. The NA-indicator
run (+.0033 over linear) is the only positive, and it is not an interaction of
articulated criteria — it is an *extra feature block* (whether a rubric was
applicable at all). Worth remembering as a small, cheap, honest source of signal,
but it does not belong in Δ_interact.

---

## 5. SHAP interactions (descriptive only)

Method: `shap` 0.52 `TreeExplainer`. Exact TreeSHAP interaction values over 101
features × 400 trees are O(F²) and not worth the wall-clock, so this is a
**screened** computation: fit the frozen-grid model (`max_leaf_nodes=31`) on all
6,030 rows, rank features by mean |SHAP|, refit on the top 15, and compute exact
interaction values on that reduced model over a 300-row subsample. Reported as a
screen, not an exhaustive search.

Off-diagonal (interaction) share of total attribution mass in the top-15 model: **.505**
— which sounds large, but Tree SHAP splits every pairwise term symmetrically across
two off-diagonal cells while main effects sit in one diagonal cell, so ~.5 is the
routine value for a mildly non-additive fit and should not be read as "half the
signal is interaction". The OOF numbers above are the arbiter, and they say the
non-additivity buys nothing out of sample.

Top-10 pairs by mean |interaction|:

| # | feature A | feature B | mean abs interaction |
|---|---|---|---|
| 1 | `v_kw_code` | A: Data/code/materials availability statement with access details | **.1681** |
| 2 | A: Novelty and significance of the contribution | A: Proposal coherence, feasibility, and potential impact | .0459 |
| 3 | `v_kw_code` | A: Novelty and significance of the contribution | .0342 |
| 4 | `v_kw_code` | `v_avg_sent_len` | .0337 |
| 5 | `v_char_len` | `v_word_len` | .0329 |
| 6 | A: Novelty and significance of the contribution | `v_char_len` | .0310 |
| 7 | A: Novelty and significance of the contribution | `v_word_len` | .0276 |
| 8 | `v_kw_novel` | A: Theoretical framing, coherence, and use of theory | .0263 |
| 9 | `v_char_len` | `v_avg_sent_len` | .0263 |
| 10 | `v_avg_word_len` | `v_kw_novel` | .0257 |

Top main effects for context: `v_kw_code` (.410), Novelty and significance (.289),
Proposal coherence/feasibility/impact (.115), Study design description (.100),
`v_char_len` (.091).

**What the interactions actually are.** The list is dominated by *redundancy*, not
synergy. #1 is nearly four times the size of anything else and pairs a surface
keyword counter (`v_kw_code`) with the rubric criterion that measures the same
construct (code/data availability) — two instruments for one signal, so the tree
spends structure on not double-counting them. The same pattern repeats at #8
(`v_kw_novel` × theoretical framing) and #3 (`v_kw_code` × novelty). Most of the
rest is length-feature collinearity (#5, #9, #4, #6, #7: `char_len`/`word_len`/
`avg_sent_len` against each other and against the novelty score). Only #2 —
novelty × proposal coherence — is a plausible *substantive* interaction between two
articulated quality criteria ("a bold claim is judged conditionally on whether the
plan behind it holds up"), and at .046 it is a fifth the size of the redundancy
term and buys no OOF AUC.

---

## 6. Protocol notes a second cell will need

1. **Group column is per-cell.** Here `ntitle`; report the group count against n and
   flag when grouping is near-vacuous (as here) or genuinely coarse.
2. **Degeneracy guard first, always.** 70 of 154 A columns are dropped before
   modelling. Both models must see the *same* post-guard matrix or Δ_interact is
   confounded with feature-set size.
3. **Seeds: use 0/1/2 and report the mean and spread.** Single-seed VA_nl at n≈6K
   carries ±.005; any Δ_interact smaller than the spread must be reported as null,
   not as a small positive.
4. **Early stopping uses a random 10% split inside each train fold.** Harmless here
   (groups ≈ unique) but on group-heavy cells it should be swapped for a grouped
   validation split, otherwise VA_nl is optimistically stopped.
5. **T population mismatch is inherited, not introduced.** T = .753 comes from the
   dense model's own title-grouped eval split (5,408 of the 53,339-row verdict
   corpus), while VA_lin/VA_nl live on the 6,030 A/V-scored rows. The existing VAT
   registry already compares .753 against VA .690, so Δ_beyond inherits exactly the
   convention already in use — but a same-rows dense rescore would remove the
   caveat, and should be listed in the prereg as the tightening step.
6. **SHAP is a screen, not evidence.** Quote it descriptively; the OOF ledger is the
   only thing that adjudicates whether interactions matter.
7. **Runtime budget.** 80 s per cell on a laptop CPU (3 matrices × nested CV + 2
   extra seeds + SHAP). Layer 1 across all seven cells in §4 of the design is a
   coffee break, not a compute project.

## 7. Bottom line

The peer-verdict residual is not an aggregation artefact. Boosting the same 154+17
articulated features buys nothing over logistic regression (Δ_interact = −.002,
CI [−.009, +.005]), so the +.063 gap to the dense model does not live in tacit
*combinations* of what has already been articulated. The taste-eligible bound is
essentially unchanged at Δ_beyond = +.065, and this cell clears the design's
Δ_beyond > .02 gate for Layer 3 articulation closure.
