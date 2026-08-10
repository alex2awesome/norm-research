# Layer-1 nonlinear stack — peer-review CURATION (oral/spotlight) + REVEALED (citation-pct)

Date: 2026-08-05. Status: Wave 2 of the taste-residual decomposition rollout
("same-matrix extensions" per `notes/2026-08-05__taste-decomposition-design.md`
§4b). Both cells reuse the pilot's matrix and mirrored linear pipeline; design
spec: §0 (quantity ledger) and §1 (frozen Layer-1 protocol); freeze changes
carried in from §6/pilot outcomes: **VA_nl (and V_nl) = mean over seeds
{0,1,2} with spread** (freeze change 1) and **Δ_interact's 95% CI = GROUP-LEVEL
(ntitle-resampled) bootstrap, not row-level** (freeze change 3).

Code: `methods/taste_decomposition/layer1_peer_curation_revealed.py` — imports
and reuses `methods/taste_decomposition/layer1_stack.py`'s data loader
(`load_cell`, generic over cell name), fold builder (`outer_folds`), linear/GBM
OOF fitters (`linear_oof`, `gbm_oof`), and SHAP screen (`shap_interactions`)
directly rather than re-typing them; adds only the seed-mean orchestration and
the group-level bootstrap that postdate the pilot.

Results: `methods/taste_decomposition/results/peer_curation_layer1.json`,
`methods/taste_decomposition/results/peer_revealed_layer1.json`. OOF arrays:
`*_va_nl_oof_seed0.npy`, `*_va_nl_oof_mean3.npy` per cell.

CPU only. Total runtime 265s (140s curation + 125s revealed) on a laptop. No
GPU, no new judging, nothing killed, `latex/` untouched.

Terminology (spelled out once, per the spell-out rule): **V** = verifiable/
surface feature block (17 cheap text features); **A** = articulated criterion
block (154-rubric A bank, degeneracy-guarded down per cell); **VA** = V and A
concatenated; **lin** = existing linear aggregation (`aggregate_3y.py`'s
`StandardScaler + LogisticRegression(C=1)`, `GroupKFold(5)` on `ntitle`);
**nl** = nonlinear (HistGradientBoostingClassifier) aggregation of the SAME
matrix, same folds; **T** = dense clean-eval AUC (Llama-3.1-8B LoRA,
title-grouped eval split); **Δ_interact** = VA_nl − VA_lin (interaction gain
*of already-articulated criteria*, not taste); **Δ_beyond** = T − VA_nl (the
only part eligible to be called taste); **OOF** = out-of-fold; **AUC** = area
under the ROC curve; **GBM** = gradient-boosted (tree) model.

---

## 1. Gate — linear reproduction (both PASS, machine precision)

Mirrored pipeline (`aggregate_3y.py::rung_row`, same `union_scores.npz`,
`clean_cols` degeneracy guard, `GroupKFold(5)` on `ntitle`) reproduces
`vat_3y_results.json`'s `rungs.curation` / `rungs.revealed` to `abs_diff = 0.0`
on V, A, VA, and n — not merely within the ±.005 tolerance.

| cell | n | V | A | VA | gate |
|---|---|---|---|---|---|
| curation | 7,941 | .5505 | .5628 | .5669 | **PASS** (0.0 diff, all) |
| revealed | 2,387 | .7051 | .7509 | .7606 | **PASS** (0.0 diff, all) |

Grouping is `ntitle`, identical to `aggregate_3y.py`. Both cells have
**groups == n** (7,941 groups / 7,941 rows; 2,387 groups / 2,387 rows) — unlike
the verdict pilot's near-vacuous 5,999/6,030, grouping here is exactly vacuous
(one paper per row, no duplicate titles survive each cell's label filter), so
`GroupKFold(5)` is plain 5-fold and the group-level vs row-level bootstrap
distinction below is not expected to matter much for CI width (confirmed: the
two bootstrap CIs are nearly identical in both cells).

Degeneracy guard: curation keeps 93 of 154 A columns (110 VA total), revealed
keeps 68 of 154 (85 VA total) — both cells see fewer surviving A columns than
the verdict pilot's 84, reflecting each cell's own NA/degeneracy pattern.

---

## 2. Ledger (design §0 names)

| symbol | curation | revealed |
|---|---|---|
| `V_lin` | .5505 | .7051 |
| `V_nl` (mean seeds 0/1/2) | .5314 | .6947 |
| `V_nl` spread | .0023 | .0065 |
| `A_lin` | .5628 | .7509 |
| `A_nl` (seed 0 only, optional) | .5549 | .7607 |
| `VA_lin` | .5669 | .7606 |
| `VA_nl` (mean seeds 0/1/2) | **.5588** | **.7667** |
| `VA_nl` spread | .0035 | .0028 |
| `T` (eval / test) | .593 / .588 | .871 / .896 |
| `Δ_total` = T(eval) − VA_lin | +.0261 | +.1104 |
| `Δ_interact` = VA_nl − VA_lin | **−.0081** | **+.0062** |
| `Δ_interact` group-level 95% CI (PRIMARY) | [−.0145, +.0088], P(>0)=.31 | **[+.0003, +.0199]**, P(>0)=.98 |
| `Δ_interact` row-level 95% CI (secondary) | [−.0151, +.0086], P(>0)=.30 | [+.0005, +.0212], P(>0)=.98 |
| `Δ_beyond` = T(eval) − VA_nl | +.0342 | +.1043 |
| `V_interact` = V_nl − V_lin | −.0191 | −.0104 |

**Reading, curation.** Δ_interact is negative and its group-level CI straddles
zero (P(Δ>0)=.31) — same null pattern as the verdict pilot: boosting the same
93+17 articulated features buys nothing over logistic regression, and the
point estimate is slightly negative. Both V_nl and A_nl trail their linear
counterparts too (V_interact −.019), consistent with the tree spending
capacity on redundancy rather than finding real synergy (see §3). Δ_beyond
(+.034) is essentially the whole of the modest Δ_total (+.026) — actually
slightly larger than Δ_total, i.e. the nonlinear stack does not shrink the gap
to T at all here, it very slightly widens it.

**Reading, revealed.** Δ_interact is small but its group-level 95% CI is
**entirely above zero** ([+.0003, +.0199], P(>0)=.98) — the one cell so far in
this rollout where the nonlinear stack detectably beats the linear one, though
the margin (.006) is barely outside the ±.0028 seed-spread band and the CI
lower bound sits at +.0003, a hair above zero. `V_interact` is *negative*
(−.0104: V_nl trails V_lin) while `A_nl` (single seed) exceeds `A_lin` by
about +.0098 — per the design's rollout-observation routing rule (§6, "only
A-side synergy with small V-only gain... is a candidate genuine interaction"),
this pattern (small/negative V-only gain + A-side gain) marks revealed as a
**candidate genuine interaction**, not the "surface nonlinearity from
length-feature overfitting" pattern seen in curation and the verdict pilot.
That said the effect is small (.006 AUC) and the A_nl side of this reading is
a single-seed number, not the frozen mean+spread protocol, so it should be
read as suggestive, not confirmed to the same standard as V_nl/VA_nl. Δ_beyond
(+.104) is essentially unchanged from Δ_total (+.110) — the nonlinear stack
closes only about 6% of the total gap to T.

---

## 3. SHAP interaction screen (descriptive only) — curation (larger |Δ_interact|)

Per protocol item 4, the SHAP screen ran only for **curation**
(|Δ_interact|=.0081 > revealed's .0062). Method: `shap` 0.52 `TreeExplainer`,
same screened protocol as the pilot (fit frozen-grid model on all 7,941 rows,
rank by mean|SHAP|, refit top-15, exact TreeSHAP interactions on a 300-row
subsample). Off-diagonal mass fraction .593 (routine for a mildly non-additive
fit, not evidence of large interaction — see pilot note on why ~.5 is typical).

Top-10 pairs by mean |interaction|:

| # | feature A | feature B | mean abs interaction |
|---|---|---|---|
| 1 | A: Dataset provenance, composition, and representativeness | `v_num_density` | .0290 |
| 2 | `v_char_len` | `v_word_len` | .0269 |
| 3 | A: Novelty and significance of the contribution | `v_kw_code` | .0265 |
| 4 | `v_avg_word_len` | `v_word_len` | .0238 |
| 5 | `v_avg_sent_len` | `v_sent_count` | .0234 |
| 6 | `v_avg_word_len` | `v_sent_count` | .0218 |
| 7 | A: Novelty and significance of the contribution | `v_avg_sent_len` | .0185 |
| 8 | A: Novelty and significance of the contribution | `v_num_density` | .0168 |
| 9 | `v_avg_word_len` | A: TRIPOD adherence for prediction model studies | .0168 |
| 10 | `v_avg_sent_len` | `v_word_len` | .0159 |

Top main effects: Novelty and significance (.109), Dataset provenance (.058),
`v_avg_word_len` (.046), `v_avg_sent_len` (.051), `v_char_len` (.032).

**What the interactions are.** Same character as the verdict pilot's screen:
the list is dominated by length-feature collinearity (#2, #4, #5, #6, #10 —
`char_len`/`word_len`/`avg_word_len`/`avg_sent_len`/`sent_count` against each
other) plus a couple of A×V redundancy pairs where a surface proxy and a
rubric criterion measure related things (#1 dataset-provenance × numeric
density, #3 novelty × code keyword). No pair here reads as two *distinct*
articulated criteria combining substantively (contrast the verdict pilot's #2,
novelty × proposal-coherence, at .046 — nothing in curation's top-10 reaches
that level between two A-criteria). Consistent with Δ_interact being null/
negative for this cell: the tree finds redundancy to prune, not synergy to
exploit, and it costs a little OOF AUC rather than buying any.

---

## 4. Protocol notes carried from the pilot, confirmed here

1. Group column `ntitle` generalizes cleanly to both cells (`load_cell` needed
   no changes — it already parametrizes on `{cell}.jsonl`).
2. Degeneracy guard applied identically to both models before either sees the
   matrix (curation 93/154 A-cols kept, revealed 68/154).
3. Seed spread (V_nl/VA_nl, seeds 0/1/2) stayed small in both cells (≤.0065),
   confirming the pilot's freeze-change recommendation to report the 3-seed
   mean rather than a single seed — curation's Δ_interact (−.008) is ~2.3×
   its VA_nl spread (.0035); revealed's Δ_interact (+.006) is ~2.2× its VA_nl
   spread (.0028). Both are close enough to the spread that the group-level
   bootstrap CI, not the point estimate alone, is what should be quoted.
4. Group-level (ntitle) vs row-level bootstrap CIs are nearly identical in
   both cells because `n_groups == n` here (no within-group correlation to
   correct for) — a useful confirmation that the group-level bootstrap
   machinery (mirrored from `nc_layer1_stack.py::bootstrap_delta_interact_docket`)
   behaves sanely when clustering is degenerate, not just when it bites hard
   (as it will on N&C's docket-grouped cells).
5. T population mismatch is inherited exactly as in the pilot: T is measured
   on the dense model's own title-grouped eval split, not on the smaller
   A/V-scored row population. Not re-litigated here.

---

## 5. Bottom line

| cell | gate | Δ_interact (group CI) | Δ_beyond | one-line read |
|---|---|---|---|---|
| peer curation | PASS | −.0081 [−.0145,+.0088], null | +.034 | no detectable interaction gain; residual is almost entirely Δ_beyond, same pattern as verdict pilot |
| peer revealed | PASS | **+.0062 [+.0003,+.0199]**, barely significant | +.104 | small but CI-positive interaction gain, driven by an A-side (not surface/V-side) pattern — candidate genuine interaction per the design's routing rule, though the effect is small and A_nl here is single-seed |

Caveat carried forward without reinterpretation: **peer revealed rides a topic
floor** (citation percentile is substantially topic-predictable per
`notes/2026-07-22__vat-paper-plan.md`), so its whole ledger — V, A, VA_lin,
VA_nl, and T — sits high in part because the construct itself leans on topic
popularity, not narrowly on paper quality; and T's test-split number (.896) is
separately flagged "optimistic" on a small n_eval=223 in the dense-chain
registry. Neither caveat changes the Δ_interact reading above (which lives
entirely inside the A/V-scored population, not T), but both bound how the
Δ_beyond number for revealed should be interpreted.
