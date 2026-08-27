# De-confounded thanks label (AoPS forum solutions)

*2026-06-11. Follow-up to `editorial_similarity_pilot.md` next-step #3:
remove the position/age/popularity confounds from `thanks_received` and
re-run the two key analyses. Script:
`scripts/deconfound_thanks_analysis.py`; label saved in
`thanks_deconfounded.parquet` (31,192 rows, col `thanks_resid`).*

## Method

Out-of-fold residualization (Math.SE v3.3 spirit, but residuals instead of
propensity deciles because thanks is heavy-tailed-count-like and the
confounds are continuous; decile matching kept as a robustness check):

- Fit `log1p(thanks)` on **confound features only** with
  HistGradientBoosting: `log1p(post_number)`, within-thread solution rank
  (raw + fraction), post age (years to corpus max post_time), days since the
  thread's first retained solution, `log1p(topic_num_views)`,
  `log1p(#solutions in thread)`.
- 5-fold **GroupKFold by problem** (a thread never predicts itself);
  OOF R² = **0.522** — more than half of log-thanks variance is pure
  position/age/popularity mechanics.
- `thanks_resid = log1p(thanks) − oof_pred`.

## Validation: confounds removed

| confound | raw pooled ρ | resid pooled ρ |
|---|---|---|
| post_number | −0.323 | **−0.004** |
| sol_rank (within thread) | −0.285 | +0.005 |
| post_age_years | +0.576 | +0.012 |
| days_since_first_sol | −0.415 | +0.018 |
| topic_num_views | +0.054 | −0.001 |
| within-problem post_number (mean ρ) | **−0.550** | **+0.061** |

Pooled confound correlations are dead. Within-problem there is a small
*positive* overshoot (+0.061, p=2e-7): the global model slightly
overcorrects position inside individual threads. Caveat: within-problem
residual ρ of ~±0.02–0.06 for any feature correlated with post order should
be read against this baseline.

## (a) Editorial similarity vs thanks — raw negative was pure artifact; de-confounded ≈ null

| analysis | raw log_thanks | thanks_resid |
|---|---|---|
| pooled ρ, sim_word | −0.093 (p=1e-60) | **+0.029** (p=5e-7) |
| pooled ρ, sim_char | −0.092 | +0.022 |
| within-problem mean ρ, sim_word | −0.035 (p=8e-5) | **+0.010 (p=0.21, null)** |
| within-problem mean ρ, sim_char | −0.034 | +0.005 (p=0.33) |
| AUC top-vs-bottom thanks quartile, sim_word | 0.432 | 0.528 |
| decile-matched robustness (raw thanks within oof-pred deciles) | — | sim_word +0.013, sim_char +0.006 |

The raw "editorial-likeness is *anti*-correlated with thanks" finding
disappears entirely. What remains is a whisper of positive pooled signal
(ρ≈+0.03, AUC 0.528) that does **not** survive within-problem contrasts —
consistent with cross-problem composition (computational contests have both
higher sims and different thanks economies) rather than per-solution taste.

## (b) Taste-within-correct — similarity still null; length is the one live style signal

Among 7,756 correct AIME/AMC solutions:

| feature vs label | raw pooled ρ | resid pooled ρ | resid AUC (top-vs-bottom quartile) | within-problem mean ρ (resid) |
|---|---|---|---|---|
| sim_word | −0.010 | −0.001 (p=0.91) | 0.500 | +0.020 (p=0.22) |
| sim_char | −0.017 | +0.001 | 0.500 | +0.026 (p=0.11) |
| len_chars | +0.121 | **+0.109** (p=7e-22) | **0.594** | +0.057 |
| n_display_math | −0.023 | +0.054 (p=2e-6) | 0.542 | +0.064 |
| latex_density | −0.106 | −0.010 (p=0.39) | 0.509 | +0.000 |
| num_edits | +0.019 | +0.026 (p=0.02) | 0.518 | +0.005 |

Bonus artifact check — **correctness vs thanks** (raw finding was a perverse
−0.076, "wrong answers get more thanks"):

| label | within-problem mean ρ | p |
|---|---|---|
| raw log_thanks | −0.076 | 1e-6 |
| thanks_resid | **+0.014** | 0.37 (null) |

The perverse raw correlation was the predicted age artifact (older posts =
more thanks + worse `\boxed` extraction); de-confounding kills it, though
thanks still doesn't *reward* correctness either.

## Interpretation (3 sentences)

De-confounding (OOF gradient-boosted residualization on position, age, and
thread popularity; R²=0.52) cleanly removes the position/age structure that
dominated raw thanks, and with it both spurious raw findings: the apparent
negative similarity→thanks correlation and the perverse negative
correctness→thanks correlation. Editorial-likeness remains **null** as a
taste signal — overall it retains only a tiny pooled correlation (+0.03,
AUC 0.53) that vanishes within problems, and among correct solutions it is
flat zero (AUC 0.50) — so TF-IDF closeness to the canonical solution
predicts correctness (0.262 vs 0.155, pilot) but not community appreciation
beyond it. The only style feature with real de-confounded signal among
correct solutions is **length** (ρ=+0.11, AUC 0.59; display-math count
+0.05), i.e., effortful, fuller writeups earn thanks — a thin but genuine
"articulable taste" layer, with the caveat that the within-thread
+0.06 overcorrection baseline bounds how much weight the small
within-problem positives can carry.
