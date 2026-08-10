# Layer-1 nonlinear stack — Gemma-4-31B-scored cells (creative writing + humor)

Date: 2026-08-05. Status: exploratory (design §6 declares the prereg freeze happens
after the peer-verdict pilot + this batch, before any *confirmatory* Layer-1 run).
Design spec: `notes/2026-08-05__taste-decomposition-design.md` §0 (ledger) and §1
(frozen protocol). Pilot + 7 protocol notes carried forward:
`notes/2026-08-05__layer1_peer_verdict_pilot.md`.

Code: `methods/taste_decomposition/layer1_gemma_cells.py` (new driver; the
`peer_verdict_layer1.json` pilot used `layer1_stack.py`, whose frozen GBM grid and
`shap_interactions` function this driver imports and reuses directly).
CPU only. No GPU, no new judging, nothing killed, `latex/` untouched.
sklearn **1.7.2**, shap **0.52.0**, Python 3.12.3 — noted per the design's caution
that Gemma-scored AUCs are sklearn-version-sensitive at the ~.006 level.

Terminology unpacked on first use (spell-out rule): **V** = verifiable/surface
text features; **A** = articulated-criteria (Gemma-4-31B judged rubric) block;
**VA** = V and A concatenated; **lin** = the task's existing linear aggregation
(logistic regression, grouped OOF); **nl** = nonlinear (HistGradientBoosting)
aggregation of the *same* matrix; **T** = dense-standard clean-eval AUC;
**Δ_interact** = VA_nl − VA_lin (interactions *of already-articulated criteria*,
not taste); **Δ_beyond** = T − VA_nl (the taste-eligible bound); **OOF** =
out-of-fold; **AUC** = area under the ROC curve; **GKF** = `GroupKFold`.

Two pre-existing "families" of linear pipeline were mirrored EXACTLY (imported
directly from the published scripts, not re-typed, so the gate calls the literal
production code):

- **family1** (`datasets/va_gemma_banks/readout_va_gemma.py`): `SimpleImputer
  (median, add_indicator=True)` + `StandardScaler` + `LogisticRegression(C=1,
  solver='liblinear', max_iter=2000, random_state=20260728)`, `GroupKFold(5)`.
  Cells: cw_community, hashtagwars_verdict, style_inv_toptier.
- **family2** (`datasets/humor/caption_multiy/aggregate_captions_multiy.py`):
  `clean_cols` (median-impute + degeneracy drop, shared matrix for lin+nl) +
  `StandardScaler` + `LogisticRegression(C=1, max_iter=2000)`, `GroupKFold(5)`
  group=contest. Cells: cap_crowd, cap_finalist.

GBM (`VA_nl`, `V_nl`): `HistGradientBoostingClassifier`, frozen grid
`max_leaf_nodes ∈ {15, 31}`, `learning_rate=.06`, `max_iter=400` + early stopping
(`validation_fraction=.1`, `n_iter_no_change=20`), grid picked by inner
`GroupKFold(3)` **inside each outer train fold only**, same outer folds as the
linear gate (verified identical by construction: `GroupKFold` splits depend only
on the `groups` array, not on `X`/`y` values, so building folds once from the
same `groups` array used inside the gate call reproduces the exact partition).
Per **FREEZE CHANGE 1** (pilot §6), `V_nl`/`VA_nl` are reported as the **mean
over seeds {0,1,2}**, spread reported alongside. `Δ_interact`'s point estimate
and bootstrap CI use the **seed-0** OOF array specifically (matching the pilot),
not the seed-averaged one — see the seed-spread column before trusting a sign.

---

## 1. Gate table (tolerance ±.006)

| cell | feature block | published | reproduced | abs diff | pass |
|---|---|---:|---:|---:|:---:|
| cw_community | V | .6039 | .6084 | .0045 | PASS |
| | A | .6053 | .6086 | .0033 | PASS |
| | V+A | .6266 | .6301 | .0035 | PASS |
| hashtagwars_verdict | V | .5592 | .5550 | .0042 | PASS |
| | A | .6350 | .6304 | .0046 | PASS |
| | V+A | .6478 | .6419 | .0059 | PASS |
| style_inv_toptier | V | .6227 | .6204 | .0024 | PASS |
| | A | .6090 | .6090 | .0001 | PASS |
| | V+A | .6161 | .6174 | .0013 | PASS |
| cap_crowd | V | .5274 | .5274 | .0000 | PASS |
| | A | .6478 | .6478 | .0000 | PASS |
| | V+A | .6485 | .6485 | .0000 | PASS |
| cap_finalist | V | .6247 | .6247 | .0000 | PASS |
| | A | .6299 | .6299 | .0000 | PASS |
| | V+A | .6508 | .6508 | .0000 | PASS |

**All 15 gate checks across 5 cells PASS.** The `family2` (caption) cells
reproduce to machine precision because the driver imports `clean_cols`/`load_scores`
directly and both linear and GBM read the identical precomputed matrix; the
`family1` cells (CW, HashtagWars, Style Invitational) reproduce within .0006–.0059
— hashtagwars_verdict V+A (.0059) is the tightest margin, still inside the ±.006
tolerance, and is consistent with the ±.005-level sklearn-version sensitivity the
design flags for Gemma-scored cells (published numbers were generated on whatever
sklearn version wrote `RESULTS_gemma.md`; this run used 1.7.2).

---

## 2. Full ledger

All AUCs pooled OOF. `V_nl`/`VA_nl` = mean over seeds {0,1,2}; `spread` = max−min
over the 3 seeds. `Δ_interact` point + 95% CI is the **row-level paired bootstrap**
(2,000×, seed-0 GBM vs linear OOF arrays) — see caveat in §4.

| cell | V_lin | V_nl (spread) | A_lin | VA_lin | VA_nl (spread) | Δ_interact [95% CI] | T | Δ_beyond |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| cw_community | .6084 | .5893 (.0083) | .6086 | .6301 | .6207 (.0131) | −.0095 [−.0273, +.0162] | .7801 | **+.1594** |
| hashtagwars_verdict | .5550 | .5687 (.0135) | .6304 | .6419 | .6301 (.0075) | −.0118 [−.0392, +.0082] | — | n/a |
| style_inv_toptier | .6204 | .6607 (.0009) | .6090 | .6174 | .6651 (.0005) | **+.0477 [+.0345, +.0609]** | — | n/a |
| cap_crowd | .5274 | .5343 (.0009) | .6478 | .6485 | .6656 (.0036) | **+.0171 [+.0082, +.0224]** | .5631 | **−.1025** |
| cap_finalist | .6247 | .6165 (.0049) | .6299 | .6508 | .6800 (.0022) | **+.0292 [+.0078, +.0476]** | .6252 | **−.0548** |

Bootstrap detail (P(Δ_interact > 0), row-level, 2000 draws): cw_community .31,
hashtagwars_verdict .10, style_inv_toptier **1.00**, cap_crowd **1.00**,
cap_finalist **.996**.

**Reading.**
- **Null (CI includes 0):** cw_community and hashtagwars_verdict. Boosting the
  same V+A matrix buys nothing measurable over logistic regression on these two
  cells — same qualitative finding as the peer-verdict pilot.
- **Non-null, CI excludes 0, all POSITIVE:** style_inv_toptier (+.048, by far the
  largest), cap_finalist (+.029), cap_crowd (+.017). On these three cells the
  nonlinear stack of the *same, already-articulated* criteria measurably beats
  the linear aggregation — a genuine `Δ_interact` (tacit *combination rule* over
  articulated criteria), distinct from tacit *content* not yet articulated at all.
- **Δ_beyond** (only defined where T exists): cw_community +.1594 — Layer 1
  removes essentially nothing from the +.15 residual (`Δ_total`=.1500 → `Δ_beyond`
  =.1594, i.e. `Δ_interact` is slightly *negative* so `Δ_beyond` is marginally
  larger than `Δ_total`). cap_crowd and cap_finalist both have **negative**
  `Δ_beyond` (−.1025, −.0548): `VA_nl` now EXCEEDS `T` on both caption cells —
  the same "bank > dense" pattern already on record in
  `notes/2026-07-27__vat-run-registry.md` (finalist-B/crowd-C dense chain), now
  sharpened by the nonlinear stack: the articulated bank not only ties the dense
  model on these short texts, it clears it once nonlinear combination is allowed.
  hashtagwars_verdict and style_inv_toptier have no `T` yet ("none yet" in the
  task's cell table), so only `Δ_interact` is reportable — no `Δ_total`/`Δ_beyond`
  claim is made for those two.
- **V-only interaction** (`V_nl − V_lin`, not the primary ledger column but
  computed identically): cw_community −.0190, hashtagwars_verdict +.0137,
  style_inv_toptier **+.0403**, cap_crowd +.0069, cap_finalist −.0082. Style
  Invitational shows a large nonlinear gain even on the 19-feature V-only block
  (spread .0009, tiny — this is a real, stable effect, not seed noise), which is
  the first hint (confirmed by SHAP below) that the interaction there is
  substantially a *surface-feature* effect (length × punctuation), not a
  criterion-criterion synergy.

---

## 3. Interaction findings — top 2 |Δ_interact| cells

Ranked by |Δ_interact|: style_inv_toptier (.0477) > cap_finalist (.0292) >
cap_crowd (.0171) > hashtagwars_verdict (.0118) > cw_community (.0095). SHAP
screen (per pilot protocol note 6 — descriptive only, not evidence; the OOF
ledger above is the arbiter) run on the top 2.

Method: fit the frozen-grid model (`max_leaf_nodes=31`, seed 0) on all rows,
rank features by mean |SHAP|, refit on the top-15, exact `TreeSHAP` interaction
values on a 300-row subsample of that reduced model (`shap` 0.52.0).

### style_inv_toptier (Δ_interact = +.048, the largest of the five)

Off-diagonal (interaction) mass fraction in the top-15 model: **.549**.

| # | feature A | feature B | mean abs interaction |
|---|---|---|---:|
| 1 | Linguistic polish (A) | v_punctuation_density (V) | **.1089** |
| 2 | v_char_count (V) | v_punctuation_density (V) | .0811 |
| 3 | v_char_count (V) | v_uppercase_ratio (V) | .0765 |
| 4 | v_char_count (V) | Linguistic polish (A) | .0645 |
| 5 | v_char_count (V) | v_prompt_token_jaccard (V) | .0490 |
| 6 | Linguistic polish (A) | v_uppercase_ratio (V) | .0478 |
| 7 | v_char_count (V) | Reference or target recognizability (A) | .0468 |
| 8 | v_char_count (V) | v_flesch_reading_ease (V) | .0427 |
| 9 | v_char_count (V) | v_type_token_ratio (V) | .0377 |
| 10 | v_char_count (V) | Explanation discipline (A) | .0372 |

Top main effects: v_char_count (.384), Linguistic polish (.414),
Reference/target recognizability (.073), v_punctuation_density (.120).

**Reading.** This is the one cell where `Δ_interact` is both large and clearly
non-null, and the SHAP screen says why: `v_char_count` appears in 7 of the top
10 pairs, and the #1 pair is a surface feature (punctuation density) crossed with
the A-criterion that most directly restates length/polish ("Linguistic polish").
This looks like a **length/format-mediated interaction**, not a substantive
combination of two independent quality judgments — consistent with the large,
stable (spread .0009) `V_interact` (+.040) found above: most of the nonlinear
gain is already present in the V-only block before A is even added. Style
Invitational entries are short, punchy one-liners where length and punctuation
density are themselves highly diagnostic of "polish," so a tree can extract a
length-mediated threshold effect that a linear model, forced to use `char_count`
additively, cannot. This is a real Layer-1 finding (not spurious per se — length
is a legitimate surface feature, honestly reported as V, not A) but it should be
named "nonlinear surface-feature effect," not "tacit combination of articulated
judgment criteria," when this cell reaches Layer 3.

### cap_finalist (Δ_interact = +.029)

Off-diagonal (interaction) mass fraction in the top-15 model: **.526**.

| # | feature A | feature B | mean abs interaction |
|---|---|---|---:|
| 1 | v_char_len (V) | Cross-cultural translation and translatability (A) | **.0882** |
| 2 | v_char_len (V) | v_avg_word_len (V) | .0687 |
| 3 | Cross-cultural translation and translatability (A) | Quality over pandering and empty signaling (A) | .0669 |
| 4 | v_char_len (V) | Originality and creative novelty vs derivation (A) | .0625 |
| 5 | Cross-cultural translation and translatability (A) | v_avg_word_len (V) | .0610 |
| 6 | v_char_len (V) | v_apostrophe (V) | .0573 |
| 7 | v_avg_word_len (V) | Originality and creative novelty vs derivation (A) | .0549 |
| 8 | Temporal topicality and durability (A) | Anti-comedy and anticlimax (A) | .0545 |
| 9 | v_char_len (V) | v_word_len (V) | .0537 |
| 10 | v_char_len (V) | International translatability and language choice (A) | .0415 |

Top main effects: v_char_len (.295), Cross-cultural translation and
translatability (.278), v_digit (.104), v_apostrophe (.100).

**Reading.** Also length-mediated (`v_char_len` in 6 of 10 pairs, #1 pair again
crosses a length feature with the criterion most collinear with it — captions are
one-liners, so "translatability/portability" tracks brevity), but two pairs are
plausibly substantive: #3 (Cross-cultural translatability × Quality-over-pandering
— a caption that travels across cultures is judged more favorably conditional on
not relying on cheap signaling) and #8 (Temporal topicality × Anti-comedy — a
timely reference interacts with whether the joke avoids anticlimax). Unlike
style_inv_toptier, `V_interact` here is small and slightly *negative* (−.0082),
so the +.029 `Δ_interact` is NOT mostly a V-only surface effect — a larger share
of this cell's interaction gain plausibly involves genuine A-criterion synergy,
though the dominant single pair is still length-mediated. `Δ_beyond` is negative
here (−.055, bank > dense), so this cell does not need Layer 3 under the design's
`Δ_beyond > .02` gate regardless.

---

## 4. Deviations, caveats, protocol notes for the next cell

1. **Family split.** Two distinct existing linear pipelines were mirrored, not
   one — see header. `family1` cells use `SimpleImputer(add_indicator=True)`
   fit fresh **inside each outer train fold** (no leakage) before GBM sees it,
   so GBM's input dimensionality includes missingness-indicator columns, matching
   what the linear model saw at that fold. `family2` cells use a single
   `clean_cols`-imputed matrix shared globally by both models (this is a property
   inherited from the existing published protocol, same as the peer-verdict
   pilot's `vat_3y` matrix — global, not per-fold, imputation — not something
   this driver introduced).
2. **Bootstrap is row-level, not group-level** (`bootstrap_delta_interact.note`
   in every JSON). Groups vary sharply in coarseness across these 5 cells:
   cw_community is near-singleton (1,500 groups / 2,000 rows, row-level ≈ exact),
   but hashtagwars_verdict (40 hashtags), style_inv_toptier (316 weeks), and both
   caption cells (~223–227 contests) are much coarser. A group-level bootstrap
   would likely widen the style_inv_toptier / cap_crowd / cap_finalist CIs
   somewhat; the current CIs for those three are wide enough to still clear zero
   comfortably (narrowest bound style_inv_toptier +.0345, cap_crowd +.0082,
   cap_finalist +.0078) but a group-level re-check before any confirmatory quote
   is the right next tightening step, per the same logic as pilot protocol note 1.
3. **`Δ_interact` uses seed-0 OOF, not the seed-mean OOF**, for both the point
   estimate quoted alongside the bootstrap and the CI itself — this matches the
   peer-review pilot's convention. The seed-mean AUC (`VA_nl_mean` in the ledger)
   differs slightly from `VA_lin + seed-0 Δ_interact` in every cell; always read
   `VA_nl_mean` for the headline ledger number and `Δ_interact`'s own value for
   the CI-bearing quantity — they are two different (close) estimators of the
   same thing, both reported so neither is silently preferred.
4. **A_nl was not run** (design requires only "every V and V+A calculation";
   A-alone nonlinear was in the pilot for completeness but is not required here
   and was dropped to keep the caption cells' larger A blocks — 329–348 columns
   after degeneracy dropping, vs. 45 for peer-review — inside a reasonable
   runtime). `A_lin` is still reported (from the gate) for context.
5. **cap_crowd / cap_finalist `T` values are as supplied in the task's cell
   table** (.5631, .6252) — these are described there as the same-rows-corrected
   dense numbers (FREEZE CHANGE 2 in the design doc); this run did not itself
   verify same-rows-ness, it only consumed the given T.
6. **hashtagwars_verdict / style_inv_toptier have no `Δ_beyond`** (no `T` yet per
   the task's cell table) — only `Δ_interact` is reported, as instructed.
7. **Runtime.** 115s (cw_community) to 249s (cap_crowd, largest: n=10,893,
   VA=364 columns) per cell; ~870s (14.5 min) total for all 5 cells' gate + 3
   GBM seeds × 2 matrices + bootstrap + 2 SHAP screens, all on a laptop CPU.

## 5. Artifacts

- `methods/taste_decomposition/layer1_gemma_cells.py` — driver (new).
- `methods/taste_decomposition/results/cw_community_layer1.json`
- `methods/taste_decomposition/results/hashtagwars_verdict_layer1.json`
- `methods/taste_decomposition/results/style_inv_toptier_layer1.json` (includes SHAP)
- `methods/taste_decomposition/results/cap_crowd_layer1.json`
- `methods/taste_decomposition/results/cap_finalist_layer1.json` (includes SHAP)
